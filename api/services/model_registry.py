"""
Model registry — loads and caches all ML models at startup.

Exposes a single ModelRegistry singleton used by both the /score
and /explain routers. Models are loaded once on first access and
held in memory for the lifetime of the process.

Models:
    sim_swap    LightGBM SIM swap detector (AUC 0.9922)
    gnn         FraudRingGNN federated checkpoint (AUC 0.6450)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from models.gnn.model import FraudRingGNN
from models.sim_swap.model import SimSwapDetector

logger = logging.getLogger(__name__)

SIM_SWAP_ARTIFACT_DIR = Path("models/sim_swap/artifacts")
GNN_ARTIFACT_PATH = Path("models/federated/artifacts/federated_best.pt")
GNN_FALLBACK_PATH = Path("models/gnn/artifacts_tuning")  # standalone best if no federated


class ModelRegistry:
    """Lazy-loaded model container. Thread-safe for reads after initialisation."""

    def __init__(self) -> None:
        self._sim_swap: SimSwapDetector | None = None
        self._gnn: FraudRingGNN | None = None
        self._gnn_device: torch.device = torch.device("cpu")
        self._gnn_hidden_dim: int = 64
        self._gnn_dropout: float = 0.2
        self.sim_swap_loaded: bool = False
        self.gnn_loaded: bool = False

    def load_all(self, device: str = "cpu") -> None:
        """Load both models. Called once at API startup."""
        self._gnn_device = torch.device(device)
        self._load_sim_swap()
        self._load_gnn()

    def _load_sim_swap(self) -> None:
        try:
            self._sim_swap = SimSwapDetector.load(SIM_SWAP_ARTIFACT_DIR)
            self.sim_swap_loaded = True
            logger.info(f"SIM swap model loaded  features={len(self._sim_swap.feature_names)}")
        except Exception as e:
            logger.warning(f"SIM swap model not loaded: {e}")

    def _load_gnn(self) -> None:
        # Prefer federated checkpoint; fall back to standalone tuning checkpoint
        ckpt_path: Path | None = None
        for candidate in (GNN_ARTIFACT_PATH, *sorted(GNN_FALLBACK_PATH.glob("*.pt"))):
            if Path(candidate).exists():
                ckpt_path = Path(candidate)
                break

        if ckpt_path is None:
            logger.warning("No GNN checkpoint found — GNN scoring disabled")
            return

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self._gnn_hidden_dim = ckpt.get("hidden_dim", 64)
            self._gnn_dropout = ckpt.get("dropout", 0.2)
            num_layers = ckpt.get("num_layers", 3)
            use_batchnorm = ckpt.get("use_batchnorm", True)

            self._gnn = FraudRingGNN(
                hidden_dim=self._gnn_hidden_dim,
                dropout=self._gnn_dropout,
                num_layers=num_layers,
                use_batchnorm=use_batchnorm,
            ).to(self._gnn_device)
            assert self._gnn is not None
            self._gnn.load_state_dict(ckpt["model_state_dict"])
            self._gnn.eval()
            self.gnn_loaded = True
            logger.info(
                f"GNN model loaded from {ckpt_path.name}  "
                f"hidden_dim={self._gnn_hidden_dim}  "
                f"auc={ckpt.get('global_test_auc', ckpt.get('test_auc', '?')):.4f}"
            )
        except Exception as e:
            logger.warning(f"GNN model not loaded: {e}")

    @property
    def sim_swap(self) -> SimSwapDetector:
        if self._sim_swap is None:
            raise RuntimeError("SIM swap model not loaded")
        return self._sim_swap

    @property
    def gnn(self) -> FraudRingGNN:
        if self._gnn is None:
            raise RuntimeError("GNN model not loaded")
        return self._gnn

    @property
    def gnn_device(self) -> torch.device:
        return self._gnn_device


# Module-level singleton — imported by routers
registry = ModelRegistry()
