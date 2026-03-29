"""
Federated server strategy factory and global evaluation function.

build_strategy() wires together DPFedAvg with:
  - Weighted FedAvg aggregation (num_examples from each client)
  - Global evaluation on the held-out test split of each shard
  - MLflow metric logging per round
  - Best global model saving (highest mean test AP across shards)
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from flwr.common import Parameters, Scalar, ndarrays_to_parameters
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import HeteroData

from models.federated.dp_strategy import DPFedAvg
from models.gnn.model import FraudRingGNN

logger = logging.getLogger(__name__)


def _get_initial_parameters(hidden_dim: int, dropout: float) -> Parameters:
    """Initialise a fresh GNN and return its weights as Flower Parameters."""
    model = FraudRingGNN(hidden_dim=hidden_dim, dropout=dropout, num_layers=3, use_batchnorm=True)
    ndarrays = [p.cpu().numpy() for p in model.state_dict().values()]
    return ndarrays_to_parameters(ndarrays)


def build_strategy(
    shards: dict[str, HeteroData],
    *,
    hidden_dim: int = 64,
    dropout: float = 0.2,
    noise_multiplier: float = 1.0,
    clip_norm: float = 1.0,
    delta: float = 1e-5,
    min_fit_clients: int = 5,
    min_evaluate_clients: int = 5,
    min_available_clients: int = 5,
    artifact_dir: Path = Path("models/federated/artifacts"),
    device: torch.device = torch.device("cpu"),
) -> DPFedAvg:
    """
    Build and return a configured DPFedAvg strategy.

    Args:
        shards:                Bank-name → HeteroData shards (with test masks)
        hidden_dim:            GNN hidden dimension
        dropout:               Dropout probability
        noise_multiplier:      DP noise multiplier
        clip_norm:             L2 update clip bound (must match client setting)
        delta:                 Target delta for (epsilon, delta)-DP
        min_fit_clients:       Minimum clients for a training round
        min_evaluate_clients:  Minimum clients for an eval round
        min_available_clients: Minimum clients that must be online
        artifact_dir:          Where to save the best global model
        device:                Torch device for server-side eval

    Returns:
        Configured DPFedAvg strategy instance
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_ap: list[float] = [0.0]  # mutable container for closure

    # ── Global evaluate_fn ───────────────────────────────────────────────────

    def evaluate_fn(
        server_round: int,
        parameters: list[np.ndarray],
        config: dict[str, Scalar],
    ) -> tuple[float, dict[str, Scalar]] | None:
        """Evaluate the global model on the test split of every shard."""
        ndarrays = parameters

        # Rebuild model from server weights
        model = FraudRingGNN(
            hidden_dim=hidden_dim, dropout=dropout, num_layers=3, use_batchnorm=True
        ).to(device)
        state_dict = dict(
            zip(
                model.state_dict().keys(),
                [torch.from_numpy(p.copy()) for p in ndarrays],
            )
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        shard_aucs: list[float] = []
        shard_aps: list[float] = []
        total_loss = 0.0
        total_examples = 0

        for bank, shard in shards.items():
            shard_device = shard.to(device)
            test_mask = shard_device["account"].test_mask
            y = shard_device["account"].y.float().to(device)

            n_fraud = int(y[test_mask].sum().item())
            n_clean = int((~y[test_mask].bool()).sum().item())
            pos_weight = torch.tensor(n_clean / max(n_fraud, 1), device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            with torch.no_grad():
                logits = model(shard_device)
                loss = criterion(logits[test_mask], y[test_mask]).item()
                proba = torch.sigmoid(logits[test_mask]).cpu().numpy()

            test_y = y[test_mask].cpu().numpy()
            try:
                auc = float(roc_auc_score(test_y, proba))
                ap = float(average_precision_score(test_y, proba))
            except ValueError:
                auc, ap = 0.0, 0.0

            n = int(test_mask.sum().item())
            shard_aucs.append(auc)
            shard_aps.append(ap)
            total_loss += loss * n
            total_examples += n

            logger.info(f"  {bank:<15} test_auc={auc:.4f}  test_ap={ap:.4f}  n={n}")

        mean_auc = float(np.mean(shard_aucs))
        mean_ap = float(np.mean(shard_aps))
        mean_loss = total_loss / max(total_examples, 1)

        logger.info(
            f"Round {server_round} global: loss={mean_loss:.4f}  "
            f"auc={mean_auc:.4f}  ap={mean_ap:.4f}"
        )

        # MLflow logging
        try:
            mlflow.log_metrics(
                {
                    "global_test_loss": mean_loss,
                    "global_test_auc": mean_auc,
                    "global_test_ap": mean_ap,
                    **{f"{b}_test_auc": a for b, a in zip(shards.keys(), shard_aucs)},
                    **{f"{b}_test_ap": p for b, p in zip(shards.keys(), shard_aps)},
                },
                step=server_round,
            )
        except Exception:
            pass  # MLflow may not be active outside simulate.py

        # Save best model
        if mean_ap > best_ap[0]:
            best_ap[0] = mean_ap
            save_path = artifact_dir / "federated_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "num_layers": 3,
                    "use_batchnorm": True,
                    "server_round": server_round,
                    "global_test_auc": mean_auc,
                    "global_test_ap": mean_ap,
                },
                save_path,
            )
            logger.info(f"  Saved best model → {save_path}  (ap={mean_ap:.4f})")

        return mean_loss, {
            "global_test_auc": mean_auc,
            "global_test_ap": mean_ap,
        }

    # ── Strategy ─────────────────────────────────────────────────────────────

    strategy = DPFedAvg(
        noise_multiplier=noise_multiplier,
        clip_norm=clip_norm,
        num_clients=len(shards),
        delta=delta,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=_get_initial_parameters(hidden_dim, dropout),
    )

    return strategy
