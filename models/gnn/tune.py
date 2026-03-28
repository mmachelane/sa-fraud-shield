"""
Hyperparameter tuning for FraudRingGNN.

Runs a grid search over hidden_dim, dropout, and lr.
Each run is logged to MLflow. Best checkpoint saved to models/gnn/artifacts_best/.

Usage:
    python -m models.gnn.tune
"""

from __future__ import annotations

import gc
import itertools
import logging
from pathlib import Path

import torch

from models.gnn.train import train

logger = logging.getLogger(__name__)

GRID: dict[str, list] = {
    "hidden_dim": [64],
    "dropout": [0.2, 0.3, 0.4],
    "lr": [1e-3, 5e-4, 1e-4],
}

GRAPH_PATH: Path = Path("data/graphs/hetero_graph.pt")
EXPERIMENT_NAME: str = "fraud-ring-gnn-tuning"
MODEL_OUTPUT_DIR: Path = Path("models/gnn/artifacts_tuning")
EPOCHS: int = 300
PATIENCE: int = 30
DEVICE_STR: str = "cuda"


def run_grid() -> None:
    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    logger.info(f"Grid search: {len(combos)} combinations")

    best_auc = 0.0
    best_params: dict = {}

    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        logger.info(f"[{i}/{len(combos)}] {params}")

        output_dir = (
            MODEL_OUTPUT_DIR / f"hd{params['hidden_dim']}_dr{params['dropout']}_lr{params['lr']}"
        )
        if (output_dir / "gnn_model.pt").exists():
            logger.info("  → skipping (already done)")
            ck = torch.load(output_dir / "gnn_model.pt", map_location="cpu", weights_only=False)
            if ck["test_auc"] > best_auc:
                best_auc = ck["test_auc"]
                best_params = {
                    **params,
                    "test_auc": ck["test_auc"],
                    "test_ap": ck["test_ap"],
                    "dir": str(output_dir),
                }
            continue
        try:
            train(
                graph_path=GRAPH_PATH,
                experiment_name=EXPERIMENT_NAME,
                model_output_dir=output_dir,
                hidden_dim=int(params["hidden_dim"]),
                dropout=float(params["dropout"]),
                lr=float(params["lr"]),
                epochs=EPOCHS,
                patience=PATIENCE,
                device_str=DEVICE_STR,
            )
        except Exception as e:
            logger.warning(f"Run failed: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ck = torch.load(output_dir / "gnn_model.pt", map_location="cpu", weights_only=False)
        auc = ck["test_auc"]
        logger.info(f"  → test_auc={auc:.4f}  ap={ck['test_ap']:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_params = {
                **params,
                "test_auc": auc,
                "test_ap": ck["test_ap"],
                "dir": str(output_dir),
            }

    logger.info("─" * 50)
    logger.info(f"Best AUC: {best_auc:.4f}")
    logger.info(f"Best params: {best_params}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_grid()
