"""
GNN fraud ring detector — training script.

Usage:
    python -m models.gnn.train \
        --graph-path data/graphs/hetero_graph.pt \
        --experiment-name fraud-ring-gnn

Training details:
    - Full-batch (graph fits in RAM/VRAM)
    - BCEWithLogitsLoss with pos_weight to handle class imbalance (~0.5%)
    - Adam optimiser, ReduceLROnPlateau scheduler
    - Early stopping on val AUC
    - MLflow logging with local fallback
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from models.gnn.dataset import load_graph
from models.gnn.model import FraudRingGNN

logger = logging.getLogger(__name__)


def _mlflow_reachable(uri: str) -> bool:
    import socket
    import urllib.parse

    parsed = urllib.parse.urlparse(uri)
    host, port = parsed.hostname, parsed.port or 80
    try:
        socket.create_connection((host, port), timeout=2).close()
        return True
    except OSError:
        return False


def train(
    graph_path: Path,
    experiment_name: str,
    mlflow_uri: str = "http://localhost:5001",
    model_output_dir: Path = Path("models/gnn/artifacts"),
    hidden_dim: int = 64,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 200,
    patience: int = 20,
    device_str: str = "cpu",
) -> None:
    device = torch.device(device_str)
    logger.info(f"Device: {device}")

    # ── Load graph ────────────────────────────────────────────────────────────
    data, split = load_graph(graph_path, device=device)
    y = data["account"].y.float()

    train_mask = split["train"]
    val_mask = split["val"]
    test_mask = split["test"]

    # ── Weighted loss (pos_weight = n_clean / n_fraud in train set) ───────────
    n_train_fraud = int(y[train_mask].sum())
    n_train_clean = int((~y[train_mask].bool()).sum())
    pos_weight = torch.tensor(n_train_clean / n_train_fraud, device=device)
    logger.info(
        f"Train set: {int(train_mask.sum()):,} accounts | "
        f"{n_train_fraud} fraud | pos_weight={pos_weight.item():.1f}"
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FraudRingGNN(
        hidden_dim=hidden_dim, dropout=dropout, num_layers=3, use_batchnorm=True
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=10
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    if _mlflow_reachable(mlflow_uri):
        mlflow.set_tracking_uri(mlflow_uri)
    else:
        logger.warning(f"MLflow at {mlflow_uri} unreachable — using ./mlruns")
        mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(experiment_name)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_state: dict = {}

    with mlflow.start_run(run_name="gnn-full-batch"):
        mlflow.log_params(
            {
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "lr": lr,
                "epochs": epochs,
                "patience": patience,
                "pos_weight": round(pos_weight.item(), 2),
            }
        )

        for epoch in range(1, epochs + 1):
            # ── Train ──
            model.train()
            optimiser.zero_grad()
            logits = model(data)
            loss = criterion(logits[train_mask], y[train_mask])
            loss.backward()
            optimiser.step()

            # ── Validate ──
            model.eval()
            with torch.no_grad():
                logits = model(data)
                proba = torch.sigmoid(logits).cpu().numpy()

            val_proba = proba[val_mask.cpu()]
            val_y = y[val_mask].cpu().numpy()
            val_auc = roc_auc_score(val_y, val_proba)
            val_ap = average_precision_score(val_y, val_proba)

            scheduler.step(val_auc)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:>4}  loss={loss.item():.4f}  "
                    f"val_auc={val_auc:.4f}  val_ap={val_ap:.4f}"
                )

            mlflow.log_metrics(
                {"train_loss": loss.item(), "val_auc": val_auc, "val_ap": val_ap},
                step=epoch,
            )

            # ── Early stopping ──
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                epochs_no_improve = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch} (best={best_epoch})")
                    break

        # ── Test evaluation ───────────────────────────────────────────────────
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits = model(data)
            proba = torch.sigmoid(logits).cpu().numpy()

        test_proba = proba[test_mask.cpu()]
        test_y = y[test_mask].cpu().numpy()
        test_auc = roc_auc_score(test_y, test_proba)
        test_ap = average_precision_score(test_y, test_proba)

        logger.info(
            f"Test AUC: {test_auc:.4f}  AvgPrecision: {test_ap:.4f}  (best val epoch: {best_epoch})"
        )
        mlflow.log_metrics({"test_auc": test_auc, "test_ap": test_ap})
        mlflow.log_param("best_epoch", best_epoch)

        # ── Save ─────────────────────────────────────────────────────────────
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_output_dir / "gnn_model.pt"
        torch.save(
            {
                "model_state_dict": best_state,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "num_layers": 3,
                "use_batchnorm": True,
                "best_epoch": best_epoch,
                "test_auc": test_auc,
                "test_ap": test_ap,
            },
            model_path,
        )
        try:
            mlflow.log_artifact(str(model_path), artifact_path="model")
        except Exception as e:
            logger.warning(f"MLflow artifact upload skipped: {e}")
        logger.info(f"Model saved to {model_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train GNN fraud ring detector")
    parser.add_argument("--graph-path", default="data/graphs/hetero_graph.pt")
    parser.add_argument("--experiment-name", default="fraud-ring-gnn")
    parser.add_argument("--mlflow-uri", default="http://localhost:5001")
    parser.add_argument("--model-output-dir", default="models/gnn/artifacts")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train(
        graph_path=Path(args.graph_path),
        experiment_name=args.experiment_name,
        mlflow_uri=args.mlflow_uri,
        model_output_dir=Path(args.model_output_dir),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
