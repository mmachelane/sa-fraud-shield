"""
Federated learning simulation entry point.

Runs a Flower in-process simulation with 5 SA bank clients, DPFedAvg strategy,
and central Gaussian DP noise. Logs all metrics to MLflow.

Usage:
    python -m models.federated.simulate [--rounds 20] [--epochs 5] [--noise 1.0]
    # or via Makefile:
    make federated

Key hyperparameters (edit at top of file or pass via CLI):
    --rounds        Number of federation rounds          (default: 20)
    --epochs        Local training epochs per round      (default: 5)
    --noise         DP noise multiplier                  (default: 1.0)
    --clip          L2 clip norm                         (default: 1.0)
    --hidden-dim    GNN hidden dimension                 (default: 64)
    --dropout       Dropout probability                  (default: 0.2)
    --lr            Local learning rate                  (default: 1e-4)
    --delta         DP delta                             (default: 1e-5)
    --device        Torch device (cpu / cuda)            (default: cpu)
    --seed          RNG seed                             (default: 42)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import flwr as fl
import mlflow
import torch

# Ensure project root is on sys.path when run as __main__
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.federated.bank_partitioner import partition_graph  # noqa: E402
from models.federated.client import FraudShieldClient  # noqa: E402
from models.federated.server import build_strategy  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

GRAPH_PATH: Path = Path("data/graphs/hetero_graph.pt")
ACCOUNTS_PATH: Path = Path("data/raw/accounts.parquet")
ARTIFACT_DIR: Path = Path("models/federated/artifacts")
EXPERIMENT_NAME: str = "federated-dp-fraudshield"


# ── Client factory ────────────────────────────────────────────────────────────


def _client_fn(
    shards: dict,
    hidden_dim: int,
    dropout: float,
    local_epochs: int,
    lr: float,
    clip_norm: float,
    device: torch.device,
) -> fl.client.ClientFn:
    """Return a Flower client_fn closure bound to the pre-built shards."""
    bank_names = list(shards.keys())

    def client_fn(cid: str) -> FraudShieldClient:
        bank = bank_names[int(cid)]
        return FraudShieldClient(
            bank_name=bank,
            shard=shards[bank],
            hidden_dim=hidden_dim,
            dropout=dropout,
            local_epochs=local_epochs,
            lr=lr,
            clip_norm=clip_norm,
            device=device,
        )

    return client_fn


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading graph and partitioning into bank shards…")
    shards = partition_graph(
        graph_path=GRAPH_PATH,
        accounts_path=ACCOUNTS_PATH,
        device="cpu",  # keep on CPU; clients move batches as needed
        seed=args.seed,
    )
    logger.info(f"Shards ready: {list(shards.keys())}")

    strategy = build_strategy(
        shards=shards,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        noise_multiplier=args.noise,
        clip_norm=args.clip,
        delta=args.delta,
        min_fit_clients=len(shards),
        min_evaluate_clients=len(shards),
        min_available_clients=len(shards),
        artifact_dir=ARTIFACT_DIR,
        device=device,
    )

    client_fn = _client_fn(
        shards=shards,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        local_epochs=args.epochs,
        lr=args.lr,
        clip_norm=args.clip,
        device=device,
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(
        run_name=f"fl-r{args.rounds}-e{args.epochs}-nm{args.noise}-clip{args.clip}"
    ):
        mlflow.log_params(
            {
                "rounds": args.rounds,
                "local_epochs": args.epochs,
                "noise_multiplier": args.noise,
                "clip_norm": args.clip,
                "delta": args.delta,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "lr": args.lr,
                "num_clients": len(shards),
            }
        )

        logger.info(
            f"Starting simulation: {args.rounds} rounds, {len(shards)} clients, "
            f"noise_multiplier={args.noise}, clip_norm={args.clip}"
        )

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(shards),
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1},
        )

        # Log final epsilon
        final_epsilon = strategy.epsilon
        mlflow.log_metric("final_epsilon", final_epsilon)
        logger.info(
            f"Simulation complete. Final privacy budget: ε={final_epsilon:.4f} (δ={args.delta})"
        )

        # Log best model artifact
        best_model_path = ARTIFACT_DIR / "federated_best.pt"
        if best_model_path.exists():
            mlflow.log_artifact(str(best_model_path))
            ckpt = torch.load(best_model_path, map_location="cpu", weights_only=False)
            logger.info(
                f"Best model: round={ckpt['server_round']}  "
                f"auc={ckpt['global_test_auc']:.4f}  ap={ckpt['global_test_ap']:.4f}"
            )
            mlflow.log_metrics(
                {
                    "best_global_auc": ckpt["global_test_auc"],
                    "best_global_ap": ckpt["global_test_ap"],
                    "best_round": ckpt["server_round"],
                }
            )

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated fraud-shield simulation")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64, dest="hidden_dim")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
