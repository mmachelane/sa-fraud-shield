"""
Federated learning simulation entry point.

Runs an in-process federation loop (no Ray required) with 5 SA bank clients,
DPFedAvg strategy, and central Gaussian DP noise. Logs all metrics to MLflow.

Usage:
    python -m models.federated.simulate [--rounds 20] [--epochs 5] [--noise 1.0]
    # or via Makefile:
    make train-federated

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

import mlflow
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

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


# ── In-process federation loop ────────────────────────────────────────────────


def _run_federation(
    clients: list[FraudShieldClient],
    strategy,  # DPFedAvg
    num_rounds: int,
) -> dict:
    """
    Manual federation loop — equivalent to Flower's start_simulation
    but runs entirely in-process without Ray.

    Each round:
      1. Distribute current global parameters to all clients
      2. Each client runs fit() → returns updated weights + num_examples
      3. Strategy aggregates (FedAvg + DP noise) → new global parameters
      4. Strategy evaluate_fn() tests the global model across all shards
      5. Each client runs evaluate() → local val metrics collected
    """
    # Pull initial parameters from strategy
    parameters = strategy.initial_parameters
    history: dict[str, list] = {"loss": [], "metrics": []}

    for server_round in range(1, num_rounds + 1):
        logger.info(f"── Round {server_round}/{num_rounds} ──────────────────")
        ndarrays = parameters_to_ndarrays(parameters)

        # ── Fit phase ────────────────────────────────────────────────────────
        fit_results = []
        for client in clients:
            updated_params, num_examples, fit_metrics = client.fit(ndarrays, config={})
            # Wrap as Flower FitRes-compatible tuple
            fit_results.append((updated_params, num_examples, fit_metrics))
            logger.info(
                f"  {client.bank_name:<15} trained  "
                f"loss={fit_metrics.get('train_loss', 0):.4f}  n={num_examples}"
            )

        # ── Aggregate (DPFedAvg + noise) ─────────────────────────────────────
        # Build the (ClientProxy-like, FitRes-like) tuples the strategy expects.
        # We use a simple namespace since DPFedAvg only reads .parameters and
        # .num_examples from FitRes, and doesn't use ClientProxy at all.
        from types import SimpleNamespace

        from flwr.common import Code, FitRes, Status

        flower_results = []
        for updated_params, num_examples, fit_metrics in fit_results:
            fit_res = FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters(updated_params),
                num_examples=num_examples,
                metrics={k: float(v) for k, v in fit_metrics.items()},
            )
            flower_results.append((SimpleNamespace(), fit_res))

        aggregated_params, agg_metrics = strategy.aggregate_fit(server_round, flower_results, [])
        if aggregated_params is None:
            logger.warning(f"Round {server_round}: aggregation returned None — skipping")
            continue
        parameters = aggregated_params
        logger.info(f"  Aggregated  epsilon={agg_metrics.get('epsilon', 0):.4f}")

        # ── Global evaluation (server-side, across all shards) ────────────────
        eval_ndarrays = parameters_to_ndarrays(parameters)
        eval_result = strategy.evaluate_fn(server_round, eval_ndarrays, {})
        if eval_result is not None:
            loss, eval_metrics = eval_result
            history["loss"].append(loss)
            history["metrics"].append(eval_metrics)

        # ── Client-side evaluate (local val metrics) ──────────────────────────
        for client in clients:
            val_loss, val_n, val_metrics = client.evaluate(eval_ndarrays, config={})
            logger.info(
                f"  {client.bank_name:<15} val_auc={val_metrics.get('local_val_auc', 0):.4f}  "
                f"val_ap={val_metrics.get('local_val_ap', 0):.4f}"
            )

    return history


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading graph and partitioning into bank shards…")
    shards = partition_graph(
        graph_path=GRAPH_PATH,
        accounts_path=ACCOUNTS_PATH,
        device="cpu",
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

    clients = [
        FraudShieldClient(
            bank_name=bank,
            shard=shards[bank],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            local_epochs=args.epochs,
            lr=args.lr,
            clip_norm=args.clip,
            device=device,
        )
        for bank in shards
    ]

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

        history = _run_federation(clients, strategy, num_rounds=args.rounds)

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
                    "best_round": float(ckpt["server_round"]),
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
