"""CLI entry point: python -m data_generation.scripts.generate_training_data"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SA fraud dataset")
    parser.add_argument("--n-accounts", type=int, default=50_000)
    parser.add_argument("--n-days", type=int, default=180)
    parser.add_argument("--fraud-rate", type=float, default=0.023)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--no-loadshedding", action="store_true")
    args = parser.parse_args()

    from data_generation.dataset_generator import DatasetConfig, generate_dataset

    config = DatasetConfig(
        n_accounts=args.n_accounts,
        n_days=args.n_days,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
        include_loadshedding=not args.no_loadshedding,
    )

    generate_dataset(config, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
