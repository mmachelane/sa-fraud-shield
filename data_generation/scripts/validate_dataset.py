"""Validate a generated dataset for statistical sanity."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def validate(data_dir: Path) -> bool:
    tx_path = data_dir / "raw" / "transactions.parquet"
    accounts_path = data_dir / "raw" / "accounts.parquet"

    if not tx_path.exists():
        logger.error(f"Transactions file not found: {tx_path}")
        return False

    tx = pd.read_parquet(tx_path)
    accounts = pd.read_parquet(accounts_path)

    logger.info(f"Transactions: {len(tx):,}")
    logger.info(f"Accounts: {len(accounts):,}")

    fraud_rate = tx["is_fraud"].mean()
    logger.info(f"Fraud rate: {fraud_rate:.3%}")

    logger.info(
        f"Payment rail distribution:\n{tx['payment_rail'].value_counts(normalize=True).to_string()}"
    )
    logger.info(
        f"Merchant category distribution:\n{tx['merchant_category'].value_counts(normalize=True).head(10).to_string()}"
    )
    logger.info(f"Amount stats (ZAR):\n{tx['amount_zar'].describe().to_string()}")

    # Fraud breakdown
    fraud_tx = tx[tx["is_fraud"]]
    if len(fraud_tx) > 0:
        logger.info(
            f"Fraud type distribution:\n{fraud_tx['fraud_type'].value_counts().to_string()}"
        )

    checks = [
        # Transaction-level fraud rate is naturally very low (account-level SABRIC figure is ~2.3%)
        (
            0.0001 <= fraud_rate <= 0.05,
            f"Fraud rate {fraud_rate:.4%} out of expected range [0.01%, 5%]",
        ),
        (len(tx) > 0, "No transactions generated"),
        (len(accounts) > 0, "No accounts generated"),
        (tx["amount_zar"].min() > 0, "Negative amounts found"),
        (tx["transaction_id"].nunique() == len(tx), "Duplicate transaction IDs found"),
    ]

    passed = True
    for ok, msg in checks:
        if not ok:
            logger.error(f"FAIL: {msg}")
            passed = False
        else:
            logger.info(f"PASS: {msg.split(' out of')[0]}")

    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    ok = validate(Path(args.data_dir))
    exit(0 if ok else 1)


if __name__ == "__main__":
    main()
