"""
One-shot patch: backfill loadshedding_active / loadshedding_stage
for SIM_SWAP fraud transactions in data/raw/transactions.parquet.

Normal transactions already have these fields set correctly.
Fraud transactions were generated without them (None) due to a generator bug.

Fix:
  - Join fraud tx → victim account → province
  - For each fraud tx timestamp + province, look up the loadshedding schedule
  - Set loadshedding_active and loadshedding_stage accordingly
  - Re-save the parquet
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def patch(data_dir: str = "data") -> None:
    data_path = Path(data_dir)
    tx_path = data_path / "raw" / "transactions.parquet"
    accounts_path = data_path / "raw" / "accounts.parquet"

    logger.info("Loading raw data...")
    tx = pd.read_parquet(tx_path)
    accounts = pd.read_parquet(accounts_path)

    tx["timestamp"] = pd.to_datetime(tx["timestamp"])

    fraud_mask = tx["is_fraud"] & (tx["fraud_type"] == "SIM_SWAP")
    n_fraud = fraud_mask.sum()
    logger.info(f"Found {n_fraud} SIM_SWAP fraud transactions to patch")

    # Build province lookup from accounts
    acc_province = accounts.set_index("account_id")["province"]

    # Add province to fraud transactions
    fraud_idx = tx[fraud_mask].index
    tx.loc[fraud_idx, "sender_province"] = tx.loc[fraud_idx, "sender_account_id"].map(acc_province)

    # Generate loadshedding schedules for all provinces
    from shared.constants import SAProvince
    from shared.utils.load_shedding import generate_mock_schedule

    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=180)

    logger.info("Generating loadshedding schedules for all provinces...")
    from shared.utils.load_shedding import LoadSheddingSchedule

    ls_schedules: dict[str, LoadSheddingSchedule] = {}
    for province in SAProvince:
        ls_schedules[province.value] = generate_mock_schedule(
            province_code=province.value,
            start_date=start_date,
            days=180,
            stage=2,
        )

    # Patch each fraud transaction
    logger.info("Patching loadshedding fields on fraud transactions...")
    patched = 0
    for idx in fraud_idx:
        row = tx.loc[idx]
        province_code = row["sender_province"]
        ts = row["timestamp"]

        ls_active = False
        ls_stage = None

        if province_code in ls_schedules:
            outage = ls_schedules[province_code].get_active_outage(ts)
            if outage:
                ls_active = True
                ls_stage = outage.stage

        tx.at[idx, "loadshedding_active"] = ls_active
        tx.at[idx, "loadshedding_stage"] = ls_stage
        patched += 1

    ls_on_fraud = tx.loc[fraud_idx, "loadshedding_active"].sum()
    logger.info(
        f"Patched {patched} fraud transactions — "
        f"{ls_on_fraud} ({100 * ls_on_fraud / patched:.1f}%) now have loadshedding_active=True"
    )

    tx.to_parquet(tx_path, index=False)
    logger.info(f"Saved patched transactions to {tx_path}")


if __name__ == "__main__":
    patch()
