"""
SIM swap model feature engineering.

Reads raw parquet files → computes per-transaction features → saves processed parquet.
All features use vectorized pandas operations (no row-by-row loops on large data).

Run standalone:
    python -m models.sim_swap.features --data-dir data/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_features(data_dir: Path | str = "data") -> pd.DataFrame:
    """
    Build SIM swap detection features from raw data.
    Saves to data/processed/sim_swap_features.parquet and returns the DataFrame.
    """
    data_dir = Path(data_dir)
    tx_path = data_dir / "raw" / "transactions.parquet"
    accounts_path = data_dir / "raw" / "accounts.parquet"
    ss_path = data_dir / "raw" / "sim_swap_events.parquet"

    logger.info("Loading raw data...")
    tx = pd.read_parquet(tx_path)
    accounts = pd.read_parquet(accounts_path)
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])

    logger.info(f"Loaded {len(tx):,} transactions, {len(accounts):,} accounts")

    # ── Account-level lookups ─────────────────────────────────────────────────
    acc_province = accounts.set_index("account_id")["province"]
    acc_balance = (
        accounts.set_index("account_id")["balance_zar"]
        if "balance_zar" in accounts.columns
        else pd.Series(10_000.0, index=accounts["account_id"])
    )

    # ── Sort for all subsequent computations ─────────────────────────────────
    tx = tx.sort_values(["sender_account_id", "timestamp"]).reset_index(drop=True)

    # ── Device features (vectorized shift) ───────────────────────────────────
    logger.info("Computing device features...")
    tx["prev_device"] = tx.groupby("sender_account_id")["sender_device_id"].shift(1)
    tx["device_changed"] = (tx["sender_device_id"] != tx["prev_device"]).astype(int)
    # new_device_first_tx: first occurrence of this (account, device) pair
    tx["_device_rank"] = tx.groupby(["sender_account_id", "sender_device_id"]).cumcount()
    tx["new_device_first_tx"] = (tx["_device_rank"] == 0).astype(int)
    tx.drop(columns=["_device_rank"], inplace=True)

    # ── Velocity: transactions per account in rolling 1h window ──────────────
    logger.info("Computing velocity features...")
    # Group by account, rolling on time index
    tx_ts = tx.set_index("timestamp")
    velocity_1h = (
        tx_ts.groupby("sender_account_id")["transaction_id"]
        .rolling("1h")
        .count()
        .reset_index(level=0, drop=True)
        .rename("velocity_1h")
    )
    tx = tx.join(velocity_1h, rsuffix="_v")
    # Daily baseline per account: mean daily count over full dataset
    tx["date"] = tx["timestamp"].dt.date
    daily_counts = tx.groupby(["sender_account_id", "date"]).size().reset_index(name="daily_count")
    baseline = daily_counts.groupby("sender_account_id")["daily_count"].mean()
    tx["daily_baseline"] = tx["sender_account_id"].map(baseline).fillna(2.5)
    tx["velocity_spike_ratio"] = tx["velocity_1h"] / (tx["daily_baseline"] / 24 + 0.1)

    # ── SIM swap proximity features ───────────────────────────────────────────
    logger.info("Computing SIM swap proximity features...")
    tx["time_since_sim_swap_minutes"] = -1.0
    tx["is_loadshedding_coincident"] = 0

    if ss_path.exists() and ss_path.stat().st_size > 0:
        ss = pd.read_parquet(ss_path)
        ss["activation_timestamp"] = pd.to_datetime(ss["activation_timestamp"])
        ss = ss.sort_values("activation_timestamp")

        # For each account, merge_asof to find most recent swap before each tx
        tx_sorted_ts = tx.sort_values("timestamp")
        merged = pd.merge_asof(
            tx_sorted_ts[["transaction_id", "sender_account_id", "timestamp"]],
            ss[["account_id", "activation_timestamp", "loadshedding_overlap"]].rename(
                columns={"account_id": "sender_account_id", "activation_timestamp": "swap_ts"}
            ),
            left_on="timestamp",
            right_on="swap_ts",
            by="sender_account_id",
            direction="backward",
        )
        merged["time_since_sim_swap_minutes"] = (
            ((merged["timestamp"] - merged["swap_ts"]).dt.total_seconds() / 60)
            .clip(upper=1440)
            .fillna(-1)
        )
        merged["is_loadshedding_coincident"] = (
            merged["loadshedding_overlap"].fillna(False).astype(int)
        )

        # Merge back to main tx (join on transaction_id, preserving original sort)
        proxy = merged.set_index("transaction_id")[
            ["time_since_sim_swap_minutes", "is_loadshedding_coincident"]
        ]
        tx = tx.join(proxy, on="transaction_id", rsuffix="_ss")
        tx["time_since_sim_swap_minutes"] = tx["time_since_sim_swap_minutes_ss"].fillna(-1.0)
        tx["is_loadshedding_coincident"] = tx["is_loadshedding_coincident_ss"].fillna(0).astype(int)
        tx.drop(columns=[c for c in tx.columns if c.endswith("_ss")], inplace=True)

    # ── Geography ─────────────────────────────────────────────────────────────
    tx["registered_province"] = tx["sender_account_id"].map(acc_province)
    tx["province_mismatch"] = (
        tx["sender_province"].fillna("") != tx["registered_province"].fillna("")
    ).astype(int)

    # ── Amount features ───────────────────────────────────────────────────────
    tx["account_balance"] = tx["sender_account_id"].map(acc_balance).fillna(10_000.0)
    tx["amount_to_balance_ratio"] = tx["amount_zar"] / (tx["account_balance"] + 1.0)
    tx["log_amount"] = np.log1p(tx["amount_zar"])

    # ── Temporal features ─────────────────────────────────────────────────────
    tx["hour_of_day"] = tx["timestamp"].dt.hour
    tx["is_weekend"] = (tx["timestamp"].dt.weekday >= 5).astype(int)

    # ── Label: SIM swap fraud transactions only ───────────────────────────────
    tx["label"] = (tx["is_fraud"] & (tx["fraud_type"].str.lower().fillna("") == "sim_swap")).astype(
        int
    )

    # ── Select output columns ─────────────────────────────────────────────────
    output_cols = [
        "transaction_id",
        "sender_account_id",
        "timestamp",
        # SIM swap signals
        "time_since_sim_swap_minutes",
        "is_loadshedding_coincident",
        # Device signals
        "device_changed",
        "new_device_first_tx",
        # Velocity
        "velocity_1h",
        "velocity_spike_ratio",
        # Geography
        "province_mismatch",
        # Amount
        "amount_zar",
        "amount_to_balance_ratio",
        "log_amount",
        # Transaction context
        "payment_rail",
        "merchant_category",
        "loadshedding_active",
        "loadshedding_stage",
        "hour_of_day",
        "is_weekend",
        # Label
        "label",
    ]
    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in tx.columns]
    feat_df = tx[output_cols].copy()

    # ── Encode categoricals ───────────────────────────────────────────────────
    feat_df = pd.get_dummies(feat_df, columns=["payment_rail", "merchant_category"])
    feat_df["loadshedding_active"] = feat_df["loadshedding_active"].fillna(False).astype(int)
    feat_df["loadshedding_stage"] = feat_df["loadshedding_stage"].fillna(0).astype(float)

    fraud_count = feat_df["label"].sum()
    logger.info(
        f"Feature engineering complete: {len(feat_df):,} rows | "
        f"{fraud_count:,} positive ({100 * fraud_count / len(feat_df):.3f}%)"
    )

    out_path = data_dir / "processed" / "sim_swap_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(out_path, index=False)
    logger.info(f"Saved to {out_path}")

    return feat_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    build_features(Path(args.data_dir))


if __name__ == "__main__":
    main()
