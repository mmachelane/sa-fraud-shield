"""
Dataset orchestrator: generates a complete synthetic SA fraud dataset.

Outputs:
  data/raw/transactions.parquet     — all transactions (labelled)
  data/raw/accounts.parquet         — all accounts
  data/raw/sim_swap_events.parquet  — SIM swap events (for feature engineering)
  data/processed/sim_swap_features.parquet  — pre-computed SIM swap model features
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from data_generation.fraud_patterns.fraud_ring import generate_fraud_ring
from data_generation.fraud_patterns.sim_swap_sequence import generate_sim_swap_sequence
from data_generation.generators.banking import generate_account
from data_generation.generators.sa_identity import generate_identity
from data_generation.generators.transactions import (
    generate_amount,
    generate_device_id,
    generate_timestamp,
    sample_merchant_category,
    sample_payment_rail,
)
from shared.constants import PROVINCE_POPULATION_WEIGHTS, SAProvince
from shared.utils.load_shedding import generate_mock_schedule

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    n_accounts: int = 50_000
    n_days: int = 180  # 6 months of history
    fraud_rate: float = 0.023  # ~2.3% overall (SABRIC 2024)
    sim_swap_rate: float = 0.008  # 0.8% of accounts
    fraud_ring_rate: float = 0.005  # 0.5% of accounts
    avg_tx_per_account_per_day: float = 2.5
    seed: int = 42
    include_loadshedding: bool = True
    loadshedding_stage: int = 2


def _sample_province(rng: random.Random) -> SAProvince:
    provinces = list(PROVINCE_POPULATION_WEIGHTS.keys())
    weights = list(PROVINCE_POPULATION_WEIGHTS.values())
    return rng.choices(provinces, weights=weights, k=1)[0]


def generate_dataset(
    config: DatasetConfig,
    output_dir: Path | str = "data",
) -> dict[str, pd.DataFrame]:
    """
    Generate a complete synthetic SA fraud dataset.

    Returns a dict of DataFrames: transactions, accounts, sim_swap_events
    """
    rng = random.Random(config.seed)
    output_dir = Path(output_dir)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    (output_dir / "processed").mkdir(parents=True, exist_ok=True)

    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=config.n_days)

    logger.info(f"Generating {config.n_accounts:,} accounts over {config.n_days} days...")

    # ── Generate accounts ─────────────────────────────────────────────────────
    accounts: list[dict] = []
    for i in range(config.n_accounts):
        province = _sample_province(rng)
        identity = generate_identity(province=province, rng=rng)
        bank_data = generate_account(
            phone=identity["phone"],
            province=province.value,
            rng=rng,
        )
        account_id = f"acc_{uuid.uuid4().hex[:16]}"
        accounts.append(
            {
                "account_id": account_id,
                "first_name": identity["first_name"],
                "last_name": identity["last_name"],
                "id_number": identity["id_number"],
                "registration_phone": identity["phone"],
                "province": province.value,
                **bank_data,
                "is_compromised": False,
            }
        )

    logger.info(f"Generated {len(accounts):,} accounts")

    # ── Assign fraud roles ────────────────────────────────────────────────────
    # Derive victim counts from fraud_rate (account-level):
    # ~35% of fraud-affected accounts are SIM swap victims, ~22% are ring members
    sim_swap_rate = config.sim_swap_rate or (config.fraud_rate * 0.35)
    fraud_ring_rate = config.fraud_ring_rate or (config.fraud_rate * 0.22)
    n_sim_swap_victims = int(config.n_accounts * sim_swap_rate)
    n_ring_accounts = int(config.n_accounts * fraud_ring_rate)

    sim_swap_victims = rng.sample(accounts, n_sim_swap_victims)
    for a in sim_swap_victims:
        a["is_compromised"] = True

    # ── Load shedding schedules ───────────────────────────────────────────────
    ls_schedules = {}
    if config.include_loadshedding:
        for province in SAProvince:
            ls_schedules[province.value] = generate_mock_schedule(
                province_code=province.value,
                start_date=start_date,
                days=config.n_days,
                stage=config.loadshedding_stage,
            )

    # ── Generate normal transactions ──────────────────────────────────────────
    all_transactions: list[dict] = []
    account_devices: dict[str, list[str]] = {}

    logger.info("Generating normal transactions...")
    for account in accounts:
        # Assign 1-3 devices to each account
        n_devices = rng.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
        devices = [generate_device_id(rng) for _ in range(n_devices)]
        account_devices[account["account_id"]] = devices

        # Generate transactions across the date range
        days_with_tx = rng.randint(
            int(config.n_days * 0.3),
            config.n_days,
        )
        tx_days = rng.sample(range(config.n_days), days_with_tx)

        for day_offset in tx_days:
            tx_date = start_date + timedelta(days=day_offset)
            n_tx = max(1, int(rng.gauss(config.avg_tx_per_account_per_day, 1.5)))
            n_tx = min(n_tx, 15)  # Cap at 15 per day

            # Salary day boost
            if tx_date.day == 25:
                n_tx = int(n_tx * 1.5)

            for _ in range(n_tx):
                ts = generate_timestamp(tx_date, rng)
                category = sample_merchant_category(rng)
                rail = sample_payment_rail(category, rng)
                amount = generate_amount(category, rng)

                # Check load shedding
                ls_active = False
                ls_stage = None
                province_code = account["province"]
                if province_code in ls_schedules:
                    sched = ls_schedules[province_code]
                    outage = sched.get_active_outage(ts)
                    if outage:
                        ls_active = True
                        ls_stage = outage.stage

                all_transactions.append(
                    {
                        "transaction_id": str(uuid.uuid4()),
                        "timestamp": ts,
                        "sender_account_id": account["account_id"],
                        "receiver_account_id": None,
                        "amount_zar": float(amount),
                        "payment_rail": rail.value,
                        "merchant_category": category.value,
                        "merchant_id": f"merch_{rng.randint(1000, 9999)}",
                        "sender_device_id": rng.choice(devices),
                        "sender_province": province_code,
                        "is_fraud": False,
                        "fraud_type": None,
                        "loadshedding_active": ls_active,
                        "loadshedding_stage": ls_stage,
                        "sim_swap_detected": False,
                        "sim_swap_timestamp": None,
                    }
                )

    logger.info(f"Generated {len(all_transactions):,} normal transactions")

    # ── Generate SIM swap fraud sequences ─────────────────────────────────────
    logger.info(f"Generating {n_sim_swap_victims} SIM swap sequences...")
    sim_swap_events: list[dict] = []
    mule_pool = [a for a in accounts if not a["is_compromised"]]

    for victim in sim_swap_victims:
        n_mules = rng.randint(1, 4)
        mules = rng.sample(mule_pool, n_mules)

        # Random time in dataset range
        days_offset = rng.randint(30, config.n_days - 7)
        attack_time = start_date + timedelta(
            days=days_offset,
            hours=rng.randint(8, 22),
            minutes=rng.randint(0, 59),
        )

        province_code = victim["province"]
        ls_window = None
        if province_code in ls_schedules:
            # 30% of SIM swaps overlap with load shedding (modeling challenge)
            if rng.random() < 0.30:
                sched = ls_schedules[province_code]
                ls_window = sched.get_nearest_outage(attack_time)

        sequence = generate_sim_swap_sequence(
            victim_account=victim,
            mule_accounts=mules,
            start_time=attack_time,
            account_balance_zar=rng.uniform(5000, 50000),
            loadshedding_window=ls_window,
            rng=rng,
        )

        all_transactions.extend(sequence.transactions)
        sim_swap_events.append(
            {
                "event_id": sequence.sim_swap_event.event_id,
                "account_id": sequence.sim_swap_event.account_id,
                "phone_number": sequence.sim_swap_event.phone_number,
                "swap_timestamp": sequence.sim_swap_event.swap_timestamp,
                "activation_timestamp": sequence.sim_swap_event.activation_timestamp,
                "insider_collusion": sequence.sim_swap_event.insider_collusion,
                "carrier": sequence.sim_swap_event.carrier,
                "loadshedding_overlap": sequence.loadshedding_overlap,
                "total_loss_zar": float(sequence.total_fraud_loss_zar),
            }
        )

    # ── Generate fraud ring sequences ─────────────────────────────────────────
    logger.info("Generating fraud ring sequences...")
    ring_accounts_pool = [a for a in accounts if not a["is_compromised"]]
    assigned_to_ring: set[str] = set()
    n_rings = max(1, n_ring_accounts // 12)  # Average ring size ~12

    for _ in range(n_rings):
        ring_size = rng.randint(5, 20)
        available = [a for a in ring_accounts_pool if a["account_id"] not in assigned_to_ring]
        if len(available) < ring_size:
            break

        ring_members = rng.sample(available, ring_size)
        for m in ring_members:
            assigned_to_ring.add(m["account_id"])
            m["is_compromised"] = True

        days_offset = rng.randint(14, config.n_days - 14)
        ring_start = start_date + timedelta(days=days_offset, hours=rng.randint(0, 23))

        ring = generate_fraud_ring(ring_members, ring_start, rng=rng)
        all_transactions.extend(ring.transactions)

    # ── Assemble DataFrames ───────────────────────────────────────────────────
    logger.info("Assembling DataFrames...")
    tx_df = pd.DataFrame(all_transactions)
    tx_df["timestamp"] = pd.to_datetime(tx_df["timestamp"])
    tx_df["amount_zar"] = tx_df["amount_zar"].astype(float)  # Decimal → float for parquet
    tx_df = tx_df.sort_values("timestamp").reset_index(drop=True)

    accounts_df = pd.DataFrame(accounts)
    sim_swap_df = pd.DataFrame(sim_swap_events) if sim_swap_events else pd.DataFrame()
    if not sim_swap_df.empty and "total_loss_zar" in sim_swap_df.columns:
        sim_swap_df["total_loss_zar"] = sim_swap_df["total_loss_zar"].astype(float)

    # ── Save ──────────────────────────────────────────────────────────────────
    tx_path = output_dir / "raw" / "transactions.parquet"
    accounts_path = output_dir / "raw" / "accounts.parquet"
    ss_path = output_dir / "raw" / "sim_swap_events.parquet"

    tx_df.to_parquet(tx_path, index=False)
    accounts_df.to_parquet(accounts_path, index=False)
    if not sim_swap_df.empty:
        sim_swap_df.to_parquet(ss_path, index=False)

    fraud_count = tx_df["is_fraud"].sum()
    total = len(tx_df)
    logger.info(
        f"Dataset saved: {total:,} transactions | "
        f"{fraud_count:,} fraud ({100 * fraud_count / total:.2f}%) | "
        f"{len(accounts):,} accounts"
    )

    return {"transactions": tx_df, "accounts": accounts_df, "sim_swap_events": sim_swap_df}
