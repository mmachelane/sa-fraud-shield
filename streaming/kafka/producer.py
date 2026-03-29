"""
Kafka transaction producer.

Replays transactions.parquet to the `transactions` Kafka topic, simulating a
live payment stream. Each message is a KafkaTransactionEvent JSON envelope.

Usage:
    python -m streaming.kafka.producer                        # full dataset, real-time
    python -m streaming.kafka.producer --speed 100           # 100 tx/s
    python -m streaming.kafka.producer --limit 50000         # first 50k rows only
    python -m streaming.kafka.producer --fraud-only          # fraud transactions only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from aiokafka import AIOKafkaProducer

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.schemas import KafkaTransactionEvent, Transaction  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TOPIC = "transactions"
TRANSACTIONS_PATH = Path("data/raw/transactions.parquet")
BOOTSTRAP_SERVERS = "localhost:9092"

# Columns needed — avoids loading all 50+ columns into memory
_REQUIRED_COLS = [
    "transaction_id",
    "timestamp",
    "sender_account_id",
    "receiver_account_id",
    "amount_zar",
    "payment_rail",
    "merchant_category",
    "merchant_id",
    "sender_device_id",
    "sender_province",
    "is_fraud",
    "fraud_type",
    "sim_swap_detected",
    "sim_swap_timestamp",
    "loadshedding_active",
    "loadshedding_stage",
]


def _load_transactions(path: Path, limit: int | None, fraud_only: bool) -> pd.DataFrame:
    cols = [c for c in _REQUIRED_COLS]
    df = pd.read_parquet(path, columns=cols)
    if fraud_only:
        df = df[df["is_fraud"]].reset_index(drop=True)
    if limit:
        df = df.head(limit)
    # Sort by timestamp to replay in chronological order
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} transactions  (fraud={df['is_fraud'].sum():,})")
    return df


def _row_to_transaction(row: pd.Series) -> Transaction:
    def _opt(col: str) -> str | None:
        val = row.get(col)
        return str(val) if val is not None and pd.notna(val) else None

    def _opt_bool(col: str) -> bool:
        val = row.get(col)
        return bool(val) if val is not None and pd.notna(val) else False

    def _opt_int(col: str) -> int | None:
        val = row.get(col)
        return int(val) if val is not None and pd.notna(val) else None

    return Transaction(
        transaction_id=str(row["transaction_id"]),
        timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
        sender_account_id=str(row["sender_account_id"]),
        receiver_account_id=_opt("receiver_account_id"),
        amount_zar=float(row["amount_zar"]),
        payment_rail=row["payment_rail"],
        merchant_category=row.get("merchant_category") or "unknown",
        merchant_id=_opt("merchant_id"),
        sender_device_id=_opt("sender_device_id"),
        sender_province=_opt("sender_province"),
        is_fraud=bool(row["is_fraud"]) if pd.notna(row.get("is_fraud")) else None,
        fraud_type=_opt("fraud_type"),
        sim_swap_detected=_opt_bool("sim_swap_detected"),
        sim_swap_timestamp=pd.Timestamp(row["sim_swap_timestamp"]).to_pydatetime()
        if pd.notna(row.get("sim_swap_timestamp"))
        else None,
        loadshedding_active=_opt_bool("loadshedding_active"),
        loadshedding_stage=_opt_int("loadshedding_stage"),
    )


def _build_event(tx: Transaction) -> KafkaTransactionEvent:
    return KafkaTransactionEvent(
        produced_at=datetime.now(UTC),
        partition_key=tx.sender_account_id,
        payload=tx,
    )


async def _produce(df: pd.DataFrame, speed: float | None) -> None:
    producer = AIOKafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        compression_type="gzip",
        linger_ms=5,
        batch_size=32768,
    )
    await producer.start()

    sent = 0
    errors = 0
    t_start = time.monotonic()

    try:
        for _, row in df.iterrows():
            try:
                tx = _row_to_transaction(row)
                event = _build_event(tx)
                await producer.send(
                    TOPIC,
                    key=event.partition_key,
                    value=json.loads(event.model_dump_json()),
                )
                sent += 1

                # Rate limiting
                if speed is not None:
                    elapsed = time.monotonic() - t_start
                    expected = sent / speed
                    if expected > elapsed:
                        await asyncio.sleep(expected - elapsed)

                if sent % 1000 == 0:
                    rate = sent / max(time.monotonic() - t_start, 0.001)
                    logger.info(f"Sent {sent:,} messages  ({rate:.0f} tx/s)")

            except Exception as e:
                errors += 1
                if errors <= 5:
                    logger.warning(f"Skipped row {sent}: {e}")

    finally:
        await producer.flush()
        await producer.stop()
        elapsed = time.monotonic() - t_start
        logger.info(
            f"Done. sent={sent:,}  errors={errors}  "
            f"elapsed={elapsed:.1f}s  rate={sent / max(elapsed, 0.001):.0f} tx/s"
        )


def main(args: argparse.Namespace) -> None:
    df = _load_transactions(
        TRANSACTIONS_PATH,
        limit=args.limit,
        fraud_only=args.fraud_only,
    )
    asyncio.run(_produce(df, speed=args.speed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay transactions to Kafka")
    parser.add_argument("--speed", type=float, default=None, help="Max tx/s (default: unlimited)")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to send")
    parser.add_argument("--fraud-only", action="store_true", help="Send only fraud transactions")
    main(parser.parse_args())
