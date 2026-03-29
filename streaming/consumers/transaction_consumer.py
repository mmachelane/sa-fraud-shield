"""
Async Kafka consumer for real-time transaction processing.

Consumes the `transactions` topic, enriches each event with velocity features
from Redis, then publishes enriched records to the `feature-updates` topic for
downstream scoring.

Architecture:
    Kafka[transactions] → consumer → VelocityChecker (Redis) → Kafka[feature-updates]

Usage:
    python -m streaming.consumers.transaction_consumer
    python -m streaming.consumers.transaction_consumer --group my-group --workers 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.feature_store import AsyncFeatureStore  # noqa: E402
from shared.schemas import AccountFeatures, KafkaTransactionEvent, Transaction  # noqa: E402
from streaming.velocity_checker import VelocityChecker  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_IN = "transactions"
TOPIC_OUT = "feature-updates"
GROUP_ID = "fraudshield-enrichment"
REDIS_URL = "redis://localhost:6379/0"


# ── Enrichment logic ──────────────────────────────────────────────────────────


async def _enrich(
    tx: Transaction,
    velocity_checker: VelocityChecker,
    feature_store: AsyncFeatureStore,
) -> dict:
    """
    Compute velocity features and merge with stored account features.
    Returns a flat dict ready to publish to feature-updates topic.
    """
    # Get velocity snapshot before recording this transaction
    velocity = await velocity_checker.get_velocity(tx.sender_account_id, tx.timestamp)

    # Record this transaction into velocity windows
    await velocity_checker.record_transaction(
        account_id=tx.sender_account_id,
        tx_id=tx.transaction_id,
        amount=float(tx.amount_zar),
        timestamp=tx.timestamp,
    )

    # Track device for unique-device velocity
    if tx.sender_device_id:
        await velocity_checker.record_device(
            account_id=tx.sender_account_id,
            device_id=tx.sender_device_id,
            timestamp=tx.timestamp,
        )

    # Fetch pre-computed account features from batch job (optional)
    stored: AccountFeatures | None = await feature_store.get_account_features(tx.sender_account_id)

    return {
        "transaction_id": tx.transaction_id,
        "sender_account_id": tx.sender_account_id,
        "receiver_account_id": tx.receiver_account_id,
        "amount_zar": tx.amount_zar,
        "payment_rail": tx.payment_rail,
        "timestamp": tx.timestamp.isoformat(),
        "sender_device_id": tx.sender_device_id,
        "is_fraud": tx.is_fraud,
        # Velocity features
        "tx_count_5min": velocity["tx_count_5min"],
        "tx_count_1hr": velocity["tx_count_1hr"],
        "tx_count_24hr": velocity["tx_count_24hr"],
        "tx_count_7day": velocity["tx_count_7day"],
        "amount_sum_1hr": velocity["amount_sum_1hr"],
        "amount_sum_24hr": velocity["amount_sum_24hr"],
        "unique_devices_24hr": velocity["unique_devices_24hr"],
        # Account features (from batch pre-computation, or safe defaults)
        "device_change_24h": stored.device_change_24h if stored else False,
        "new_device_first_tx": stored.new_device_first_tx if stored else False,
        "days_since_account_open": stored.days_since_account_open if stored else 0,
        "prior_fraud_alerts_30d": stored.prior_fraud_alerts_30d if stored else 0,
        "enriched_at": datetime.now(UTC).isoformat(),
    }


# ── Consumer loop ─────────────────────────────────────────────────────────────


async def run_consumer(group_id: str = GROUP_ID) -> None:
    """
    Main consumer loop. Runs until SIGINT/SIGTERM.

    Processes one message at a time; for production scale-out, run multiple
    worker processes (each gets its own partition assignment from Kafka).
    """
    velocity_checker = VelocityChecker(REDIS_URL)
    feature_store = AsyncFeatureStore(REDIS_URL)
    await feature_store.connect()

    consumer = AIOKafkaConsumer(
        TOPIC_IN,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=group_id,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        auto_commit_interval_ms=1000,
    )

    producer = AIOKafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        compression_type="gzip",
    )

    await consumer.start()
    await producer.start()
    logger.info(f"Consumer started  topic={TOPIC_IN}  group={group_id}")

    processed = 0
    errors = 0

    # Graceful shutdown
    stop_event = asyncio.Event()

    def _handle_signal(*_):
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except (NotImplementedError, RuntimeError):
            # Windows doesn't support add_signal_handler — use KeyboardInterrupt instead
            pass

    try:
        async for msg in consumer:
            if stop_event.is_set():
                break
            try:
                event = KafkaTransactionEvent.model_validate(msg.value)
                tx = event.payload

                enriched = await _enrich(tx, velocity_checker, feature_store)

                await producer.send(
                    TOPIC_OUT,
                    key=tx.sender_account_id,
                    value=enriched,
                )

                processed += 1
                if processed % 1000 == 0:
                    logger.info(f"Processed {processed:,} messages  errors={errors}")

                if tx.is_fraud:
                    logger.warning(
                        f"FRAUD detected: {tx.transaction_id}  "
                        f"R{tx.amount_zar:.2f}  vel_1hr={enriched['tx_count_1hr']}"
                    )

            except KeyboardInterrupt:
                break
            except Exception as e:
                errors += 1
                logger.error(f"Failed to process message: {e}", exc_info=errors <= 3)

    finally:
        await producer.flush()
        await consumer.stop()
        await producer.stop()
        await velocity_checker.close()
        await feature_store.close()
        logger.info(f"Consumer stopped. processed={processed:,}  errors={errors}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    asyncio.run(run_consumer(group_id=args.group))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud transaction stream consumer")
    parser.add_argument("--group", type=str, default=GROUP_ID, help="Kafka consumer group ID")
    main(parser.parse_args())
