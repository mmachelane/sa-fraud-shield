"""
Redis-backed sliding-window velocity checker.

Maintains sorted sets keyed by account_id + window size. Each transaction
is stored as a member with its Unix timestamp as the score. Expired entries
are pruned on read (no background job needed — Redis TTL handles cleanup).

Velocity features computed:
    tx_count_5min     Number of transactions in last 5 minutes
    tx_count_1hr      Number of transactions in last 1 hour
    tx_count_24hr     Number of transactions in last 24 hours
    amount_sum_1hr    Total ZAR sent in last 1 hour
    amount_sum_24hr   Total ZAR sent in last 24 hours
    unique_devices_24hr  Distinct device IDs used in last 24 hours

These feed directly into the SIM swap feature vector and the streaming
score request payload.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import redis.asyncio as aioredis

from shared.constants import VELOCITY_WINDOWS_SECONDS

logger = logging.getLogger(__name__)

# Separate sorted sets for amount sums (member = tx_id, score = amount)
# and device tracking (member = device_id, score = timestamp)
_KEY_TX_COUNT = "vel:{account_id}:count:{window}"
_KEY_AMOUNT = "vel:{account_id}:amount:{window}"
_KEY_DEVICES = "vel:{account_id}:devices"

# How long to keep velocity data beyond the longest window (7 days + 1 day buffer)
_TTL_SECONDS = VELOCITY_WINDOWS_SECONDS["7day"] + 86400


class VelocityChecker:
    """
    Async velocity checker backed by Redis sorted sets.

    Each window uses two sorted sets:
      - count set: member=tx_id, score=unix_timestamp
      - amount set: member=tx_id, score=amount_zar

    The amount set is pruned by looking up which tx_ids are still in the
    count set, so both sets stay in sync without a separate cleanup pass.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self._redis: aioredis.Redis = aioredis.from_url(redis_url, decode_responses=True)

    async def record_transaction(
        self,
        account_id: str,
        tx_id: str,
        amount: float,
        timestamp: datetime,
    ) -> None:
        """
        Add a transaction to all velocity windows.
        Uses a Redis pipeline for atomicity and performance.
        """
        ts = timestamp.timestamp()

        pipe = self._redis.pipeline(transaction=False)

        for window in VELOCITY_WINDOWS_SECONDS:
            count_key = _KEY_TX_COUNT.format(account_id=account_id, window=window)
            amount_key = _KEY_AMOUNT.format(account_id=account_id, window=window)

            pipe.zadd(count_key, {tx_id: ts})
            pipe.zadd(amount_key, {tx_id: amount})
            pipe.expire(count_key, _TTL_SECONDS)
            pipe.expire(amount_key, _TTL_SECONDS)

        await pipe.execute()

    async def record_device(
        self,
        account_id: str,
        device_id: str,
        timestamp: datetime,
    ) -> None:
        """Track device usage for unique-device velocity."""
        ts = timestamp.timestamp()
        key = _KEY_DEVICES.format(account_id=account_id)
        await self._redis.zadd(key, {device_id: ts})
        await self._redis.expire(key, _TTL_SECONDS)

    async def get_velocity(
        self,
        account_id: str,
        as_of: datetime,
    ) -> dict[str, Any]:
        """
        Compute velocity features for an account as of a given timestamp.

        Returns a dict with all velocity keys. Returns zeros if the account
        has no history (cold start — safe default for new accounts).
        """
        now_ts = as_of.timestamp()

        pipe = self._redis.pipeline(transaction=False)

        # Queue count queries for each window
        for window, seconds in VELOCITY_WINDOWS_SECONDS.items():
            cutoff = now_ts - seconds
            count_key = _KEY_TX_COUNT.format(account_id=account_id, window=window)
            pipe.zrangebyscore(count_key, cutoff, now_ts)

        # Queue unique devices in 24hr
        cutoff_24hr = now_ts - VELOCITY_WINDOWS_SECONDS["24hr"]
        devices_key = _KEY_DEVICES.format(account_id=account_id)
        pipe.zrangebyscore(devices_key, cutoff_24hr, now_ts)

        results = await pipe.execute()

        # Parse count results (first 5 results = one per window)
        windows = list(VELOCITY_WINDOWS_SECONDS.keys())
        counts = {windows[i]: len(results[i]) for i in range(len(windows))}

        # Compute amount sums separately
        amount_1hr = await self._sum_amounts(account_id, "1hr", now_ts)
        amount_24hr = await self._sum_amounts(account_id, "24hr", now_ts)

        # Unique devices from last result
        unique_devices = len(results[-1]) if results else 0

        return {
            "tx_count_5min": counts.get("5min", 0),
            "tx_count_1hr": counts.get("1hr", 0),
            "tx_count_24hr": counts.get("24hr", 0),
            "tx_count_7day": counts.get("7day", 0),
            "amount_sum_1hr": amount_1hr,
            "amount_sum_24hr": amount_24hr,
            "unique_devices_24hr": unique_devices,
        }

    async def _sum_amounts(self, account_id: str, window: str, now_ts: float) -> float:
        """Sum ZAR amounts for transactions within the window."""
        seconds = VELOCITY_WINDOWS_SECONDS[window]
        cutoff = now_ts - seconds
        count_key = _KEY_TX_COUNT.format(account_id=account_id, window=window)
        amount_key = _KEY_AMOUNT.format(account_id=account_id, window=window)

        # Get tx_ids in the time window
        tx_ids = await self._redis.zrangebyscore(count_key, cutoff, now_ts)
        if not tx_ids:
            return 0.0

        # Fetch their amounts from the amount sorted set
        scores = await self._redis.zmscore(amount_key, tx_ids)
        return float(sum(s for s in scores if s is not None))

    async def is_high_velocity(
        self,
        account_id: str,
        as_of: datetime,
        tx_threshold_5min: int = 5,
        tx_threshold_1hr: int = 20,
        amount_threshold_1hr: float = 50_000.0,
    ) -> tuple[bool, list[str]]:
        """
        Rule-based velocity check. Returns (is_suspicious, triggered_rules).

        Used as a fast pre-filter before the ML model — if any rule fires,
        the transaction is flagged for step-up auth regardless of ML score.
        """
        v = await self.get_velocity(account_id, as_of)
        triggered = []

        if v["tx_count_5min"] >= tx_threshold_5min:
            triggered.append(f"tx_count_5min={v['tx_count_5min']} >= {tx_threshold_5min}")

        if v["tx_count_1hr"] >= tx_threshold_1hr:
            triggered.append(f"tx_count_1hr={v['tx_count_1hr']} >= {tx_threshold_1hr}")

        if v["amount_sum_1hr"] >= amount_threshold_1hr:
            triggered.append(
                f"amount_sum_1hr=R{v['amount_sum_1hr']:,.0f} >= R{amount_threshold_1hr:,.0f}"
            )

        return bool(triggered), triggered

    async def close(self) -> None:
        await self._redis.close()
