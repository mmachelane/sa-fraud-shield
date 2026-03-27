"""
Redis-backed online feature store.

Provides sub-millisecond access to pre-computed account features during
real-time fraud scoring. Both the Faust streaming consumer and the FastAPI
serving layer import this module.

Key patterns:
  account:{account_id}:features  → AccountFeatures JSON
  account:{account_id}:velocity:{window}  → sorted set (score=ts, member=tx_id)
  device:{device_id}:accounts  → set of account_ids
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import redis as syncredis
import redis.asyncio as aioredis

from shared.constants import VELOCITY_WINDOWS_SECONDS
from shared.schemas import AccountFeatures, Transaction

logger = logging.getLogger(__name__)

# Redis key templates
_KEY_ACCOUNT_FEATURES = "account:{account_id}:features"
_KEY_VELOCITY = "account:{account_id}:velocity:{window}"
_KEY_DEVICE_ACCOUNTS = "device:{device_id}:accounts"
_KEY_PHONE_ACCOUNTS = "phone:{phone}:accounts"


def _account_features_key(account_id: str) -> str:
    return _KEY_ACCOUNT_FEATURES.format(account_id=account_id)


def _velocity_key(account_id: str, window: str) -> str:
    return _KEY_VELOCITY.format(account_id=account_id, window=window)


def _device_accounts_key(device_id: str) -> str:
    return _KEY_DEVICE_ACCOUNTS.format(device_id=device_id)


def _serialize_features(features: AccountFeatures) -> str:
    return features.model_dump_json()


def _deserialize_features(data: str | bytes) -> AccountFeatures:
    return AccountFeatures.model_validate_json(data)


class FeatureStore:
    """
    Synchronous feature store client (for training scripts and notebooks).
    For async use (streaming consumer, API), use AsyncFeatureStore.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._client = syncredis.from_url(redis_url, decode_responses=True)

    def get_account_features(self, account_id: str) -> AccountFeatures | None:
        key = _account_features_key(account_id)
        data = self._client.get(key)
        if data is None:
            return None
        return _deserialize_features(data)

    def set_account_features(self, features: AccountFeatures, ttl_seconds: int = 86400) -> None:
        key = _account_features_key(features.account_id)
        self._client.setex(key, ttl_seconds, _serialize_features(features))

    def get_velocity(self, account_id: str, window: str) -> int:
        """Return transaction count in velocity window."""
        if window not in VELOCITY_WINDOWS_SECONDS:
            raise ValueError(f"Unknown window: {window}. Valid: {list(VELOCITY_WINDOWS_SECONDS)}")
        key = _velocity_key(account_id, window)
        now = datetime.now(tz=UTC).timestamp()
        min_score = now - VELOCITY_WINDOWS_SECONDS[window]
        return self._client.zcount(key, min_score, "+inf")

    def record_transaction(self, tx: Transaction) -> None:
        """Atomically update all velocity windows for a transaction."""
        now = datetime.now(tz=UTC).timestamp()
        pipe = self._client.pipeline()

        for window, seconds in VELOCITY_WINDOWS_SECONDS.items():
            key = _velocity_key(tx.sender_account_id, window)
            pipe.zadd(key, {tx.transaction_id: now})
            # Evict old entries
            pipe.zremrangebyscore(key, "-inf", now - seconds)
            pipe.expire(key, seconds * 2)

        if tx.sender_device_id:
            device_key = _device_accounts_key(tx.sender_device_id)
            pipe.sadd(device_key, tx.sender_account_id)
            pipe.expire(device_key, 86400 * 30)

        pipe.execute()

    def get_device_accounts(self, device_id: str) -> set[str]:
        """Return all account IDs that have used this device."""
        return self._client.smembers(_device_accounts_key(device_id))

    def ping(self) -> bool:
        try:
            return self._client.ping()
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()


class AsyncFeatureStore:
    """
    Async feature store client for use in FastAPI and Faust consumers.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._redis_url = redis_url
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._client = aioredis.from_url(self._redis_url, decode_responses=True)

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    def _require_client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("AsyncFeatureStore not connected. Call await store.connect() first.")
        return self._client

    async def get_account_features(self, account_id: str) -> AccountFeatures | None:
        client = self._require_client()
        data = await client.get(_account_features_key(account_id))
        if data is None:
            return None
        return _deserialize_features(data)

    async def set_account_features(
        self, features: AccountFeatures, ttl_seconds: int = 86400
    ) -> None:
        client = self._require_client()
        await client.setex(
            _account_features_key(features.account_id),
            ttl_seconds,
            _serialize_features(features),
        )

    async def get_velocity(self, account_id: str, window: str) -> int:
        client = self._require_client()
        if window not in VELOCITY_WINDOWS_SECONDS:
            raise ValueError(f"Unknown window: {window}")
        key = _velocity_key(account_id, window)
        now = datetime.now(tz=UTC).timestamp()
        min_score = now - VELOCITY_WINDOWS_SECONDS[window]
        return await client.zcount(key, min_score, "+inf")

    async def record_transaction(self, tx: Transaction) -> None:
        client = self._require_client()
        now = datetime.now(tz=UTC).timestamp()
        pipe = client.pipeline()

        for window, seconds in VELOCITY_WINDOWS_SECONDS.items():
            key = _velocity_key(tx.sender_account_id, window)
            pipe.zadd(key, {tx.transaction_id: now})
            pipe.zremrangebyscore(key, "-inf", now - seconds)
            pipe.expire(key, seconds * 2)

        if tx.sender_device_id:
            device_key = _device_accounts_key(tx.sender_device_id)
            pipe.sadd(device_key, tx.sender_account_id)
            pipe.expire(device_key, 86400 * 30)

        await pipe.execute()

    async def get_device_accounts(self, device_id: str) -> set[str]:
        client = self._require_client()
        return await client.smembers(_device_accounts_key(device_id))

    async def ping(self) -> bool:
        try:
            client = self._require_client()
            return await client.ping()
        except Exception:
            return False
