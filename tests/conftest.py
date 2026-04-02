"""
Shared pytest fixtures for sa-fraud-shield tests.

Provides:
    sample_transaction      — a valid Transaction with no fraud signals
    high_risk_transaction   — a Transaction with SIM swap + high amount
    mock_registry           — a ModelRegistry stub with a fake SIM swap model
    test_app / test_client  — FastAPI TestClient with mocked startup state
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from shared.constants import PaymentRail, SAMerchantCategory, SAProvince
from shared.schemas import Transaction

# ── Transactions ──────────────────────────────────────────────────────────────


@pytest.fixture
def sample_transaction() -> Transaction:
    """Low-risk PayShap transaction — no fraud indicators."""
    return Transaction(
        transaction_id="tx-unit-001",
        timestamp=datetime(2026, 3, 15, 10, 30, 0, tzinfo=UTC),
        sender_account_id="acc-sender-001",
        receiver_account_id="acc-receiver-001",
        amount_zar=Decimal("1500.00"),
        payment_rail=PaymentRail.PAYSHAP,
        merchant_category=SAMerchantCategory.GROCERY,
        sender_province=SAProvince.GAUTENG,
        sim_swap_detected=False,
        loadshedding_active=False,
    )


@pytest.fixture
def high_risk_transaction() -> Transaction:
    """High-risk transaction: SIM swap detected 5 minutes before R45 000 PayShap."""
    swap_ts = datetime(2026, 3, 15, 10, 25, 0, tzinfo=UTC)
    tx_ts = swap_ts + timedelta(minutes=5)
    return Transaction(
        transaction_id="tx-unit-002",
        timestamp=tx_ts,
        sender_account_id="acc-sender-002",
        receiver_account_id="acc-receiver-002",
        amount_zar=Decimal("45000.00"),
        payment_rail=PaymentRail.PAYSHAP,
        merchant_category=SAMerchantCategory.PEER_TRANSFER,
        sender_province=SAProvince.GAUTENG,
        sim_swap_detected=True,
        sim_swap_timestamp=swap_ts,
        loadshedding_active=False,
    )


# ── Mock model registry ───────────────────────────────────────────────────────


class _FakeSimSwapDetector:
    """Minimal stub that mimics SimSwapDetector.predict_proba()."""

    def __init__(self, fixed_score: float = 0.75) -> None:
        self._score = fixed_score
        self.feature_names = list(
            [
                "time_since_sim_swap_minutes",
                "is_loadshedding_coincident",
                "device_changed",
                "new_device_first_tx",
                "velocity_1h",
                "velocity_spike_ratio",
                "province_mismatch",
                "amount_zar",
                "amount_to_balance_ratio",
                "log_amount",
                "loadshedding_active",
                "loadshedding_stage",
                "hour_of_day",
                "is_weekend",
                "payment_rail_ATM",
                "payment_rail_CARD_CNP",
                "payment_rail_CARD_PRESENT",
                "payment_rail_CASH_DEPOSIT",
                "payment_rail_EFT",
                "payment_rail_MOBILE_APP",
                "payment_rail_PAYSHAP",
                "merchant_category_clothing",
                "merchant_category_crypto_exchange",
                "merchant_category_electronics",
                "merchant_category_fuel",
                "merchant_category_gambling",
                "merchant_category_grocery",
                "merchant_category_peer_transfer",
                "merchant_category_pharmacy",
                "merchant_category_restaurant",
                "merchant_category_spaza_shop",
                "merchant_category_taxi",
                "merchant_category_unknown",
                "merchant_category_utilities",
            ]
        )

    def predict_proba(self, df: Any) -> list[float]:  # noqa: ANN001
        return [self._score]


def make_mock_registry(sim_score: float = 0.75, load_gnn: bool = False):
    """Return a mock ModelRegistry with a fake SIM swap model."""
    from api.services.model_registry import ModelRegistry

    reg = ModelRegistry.__new__(ModelRegistry)
    reg._sim_swap = _FakeSimSwapDetector(sim_score)  # type: ignore[assignment]
    reg._gnn = None
    reg._gnn_device = None
    reg._gnn_hidden_dim = 64
    reg._gnn_dropout = 0.2
    reg.sim_swap_loaded = True
    reg.gnn_loaded = False
    return reg


@pytest.fixture
def mock_registry():
    return make_mock_registry()


# ── FastAPI test client ───────────────────────────────────────────────────────


@pytest.fixture
def test_client(mock_registry):
    """
    TestClient with mocked startup: skips model loading from disk,
    injects mock_registry and a no-op DriftDetector into app state.
    """
    from monitoring.drift_detector import DriftDetector

    drift = DriftDetector(baseline_stats={}, window_size=100)

    def _fake_lifespan(app):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _ctx(a):
            a.state.registry = mock_registry
            a.state.drift_detector = drift
            yield

        return _ctx(app)

    # Patch the lifespan so no disk I/O happens during test startup
    with patch("api.main.lifespan", side_effect=_fake_lifespan):
        from api.main import app

        # Override state directly (lifespan patch sets it above)
        app.state.registry = mock_registry
        app.state.drift_detector = drift

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client
