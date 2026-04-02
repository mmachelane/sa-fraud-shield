"""
End-to-end test: generate → score → verify.

Exercises the full scoring pipeline without any external services:
  1. Generate a synthetic Transaction using the data generators
  2. POST it to /score via TestClient with a mocked registry
  3. Assert the response shape, bounds, and decision logic
  4. Assert a BLOCK decision produces a FraudAlert with correct fields

Does NOT require:
  - Trained model files on disk
  - Redis / Kafka
  - Network access

Skip conditions:
  - If data_generation package fails to import (optional dependency missing),
    tests are skipped rather than failing.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _build_transaction(
    amount: float = 1500.0,
    sim_swap_minutes_ago: float | None = None,
    payment_rail: str = "PAYSHAP",
) -> dict:
    now = datetime(2026, 3, 20, 14, 15, 0)
    tx: dict = {
        "transaction_id": f"e2e-tx-{int(amount)}",
        "timestamp": now.isoformat() + "Z",
        "sender_account_id": "e2e-acc-sender-001",
        "receiver_account_id": "e2e-acc-receiver-001",
        "amount_zar": str(amount),
        "payment_rail": payment_rail,
        "merchant_category": "peer_transfer",
        "sender_province": "GP",
    }
    if sim_swap_minutes_ago is not None:
        swap_ts = now - timedelta(minutes=sim_swap_minutes_ago)
        tx["sim_swap_detected"] = True
        tx["sim_swap_timestamp"] = swap_ts.isoformat() + "Z"
    return {"transaction": tx}


# ── E2E: normal transaction flow ──────────────────────────────────────────────


class TestNormalTransactionFlow:
    def test_score_response_structure(self, test_client):
        payload = _build_transaction(amount=2500.0)
        resp = test_client.post("/score", json=payload)
        assert resp.status_code == 200

        body = resp.json()
        # Required fields
        assert "transaction_id" in body
        assert "ensemble_score" in body
        assert "decision" in body
        assert body["transaction_id"] == "e2e-tx-2500"

    def test_ensemble_score_in_bounds(self, test_client):
        payload = _build_transaction(amount=500.0)
        resp = test_client.post("/score", json=payload)
        body = resp.json()
        assert 0.0 <= body["ensemble_score"] <= 1.0

    def test_valid_decision_values(self, test_client):
        payload = _build_transaction(amount=1000.0)
        resp = test_client.post("/score", json=payload)
        body = resp.json()
        assert body["decision"] in ("APPROVE", "STEP_UP", "BLOCK")

    def test_latency_ms_recorded(self, test_client):
        payload = _build_transaction(amount=3000.0)
        resp = test_client.post("/score", json=payload)
        body = resp.json()
        assert body.get("latency_ms") is not None
        assert body["latency_ms"] >= 0.0


# ── E2E: SIM swap fraud flow ──────────────────────────────────────────────────


class TestSIMSwapFraudFlow:
    def test_sim_swap_produces_block(self, test_client):
        """
        SIM swap 3 min before R50 000 PayShap transfer.
        Mock model returns 0.75 → ensemble = 0.75 → BLOCK.
        """
        payload = _build_transaction(amount=50000.0, sim_swap_minutes_ago=3)
        resp = test_client.post("/score", json=payload)
        assert resp.status_code == 200

        body = resp.json()
        assert body["decision"] == "BLOCK"
        assert body["ensemble_score"] >= 0.7

    def test_block_decision_includes_alert(self, test_client):
        payload = _build_transaction(amount=50000.0, sim_swap_minutes_ago=3)
        resp = test_client.post("/score", json=payload)
        body = resp.json()

        assert body["alert"] is not None
        alert = body["alert"]
        assert alert["status"] == "OPEN"
        assert alert["ensemble_score"] >= 0.7
        assert "transaction_id" in alert
        assert "account_id" in alert
        assert "severity" in alert

    def test_alert_severity_is_high_or_critical(self, test_client):
        payload = _build_transaction(amount=50000.0, sim_swap_minutes_ago=3)
        resp = test_client.post("/score", json=payload)
        alert = resp.json()["alert"]
        assert alert["severity"] in ("HIGH", "CRITICAL")

    def test_sim_swap_score_present(self, test_client):
        payload = _build_transaction(amount=50000.0, sim_swap_minutes_ago=3)
        resp = test_client.post("/score", json=payload)
        body = resp.json()
        assert body.get("sim_swap_score") is not None
        assert 0.0 <= body["sim_swap_score"] <= 1.0


# ── E2E: drift endpoint after scoring ────────────────────────────────────────


class TestDriftAfterScoring:
    def test_drift_status_after_multiple_scores(self, test_client):
        """
        After scoring 20 transactions, /drift should return a valid report.
        With empty baseline stats (no training data), PSI is always 0 → stable.
        """
        for i in range(20):
            test_client.post(
                "/score",
                json=_build_transaction(amount=float(1000 + i * 100)),
            )

        resp = test_client.get("/drift")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("stable", "warning", "drift")
        assert isinstance(body["n_live_samples"], int)


# ── E2E: multiple payment rails ───────────────────────────────────────────────


class TestPaymentRailVariants:
    @pytest.mark.parametrize(
        "rail",
        ["PAYSHAP", "EFT", "MOBILE_APP", "CARD_CNP", "ATM"],
    )
    def test_all_rails_score_successfully(self, test_client, rail):
        payload = _build_transaction(amount=1000.0, payment_rail=rail)
        resp = test_client.post("/score", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["decision"] in ("APPROVE", "STEP_UP", "BLOCK")
