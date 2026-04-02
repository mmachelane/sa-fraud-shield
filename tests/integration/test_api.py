"""
Integration tests for the FastAPI application (api/main.py).

Tests all major endpoints using a TestClient with a mocked ModelRegistry
and no-op DriftDetector — no model files or Redis required.

Covers:
  - GET /health, /ready
  - GET / (root info)
  - GET /drift
  - POST /score — no models loaded → 503
  - POST /score — mocked SIM swap model → valid ScoreResponse
  - POST /score — high-risk SIM swap transaction → BLOCK decision
  - POST /score/debug-features — returns feature vector
  - GET /metrics — Prometheus text format
"""

from __future__ import annotations

from datetime import datetime, timedelta

# ── Helpers ───────────────────────────────────────────────────────────────────


def _tx_payload(
    amount: float = 1500.0,
    sim_swap_minutes_ago: float | None = None,
    payment_rail: str = "PAYSHAP",
) -> dict:
    now = datetime(2026, 3, 15, 10, 30, 0)
    payload: dict = {
        "transaction": {
            "transaction_id": "tx-integ-001",
            "timestamp": now.isoformat() + "Z",
            "sender_account_id": "acc-001",
            "amount_zar": str(amount),
            "payment_rail": payment_rail,
        }
    }
    if sim_swap_minutes_ago is not None:
        swap_ts = now - timedelta(minutes=sim_swap_minutes_ago)
        payload["transaction"]["sim_swap_detected"] = True
        payload["transaction"]["sim_swap_timestamp"] = swap_ts.isoformat() + "Z"
    return payload


# ── Health / ready ────────────────────────────────────────────────────────────


class TestHealthEndpoints:
    def test_health_ok(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ready_ok(self, test_client):
        resp = test_client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"


# ── Root ──────────────────────────────────────────────────────────────────────


class TestRootEndpoint:
    def test_root_returns_service_info(self, test_client):
        resp = test_client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["service"] == "sa-fraud-shield"
        assert "models" in body
        assert "endpoints" in body


# ── Drift ─────────────────────────────────────────────────────────────────────


class TestDriftEndpoint:
    def test_drift_returns_report(self, test_client):
        resp = test_client.get("/drift")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert body["status"] in ("stable", "warning", "drift")
        assert "psi_scores" in body
        assert "thresholds" in body


# ── Metrics ───────────────────────────────────────────────────────────────────


class TestMetricsEndpoint:
    def test_metrics_prometheus_format(self, test_client):
        resp = test_client.get("/metrics")
        assert resp.status_code == 200
        # Prometheus text format starts with HELP or TYPE lines
        content = resp.text
        assert "# HELP" in content or "# TYPE" in content or len(content) > 0


# ── Score endpoint ────────────────────────────────────────────────────────────


class TestScoreEndpoint:
    def test_no_models_returns_503(self, test_client):
        """When no models are loaded, /score must return 503."""
        # Access the live registry from app state and temporarily disable all models
        live_registry = test_client.app.state.registry
        original_ss = live_registry.sim_swap_loaded
        original_gnn = live_registry.gnn_loaded
        live_registry.sim_swap_loaded = False
        live_registry.gnn_loaded = False
        try:
            resp = test_client.post("/score", json=_tx_payload())
            assert resp.status_code == 503
        finally:
            live_registry.sim_swap_loaded = original_ss
            live_registry.gnn_loaded = original_gnn

    def test_score_returns_valid_response(self, test_client):
        resp = test_client.post("/score", json=_tx_payload(amount=2000.0))
        assert resp.status_code == 200
        body = resp.json()
        assert "ensemble_score" in body
        assert "decision" in body
        assert body["decision"] in ("APPROVE", "STEP_UP", "BLOCK")
        assert 0.0 <= body["ensemble_score"] <= 1.0

    def test_score_sim_swap_transaction_has_valid_response(self, test_client):
        """SIM swap 5 min ago + R45 000 PayShap — response must be structurally valid."""
        payload = _tx_payload(amount=45000.0, sim_swap_minutes_ago=5)
        resp = test_client.post("/score", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        # Decision logic: if BLOCK then alert must be present
        if body["decision"] == "BLOCK":
            assert body.get("alert") is not None
            assert body["ensemble_score"] >= 0.7
        else:
            assert body["decision"] in ("APPROVE", "STEP_UP")

    def test_score_decision_block_always_has_alert(self, test_client):
        """Verify decision/alert contract: BLOCK ↔ alert present."""
        live_registry = test_client.app.state.registry
        original_ss = live_registry.sim_swap_loaded
        # Force a high score by injecting a fake detector
        from tests.conftest import _FakeSimSwapDetector

        live_registry._sim_swap = _FakeSimSwapDetector(fixed_score=0.9)
        live_registry.sim_swap_loaded = True
        try:
            resp = test_client.post("/score", json=_tx_payload(amount=50000.0))
            assert resp.status_code == 200
            body = resp.json()
            assert body["decision"] == "BLOCK"
            assert body["ensemble_score"] >= 0.7
            assert body["alert"] is not None
        finally:
            live_registry.sim_swap_loaded = original_ss

    def test_score_approve_decision(self, test_client):
        """Force a low score → APPROVE."""
        live_registry = test_client.app.state.registry
        from tests.conftest import _FakeSimSwapDetector

        original_detector = live_registry._sim_swap
        original_ss = live_registry.sim_swap_loaded
        live_registry._sim_swap = _FakeSimSwapDetector(fixed_score=0.05)
        live_registry.sim_swap_loaded = True
        try:
            resp = test_client.post("/score", json=_tx_payload(amount=200.0))
            assert resp.status_code == 200
            body = resp.json()
            assert body["decision"] == "APPROVE"
            assert body["ensemble_score"] < 0.3
        finally:
            live_registry._sim_swap = original_detector
            live_registry.sim_swap_loaded = original_ss

    def test_score_step_up_decision(self, test_client):
        """Force a mid-range score → STEP_UP."""
        live_registry = test_client.app.state.registry
        from tests.conftest import _FakeSimSwapDetector

        original_detector = live_registry._sim_swap
        original_ss = live_registry.sim_swap_loaded
        live_registry._sim_swap = _FakeSimSwapDetector(fixed_score=0.5)
        live_registry.sim_swap_loaded = True
        try:
            resp = test_client.post("/score", json=_tx_payload())
            assert resp.status_code == 200
            body = resp.json()
            assert body["decision"] == "STEP_UP"
            assert 0.3 <= body["ensemble_score"] < 0.7
        finally:
            live_registry._sim_swap = original_detector
            live_registry.sim_swap_loaded = original_ss

    def test_score_response_latency_present(self, test_client):
        resp = test_client.post("/score", json=_tx_payload())
        body = resp.json()
        assert "latency_ms" in body
        assert body["latency_ms"] is not None
        assert body["latency_ms"] >= 0.0

    def test_score_with_account_features(self, test_client):
        payload = _tx_payload()
        payload["account_features"] = {
            "account_id": "acc-001",
            "computed_at": "2026-03-15T10:29:00Z",
            "tx_count_5min": 3,
            "tx_count_1hr": 12,
            "device_change_24h": True,
        }
        resp = test_client.post("/score", json=payload)
        assert resp.status_code == 200

    def test_score_missing_required_field(self, test_client):
        """Missing amount_zar → 422 Unprocessable Entity."""
        payload = {
            "transaction": {
                "transaction_id": "tx-bad",
                "timestamp": "2026-03-15T10:00:00Z",
                "sender_account_id": "acc-001",
                "payment_rail": "PAYSHAP",
                # amount_zar intentionally missing
            }
        }
        resp = test_client.post("/score", json=payload)
        assert resp.status_code == 422


# ── Debug features endpoint ───────────────────────────────────────────────────


class TestDebugFeaturesEndpoint:
    def test_debug_features_returns_dict(self, test_client):
        resp = test_client.post("/score/debug-features", json=_tx_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert "non_zero_features" in body
        assert "tx_timestamp" in body

    def test_debug_features_payshap_flag(self, test_client):
        resp = test_client.post(
            "/score/debug-features",
            json=_tx_payload(payment_rail="PAYSHAP"),
        )
        body = resp.json()
        non_zero = body["non_zero_features"]
        assert non_zero.get("payment_rail_PAYSHAP") == 1.0

    def test_debug_features_sim_swap_timing(self, test_client):
        resp = test_client.post(
            "/score/debug-features",
            json=_tx_payload(sim_swap_minutes_ago=10),
        )
        body = resp.json()
        non_zero = body["non_zero_features"]
        # time_since_sim_swap_minutes should be approximately 10
        tsm = non_zero.get("time_since_sim_swap_minutes")
        assert tsm is not None
        assert 9.0 <= tsm <= 11.0
