"""
Unit tests for monitoring/drift_detector.py.

Covers:
  - DriftDetector with empty baseline (always stable)
  - DriftDetector.update() and buffer size cap
  - PSI = 0 when live distribution matches baseline exactly
  - PSI > threshold when live distribution shifts dramatically
  - report() status logic: stable / warning / drift
"""

from __future__ import annotations

import numpy as np

from monitoring.drift_detector import (
    _PSI_ALERT_THRESHOLD,
    _PSI_WARN_THRESHOLD,
    DriftDetector,
    DriftReport,
)


def _make_detector_with_baseline(
    feature: str = "amount_zar",
    n_samples: int = 5000,
    window_size: int = 1000,
) -> tuple[DriftDetector, np.ndarray]:
    """
    Build a DriftDetector whose baseline is a normal distribution.
    Returns the detector and the baseline samples (for constructing matched live data).
    """
    rng = np.random.default_rng(42)
    baseline_vals = rng.normal(loc=1500, scale=500, size=n_samples)

    bins = np.percentile(baseline_vals, np.linspace(0, 100, 11))
    bins = np.unique(bins)
    counts, _ = np.histogram(baseline_vals, bins=bins)
    pct = counts / counts.sum()
    pct = np.where(pct == 0, 1e-6, pct)

    baseline_stats = {
        feature: {
            "bins": bins.tolist(),
            "expected_pct": pct.tolist(),
        }
    }
    return DriftDetector(baseline_stats=baseline_stats, window_size=window_size), baseline_vals


# ── Empty / disabled detector ─────────────────────────────────────────────────


class TestEmptyDetector:
    def test_empty_baseline_stable(self):
        detector = DriftDetector(baseline_stats={}, window_size=100)
        report = detector.report()
        assert report.status == "stable"
        assert report.psi_scores == {}
        assert report.drifted_features == []
        assert report.n_live_samples == 0

    def test_empty_buffer_stable(self):
        detector, _ = _make_detector_with_baseline()
        # No updates — buffer is empty
        report = detector.report()
        assert report.status == "stable"

    def test_from_training_data_missing_file(self, tmp_path):
        detector = DriftDetector.from_training_data(
            tmp_path / "nonexistent.parquet",
            window_size=100,
        )
        assert detector._baseline == {}


# ── update() ─────────────────────────────────────────────────────────────────


class TestUpdate:
    def test_update_adds_rows(self):
        detector, _ = _make_detector_with_baseline()
        detector.update([{"amount_zar": 1500.0}] * 10)
        assert len(detector._live_buffer) == 10

    def test_update_caps_buffer(self):
        detector, _ = _make_detector_with_baseline(window_size=50)
        detector.update([{"amount_zar": 1500.0}] * 60)
        assert len(detector._live_buffer) == 50

    def test_update_keeps_most_recent(self):
        detector, _ = _make_detector_with_baseline(window_size=5)
        detector.update([{"amount_zar": float(i)} for i in range(10)])
        # Buffer should contain the last 5 entries
        assert detector._live_buffer[0]["amount_zar"] == 5.0
        assert detector._live_buffer[-1]["amount_zar"] == 9.0


# ── PSI computation ───────────────────────────────────────────────────────────


class TestPSIComputation:
    def test_matched_distribution_psi_near_zero(self):
        """Live distribution matches baseline → PSI should be very small."""
        detector, baseline_vals = _make_detector_with_baseline(feature="amount_zar")
        # Use baseline values as live data — should have near-zero PSI
        live_rows = [{"amount_zar": float(v)} for v in baseline_vals[:1000]]
        detector.update(live_rows)
        report = detector.report()
        psi = report.psi_scores.get("amount_zar", 1.0)
        assert psi < _PSI_WARN_THRESHOLD

    def test_shifted_distribution_psi_above_alert(self):
        """Live distribution concentrated at one tail of the baseline range → PSI > 0.2."""
        detector, baseline_vals = _make_detector_with_baseline(feature="amount_zar")
        # Use values at the extreme high end of the baseline range (within-range but skewed)
        # baseline is N(1500, 500) so max is ~3500; use values near the max percentile
        max_val = float(np.max(baseline_vals))
        min_val = float(np.min(baseline_vals))
        # Concentrate all live values in the top 10% of the baseline range
        high_end = min_val + (max_val - min_val) * 0.9
        live_vals = np.linspace(high_end, max_val, 1000)
        live_rows = [{"amount_zar": float(v)} for v in live_vals]
        detector.update(live_rows)
        report = detector.report()
        psi = report.psi_scores.get("amount_zar", 0.0)
        assert psi > _PSI_ALERT_THRESHOLD

    def test_unknown_feature_returns_zero_psi(self):
        detector, _ = _make_detector_with_baseline(feature="amount_zar")
        # Feed data with a completely different feature name
        detector.update([{"some_other_feature": 1.0}] * 100)
        report = detector.report()
        # amount_zar has no live data in buffer → no PSI entry
        assert report.psi_scores.get("amount_zar") is None or report.psi_scores == {}


# ── report() status ───────────────────────────────────────────────────────────


class TestReportStatus:
    def test_stable_status(self):
        detector, baseline_vals = _make_detector_with_baseline()
        live_rows = [{"amount_zar": float(v)} for v in baseline_vals[:500]]
        detector.update(live_rows)
        report = detector.report()
        assert report.status in ("stable", "warning")  # allow minor variance
        assert isinstance(report.drifted_features, list)
        assert isinstance(report.warned_features, list)

    def test_drift_status(self):
        detector, baseline_vals = _make_detector_with_baseline(feature="amount_zar")
        max_val = float(np.max(baseline_vals))
        min_val = float(np.min(baseline_vals))
        high_end = min_val + (max_val - min_val) * 0.9
        live_vals = np.linspace(high_end, max_val, 1000)
        detector.update([{"amount_zar": float(v)} for v in live_vals])
        report = detector.report()
        assert report.status == "drift"
        assert "amount_zar" in report.drifted_features

    def test_report_n_live_samples(self):
        detector, _ = _make_detector_with_baseline()
        detector.update([{"amount_zar": 1500.0}] * 42)
        report = detector.report()
        assert report.n_live_samples == 42

    def test_drift_report_is_dataclass(self):
        detector = DriftDetector(baseline_stats={}, window_size=100)
        report = detector.report()
        assert isinstance(report, DriftReport)
        assert hasattr(report, "psi_scores")
        assert hasattr(report, "drifted_features")
        assert hasattr(report, "warned_features")
        assert hasattr(report, "n_live_samples")
        assert hasattr(report, "status")
