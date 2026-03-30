"""
PSI-based model drift detector.

Population Stability Index (PSI) measures how much a feature's distribution
has shifted between training (baseline) and live scoring data.

PSI interpretation:
    PSI < 0.1   — No significant drift (stable)
    PSI 0.1–0.2 — Moderate drift (monitor)
    PSI > 0.2   — Significant drift (retrain alert)

PSI formula:
    PSI = sum((actual% - expected%) * ln(actual% / expected%))

Usage:
    detector = DriftDetector.from_training_data("data/processed/sim_swap_features.parquet")
    detector.update(score_results_batch)
    report = detector.report()
    # report.psi_scores → {feature: psi_value}
    # report.drifted_features → features with PSI > 0.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Features to monitor for drift
_MONITORED_FEATURES = [
    "time_since_sim_swap_minutes",
    "amount_zar",
    "log_amount",
    "velocity_1h",
    "velocity_spike_ratio",
    "hour_of_day",
    "loadshedding_active",
    "new_device_first_tx",
    "device_changed",
]

_N_BINS = 10
_PSI_ALERT_THRESHOLD = 0.2
_PSI_WARN_THRESHOLD = 0.1


@dataclass
class DriftReport:
    psi_scores: dict[str, float]
    drifted_features: list[str]
    warned_features: list[str]
    n_live_samples: int
    status: str  # "stable" | "warning" | "drift"


class DriftDetector:
    """
    Computes PSI between training baseline and a rolling window of live scores.
    Thread-safe for reads; call update() from a single background thread.
    """

    def __init__(
        self,
        baseline_stats: dict[str, dict],
        window_size: int = 1000,
    ) -> None:
        self._baseline = baseline_stats  # {feature: {bins: [...], expected_pct: [...]}}
        self._window_size = window_size
        self._live_buffer: list[dict] = []

    @classmethod
    def from_training_data(
        cls,
        features_path: str | Path,
        window_size: int = 1000,
        sample_size: int = 50_000,
    ) -> DriftDetector:
        """Build baseline statistics from the training feature parquet."""
        path = Path(features_path)
        if not path.exists():
            logger.warning(f"Training data not found at {path} — drift detection disabled")
            return cls(baseline_stats={}, window_size=window_size)

        logger.info(f"Building PSI baseline from {path}...")
        df = pd.read_parquet(path, columns=_MONITORED_FEATURES)

        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)

        baseline: dict[str, dict] = {}
        for feat in _MONITORED_FEATURES:
            if feat not in df.columns:
                continue
            vals = df[feat].fillna(0).astype(float).values
            bins = np.percentile(vals, np.linspace(0, 100, _N_BINS + 1))
            bins = np.unique(bins)  # remove duplicate edges
            if len(bins) < 2:
                continue
            counts, _ = np.histogram(vals, bins=bins)
            pct = counts / counts.sum()
            pct = np.where(pct == 0, 1e-6, pct)  # avoid log(0)
            baseline[feat] = {"bins": bins.tolist(), "expected_pct": pct.tolist()}

        logger.info(f"PSI baseline built for {len(baseline)} features")
        return cls(baseline_stats=baseline, window_size=window_size)

    def update(self, feature_rows: list[dict]) -> None:
        """Add live feature rows to the rolling buffer."""
        self._live_buffer.extend(feature_rows)
        if len(self._live_buffer) > self._window_size:
            self._live_buffer = self._live_buffer[-self._window_size :]

    def _compute_psi(self, feature: str, live_vals: np.ndarray) -> float:
        if feature not in self._baseline:
            return 0.0
        baseline = self._baseline[feature]
        bins = np.array(baseline["bins"])
        expected_pct = np.array(baseline["expected_pct"])

        counts, _ = np.histogram(live_vals, bins=bins)
        if counts.sum() == 0:
            return 0.0
        actual_pct = counts / counts.sum()
        actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    def report(self) -> DriftReport:
        """Compute PSI for all monitored features from current live buffer."""
        if not self._live_buffer or not self._baseline:
            return DriftReport(
                psi_scores={},
                drifted_features=[],
                warned_features=[],
                n_live_samples=len(self._live_buffer),
                status="stable",
            )

        live_df = pd.DataFrame(self._live_buffer)
        psi_scores: dict[str, float] = {}

        for feat in _MONITORED_FEATURES:
            if feat not in live_df.columns or feat not in self._baseline:
                continue
            vals = live_df[feat].fillna(0).astype(float).values
            psi_scores[feat] = self._compute_psi(feat, vals)

        drifted = [f for f, v in psi_scores.items() if v > _PSI_ALERT_THRESHOLD]
        warned = [
            f for f, v in psi_scores.items() if _PSI_WARN_THRESHOLD < v <= _PSI_ALERT_THRESHOLD
        ]

        if drifted:
            status = "drift"
        elif warned:
            status = "warning"
        else:
            status = "stable"

        return DriftReport(
            psi_scores=psi_scores,
            drifted_features=drifted,
            warned_features=warned,
            n_live_samples=len(self._live_buffer),
            status=status,
        )
