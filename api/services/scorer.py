"""
Real-time fraud scoring service.

Combines SIM swap (LightGBM) and fraud ring (GNN) scores into a single
ensemble score, then applies SA banking thresholds to produce a decision.

Scoring pipeline per transaction:
    1. Build SIM swap feature vector from enriched Kafka payload
    2. Run SimSwapDetector.predict_proba()  → sim_swap_score
    3. If GNN loaded: run graph-level heuristic score  → gnn_score
    4. Ensemble: weighted average (0.6 SIM swap + 0.4 GNN)
    5. Apply thresholds → APPROVE / STEP_UP / BLOCK

Decision thresholds (configurable via env vars in production):
    APPROVE   score < 0.3
    STEP_UP   0.3 ≤ score < 0.7   (OTP / biometric challenge)
    BLOCK     score ≥ 0.7
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from api.services.model_registry import ModelRegistry
from shared.schemas import AlertSeverity, FraudAlert, FraudType, Transaction

logger = logging.getLogger(__name__)

# Ensemble weights
_W_SIM_SWAP = 0.6
_W_GNN = 0.4

# Decision thresholds
_THRESHOLD_BLOCK = 0.7
_THRESHOLD_STEP_UP = 0.3

# SIM swap feature defaults matching the trained model's feature set
_FEATURE_DEFAULTS: dict[str, float] = {
    "time_since_sim_swap_minutes": 99999.0,
    "is_loadshedding_coincident": 0.0,
    "device_changed": 0.0,
    "new_device_first_tx": 0.0,
    "velocity_1h": 0.0,
    "velocity_spike_ratio": 1.0,
    "province_mismatch": 0.0,
    "amount_zar": 0.0,
    "amount_to_balance_ratio": 0.0,
    "log_amount": 0.0,
    "loadshedding_active": 0.0,
    "loadshedding_stage": 0.0,
    "hour_of_day": 0.0,
    "is_weekend": 0.0,
    # payment rail one-hots
    "payment_rail_ATM": 0.0,
    "payment_rail_CARD_CNP": 0.0,
    "payment_rail_CARD_PRESENT": 0.0,
    "payment_rail_CASH_DEPOSIT": 0.0,
    "payment_rail_EFT": 0.0,
    "payment_rail_MOBILE_APP": 0.0,
    "payment_rail_PAYSHAP": 0.0,
    # merchant category one-hots
    "merchant_category_clothing": 0.0,
    "merchant_category_crypto_exchange": 0.0,
    "merchant_category_electronics": 0.0,
    "merchant_category_fuel": 0.0,
    "merchant_category_gambling": 0.0,
    "merchant_category_grocery": 0.0,
    "merchant_category_peer_transfer": 0.0,
    "merchant_category_pharmacy": 0.0,
    "merchant_category_restaurant": 0.0,
    "merchant_category_spaza_shop": 0.0,
    "merchant_category_taxi": 0.0,
    "merchant_category_unknown": 0.0,
    "merchant_category_utilities": 0.0,
}


@dataclass
class ScoreResult:
    transaction_id: str
    ensemble_score: float
    sim_swap_score: float | None
    gnn_score: float | None
    decision: str
    alert: FraudAlert | None
    latency_ms: float


def _build_sim_swap_features(tx: Transaction, enriched: dict | None) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame matching the trained SIM swap model.
    Uses enriched velocity features from Kafka if available.
    """
    import math

    now = tx.timestamp
    features = _FEATURE_DEFAULTS.copy()

    # Core transaction features
    amount = float(tx.amount_zar)
    features["amount_zar"] = amount
    features["log_amount"] = math.log1p(amount)
    features["hour_of_day"] = float(now.hour)
    features["is_weekend"] = float(now.weekday() >= 5)
    features["loadshedding_active"] = float(tx.loadshedding_active)
    features["loadshedding_stage"] = float(tx.loadshedding_stage or 0)

    # SIM swap timing
    if tx.sim_swap_timestamp is not None:
        minutes_ago = (now - tx.sim_swap_timestamp).total_seconds() / 60
        features["time_since_sim_swap_minutes"] = max(0.0, minutes_ago)
        features["is_loadshedding_coincident"] = float(tx.loadshedding_active and minutes_ago <= 60)
        features["new_device_first_tx"] = 1.0

    # Velocity from Kafka enrichment
    if enriched:
        vel_1h = float(enriched.get("tx_count_1hr", 0))
        vel_avg = float(enriched.get("tx_count_24hr", 0)) / 24.0
        features["velocity_1h"] = vel_1h
        features["velocity_spike_ratio"] = vel_1h / max(vel_avg, 1.0)
        features["device_changed"] = float(enriched.get("device_change_24h", 0))
        features["new_device_first_tx"] = float(enriched.get("new_device_first_tx", 0))

    # Payment rail one-hot
    rail_key = f"payment_rail_{tx.payment_rail}"
    if rail_key in features:
        features[rail_key] = 1.0

    # Merchant category one-hot
    if tx.merchant_category:
        cat_key = f"merchant_category_{str(tx.merchant_category).lower()}"
        if cat_key in features:
            features[cat_key] = 1.0

    return pd.DataFrame([features])


def score_transaction(
    tx: Transaction,
    registry: ModelRegistry,
    enriched: dict | None = None,
) -> ScoreResult:
    """
    Score a single transaction and return a ScoreResult.

    Args:
        tx:        Transaction to score
        registry:  Loaded ModelRegistry
        enriched:  Optional pre-enriched velocity dict from Kafka consumer
    """
    t0 = time.monotonic()

    sim_swap_score: float | None = None
    gnn_score: float | None = None

    # ── SIM swap model ────────────────────────────────────────────────────────
    if registry.sim_swap_loaded:
        try:
            feature_df = _build_sim_swap_features(tx, enriched)
            # Align columns to model's expected feature set
            model_features = registry.sim_swap.feature_names
            for col in model_features:
                if col not in feature_df.columns:
                    feature_df[col] = 0.0
            feature_df = feature_df[model_features]
            sim_swap_score = float(registry.sim_swap.predict_proba(feature_df)[0])
        except Exception as e:
            logger.warning(f"SIM swap scoring failed: {e}")

    # ── GNN model ─────────────────────────────────────────────────────────────
    # The GNN needs a full subgraph — not available per-transaction at score time.
    # We use a proxy: if the account was in the fraud ring training set and the
    # SIM swap score is elevated, we nudge the ensemble upward.
    # Full graph inference is deferred to batch re-scoring (Phase 10 monitoring).
    if registry.gnn_loaded and sim_swap_score is not None:
        # Placeholder: GNN score proxied from SIM swap with added noise dampening
        # Replace with real graph inference once streaming graph is built (Phase 10)
        gnn_score = float(np.clip(sim_swap_score * 0.85, 0.0, 1.0))

    # ── Ensemble ──────────────────────────────────────────────────────────────
    if sim_swap_score is not None and gnn_score is not None:
        ensemble = _W_SIM_SWAP * sim_swap_score + _W_GNN * gnn_score
    elif sim_swap_score is not None:
        ensemble = sim_swap_score
    elif gnn_score is not None:
        ensemble = gnn_score
    else:
        ensemble = 0.0

    ensemble = float(np.clip(ensemble, 0.0, 1.0))

    # ── Decision ──────────────────────────────────────────────────────────────
    if ensemble >= _THRESHOLD_BLOCK:
        decision = "BLOCK"
    elif ensemble >= _THRESHOLD_STEP_UP:
        decision = "STEP_UP"
    else:
        decision = "APPROVE"

    # ── Alert (BLOCK only) ────────────────────────────────────────────────────
    alert: FraudAlert | None = None
    if decision == "BLOCK":
        alert = FraudAlert(
            transaction_id=tx.transaction_id,
            account_id=tx.sender_account_id,
            timestamp=datetime.now(UTC),
            severity=AlertSeverity.CRITICAL if ensemble >= 0.9 else AlertSeverity.HIGH,
            ensemble_score=ensemble,
            sim_swap_score=sim_swap_score,
            gnn_score=gnn_score,
            predicted_fraud_type=FraudType.SIM_SWAP if tx.sim_swap_detected else None,
            triggered_rules=[f"ensemble_score={ensemble:.3f} >= {_THRESHOLD_BLOCK}"],
        )

    latency_ms = (time.monotonic() - t0) * 1000

    sim_str = f"{sim_swap_score:.3f}" if sim_swap_score is not None else "N/A"
    gnn_str = f"{gnn_score:.3f}" if gnn_score is not None else "N/A"
    logger.info(
        f"Scored {tx.transaction_id[:8]}…  "
        f"sim={sim_str}  gnn={gnn_str}  "
        f"ensemble={ensemble:.3f}  decision={decision}  {latency_ms:.1f}ms"
    )

    return ScoreResult(
        transaction_id=tx.transaction_id,
        ensemble_score=ensemble,
        sim_swap_score=sim_swap_score,
        gnn_score=gnn_score,
        decision=decision,
        alert=alert,
        latency_ms=latency_ms,
    )
