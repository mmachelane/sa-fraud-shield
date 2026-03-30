"""
POST /score  — real-time fraud scoring endpoint.

Accepts a transaction (and optional pre-enriched velocity features),
returns an ensemble fraud score and APPROVE / STEP_UP / BLOCK decision.

Example request:
    {
        "transaction": {
            "transaction_id": "abc-123",
            "timestamp": "2026-03-29T10:00:00Z",
            "sender_account_id": "acc-001",
            "amount_zar": 5000.00,
            "payment_rail": "payshap"
        }
    }

Example response:
    {
        "transaction_id": "abc-123",
        "ensemble_score": 0.82,
        "sim_swap_score": 0.87,
        "gnn_score": 0.74,
        "decision": "BLOCK",
        "alert": { ... },
        "latency_ms": 4.2
    }
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.services.scorer import _build_sim_swap_features, score_transaction
from monitoring.metrics import record_score_result_with_rail
from shared.schemas import ScoreRequest, ScoreResponse

router = APIRouter(prefix="/score", tags=["scoring"])


@router.post("", response_model=ScoreResponse)
async def score(request: Request, body: ScoreRequest) -> ScoreResponse:
    """
    Score a single transaction for fraud.

    - **transaction**: Required. The payment event to score.
    - **account_features**: Optional. Pre-fetched account features (device history,
      prior alerts). If omitted, defaults are used.
    """
    registry = request.app.state.registry

    if not registry.sim_swap_loaded and not registry.gnn_loaded:
        raise HTTPException(
            status_code=503,
            detail="No models loaded — check startup logs",
        )

    enriched = body.account_features.model_dump() if body.account_features else None

    result = score_transaction(
        tx=body.transaction,
        registry=registry,
        enriched=enriched,
    )

    # Record Prometheus metrics
    rail_val = (
        body.transaction.payment_rail.value
        if hasattr(body.transaction.payment_rail, "value")
        else str(body.transaction.payment_rail)
    )
    record_score_result_with_rail(result, rail_val)

    # Feed drift detector
    try:
        feature_df = _build_sim_swap_features(body.transaction, enriched)
        request.app.state.drift_detector.update([feature_df.iloc[0].to_dict()])
    except Exception:
        pass

    return ScoreResponse(
        transaction_id=result.transaction_id,
        ensemble_score=result.ensemble_score,
        sim_swap_score=result.sim_swap_score,
        gnn_score=result.gnn_score,
        decision=result.decision,
        alert=result.alert,
        latency_ms=result.latency_ms,
    )


@router.post("/debug-features")
async def debug_features(request: Request, body: ScoreRequest) -> dict:
    """Return the feature vector that would be passed to the SIM swap model."""
    registry = request.app.state.registry
    enriched = body.account_features.model_dump() if body.account_features else None
    feature_df = _build_sim_swap_features(body.transaction, enriched)

    if registry.sim_swap_loaded:
        model_features = registry.sim_swap.feature_names
        for col in model_features:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        feature_df = feature_df[model_features]
        raw_score = float(registry.sim_swap.predict_proba(feature_df)[0])
    else:
        raw_score = None

    non_zero = {k: v for k, v in feature_df.iloc[0].to_dict().items() if v != 0.0}
    return {
        "non_zero_features": non_zero,
        "raw_sim_swap_score": raw_score,
        "sim_swap_timestamp": str(body.transaction.sim_swap_timestamp),
        "tx_timestamp": str(body.transaction.timestamp),
    }
