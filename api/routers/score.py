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

from api.services.scorer import score_transaction
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

    return ScoreResponse(
        transaction_id=result.transaction_id,
        ensemble_score=result.ensemble_score,
        sim_swap_score=result.sim_swap_score,
        gnn_score=result.gnn_score,
        decision=result.decision,
        alert=result.alert,
        latency_ms=result.latency_ms,
    )
