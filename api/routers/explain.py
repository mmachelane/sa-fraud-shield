"""
POST /explain  — POPIA-compliant fraud explanation endpoint.

Returns the same score as /score PLUS:
  - SHAP values for each SIM swap feature
  - Top N features ranked by absolute SHAP value
  - Plain-English narrative (LLM-generated via Claude, template fallback)
  - isiZulu narrative for SA language compliance
  - Graph attribution scores from GNN explainer
    (POPIA Section 71: right to know why an automated decision was made)

Example response:
    {
        "transaction_id": "abc-123",
        "shap_values": {"new_device_first_tx": 261.28, "amount_zar": 107.84, ...},
        "top_features": [["new_device_first_tx", 261.28], ["amount_zar", 107.84]],
        "narrative_en": "This transaction was blocked because a new device...",
        "narrative_zu": "Le ntlawulelwano ivinjelwe...",
        "graph_attribution": {"account_uses_device": 0.68, "sim_swap_ring_pattern": 0.80},
        "model_version": "sim_swap_v1+federated_gnn_r9"
    }
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from api.services.scorer import _build_sim_swap_features, score_transaction
from explainability.gnn_explainer import compute_online_attribution
from explainability.llm_narrative import generate_narratives
from shared.schemas import ExplanationResponse, ScoreRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["explainability"])


@router.post("", response_model=ExplanationResponse)
async def explain(request: Request, body: ScoreRequest) -> ExplanationResponse:
    """
    Score a transaction and return a POPIA-compliant explanation.

    Includes SHAP values, top contributing features, LLM-generated
    narratives in English and isiZulu, and GNN graph attribution scores.
    Falls back to template narratives if ANTHROPIC_API_KEY is not set.
    """
    registry = request.app.state.registry

    if not registry.sim_swap_loaded:
        raise HTTPException(
            status_code=503,
            detail="SIM swap model not loaded — SHAP explanation unavailable",
        )

    tx = body.transaction
    enriched = body.account_features.model_dump() if body.account_features else None

    # ── Score ─────────────────────────────────────────────────────────────────
    result = score_transaction(tx=tx, registry=registry, enriched=enriched)

    # ── SHAP explanation ──────────────────────────────────────────────────────
    shap_values: dict[str, float] = {}
    top_features: list[tuple[str, float]] = []

    try:
        feature_df = _build_sim_swap_features(tx, enriched)
        model_features = registry.sim_swap.feature_names
        for col in model_features:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        feature_df = feature_df[model_features]

        shap_df = registry.sim_swap.explain(feature_df)
        shap_row = shap_df.iloc[0].to_dict()
        shap_values = {k: float(v) for k, v in shap_row.items()}
        top_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")

    # ── LLM narratives (English + isiZulu) ────────────────────────────────────
    rail_val = tx.payment_rail.value if hasattr(tx.payment_rail, "value") else str(tx.payment_rail)
    narrative_en, narrative_zu = generate_narratives(
        decision=result.decision,
        ensemble_score=result.ensemble_score,
        top_features=top_features,
        amount_zar=float(tx.amount_zar),
        payment_rail=rail_val,
        sim_swap_detected=bool(tx.sim_swap_detected),
    )

    # ── GNN graph attribution ─────────────────────────────────────────────────
    graph_attribution: dict[str, float] | None = None
    if result.gnn_score is not None:
        try:
            graph_attribution = compute_online_attribution(
                account_id=tx.sender_account_id,
                sender_device_id=tx.sender_device_id or "",
                receiver_account_id=tx.receiver_account_id,
                sim_swap_detected=bool(tx.sim_swap_detected),
                gnn_score=result.gnn_score,
                enriched=enriched,
            )
        except Exception as e:
            logger.warning(f"GNN attribution failed: {e}")

    # ── Model version ─────────────────────────────────────────────────────────
    model_version = "sim_swap_v1"
    if registry.gnn_loaded:
        model_version += "+federated_gnn_r9"

    return ExplanationResponse(
        transaction_id=result.transaction_id,
        shap_values=shap_values,
        top_features=top_features,
        narrative_en=narrative_en,
        narrative_zu=narrative_zu,
        graph_attribution=graph_attribution if graph_attribution else None,
        model_version=model_version,
    )
