"""
POST /explain  — POPIA-compliant fraud explanation endpoint.

Returns the same score as /score PLUS:
  - SHAP values for each SIM swap feature
  - Top N features ranked by absolute SHAP value
  - Plain-English narrative suitable for customer communication
    (POPIA Section 71: right to know why an automated decision was made)

The narrative uses a template approach — no LLM dependency required.
LLM-powered narratives (litellm) are wired in Phase 9 (explainability/).

Example response:
    {
        "transaction_id": "abc-123",
        "shap_values": {"sim_swap_detected": 0.42, "tx_count_5min": 0.18, ...},
        "top_features": [["sim_swap_detected", 0.42], ["tx_count_5min", 0.18]],
        "narrative_en": "This transaction was blocked because ...",
        "graph_attribution": null,
        "model_version": "sim_swap_v1+federated_gnn_r9"
    }
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from api.services.scorer import _build_sim_swap_features, score_transaction
from shared.schemas import ExplanationResponse, ScoreRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["explainability"])

_DECISION_TEMPLATES = {
    "BLOCK": (
        "This transaction was blocked by our fraud detection system. "
        "The top indicators were: {top_features}. "
        "If you believe this is an error, please contact your bank."
    ),
    "STEP_UP": (
        "Additional verification was required for this transaction. "
        "The system detected elevated risk based on: {top_features}."
    ),
    "APPROVE": ("This transaction was approved. No significant fraud indicators were detected."),
}


def _format_feature_name(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _build_narrative(decision: str, top_features: list[tuple[str, float]]) -> str:
    top_str = ", ".join(f"{_format_feature_name(f)} ({v:+.3f})" for f, v in top_features[:3])
    template = _DECISION_TEMPLATES.get(decision, _DECISION_TEMPLATES["APPROVE"])
    return template.format(top_features=top_str)


@router.post("", response_model=ExplanationResponse)
async def explain(request: Request, body: ScoreRequest) -> ExplanationResponse:
    """
    Score a transaction and return a POPIA-compliant explanation.

    Includes SHAP values, top contributing features, and a plain-English
    narrative explaining the automated decision (POPIA Section 71).
    """
    registry = request.app.state.registry

    if not registry.sim_swap_loaded:
        raise HTTPException(
            status_code=503,
            detail="SIM swap model not loaded — SHAP explanation unavailable",
        )

    tx = body.transaction
    enriched = body.account_features.model_dump() if body.account_features else None

    # Score first
    result = score_transaction(tx=tx, registry=registry, enriched=enriched)

    # SHAP explanation from SIM swap model
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

    narrative = _build_narrative(result.decision, top_features)

    model_version = "sim_swap_v1"
    if registry.gnn_loaded:
        model_version += "+federated_gnn_r9"

    return ExplanationResponse(
        transaction_id=result.transaction_id,
        shap_values=shap_values,
        top_features=top_features,
        narrative_en=narrative,
        graph_attribution=None,  # wired in Phase 9
        model_version=model_version,
    )
