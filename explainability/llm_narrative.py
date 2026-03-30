"""
LLM-powered POPIA-compliant fraud explanation narratives.

Generates plain-English and isiZulu explanations for fraud decisions
using Claude via litellm. Falls back to template narratives if the
API is unavailable or not configured.

POPIA Section 71: data subjects have the right to be informed of the
reasons for any automated decision that affects them. This module
satisfies that obligation with human-readable, non-technical narratives.

Usage:
    from explainability.llm_narrative import generate_narratives

    en, zu = generate_narratives(
        decision="BLOCK",
        ensemble_score=0.94,
        top_features=[("new_device_first_tx", 261.28), ("amount_zar", 107.84)],
        amount_zar=15000.0,
        payment_rail="PAYSHAP",
    )
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Template fallbacks (used when LLM unavailable)
_TEMPLATES_EN = {
    "BLOCK": (
        "This transaction was blocked by our automated fraud detection system. "
        "The primary risk indicators were: {top_features}. "
        "The transaction scored {score:.0%} on our fraud risk scale. "
        "If you believe this is an error, please contact your bank immediately."
    ),
    "STEP_UP": (
        "Additional verification is required for this transaction. "
        "Our system detected elevated risk based on: {top_features}. "
        "Please complete the verification step to proceed."
    ),
    "APPROVE": ("This transaction was approved. No significant fraud indicators were detected."),
}

_TEMPLATES_ZU = {
    "BLOCK": (
        "Le ntlawulelwano ivinjelwe uhlelo lwethu lokutholwa kobugebengu. "
        "Izimpawu zengozi eziyinhloko zaziwukuthi: {top_features}. "
        "Uma ucabanga ukuthi lena iphutha, xhumana nebhange lakho ngokushesha."
    ),
    "STEP_UP": (
        "Ukuqinisekiswa okwengeziwe kuyadingeka le ntlawulelwano. "
        "Uhlelo lwethu luthole ingozi ephakeme."
    ),
    "APPROVE": ("Le ntlawulelwano ivunyelwe. Ayikho izimpawu ezibalulekile zobugebengu ezitoliwe."),
}


def _format_feature_name(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _template_narrative(
    decision: str,
    ensemble_score: float,
    top_features: list[tuple[str, float]],
    lang: str = "en",
) -> str:
    templates = _TEMPLATES_EN if lang == "en" else _TEMPLATES_ZU
    top_str = ", ".join(_format_feature_name(f) for f, v in top_features[:3] if v > 0)
    if not top_str:
        top_str = ", ".join(_format_feature_name(f) for f, _ in top_features[:3])
    template = templates.get(decision, templates["APPROVE"])
    return template.format(top_features=top_str, score=ensemble_score)


def generate_narratives(
    decision: str,
    ensemble_score: float,
    top_features: list[tuple[str, float]],
    amount_zar: float,
    payment_rail: str,
    sim_swap_detected: bool = False,
) -> tuple[str, str | None]:
    """
    Generate English and isiZulu POPIA narratives for a fraud decision.

    Returns (narrative_en, narrative_zu). Falls back to templates if
    ANTHROPIC_API_KEY is not set or the LLM call fails.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.debug("ANTHROPIC_API_KEY not set — using template narratives")
        return (
            _template_narrative(decision, ensemble_score, top_features, "en"),
            _template_narrative(decision, ensemble_score, top_features, "zu"),
        )

    try:
        return _llm_narratives(
            decision=decision,
            ensemble_score=ensemble_score,
            top_features=top_features,
            amount_zar=amount_zar,
            payment_rail=payment_rail,
            sim_swap_detected=sim_swap_detected,
        )
    except Exception as e:
        logger.warning(f"LLM narrative generation failed: {e} — using templates")
        return (
            _template_narrative(decision, ensemble_score, top_features, "en"),
            _template_narrative(decision, ensemble_score, top_features, "zu"),
        )


def _llm_narratives(
    decision: str,
    ensemble_score: float,
    top_features: list[tuple[str, float]],
    amount_zar: float,
    payment_rail: str,
    sim_swap_detected: bool,
) -> tuple[str, str]:
    import litellm

    top_str = "\n".join(f"  - {_format_feature_name(f)}: {v:+.3f}" for f, v in top_features[:5])

    decision_label = {
        "BLOCK": "blocked",
        "STEP_UP": "flagged for additional verification",
        "APPROVE": "approved",
    }.get(decision, decision.lower())

    sim_context = (
        " A SIM swap was detected on this account shortly before the transaction."
        if sim_swap_detected
        else ""
    )

    prompt = f"""You are a POPIA-compliant fraud explanation system for a South African bank.

A transaction of R{amount_zar:,.2f} via {payment_rail} was {decision_label} by the automated fraud detection system (risk score: {ensemble_score:.0%}).{sim_context}

The top contributing risk factors were:
{top_str}

Write two short explanations (2-3 sentences each):
1. ENGLISH: A clear, non-technical explanation for the customer explaining why this decision was made. Use plain language. Mention the most important 2-3 risk factors naturally. End with guidance (contact bank if error, or complete verification if step-up).
2. ISIZULU: The same explanation translated into isiZulu, keeping it natural and respectful.

Respond in this exact format:
ENGLISH: <explanation>
ISIZULU: <explanation>"""

    response = litellm.completion(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3,
    )

    text: str = response.choices[0].message.content.strip()

    narrative_en = ""
    narrative_zu = ""

    for line in text.splitlines():
        if line.startswith("ENGLISH:"):
            narrative_en = line[len("ENGLISH:") :].strip()
        elif line.startswith("ISIZULU:"):
            narrative_zu = line[len("ISIZULU:") :].strip()

    if not narrative_en:
        narrative_en = _template_narrative("BLOCK", ensemble_score, top_features, "en")
    if not narrative_zu:
        narrative_zu = _template_narrative("BLOCK", ensemble_score, top_features, "zu")

    return narrative_en, narrative_zu
