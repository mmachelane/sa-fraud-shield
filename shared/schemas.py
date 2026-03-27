"""
Shared Pydantic data contracts — imported by every module.

These schemas are the single source of truth for data shapes flowing through
the entire pipeline: generator → Kafka → streaming consumer → feature store
→ model serving → API → alert queue.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from shared.constants import (
    FraudType,
    PaymentRail,
    SABank,
    SAMerchantCategory,
    SAProvince,
)

# ── Identity ──────────────────────────────────────────────────────────────────


class SAIDNumber(BaseModel):
    """South African 13-digit identity number."""

    value: str = Field(..., pattern=r"^\d{13}$")

    @property
    def date_of_birth(self) -> str:
        """YYMMDD from first 6 digits."""
        return self.value[:6]

    @property
    def gender(self) -> str:
        """M if 5000-9999, F if 0000-4999."""
        seq = int(self.value[6:10])
        return "M" if seq >= 5000 else "F"

    @property
    def is_citizen(self) -> bool:
        """11th digit: 0 = SA citizen, 1 = permanent resident."""
        return self.value[10] == "0"


# ── Device ────────────────────────────────────────────────────────────────────


class DeviceFingerprint(BaseModel):
    device_id: str
    imei: str | None = None
    user_agent: str | None = None
    os_type: str | None = None  # android | ios | web
    app_version: str | None = None
    screen_resolution: str | None = None
    timezone: str | None = None
    ip_address: str | None = None
    province: SAProvince | None = None


# ── Account ───────────────────────────────────────────────────────────────────


class Account(BaseModel):
    account_id: str = Field(default_factory=lambda: str(uuid4()))
    bank: SABank
    account_number: str
    payshap_id: str | None = None  # +27XXXXXXXXX@{bank_shortcode}
    registration_phone: str  # E.164 format: +27XXXXXXXXX
    id_number: str | None = None  # SA ID number
    province: SAProvince
    account_age_days: int = Field(ge=0)
    is_compromised: bool = False  # Ground truth label for synthetic data
    monthly_income_zar: Decimal | None = None

    @field_validator("registration_phone")
    @classmethod
    def validate_sa_phone(cls, v: str) -> str:
        if not v.startswith("+27") or len(v) != 12:
            raise ValueError(f"Phone must be E.164 SA format (+27XXXXXXXXX), got: {v}")
        return v


# ── Transaction ───────────────────────────────────────────────────────────────


class Transaction(BaseModel):
    """Core transaction event — the fundamental unit flowing through the pipeline."""

    transaction_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime
    sender_account_id: str
    receiver_account_id: str | None = None  # None for ATM withdrawals
    amount_zar: Decimal = Field(gt=Decimal("0"))
    payment_rail: PaymentRail
    merchant_category: SAMerchantCategory = SAMerchantCategory.UNKNOWN
    merchant_id: str | None = None
    sender_device_id: str | None = None
    sender_province: SAProvince | None = None
    receiver_province: SAProvince | None = None

    # Ground truth (None = unlabeled, e.g. live streaming events)
    is_fraud: bool | None = None
    fraud_type: FraudType | None = None

    # Enrichment fields (populated by streaming consumer)
    sim_swap_detected: bool = False
    sim_swap_timestamp: datetime | None = None
    loadshedding_active: bool = False
    loadshedding_stage: int | None = None

    @field_validator("amount_zar", mode="before")
    @classmethod
    def coerce_to_decimal(cls, v: Any) -> Decimal:
        return Decimal(str(v))

    @model_validator(mode="after")
    def fraud_type_requires_is_fraud(self) -> Transaction:
        if self.fraud_type is not None and self.is_fraud is False:
            raise ValueError("fraud_type set but is_fraud=False")
        return self


# ── Fraud Alert ───────────────────────────────────────────────────────────────


class AlertSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FraudAlert(BaseModel):
    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    transaction_id: str
    account_id: str
    timestamp: datetime
    severity: AlertSeverity
    ensemble_score: float = Field(ge=0.0, le=1.0)
    sim_swap_score: float | None = Field(default=None, ge=0.0, le=1.0)
    gnn_score: float | None = Field(default=None, ge=0.0, le=1.0)
    predicted_fraud_type: FraudType | None = None
    triggered_rules: list[str] = Field(default_factory=list)
    explanation_id: str | None = None  # FK to cached SHAP/LLM explanation
    status: str = "OPEN"  # OPEN | INVESTIGATING | CLOSED_TP | CLOSED_FP


# ── Feature Store ─────────────────────────────────────────────────────────────


class AccountFeatures(BaseModel):
    """Pre-computed account-level features stored in Redis."""

    account_id: str
    computed_at: datetime

    # Velocity
    tx_count_5min: int = 0
    tx_count_1hr: int = 0
    tx_count_24hr: int = 0
    tx_count_7day: int = 0
    amount_sum_24hr: Decimal = Decimal("0")
    amount_sum_7day: Decimal = Decimal("0")

    # Behavioural baseline
    avg_tx_amount_30d: Decimal | None = None
    avg_daily_tx_count_30d: float | None = None
    usual_provinces: list[str] = Field(default_factory=list)
    usual_device_ids: list[str] = Field(default_factory=list)
    usual_merchant_categories: list[str] = Field(default_factory=list)

    # SIM swap signals
    sim_swap_detected: bool = False
    sim_swap_timestamp: datetime | None = None
    device_change_24h: bool = False
    new_device_first_tx: bool = False

    # Account risk
    days_since_account_open: int = 0
    is_high_risk_province: bool = False
    prior_fraud_alerts_30d: int = 0


# ── Scoring Request / Response (API contracts) ────────────────────────────────


class ScoreRequest(BaseModel):
    transaction: Transaction
    account_features: AccountFeatures | None = None  # optional pre-fetched


class ScoreResponse(BaseModel):
    transaction_id: str
    ensemble_score: float = Field(ge=0.0, le=1.0)
    sim_swap_score: float | None = None
    gnn_score: float | None = None
    decision: str  # APPROVE | STEP_UP | BLOCK
    alert: FraudAlert | None = None
    latency_ms: float | None = None


class ExplanationResponse(BaseModel):
    transaction_id: str
    shap_values: dict[str, float]  # feature_name → SHAP value
    top_features: list[tuple[str, float]]  # sorted by abs(SHAP value)
    narrative_en: str  # POPIA Section 71 narrative (English)
    narrative_zu: str | None = None  # isiZulu narrative (if available)
    graph_attribution: dict[str, float] | None = None  # edge_id → attribution
    model_version: str | None = None


# ── Kafka Event Envelope ──────────────────────────────────────────────────────


class KafkaTransactionEvent(BaseModel):
    """Envelope wrapping a Transaction for Kafka transport."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str = "transaction.created"
    schema_version: str = "1.0"
    produced_at: datetime
    partition_key: str  # sender_account_id — ensures same-account ordering
    payload: Transaction


class KafkaAlertEvent(BaseModel):
    """Envelope wrapping a FraudAlert for Kafka transport."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str = "fraud.alert.created"
    schema_version: str = "1.0"
    produced_at: datetime
    payload: FraudAlert
