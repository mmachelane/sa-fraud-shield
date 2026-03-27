"""
SA transaction generator with realistic ZAR amounts, timing, and merchant categories.
"""

from __future__ import annotations

import math
import random
import uuid
from datetime import datetime
from decimal import Decimal

from shared.constants import (
    PAYSHAP_PER_TRANSACTION_LIMIT_ZAR,
    PaymentRail,
    SAMerchantCategory,
)

# ── Amount distributions ──────────────────────────────────────────────────────

# (mean_log, std_log, min_zar, max_zar) for log-normal distributions
_AMOUNT_PARAMS: dict[SAMerchantCategory, tuple[float, float, float, float]] = {
    SAMerchantCategory.SPAZA_SHOP: (math.log(50), 0.6, 5, 500),
    SAMerchantCategory.GROCERY: (math.log(350), 0.7, 20, 5000),
    SAMerchantCategory.FUEL: (math.log(600), 0.4, 100, 2500),
    SAMerchantCategory.TAXI: (math.log(25), 0.5, 5, 150),
    SAMerchantCategory.RESTAURANT: (math.log(200), 0.6, 30, 2000),
    SAMerchantCategory.CLOTHING: (math.log(500), 0.7, 50, 5000),
    SAMerchantCategory.ELECTRONICS: (math.log(2000), 0.8, 100, 50000),
    SAMerchantCategory.PHARMACY: (math.log(250), 0.6, 20, 3000),
    SAMerchantCategory.UTILITIES: (math.log(800), 0.5, 50, 5000),
    SAMerchantCategory.GAMBLING: (math.log(300), 1.0, 10, 10000),
    SAMerchantCategory.CRYPTO_EXCHANGE: (math.log(3000), 1.2, 100, 100000),
    SAMerchantCategory.PEER_TRANSFER: (math.log(1500), 0.9, 50, PAYSHAP_PER_TRANSACTION_LIMIT_ZAR),
    SAMerchantCategory.SALARY: (math.log(15000), 0.7, 3000, 150000),
    SAMerchantCategory.UNKNOWN: (math.log(200), 0.8, 5, 10000),
}

# Merchant category weights for normal (non-fraud) transactions
_CATEGORY_WEIGHTS: dict[SAMerchantCategory, float] = {
    SAMerchantCategory.GROCERY: 0.25,
    SAMerchantCategory.SPAZA_SHOP: 0.15,
    SAMerchantCategory.FUEL: 0.12,
    SAMerchantCategory.RESTAURANT: 0.10,
    SAMerchantCategory.TAXI: 0.08,
    SAMerchantCategory.CLOTHING: 0.07,
    SAMerchantCategory.PHARMACY: 0.06,
    SAMerchantCategory.UTILITIES: 0.05,
    SAMerchantCategory.ELECTRONICS: 0.03,
    SAMerchantCategory.GAMBLING: 0.03,
    SAMerchantCategory.CRYPTO_EXCHANGE: 0.01,
    SAMerchantCategory.PEER_TRANSFER: 0.04,
    SAMerchantCategory.UNKNOWN: 0.01,
}

# Payment rail weights per merchant category
_RAIL_WEIGHTS: dict[SAMerchantCategory, dict[PaymentRail, float]] = {
    SAMerchantCategory.GROCERY: {
        PaymentRail.CARD_PRESENT: 0.5,
        PaymentRail.CARD_CNP: 0.1,
        PaymentRail.MOBILE_APP: 0.3,
        PaymentRail.CASH_DEPOSIT: 0.1,
    },
    SAMerchantCategory.SPAZA_SHOP: {
        PaymentRail.CASH_DEPOSIT: 0.6,
        PaymentRail.MOBILE_APP: 0.3,
        PaymentRail.CARD_PRESENT: 0.1,
    },
    SAMerchantCategory.FUEL: {
        PaymentRail.CARD_PRESENT: 0.7,
        PaymentRail.CARD_CNP: 0.05,
        PaymentRail.MOBILE_APP: 0.2,
        PaymentRail.CASH_DEPOSIT: 0.05,
    },
    SAMerchantCategory.TAXI: {PaymentRail.CASH_DEPOSIT: 0.7, PaymentRail.MOBILE_APP: 0.3},
    SAMerchantCategory.PEER_TRANSFER: {
        PaymentRail.PAYSHAP: 0.6,
        PaymentRail.EFT: 0.3,
        PaymentRail.MOBILE_APP: 0.1,
    },
    SAMerchantCategory.SALARY: {PaymentRail.EFT: 1.0},
    SAMerchantCategory.CRYPTO_EXCHANGE: {
        PaymentRail.EFT: 0.5,
        PaymentRail.CARD_CNP: 0.3,
        PaymentRail.PAYSHAP: 0.2,
    },
}

_DEFAULT_RAIL_WEIGHTS = {
    PaymentRail.CARD_PRESENT: 0.4,
    PaymentRail.CARD_CNP: 0.2,
    PaymentRail.MOBILE_APP: 0.25,
    PaymentRail.EFT: 0.1,
    PaymentRail.PAYSHAP: 0.05,
}


def generate_amount(
    category: SAMerchantCategory,
    rng: random.Random | None = None,
) -> Decimal:
    """Sample a realistic ZAR transaction amount for a merchant category."""
    r = rng or random
    mu, sigma, min_val, max_val = _AMOUNT_PARAMS[category]
    amount = math.exp(r.gauss(mu, sigma))
    amount = max(min_val, min(max_val, amount))
    # Round to nearest R1 for larger amounts, 50c for small
    if amount >= 10:
        amount = round(amount)
    else:
        amount = round(amount * 2) / 2
    return Decimal(str(amount))


def sample_merchant_category(rng: random.Random | None = None) -> SAMerchantCategory:
    """Sample a merchant category weighted by realistic transaction frequency."""
    r = rng or random
    categories = list(_CATEGORY_WEIGHTS.keys())
    weights = list(_CATEGORY_WEIGHTS.values())
    return r.choices(categories, weights=weights, k=1)[0]


def sample_payment_rail(
    category: SAMerchantCategory,
    rng: random.Random | None = None,
) -> PaymentRail:
    """Sample a payment rail appropriate for the merchant category."""
    r = rng or random
    rail_weights = _RAIL_WEIGHTS.get(category, _DEFAULT_RAIL_WEIGHTS)
    rails = list(rail_weights.keys())
    weights = list(rail_weights.values())
    return r.choices(rails, weights=weights, k=1)[0]


# ── Timing ────────────────────────────────────────────────────────────────────


def generate_timestamp(
    base_date: datetime,
    rng: random.Random | None = None,
) -> datetime:
    """
    Generate a realistic transaction timestamp for a given day.

    SA transaction patterns:
    - Peak hours: 8am-12pm, 4pm-7pm
    - Low volume: 11pm-5am
    - Salary day (25th): higher volume
    """
    r = rng or random

    # Hour distribution: weighted toward business hours
    hours = list(range(24))
    hour_weights = [
        0.3,
        0.2,
        0.15,
        0.1,
        0.1,
        0.2,  # 0-5am (night/early)
        0.5,
        1.5,
        3.0,
        3.5,
        3.5,
        3.0,  # 6-11am
        2.5,
        2.5,
        2.0,
        2.0,
        3.5,
        3.5,  # 12-17pm
        3.0,
        2.5,
        2.0,
        1.5,
        1.0,
        0.5,  # 18-23pm
    ]
    # Boost salary day (25th) volume
    if base_date.day == 25:
        hour_weights = [w * 1.5 for w in hour_weights]

    hour = r.choices(hours, weights=hour_weights, k=1)[0]
    minute = r.randint(0, 59)
    second = r.randint(0, 59)

    return base_date.replace(hour=hour, minute=minute, second=second, microsecond=0)


# ── Device fingerprint ────────────────────────────────────────────────────────

_USER_AGENTS = [
    "CapitecBankApp/5.2.1 Android/13",
    "CapitecBankApp/5.1.0 iOS/17.1",
    "FNBApp/9.4.2 Android/13",
    "FNBApp/9.3.1 iOS/16.5",
    "AbsaBankingApp/6.8.0 Android/12",
    "NedBankMoney/4.2.1 Android/13",
    "StandardBankApp/7.1.0 iOS/17.0",
    "TymeBankApp/3.5.2 Android/13",
    "Mozilla/5.0 (Linux; Android 13) Chrome/120",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) Safari/604.1",
]


def generate_device_id(rng: random.Random | None = None) -> str:
    r = rng or random
    return f"dev_{uuid.UUID(int=r.getrandbits(128)).hex[:16]}"


def generate_device_fingerprint(rng: random.Random | None = None) -> dict:
    r = rng or random
    os_type = r.choice(["android", "ios", "web"])
    return {
        "device_id": generate_device_id(r),  # type: ignore[arg-type]
        "user_agent": r.choice(_USER_AGENTS),
        "os_type": os_type,
        "app_version": f"{r.randint(3, 9)}.{r.randint(0, 9)}.{r.randint(0, 9)}",
    }
