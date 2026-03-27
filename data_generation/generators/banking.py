"""
SA banking data generators: account numbers, PayShap IDs, and card numbers.
"""

from __future__ import annotations

import random

from shared.constants import (
    BANK_ACCOUNT_LENGTHS,
    BANK_BRANCH_CODES,
    BANK_PAYSHAP_SHORTCODES,
    SABank,
)
from shared.utils.sa_validators import validate_payshap_id

# Bank distribution weights (approximates SA market share)
_BANK_WEIGHTS: dict[SABank, float] = {
    SABank.CAPITEC: 0.30,  # Largest by account count
    SABank.FNB: 0.22,
    SABank.ABSA: 0.18,
    SABank.STANDARD_BANK: 0.15,
    SABank.NEDBANK: 0.08,
    SABank.TYMEBANK: 0.04,
    SABank.DISCOVERY: 0.02,
    SABank.AFRICAN_BANK: 0.01,
}

# % of customers on PayShap per bank (estimated)
_PAYSHAP_ADOPTION: dict[SABank, float] = {
    SABank.CAPITEC: 0.70,
    SABank.FNB: 0.65,
    SABank.ABSA: 0.55,
    SABank.STANDARD_BANK: 0.50,
    SABank.NEDBANK: 0.45,
    SABank.TYMEBANK: 0.80,
    SABank.DISCOVERY: 0.60,
    SABank.AFRICAN_BANK: 0.20,
}


def sample_bank(rng: random.Random | None = None) -> SABank:
    """Sample a SA bank according to realistic market share weights."""
    r = rng or random
    banks = list(_BANK_WEIGHTS.keys())
    weights = list(_BANK_WEIGHTS.values())
    return r.choices(banks, weights=weights, k=1)[0]


def generate_account_number(bank: SABank, rng: random.Random | None = None) -> str:
    """Generate a syntactically valid SA bank account number for a given bank."""
    r = rng or random
    length = BANK_ACCOUNT_LENGTHS[bank]
    # Capitec accounts start with 4 (real prefix pattern)
    if bank == SABank.CAPITEC:
        digits = "4" + "".join(str(r.randint(0, 9)) for _ in range(length - 1))
    elif bank == SABank.FNB:
        digits = "6" + "".join(str(r.randint(0, 9)) for _ in range(length - 1))
    elif bank == SABank.ABSA:
        digits = "4" + "".join(str(r.randint(0, 9)) for _ in range(length - 1))
    elif bank == SABank.STANDARD_BANK:
        digits = "0" + "".join(str(r.randint(0, 9)) for _ in range(length - 1))
    else:
        digits = "".join(str(r.randint(0, 9)) for _ in range(length))
    return digits


def generate_payshap_id(phone: str, bank: SABank) -> str | None:
    """
    Generate a PayShap ShapID for a customer.

    Format: +27XXXXXXXXX@{bank_shortcode}
    Returns None if the bank or customer doesn't have PayShap.
    """
    shortcode = BANK_PAYSHAP_SHORTCODES.get(bank)
    if shortcode is None:
        return None
    shapid = f"{phone}@{shortcode}"
    assert validate_payshap_id(shapid), f"Generated invalid ShapID: {shapid}"
    return shapid


def generate_account(
    phone: str,
    province: str,
    bank: SABank | None = None,
    rng: random.Random | None = None,
) -> dict:
    """Generate a complete SA bank account record."""
    r = rng or random

    if bank is None:
        bank = sample_bank(r)  # type: ignore[arg-type]

    account_number = generate_account_number(bank, r)  # type: ignore[arg-type]
    branch_code = BANK_BRANCH_CODES[bank]

    # PayShap adoption varies by bank
    has_payshap = r.random() < _PAYSHAP_ADOPTION[bank]
    payshap_id = generate_payshap_id(phone, bank) if has_payshap else None

    # Account age: mixture of old and new accounts
    # New accounts (<90 days) are higher fraud risk
    age_weights = [0.05, 0.10, 0.15, 0.70]  # <30d, 30-90d, 90-365d, >365d
    age_bucket = r.choices([0, 1, 2, 3], weights=age_weights)[0]
    if age_bucket == 0:
        account_age_days = r.randint(1, 30)
    elif age_bucket == 1:
        account_age_days = r.randint(31, 90)
    elif age_bucket == 2:
        account_age_days = r.randint(91, 365)
    else:
        account_age_days = r.randint(366, 365 * 15)

    # Monthly income: log-normal distribution centered around SA median (~R15,000)
    import math

    log_income = r.gauss(math.log(15000), 0.8)
    monthly_income = round(math.exp(log_income), 2)

    return {
        "bank": bank.value,
        "account_number": account_number,
        "branch_code": branch_code,
        "payshap_id": payshap_id,
        "account_age_days": account_age_days,
        "monthly_income_zar": monthly_income,
        "province": province,
    }
