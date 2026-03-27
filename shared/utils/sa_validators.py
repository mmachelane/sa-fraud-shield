"""
South African identity and financial validators.

SA ID number structure (13 digits): YYMMDDGGGGSAZ
  YYMMDD  — date of birth
  GGGG    — gender sequence (0000-4999 = female, 5000-9999 = male)
  S       — citizenship (0 = SA citizen, 1 = permanent resident)
  A       — race (obsolete, always 8 in modern IDs)
  Z       — Luhn check digit
"""

from __future__ import annotations

import re
from datetime import date

# ── SA ID Number ──────────────────────────────────────────────────────────────


def luhn_checksum(number: str) -> int:
    """Standard Luhn algorithm checksum (returns 0 if valid)."""
    digits = [int(d) for d in number]
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    total = sum(odd_digits)
    for d in even_digits:
        total += sum(divmod(d * 2, 10))
    return total % 10


def validate_sa_id_number(id_number: str) -> bool:
    """
    Validate a South African 13-digit identity number.

    Checks:
    - Exactly 13 digits
    - Valid date of birth (YYMMDD)
    - Luhn check digit
    """
    if not re.fullmatch(r"\d{13}", id_number):
        return False

    # Validate date of birth
    yy, mm, dd = id_number[:2], id_number[2:4], id_number[4:6]
    try:
        year = int(yy)
        # Two-digit year: >= 00 but current year's last two digits
        current_yy = date.today().year % 100
        full_year = 1900 + year if year > current_yy else 2000 + year
        date(full_year, int(mm), int(dd))
    except ValueError:
        return False

    # Luhn check
    return luhn_checksum(id_number) == 0


def generate_luhn_check_digit(partial: str) -> str:
    """Given a string of N-1 digits, compute and return the Luhn check digit."""
    digits = [int(d) for d in partial]
    # Pad to make it as if computing for position len(partial)+1
    # Luhn: double every second-from-right in the full number
    # For generation: the partial has even length after adding check digit
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:  # these will be at odd positions in full number
            d *= 2
            if d > 9:
                d -= 9
        total += d
    check = (10 - (total % 10)) % 10
    return str(check)


# ── Phone Numbers ─────────────────────────────────────────────────────────────

_SA_MOBILE_PREFIXES = {
    # Vodacom
    "060",
    "061",
    "062",
    "063",
    "064",
    "065",
    "066",
    "067",
    "068",
    "069",
    "071",
    "072",
    "073",
    "074",
    "076",
    "078",
    "079",
    # MTN
    "083",
    "084",
    # Cell C
    "074",
    "076",
    "084",
    # Telkom Mobile
    "081",
}

_SA_PHONE_LOCAL_RE = re.compile(r"^0[6-8]\d{8}$")
_SA_PHONE_E164_RE = re.compile(r"^\+27[6-8]\d{8}$")


def normalize_sa_phone(phone: str) -> str | None:
    """
    Normalize a SA phone number to E.164 format (+27XXXXXXXXX).
    Returns None if the number cannot be normalized.
    """
    phone = re.sub(r"[\s\-()]", "", phone)

    if _SA_PHONE_E164_RE.match(phone):
        return phone
    if _SA_PHONE_LOCAL_RE.match(phone):
        return "+27" + phone[1:]
    if phone.startswith("27") and len(phone) == 11:
        return "+" + phone
    return None


def validate_sa_phone(phone: str) -> bool:
    """Return True if phone is a valid SA mobile number in E.164 format."""
    return bool(_SA_PHONE_E164_RE.match(phone))


# ── Bank Account Numbers ──────────────────────────────────────────────────────


def validate_sa_account_number(account_number: str, bank: str | None = None) -> bool:
    """
    Validate a SA bank account number.
    Digit-only, length between 9-11 digits.
    """
    if not re.fullmatch(r"\d{9,11}", account_number):
        return False
    return True


def validate_payshap_id(shapid: str) -> bool:
    """
    Validate a PayShap ShapID in the format +27XXXXXXXXX@{bank_shortcode}.

    Example: +27831234567@capitec
    """
    pattern = r"^\+27[6-8]\d{8}@[a-z]+$"
    return bool(re.match(pattern, shapid))


# ── SA Postal Codes ───────────────────────────────────────────────────────────


def validate_sa_postal_code(code: str) -> bool:
    """SA postal codes are 4 digits."""
    return bool(re.fullmatch(r"\d{4}", code))
