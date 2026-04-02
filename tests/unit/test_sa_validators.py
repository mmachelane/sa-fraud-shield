"""
Unit tests for shared/utils/sa_validators.py.

Covers:
  - SA ID number validation (format, date, Luhn)
  - Phone number normalization and validation
  - Bank account number validation
  - PayShap ID validation
  - SA postal code validation
  - Luhn check digit generation
"""

from __future__ import annotations

from shared.utils.sa_validators import (
    generate_luhn_check_digit,
    luhn_checksum,
    normalize_sa_phone,
    validate_payshap_id,
    validate_sa_account_number,
    validate_sa_id_number,
    validate_sa_phone,
    validate_sa_postal_code,
)

# ── Luhn checksum ─────────────────────────────────────────────────────────────


class TestLuhnChecksum:
    def test_valid_card_number(self):
        # Classic Luhn-valid number
        assert luhn_checksum("4532015112830366") == 0

    def test_invalid_card_number(self):
        assert luhn_checksum("4532015112830367") != 0

    def test_single_digit(self):
        # Edge case: single digit
        assert isinstance(luhn_checksum("0"), int)


class TestGenerateLuhnCheckDigit:
    def test_generated_digit_makes_valid_number(self):
        partial = "900101500008"  # 12 digits
        check = generate_luhn_check_digit(partial)
        full = partial + check
        assert luhn_checksum(full) == 0

    def test_returns_single_digit_string(self):
        check = generate_luhn_check_digit("900101500008")
        assert len(check) == 1
        assert check.isdigit()


# ── SA ID number ──────────────────────────────────────────────────────────────


class TestValidateSAIDNumber:
    # Known-valid SA ID: DOB 1990-01-15, female (seq 0002), citizen, Luhn-valid
    # We'll compute a valid one using the validator logic

    def _make_valid_id(self, dob: str = "900115", gender_seq: str = "0002") -> str:
        """Build a valid 13-digit SA ID number."""
        partial = f"{dob}{gender_seq}08"
        check = generate_luhn_check_digit(partial)
        return partial + check

    def test_valid_id_number(self):
        id_num = self._make_valid_id()
        assert validate_sa_id_number(id_num) is True

    def test_valid_male_id(self):
        id_num = self._make_valid_id(gender_seq="5001")
        assert validate_sa_id_number(id_num) is True

    def test_too_short(self):
        assert validate_sa_id_number("900101500") is False

    def test_too_long(self):
        assert validate_sa_id_number("90010150000881") is False

    def test_contains_letters(self):
        assert validate_sa_id_number("9001015000A82") is False

    def test_invalid_month(self):
        # Month 13 — invalid date
        id_num = "9013015000082"  # YYMMDD = 901301
        assert validate_sa_id_number(id_num) is False

    def test_invalid_day(self):
        # Day 32 — invalid date
        id_num = "9001325000082"
        assert validate_sa_id_number(id_num) is False

    def test_wrong_luhn(self):
        valid = self._make_valid_id()
        # Flip last digit
        wrong_last = str((int(valid[-1]) + 1) % 10)
        tampered = valid[:-1] + wrong_last
        assert validate_sa_id_number(tampered) is False

    def test_all_zeros(self):
        assert validate_sa_id_number("0000000000000") is False

    def test_empty_string(self):
        assert validate_sa_id_number("") is False


# ── SA phone numbers ──────────────────────────────────────────────────────────


class TestNormalizeSAPhone:
    def test_local_format_to_e164(self):
        assert normalize_sa_phone("0831234567") == "+27831234567"

    def test_already_e164(self):
        assert normalize_sa_phone("+27831234567") == "+27831234567"

    def test_27_prefix_without_plus(self):
        assert normalize_sa_phone("27831234567") == "+27831234567"

    def test_strips_spaces(self):
        assert normalize_sa_phone("083 123 4567") == "+27831234567"

    def test_strips_hyphens(self):
        assert normalize_sa_phone("083-123-4567") == "+27831234567"

    def test_invalid_prefix(self):
        # SA mobiles start with 06x, 07x, 08x
        assert normalize_sa_phone("0121234567") is None

    def test_too_short(self):
        assert normalize_sa_phone("083123") is None

    def test_empty(self):
        assert normalize_sa_phone("") is None


class TestValidateSAPhone:
    def test_valid_e164(self):
        assert validate_sa_phone("+27831234567") is True

    def test_local_format_rejected(self):
        # validate_sa_phone requires E.164 — local format should fail
        assert validate_sa_phone("0831234567") is False

    def test_invalid_country_code(self):
        assert validate_sa_phone("+44831234567") is False

    def test_too_long(self):
        assert validate_sa_phone("+278312345678") is False


# ── Bank account numbers ──────────────────────────────────────────────────────


class TestValidateSAAccountNumber:
    def test_valid_9_digits(self):
        assert validate_sa_account_number("123456789") is True

    def test_valid_11_digits(self):
        assert validate_sa_account_number("12345678901") is True

    def test_too_short(self):
        assert validate_sa_account_number("12345678") is False

    def test_too_long(self):
        assert validate_sa_account_number("123456789012") is False

    def test_contains_letters(self):
        assert validate_sa_account_number("12345678A") is False

    def test_capitec_style(self):
        # Capitec accounts are 10 digits
        assert validate_sa_account_number("1234567890") is True


# ── PayShap IDs ───────────────────────────────────────────────────────────────


class TestValidatePayShapID:
    def test_valid_capitec(self):
        assert validate_payshap_id("+27831234567@capitec") is True

    def test_valid_fnb(self):
        assert validate_payshap_id("+27761234567@fnb") is True

    def test_missing_at_sign(self):
        assert validate_payshap_id("+27831234567capitec") is False

    def test_wrong_country_code(self):
        assert validate_payshap_id("+44831234567@capitec") is False

    def test_local_number_format(self):
        # Must use E.164 with +27
        assert validate_payshap_id("0831234567@capitec") is False

    def test_uppercase_bank(self):
        # Pattern requires lowercase bank name
        assert validate_payshap_id("+27831234567@Capitec") is False

    def test_empty(self):
        assert validate_payshap_id("") is False


# ── SA postal codes ───────────────────────────────────────────────────────────


class TestValidateSAPostalCode:
    def test_valid_4_digits(self):
        assert validate_sa_postal_code("2000") is True  # Johannesburg

    def test_valid_cape_town(self):
        assert validate_sa_postal_code("8001") is True

    def test_too_short(self):
        assert validate_sa_postal_code("200") is False

    def test_too_long(self):
        assert validate_sa_postal_code("20001") is False

    def test_with_letters(self):
        assert validate_sa_postal_code("20AB") is False

    def test_empty(self):
        assert validate_sa_postal_code("") is False
