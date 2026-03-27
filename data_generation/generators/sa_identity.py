"""
South African identity data generator.

Generates realistic SA names, ID numbers (Luhn-valid), phone numbers,
and addresses using the Faker en_ZA locale with custom extensions.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from faker import Faker

from shared.constants import PROVINCE_POPULATION_WEIGHTS, SAProvince
from shared.utils.sa_validators import generate_luhn_check_digit, validate_sa_id_number

fake = Faker("en")  # en_ZA removed in Faker 40.x; SA-specific data is custom
Faker.seed(0)


# ── SA Phone number generator ─────────────────────────────────────────────────

# SA mobile prefixes by carrier
_MOBILE_PREFIXES = [
    # Vodacom (largest)
    "060",
    "061",
    "062",
    "063",
    "064",
    "071",
    "072",
    "073",
    "076",
    "078",
    "079",
    # MTN
    "066",
    "083",
    # Cell C
    "074",
    # Telkom Mobile
    "081",
]

_PREFIX_WEIGHTS = [
    # Vodacom ~40%
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    # MTN ~30%
    15,
    15,
    # Cell C ~20%
    20,
    # Telkom ~10%
    10,
]


def generate_sa_phone(rng: random.Random | None = None) -> str:
    """Generate a valid SA mobile number in E.164 format (+27XXXXXXXXX)."""
    r = rng or random
    prefix = r.choices(_MOBILE_PREFIXES, weights=_PREFIX_WEIGHTS, k=1)[0]
    suffix = "".join([str(r.randint(0, 9)) for _ in range(7)])
    local = prefix + suffix
    return "+27" + local[1:]  # replace leading 0 with +27


# ── SA ID number generator ────────────────────────────────────────────────────


def generate_sa_id_number(
    dob: date | None = None,
    gender: str | None = None,  # 'M' or 'F'
    is_citizen: bool = True,
    rng: random.Random | None = None,
) -> str:
    """
    Generate a valid SA 13-digit identity number with correct Luhn check digit.

    Format: YYMMDDGGGGSAZ
    """
    r = rng or random

    if dob is None:
        # Ages 18-70
        today = date.today()
        days_back = r.randint(18 * 365, 70 * 365)
        dob = today - timedelta(days=days_back)

    yy = str(dob.year % 100).zfill(2)
    mm = str(dob.month).zfill(2)
    dd = str(dob.day).zfill(2)

    # Gender sequence: 5000-9999 = male, 0000-4999 = female
    if gender == "M" or (gender is None and r.random() > 0.5):
        seq = r.randint(5000, 9999)
    else:
        seq = r.randint(0, 4999)
    gender_seq = str(seq).zfill(4)

    citizenship = "0" if is_citizen else "1"
    race_digit = "8"  # obsolete, always 8

    partial = yy + mm + dd + gender_seq + citizenship + race_digit
    check = generate_luhn_check_digit(partial)
    id_number = partial + check

    assert validate_sa_id_number(id_number), f"Generated invalid ID: {id_number}"
    return id_number


# ── SA Address generator ──────────────────────────────────────────────────────

# Suburb/township names per province for realistic SA addresses
_SUBURBS: dict[SAProvince, list[str]] = {
    SAProvince.GAUTENG: [
        "Soweto",
        "Alexandra",
        "Sandton",
        "Midrand",
        "Centurion",
        "Boksburg",
        "Germiston",
        "Benoni",
        "Tembisa",
        "Katlehong",
        "Mamelodi",
        "Soshanguve",
        "Pretoria CBD",
        "Randburg",
        "Roodepoort",
    ],
    SAProvince.WESTERN_CAPE: [
        "Khayelitsha",
        "Mitchell's Plain",
        "Bellville",
        "Stellenbosch",
        "George",
        "Paarl",
        "Cape Town CBD",
        "Gugulethu",
        "Nyanga",
        "Somerset West",
        "Atlantis",
        "Strand",
    ],
    SAProvince.KWAZULU_NATAL: [
        "Umlazi",
        "KwaMashu",
        "Pinetown",
        "Chatsworth",
        "Durban CBD",
        "Pietermaritzburg",
        "Newcastle",
        "Richards Bay",
        "Phoenix",
        "Tongaat",
        "Stanger",
        "Empangeni",
    ],
    SAProvince.EASTERN_CAPE: [
        "Mdantsane",
        "Motherwell",
        "Port Elizabeth CBD",
        "East London CBD",
        "Queenstown",
        "Mthatha",
        "King William's Town",
        "Bhisho",
    ],
    SAProvince.LIMPOPO: [
        "Polokwane",
        "Seshego",
        "Thohoyandou",
        "Phalaborwa",
        "Lephalale",
        "Mokopane",
        "Burgersfort",
    ],
    SAProvince.MPUMALANGA: [
        "Nelspruit",
        "Witbank",
        "Secunda",
        "Middelburg",
        "Barberton",
        "Lydenburg",
        "White River",
    ],
    SAProvince.NORTH_WEST: [
        "Rustenburg",
        "Mahikeng",
        "Klerksdorp",
        "Potchefstroom",
        "Brits",
        "Lichtenburg",
    ],
    SAProvince.FREE_STATE: [
        "Bloemfontein",
        "Welkom",
        "Botshabelo",
        "Phuthaditjhaba",
        "Sasolburg",
        "Kroonstad",
    ],
    SAProvince.NORTHERN_CAPE: [
        "Kimberley",
        "Upington",
        "Springbok",
        "De Aar",
        "Kuruman",
    ],
}

# SA postal codes per province (representative ranges)
_POSTAL_CODE_RANGES: dict[SAProvince, tuple[int, int]] = {
    SAProvince.GAUTENG: (1, 2199),
    SAProvince.WESTERN_CAPE: (7400, 8099),
    SAProvince.KWAZULU_NATAL: (3200, 4699),
    SAProvince.EASTERN_CAPE: (5200, 6499),
    SAProvince.LIMPOPO: (700, 999),
    SAProvince.MPUMALANGA: (1200, 1399),
    SAProvince.NORTH_WEST: (2500, 2899),
    SAProvince.FREE_STATE: (9300, 9999),
    SAProvince.NORTHERN_CAPE: (8200, 8899),
}


def generate_sa_address(
    province: SAProvince | None = None,
    rng: random.Random | None = None,
) -> dict[str, str]:
    """Generate a realistic SA address dict."""
    r = rng or random

    if province is None:
        province = r.choices(
            list(PROVINCE_POPULATION_WEIGHTS.keys()),
            weights=list(PROVINCE_POPULATION_WEIGHTS.values()),
        )[0]

    suburb = r.choice(_SUBURBS[province])
    street_num = r.randint(1, 999)
    street_name = fake.street_name()
    postal_min, postal_max = _POSTAL_CODE_RANGES[province]
    postal_code = str(r.randint(postal_min, postal_max)).zfill(4)

    return {
        "street_address": f"{street_num} {street_name}",
        "suburb": suburb,
        "province": province.value,
        "postal_code": postal_code,
        "country": "ZA",
    }


# ── Full identity bundle ──────────────────────────────────────────────────────


def generate_identity(
    province: SAProvince | None = None,
    rng: random.Random | None = None,
) -> dict:
    """Generate a complete SA identity bundle."""
    r = rng or random

    gender = r.choice(["M", "F"])
    first_name = fake.first_name_male() if gender == "M" else fake.first_name_female()
    last_name = fake.last_name()

    id_number = generate_sa_id_number(gender=gender, rng=r)  # type: ignore[arg-type]
    phone = generate_sa_phone(rng=r)  # type: ignore[arg-type]
    address = generate_sa_address(province=province, rng=r)  # type: ignore[arg-type]

    return {
        "first_name": first_name,
        "last_name": last_name,
        "full_name": f"{first_name} {last_name}",
        "id_number": id_number,
        "gender": gender,
        "phone": phone,
        "email": f"{first_name.lower()}.{last_name.lower()}{r.randint(1, 99)}@{r.choice(['gmail.com', 'webmail.co.za', 'icloud.com', 'outlook.com'])}",
        **address,
    }
