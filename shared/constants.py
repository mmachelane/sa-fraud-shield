"""SA-specific constants used across all modules."""

from enum import Enum

# ── SA Banks ──────────────────────────────────────────────────────────────────


class SABank(str, Enum):
    CAPITEC = "capitec"
    FNB = "fnb"
    NEDBANK = "nedbank"
    STANDARD_BANK = "standard_bank"
    ABSA = "absa"
    TYMEBANK = "tymebank"
    DISCOVERY = "discovery"
    AFRICAN_BANK = "african_bank"


# Universal branch codes (used in EFT routing)
BANK_BRANCH_CODES: dict[SABank, str] = {
    SABank.CAPITEC: "470010",
    SABank.FNB: "250655",
    SABank.NEDBANK: "198765",
    SABank.STANDARD_BANK: "051001",
    SABank.ABSA: "632005",
    SABank.TYMEBANK: "678910",
    SABank.DISCOVERY: "679000",
    SABank.AFRICAN_BANK: "430000",
}

# PayShap shortcodes used in ShapIDs (+27XXXXXXXXX@{shortcode})
BANK_PAYSHAP_SHORTCODES: dict[SABank, str] = {
    SABank.CAPITEC: "capitec",
    SABank.FNB: "fnb",
    SABank.NEDBANK: "nedbank",
    SABank.STANDARD_BANK: "standardbank",
    SABank.ABSA: "absa",
    SABank.TYMEBANK: "tymebank",
    SABank.DISCOVERY: "discovery",
    SABank.AFRICAN_BANK: "africanbank",
}

# Account number digit lengths per bank
BANK_ACCOUNT_LENGTHS: dict[SABank, int] = {
    SABank.CAPITEC: 10,
    SABank.FNB: 11,
    SABank.NEDBANK: 11,
    SABank.STANDARD_BANK: 9,
    SABank.ABSA: 10,
    SABank.TYMEBANK: 10,
    SABank.DISCOVERY: 10,
    SABank.AFRICAN_BANK: 10,
}


# ── SA Provinces ──────────────────────────────────────────────────────────────


class SAProvince(str, Enum):
    GAUTENG = "GP"
    WESTERN_CAPE = "WC"
    KWAZULU_NATAL = "KZN"
    EASTERN_CAPE = "EC"
    LIMPOPO = "LP"
    MPUMALANGA = "MP"
    NORTH_WEST = "NW"
    FREE_STATE = "FS"
    NORTHERN_CAPE = "NC"


# Population weight for realistic geographic distribution
PROVINCE_POPULATION_WEIGHTS: dict[SAProvince, float] = {
    SAProvince.GAUTENG: 0.262,
    SAProvince.KWAZULU_NATAL: 0.199,
    SAProvince.WESTERN_CAPE: 0.115,
    SAProvince.EASTERN_CAPE: 0.127,
    SAProvince.LIMPOPO: 0.103,
    SAProvince.MPUMALANGA: 0.079,
    SAProvince.NORTH_WEST: 0.067,
    SAProvince.FREE_STATE: 0.051,
    SAProvince.NORTHERN_CAPE: 0.022,
}


# ── Payment Rails ─────────────────────────────────────────────────────────────


class PaymentRail(str, Enum):
    PAYSHAP = "PAYSHAP"  # Instant payment via SARB PayShap
    EFT = "EFT"  # Electronic funds transfer
    CARD_CNP = "CARD_CNP"  # Card-not-present (e-commerce)
    CARD_PRESENT = "CARD_PRESENT"  # POS terminal
    ATM = "ATM"  # ATM cash withdrawal
    MOBILE_APP = "MOBILE_APP"  # In-app transfer (Capitec, FNB)
    CASH_DEPOSIT = "CASH_DEPOSIT"


# ── Fraud Types ────────────────────────────────────────────────────────────────


class FraudType(str, Enum):
    SIM_SWAP = "SIM_SWAP"
    FRAUD_RING = "FRAUD_RING"
    ACCOUNT_TAKEOVER = "ACCOUNT_TAKEOVER"
    APP_FRAUD = "APP_FRAUD"  # Authorised Push Payment fraud
    CARD_NOT_PRESENT = "CARD_NOT_PRESENT"
    IDENTITY_THEFT = "IDENTITY_THEFT"


# ── Merchant Categories (SA-specific) ─────────────────────────────────────────


class SAMerchantCategory(str, Enum):
    SPAZA_SHOP = "spaza_shop"
    GROCERY = "grocery"
    FUEL = "fuel"
    TAXI = "taxi"
    RESTAURANT = "restaurant"
    CLOTHING = "clothing"
    ELECTRONICS = "electronics"
    PHARMACY = "pharmacy"
    UTILITIES = "utilities"
    GAMBLING = "gambling"
    CRYPTO_EXCHANGE = "crypto_exchange"
    PEER_TRANSFER = "peer_transfer"  # PayShap / EFT to individual
    SALARY = "salary"
    UNKNOWN = "unknown"


# ── SARB / FICA Thresholds ─────────────────────────────────────────────────────

# Cash Transaction Report threshold (must report to FIC within 48h)
FICA_CTR_THRESHOLD_ZAR = 24_999

# Suspicious Transaction Report trigger (structuring detection)
FICA_STR_STRUCTURING_THRESHOLD_ZAR = 20_000

# SARB PayShap per-transaction limit
PAYSHAP_PER_TRANSACTION_LIMIT_ZAR = 3_000

# SARB PayShap daily limit
PAYSHAP_DAILY_LIMIT_ZAR = 5_000

# Typical daily EFT limit for retail banking
DEFAULT_DAILY_EFT_LIMIT_ZAR = 25_000


# ── Load Shedding ─────────────────────────────────────────────────────────────

# EskomSePush public API base URL
ESKOM_SE_PUSH_API_BASE = "https://developer.sepush.co.za/business/2.0"

# Typical outage duration in minutes per stage
LOADSHEDDING_STAGE_DURATION_MINUTES: dict[int, int] = {
    1: 120,  # 2 hours
    2: 150,  # 2.5 hours
    3: 180,  # 3 hours
    4: 240,  # 4 hours
    5: 270,  # 4.5 hours
    6: 300,  # 5 hours
}


# ── Graph Schema ──────────────────────────────────────────────────────────────

# Heterogeneous graph node types
GRAPH_NODE_TYPES = ["account", "device", "merchant", "phone_number"]

# Heterogeneous graph edge types (src_type, edge_type, dst_type)
GRAPH_EDGE_TYPES = [
    ("account", "transacts_with", "merchant"),
    ("account", "uses_device", "device"),
    ("account", "registered_on", "phone_number"),
    ("account", "payshap_transfer", "account"),
    ("account", "eft_transfer", "account"),
    ("device", "shared_by", "account"),  # reverse of uses_device
    ("phone_number", "used_by", "account"),  # reverse of registered_on
]


# ── Velocity Windows ──────────────────────────────────────────────────────────

VELOCITY_WINDOWS_SECONDS = {
    "5min": 300,
    "1hr": 3600,
    "24hr": 86400,
    "7day": 604800,
}


# ── Model Thresholds ──────────────────────────────────────────────────────────

# Default score thresholds (tunable via env vars in production)
SIM_SWAP_ALERT_THRESHOLD = 0.70
SIM_SWAP_REVIEW_THRESHOLD = 0.40
GNN_FRAUD_RING_ALERT_THRESHOLD = 0.75
ENSEMBLE_ALERT_THRESHOLD = 0.65
ENSEMBLE_STEP_UP_THRESHOLD = 0.35
