"""
Fraud ring (syndicate) generator.

Generates coordinated networks of 5-30 accounts with:
  - Shared devices (the primary GNN detection signal)
  - Overlapping transaction timing (coordinated)
  - Circular payment flows (money mule chains)
  - Shared registration addresses
  - Multiple accounts per device

This structure is invisible to per-transaction classifiers but produces
clear relational patterns in the heterogeneous graph that GNNs detect.

Pattern based on SA fraud syndicates: account holders recruited as mules,
coordinated by a central controller, with funds flowing outward through
crypto exchanges or cross-border transfers.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from shared.constants import (
    FraudType,
    PaymentRail,
    SAMerchantCategory,
)


@dataclass
class FraudRing:
    """A coordinated fraud ring with shared graph structure."""

    ring_id: str
    member_account_ids: list[str]  # All ring members
    controller_account_id: str  # Central controller (highest risk)
    shared_device_ids: list[str]  # Devices used by multiple ring members
    transactions: list[dict[str, Any]]
    total_fraud_amount_zar: Decimal
    ring_size: int


def generate_fraud_ring(
    ring_accounts: list[dict],
    start_time: datetime,
    rng: random.Random | None = None,
) -> FraudRing:
    """
    Generate a fraud ring transaction network.

    Args:
        ring_accounts: 5-30 account dicts; first is the controller
        start_time: Start of the coordinated fraud window
        rng: Random state for reproducibility
    """
    r = rng or random

    ring_id = str(uuid.uuid4())
    n_members = len(ring_accounts)
    controller = ring_accounts[0]
    members = ring_accounts[1:]

    # ── Shared devices (the GNN signal) ──────────────────────────────────────
    # 2-5 devices shared across multiple ring members
    n_shared = r.randint(2, min(5, n_members))
    shared_devices = [f"dev_ring_{uuid.uuid4().hex[:12]}" for _ in range(n_shared)]

    # Assign devices to accounts (each device shared by 2-4 accounts)
    account_devices: dict[str, list[str]] = {}
    for account in ring_accounts:
        # Each ring member uses 1-2 shared devices + maybe a personal device
        n_assigned = r.randint(1, min(2, n_shared))
        assigned = r.sample(shared_devices, n_assigned)
        if r.random() < 0.3:  # 30% chance of personal device too
            assigned.append(f"dev_personal_{uuid.uuid4().hex[:12]}")
        account_devices[account["account_id"]] = assigned

    transactions: list[dict[str, Any]] = []
    current_time = start_time
    total_amount = Decimal("0")

    # ── Phase 1: Account seeding (small deposits to look legitimate) ─────────
    for member in r.sample(members, min(len(members), r.randint(3, len(members)))):
        seed_amount = Decimal(str(round(r.uniform(50, 500), 2)))
        seed_time = current_time + timedelta(minutes=r.randint(0, 120))

        transactions.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": seed_time,
                "sender_account_id": controller["account_id"],
                "receiver_account_id": member["account_id"],
                "amount_zar": seed_amount,
                "payment_rail": PaymentRail.PAYSHAP.value
                if controller.get("payshap_id")
                else PaymentRail.EFT.value,
                "merchant_category": SAMerchantCategory.PEER_TRANSFER.value,
                "sender_device_id": r.choice(account_devices[controller["account_id"]]),
                "is_fraud": True,
                "fraud_type": FraudType.FRAUD_RING.value,
                "ring_id": ring_id,
                "sequence_step": "seed",
            }
        )

    current_time += timedelta(hours=r.randint(1, 24))

    # ── Phase 2: Coordinated rapid transactions (the coordinated timing signal) ─
    # Multiple ring members transact within a short window
    window_start = current_time
    for i, member in enumerate(members):
        # Transactions within a 30-minute window (coordinated timing signal)
        tx_time = window_start + timedelta(minutes=r.randint(0, 30))
        # External merchant: the ring is withdrawing from victim accounts
        amount = Decimal(str(round(r.uniform(500, 5000), 2)))
        total_amount += amount

        transactions.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": tx_time,
                "sender_account_id": member["account_id"],
                "receiver_account_id": None,
                "amount_zar": amount,
                "payment_rail": r.choice([PaymentRail.ATM.value, PaymentRail.CARD_PRESENT.value]),
                "merchant_category": r.choice(
                    [
                        SAMerchantCategory.CRYPTO_EXCHANGE.value,
                        SAMerchantCategory.GAMBLING.value,
                        SAMerchantCategory.UNKNOWN.value,
                    ]
                ),
                "sender_device_id": r.choice(account_devices[member["account_id"]]),
                "is_fraud": True,
                "fraud_type": FraudType.FRAUD_RING.value,
                "ring_id": ring_id,
                "sequence_step": "coordinated_withdrawal",
            }
        )

    # ── Phase 3: Circular transfers (confound tracing) ───────────────────────
    n_circular = r.randint(2, min(6, n_members - 1))
    circular_members = r.sample(members, n_circular)
    for j in range(len(circular_members)):
        src = circular_members[j]
        dst = circular_members[(j + 1) % len(circular_members)]
        circ_time = current_time + timedelta(hours=r.randint(1, 6))
        circ_amount = Decimal(str(round(r.uniform(100, 1500), 2)))

        transactions.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": circ_time,
                "sender_account_id": src["account_id"],
                "receiver_account_id": dst["account_id"],
                "amount_zar": circ_amount,
                "payment_rail": PaymentRail.PAYSHAP.value
                if src.get("payshap_id")
                else PaymentRail.EFT.value,
                "merchant_category": SAMerchantCategory.PEER_TRANSFER.value,
                "sender_device_id": r.choice(account_devices[src["account_id"]]),
                "is_fraud": True,
                "fraud_type": FraudType.FRAUD_RING.value,
                "ring_id": ring_id,
                "sequence_step": "circular_transfer",
            }
        )

    return FraudRing(
        ring_id=ring_id,
        member_account_ids=[a["account_id"] for a in ring_accounts],
        controller_account_id=controller["account_id"],
        shared_device_ids=shared_devices,
        transactions=transactions,
        total_fraud_amount_zar=total_amount,
        ring_size=n_members,
    )
