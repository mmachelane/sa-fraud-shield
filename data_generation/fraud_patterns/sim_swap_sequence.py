"""
SIM swap fraud sequence generator — SA's signature fraud type.

Generates realistic SIM swap attack chains including:
  1. Social engineering (pre-event, not a transaction)
  2. SIM swap at carrier (timestamp T)
  3. Carrier processing delay (15-90 minutes)
  4. New SIM activation
  5. OTP interception begins
  6. Probe transaction (small amount to verify access)
  7. Balance query
  8. Drain transactions (large, approaching daily limit)
  9. PayShap transfers to mule accounts

Key challenge: load shedding outages create legitimate connectivity gaps
that mimic SIM swap indicators. The generator can overlap fraud sequences
with load shedding windows to create this modeling challenge.

Statistics (SABRIC 2024):
  - 60% of mobile banking breaches are SIM swap
  - Average loss: ~R10,000 per incident
  - 15% involve insider collusion at telecom/bank
  - Attack completes within 2-4 hours of SIM activation
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
from shared.utils.load_shedding import LoadSheddingSchedule, OutageWindow


@dataclass
class SimSwapEvent:
    """Non-transaction event marking the SIM swap at carrier level."""

    event_id: str
    account_id: str
    phone_number: str
    swap_timestamp: datetime  # When carrier processed the swap
    activation_timestamp: datetime  # When new SIM became active (swap + delay)
    insider_collusion: bool = False  # 15% of cases involve employee collusion
    carrier: str = "vodacom"


@dataclass
class SimSwapSequence:
    """Complete SIM swap fraud sequence for one victim account."""

    victim_account_id: str
    mule_account_ids: list[str]
    sim_swap_event: SimSwapEvent
    transactions: list[dict[str, Any]]
    total_fraud_loss_zar: Decimal
    loadshedding_overlap: bool = False  # Was the connectivity gap during load shedding?


def generate_sim_swap_sequence(
    victim_account: dict,
    mule_accounts: list[dict],
    start_time: datetime,
    account_balance_zar: float = 15000.0,
    loadshedding_window: OutageWindow | None = None,
    loadshedding_schedule: LoadSheddingSchedule | None = None,
    rng: random.Random | None = None,
) -> SimSwapSequence:
    """
    Generate a full SIM swap fraud sequence.

    Args:
        victim_account: Account dict with account_id, phone, bank, payshap_id
        mule_accounts: List of mule account dicts (1-5 accounts)
        start_time: When the SIM swap request was made at the carrier
        account_balance_zar: Estimated victim account balance
        loadshedding_window: Optional overlapping load shedding outage
        rng: Random state for reproducibility
    """
    r = rng or random

    # ── Step 1: SIM swap at carrier ───────────────────────────────────────────
    # Processing delay: 15-90 minutes (shorter if insider collusion)
    insider = r.random() < 0.15
    if insider:
        processing_delay_min = r.randint(3, 15)
    else:
        processing_delay_min = r.randint(15, 90)

    swap_time = start_time
    activation_time = swap_time + timedelta(minutes=processing_delay_min)

    sim_swap_event = SimSwapEvent(
        event_id=str(uuid.uuid4()),
        account_id=victim_account["account_id"],
        phone_number=victim_account.get("registration_phone", ""),
        swap_timestamp=swap_time,
        activation_timestamp=activation_time,
        insider_collusion=insider,
        carrier=r.choice(["vodacom", "mtn", "cellc", "telkom"]),
    )

    loadshedding_overlap = loadshedding_window is not None and loadshedding_window.contains(
        swap_time
    )

    def _ls_at(ts: datetime) -> tuple[bool, int | None]:
        """Return (loadshedding_active, stage) for a given timestamp."""
        if loadshedding_schedule is None:
            return False, None
        outage = loadshedding_schedule.get_active_outage(ts)
        if outage:
            return True, outage.stage
        return False, None

    transactions: list[dict[str, Any]] = []
    current_time = activation_time

    # ── Step 2: Connectivity gap (OTP interception window) ───────────────────
    # The victim's phone loses service during SIM swap activation
    # This gap is 5-30 minutes — distinguishable from load shedding if
    # we cross-reference the Eskom schedule
    connectivity_gap_minutes = processing_delay_min + r.randint(5, 25)

    # ── Step 3: Probe transaction (verify OTP access) ─────────────────────────
    # Small amount, unusual merchant, new device
    probe_delay = timedelta(minutes=r.randint(2, 15))
    probe_time = activation_time + probe_delay
    current_time = probe_time

    probe_amount = Decimal(str(round(r.uniform(1.0, 99.0), 2)))
    probe_merchant = r.choice(
        [
            SAMerchantCategory.SPAZA_SHOP,
            SAMerchantCategory.GAMBLING,
            SAMerchantCategory.CRYPTO_EXCHANGE,
        ]
    )

    probe_ls_active, probe_ls_stage = _ls_at(probe_time)
    transactions.append(
        {
            "transaction_id": str(uuid.uuid4()),
            "timestamp": probe_time,
            "sender_account_id": victim_account["account_id"],
            "receiver_account_id": None,
            "amount_zar": probe_amount,
            "payment_rail": PaymentRail.MOBILE_APP.value,
            "merchant_category": probe_merchant.value,
            "merchant_id": f"merch_{uuid.uuid4().hex[:8]}",
            "sender_device_id": f"dev_new_{uuid.uuid4().hex[:12]}",  # New/unknown device
            "sender_province": victim_account.get("province", "GP"),
            "is_fraud": True,
            "fraud_type": FraudType.SIM_SWAP.value,
            "sim_swap_detected": True,
            "sim_swap_timestamp": swap_time,
            "connectivity_gap_minutes": connectivity_gap_minutes,
            "loadshedding_active": probe_ls_active,
            "loadshedding_stage": probe_ls_stage,
            "loadshedding_coincident": loadshedding_overlap,
            "sequence_step": "probe",
        }
    )

    # ── Step 4: Drain transactions ────────────────────────────────────────────
    # 1-3 large transactions, total approaching daily EFT limit
    remaining_balance = account_balance_zar - float(probe_amount)
    daily_limit = 25000.0
    total_drained = float(probe_amount)
    n_drain = r.randint(1, 3)

    for i in range(n_drain):
        if remaining_balance < 100 or total_drained >= daily_limit:
            break

        # Each drain: takes 40-80% of remaining balance or up to daily limit
        drain_fraction = r.uniform(0.4, 0.80)
        drain_amount = min(
            remaining_balance * drain_fraction,
            daily_limit - total_drained,
            50000,  # PayShap + EFT combined practical limit
        )
        drain_amount = max(100.0, drain_amount)
        drain_zar = Decimal(str(round(drain_amount, 2)))

        # Drain delay: 5-20 minutes between transactions
        drain_time = current_time + timedelta(minutes=r.randint(3, 20))
        current_time = drain_time

        # Destination: mule account
        mule = mule_accounts[i % len(mule_accounts)]

        # PayShap preferred for speed (instant settlement)
        if victim_account.get("payshap_id") and drain_amount <= 3000:
            rail = PaymentRail.PAYSHAP
        else:
            rail = PaymentRail.EFT

        drain_ls_active, drain_ls_stage = _ls_at(drain_time)
        transactions.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": drain_time,
                "sender_account_id": victim_account["account_id"],
                "receiver_account_id": mule["account_id"],
                "amount_zar": drain_zar,
                "payment_rail": rail.value,
                "merchant_category": SAMerchantCategory.PEER_TRANSFER.value,
                "merchant_id": None,
                "sender_device_id": f"dev_new_{uuid.uuid4().hex[:12]}",
                "sender_province": victim_account.get("province", "GP"),
                "is_fraud": True,
                "fraud_type": FraudType.SIM_SWAP.value,
                "sim_swap_detected": True,
                "sim_swap_timestamp": swap_time,
                "connectivity_gap_minutes": connectivity_gap_minutes,
                "loadshedding_active": drain_ls_active,
                "loadshedding_stage": drain_ls_stage,
                "loadshedding_coincident": loadshedding_overlap,
                "sequence_step": f"drain_{i + 1}",
            }
        )

        remaining_balance -= drain_amount
        total_drained += drain_amount

    total_loss = Decimal(str(round(total_drained, 2)))

    return SimSwapSequence(
        victim_account_id=victim_account["account_id"],
        mule_account_ids=[m["account_id"] for m in mule_accounts],
        sim_swap_event=sim_swap_event,
        transactions=transactions,
        total_fraud_loss_zar=total_loss,
        loadshedding_overlap=loadshedding_overlap,
    )
