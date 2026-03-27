"""
EskomSePush API client for load shedding schedule data.

Load shedding creates legitimate connectivity gaps that mimic SIM swap
indicators. This module provides schedule data so models can distinguish
infrastructure-induced anomalies from genuine fraud signals.

API docs: https://documenter.getpostman.com/view/1296288/UzQuNk3E
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class OutageWindow:
    province_code: str
    stage: int
    start: datetime
    end: datetime

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60

    def contains(self, ts: datetime) -> bool:
        return self.start <= ts <= self.end

    def minutes_until(self, ts: datetime) -> float:
        """Minutes from ts until the outage starts (negative if already started)."""
        return (self.start - ts).total_seconds() / 60

    def minutes_since_end(self, ts: datetime) -> float | None:
        """Minutes elapsed since outage ended. None if outage hasn't ended yet."""
        if ts < self.end:
            return None
        return (ts - self.end).total_seconds() / 60


@dataclass
class LoadSheddingSchedule:
    province_code: str
    outages: list[OutageWindow] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    def get_active_outage(self, ts: datetime) -> OutageWindow | None:
        return next((o for o in self.outages if o.contains(ts)), None)

    def get_nearest_outage(self, ts: datetime, window_hours: int = 2) -> OutageWindow | None:
        """Return outage within window_hours of ts, or None."""
        cutoff = ts + timedelta(hours=window_hours)
        upcoming = [o for o in self.outages if o.start <= cutoff and o.end >= ts]
        return upcoming[0] if upcoming else None


# ── Feature extraction ────────────────────────────────────────────────────────


@dataclass
class LoadSheddingFeatures:
    """Features derived from load shedding schedule for a given timestamp/province."""

    is_active: bool
    stage: int | None
    minutes_since_outage_end: float | None  # None if no recent outage
    minutes_until_next_outage: float | None  # None if no upcoming outage
    outage_in_last_2h: bool
    outage_in_next_30min: bool


def extract_features(
    schedule: LoadSheddingSchedule | None,
    ts: datetime,
) -> LoadSheddingFeatures:
    """Compute load shedding features for a timestamp."""
    if schedule is None:
        return LoadSheddingFeatures(
            is_active=False,
            stage=None,
            minutes_since_outage_end=None,
            minutes_until_next_outage=None,
            outage_in_last_2h=False,
            outage_in_next_30min=False,
        )

    active = schedule.get_active_outage(ts)
    recent_end: float | None = None
    recent_in_2h = False

    for outage in schedule.outages:
        since_end = outage.minutes_since_end(ts)
        if since_end is not None and since_end <= 120:
            recent_in_2h = True
            if recent_end is None or since_end < recent_end:
                recent_end = since_end

    upcoming: float | None = None
    upcoming_in_30min = False
    for outage in schedule.outages:
        if outage.start > ts:
            mins = outage.minutes_until(ts)
            if upcoming is None or mins < upcoming:
                upcoming = mins
            if mins <= 30:
                upcoming_in_30min = True

    return LoadSheddingFeatures(
        is_active=active is not None,
        stage=active.stage if active else None,
        minutes_since_outage_end=recent_end,
        minutes_until_next_outage=upcoming,
        outage_in_last_2h=recent_in_2h,
        outage_in_next_30min=upcoming_in_30min,
    )


# ── Mock schedule for synthetic data generation ───────────────────────────────


def generate_mock_schedule(
    province_code: str,
    start_date: datetime,
    days: int = 30,
    stage: int = 2,
) -> LoadSheddingSchedule:
    """
    Generate a realistic mock load shedding schedule for synthetic data.

    Stage 2 in Gauteng: outages roughly every 12 hours, 2.5 hours each.
    """
    import random

    rng = random.Random(hash(province_code + str(start_date.date())))
    outages: list[OutageWindow] = []
    current = start_date.replace(hour=6, minute=0, second=0, microsecond=0)

    for _ in range(days):
        # Stage 2: ~2 outages per day
        n_outages = rng.randint(max(1, stage - 1), stage + 1)
        hours_between = 24 // (n_outages + 1)

        for j in range(n_outages):
            # Add some jitter to make it realistic
            offset_hours = hours_between * (j + 1) + rng.randint(-1, 1)
            start = current + timedelta(hours=offset_hours)
            duration_mins = 120 + (stage - 1) * 30 + rng.randint(-15, 15)
            end = start + timedelta(minutes=duration_mins)

            outages.append(
                OutageWindow(
                    province_code=province_code,
                    stage=stage,
                    start=start,
                    end=end,
                )
            )

        current += timedelta(days=1)

    return LoadSheddingSchedule(province_code=province_code, outages=outages)
