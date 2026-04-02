"""
Unit tests for shared/utils/load_shedding.py.

Covers:
  - LoadSheddingFeatures extraction with None schedule
  - Active outage detection
  - Recent-outage features (within last 2 hours)
  - Upcoming outage features (within next 30 minutes)
  - Mock schedule generation
  - OutageWindow.contains() and duration_minutes
"""

from __future__ import annotations

from datetime import datetime, timedelta

from shared.utils.load_shedding import (
    LoadSheddingSchedule,
    OutageWindow,
    extract_features,
    generate_mock_schedule,
)


def _make_outage(
    start: datetime,
    duration_minutes: int = 150,
    stage: int = 2,
    province: str = "GP",
) -> OutageWindow:
    return OutageWindow(
        province_code=province,
        stage=stage,
        start=start,
        end=start + timedelta(minutes=duration_minutes),
    )


BASE_TIME = datetime(2026, 3, 15, 14, 0, 0)  # 2 PM, naive


# ── OutageWindow ──────────────────────────────────────────────────────────────


class TestOutageWindow:
    def test_contains_during_outage(self):
        outage = _make_outage(BASE_TIME)
        assert outage.contains(BASE_TIME + timedelta(minutes=60)) is True

    def test_contains_at_start(self):
        outage = _make_outage(BASE_TIME)
        assert outage.contains(BASE_TIME) is True

    def test_not_contains_before_outage(self):
        outage = _make_outage(BASE_TIME)
        assert outage.contains(BASE_TIME - timedelta(minutes=1)) is False

    def test_not_contains_after_outage(self):
        outage = _make_outage(BASE_TIME)
        assert outage.contains(BASE_TIME + timedelta(minutes=200)) is False

    def test_duration_minutes(self):
        outage = _make_outage(BASE_TIME, duration_minutes=150)
        assert outage.duration_minutes == 150.0

    def test_minutes_since_end_before_end(self):
        outage = _make_outage(BASE_TIME, duration_minutes=60)
        ts = BASE_TIME + timedelta(minutes=30)  # still during outage
        assert outage.minutes_since_end(ts) is None

    def test_minutes_since_end_after_end(self):
        outage = _make_outage(BASE_TIME, duration_minutes=60)
        ts = BASE_TIME + timedelta(minutes=90)  # 30 min after end
        result = outage.minutes_since_end(ts)
        assert result is not None
        assert abs(result - 30.0) < 0.1

    def test_minutes_until_upcoming(self):
        outage = _make_outage(BASE_TIME + timedelta(hours=2))  # 2 hours from now
        result = outage.minutes_until(BASE_TIME)
        assert abs(result - 120.0) < 0.1

    def test_minutes_until_past_outage_is_negative(self):
        outage = _make_outage(BASE_TIME - timedelta(hours=1))  # started 1 hour ago
        result = outage.minutes_until(BASE_TIME)
        assert result < 0


# ── LoadSheddingSchedule ──────────────────────────────────────────────────────


class TestLoadSheddingSchedule:
    def test_get_active_outage_during(self):
        outage = _make_outage(BASE_TIME)
        schedule = LoadSheddingSchedule(province_code="GP", outages=[outage])
        ts = BASE_TIME + timedelta(minutes=30)
        active = schedule.get_active_outage(ts)
        assert active is outage

    def test_get_active_outage_none_outside(self):
        outage = _make_outage(BASE_TIME)
        schedule = LoadSheddingSchedule(province_code="GP", outages=[outage])
        ts = BASE_TIME + timedelta(hours=5)
        assert schedule.get_active_outage(ts) is None

    def test_get_nearest_outage_within_window(self):
        outage = _make_outage(BASE_TIME + timedelta(hours=1))
        schedule = LoadSheddingSchedule(province_code="GP", outages=[outage])
        nearest = schedule.get_nearest_outage(BASE_TIME, window_hours=2)
        assert nearest is outage

    def test_get_nearest_outage_outside_window(self):
        outage = _make_outage(BASE_TIME + timedelta(hours=5))
        schedule = LoadSheddingSchedule(province_code="GP", outages=[outage])
        assert schedule.get_nearest_outage(BASE_TIME, window_hours=2) is None


# ── extract_features ──────────────────────────────────────────────────────────


class TestExtractFeatures:
    def test_none_schedule_returns_safe_defaults(self):
        features = extract_features(schedule=None, ts=BASE_TIME)
        assert features.is_active is False
        assert features.stage is None
        assert features.minutes_since_outage_end is None
        assert features.minutes_until_next_outage is None
        assert features.outage_in_last_2h is False
        assert features.outage_in_next_30min is False

    def test_active_outage_detected(self):
        outage = _make_outage(BASE_TIME - timedelta(minutes=30))  # started 30 min ago
        schedule = LoadSheddingSchedule(province_code="GP", outages=[outage])
        ts = BASE_TIME
        features = extract_features(schedule, ts)
        assert features.is_active is True
        assert features.stage == 2

    def test_recent_outage_in_last_2h(self):
        # Outage ended 45 minutes ago
        end = BASE_TIME - timedelta(minutes=45)
        outage = _make_outage(end - timedelta(minutes=90), duration_minutes=90)
        schedule = LoadSheddingSchedule(province_code="GP", outages=[outage])
        features = extract_features(schedule, BASE_TIME)
        assert features.outage_in_last_2h is True
        assert features.minutes_since_outage_end is not None
        assert abs(features.minutes_since_outage_end - 45.0) < 1.0

    def test_old_outage_not_in_last_2h(self):
        # Outage ended 3 hours ago
        end = BASE_TIME - timedelta(hours=3)
        outage = _make_outage(end - timedelta(minutes=90), duration_minutes=90)
        schedule = LoadSheddingSchedule(province_code="GP", outages=[outage])
        features = extract_features(schedule, BASE_TIME)
        assert features.outage_in_last_2h is False

    def test_upcoming_outage_within_30min(self):
        # Outage starts in 20 minutes
        outage = _make_outage(BASE_TIME + timedelta(minutes=20))
        schedule = LoadSheddingSchedule(province_code="GP", outages=[outage])
        features = extract_features(schedule, BASE_TIME)
        assert features.outage_in_next_30min is True
        assert features.minutes_until_next_outage is not None
        assert abs(features.minutes_until_next_outage - 20.0) < 1.0

    def test_no_upcoming_outage(self):
        schedule = LoadSheddingSchedule(province_code="GP", outages=[])
        features = extract_features(schedule, BASE_TIME)
        assert features.outage_in_next_30min is False
        assert features.minutes_until_next_outage is None


# ── generate_mock_schedule ────────────────────────────────────────────────────


class TestGenerateMockSchedule:
    def test_produces_outages(self):
        schedule = generate_mock_schedule("GP", BASE_TIME, days=7, stage=2)
        assert len(schedule.outages) > 0

    def test_outages_have_correct_province(self):
        schedule = generate_mock_schedule("WC", BASE_TIME, days=3)
        assert all(o.province_code == "WC" for o in schedule.outages)

    def test_outages_have_correct_stage(self):
        schedule = generate_mock_schedule("GP", BASE_TIME, days=3, stage=4)
        assert all(o.stage == 4 for o in schedule.outages)

    def test_deterministic_given_same_seed(self):
        s1 = generate_mock_schedule("GP", BASE_TIME, days=7)
        s2 = generate_mock_schedule("GP", BASE_TIME, days=7)
        assert len(s1.outages) == len(s2.outages)

    def test_outage_start_before_end(self):
        schedule = generate_mock_schedule("GP", BASE_TIME, days=5)
        for o in schedule.outages:
            assert o.start < o.end
