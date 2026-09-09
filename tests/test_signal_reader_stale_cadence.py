"""Tests for signal_reader.is_signals_stale — the cadence-aware staleness
check shared by signal_reader._warn_if_stale and executor/main.py's
Telegram stale-signals notice.

Root cause fixed here: Research is a weekly Saturday pipeline that writes
signals.json dated with the Friday just closed. That file stays CURRENT
for the entire following Mon-Fri. A plain "age > N" test therefore fires
on some healthy weekday no matter where N is set — main.py used age > 2
(fired every Tue-Fri of a healthy week) while signal_reader used age > 7
(fired on a healthy Friday, age exactly 7). Both were wrong; this table
pins the correct behavior: silent across a whole healthy week, loud only
when a Saturday run was genuinely skipped.
"""
from __future__ import annotations

from datetime import date

import pytest

from executor.signal_reader import _expected_signal_friday, is_signals_stale

# Week W: Friday 2026-09-04. Research runs Saturday 2026-09-05, writing
# signals dated 2026-09-04. That file is current through Fri 2026-09-11
# (verified weekday(): 2026-09-04 is a Friday, 2026-09-07 is the following
# Monday, 2026-09-11 the following Friday).
HEALTHY_SIGNALS_DATE = "2026-09-04"

HEALTHY_WEEKDAYS = [
    "2026-09-07",  # Mon
    "2026-09-08",  # Tue
    "2026-09-09",  # Wed (today, per the live alert this fix addresses)
    "2026-09-10",  # Thu
    "2026-09-11",  # Fri
]


@pytest.mark.parametrize("ref_date", HEALTHY_WEEKDAYS)
def test_healthy_week_is_silent(ref_date):
    stale, age = is_signals_stale(HEALTHY_SIGNALS_DATE, ref_date)
    assert stale is False, (
        f"{ref_date} against a healthy {HEALTHY_SIGNALS_DATE} signals file "
        f"must be silent (age={age})"
    )


def test_skipped_saturday_alerts():
    # By the following Monday (2026-09-14), Research should have refreshed
    # to Friday 2026-09-11 signals via the 2026-09-12 Saturday run. If the
    # file is still dated 2026-09-04, that Saturday run was genuinely
    # skipped — this MUST alert.
    stale, age = is_signals_stale(HEALTHY_SIGNALS_DATE, "2026-09-14")
    assert stale is True
    assert age == 10


def test_skipped_saturday_alerts_on_friday_of_the_missed_week():
    # 2026-09-11 (Friday) with signals still dated one week further back
    # (2026-08-28) means the 2026-09-05 Saturday run was also skipped.
    stale, age = is_signals_stale("2026-08-28", "2026-09-11")
    assert stale is True
    assert age == 14


def test_missing_signals_date_is_silent():
    stale, age = is_signals_stale(None, "2026-09-09")
    assert (stale, age) == (False, 0)
    stale, age = is_signals_stale("", "2026-09-09")
    assert (stale, age) == (False, 0)


def test_accepts_date_objects_not_just_iso_strings():
    stale, age = is_signals_stale(date(2026, 9, 4), date(2026, 9, 9))
    assert stale is False
    assert age == 5


def test_ref_date_defaults_to_today_when_falsy():
    # Doesn't raise; exact staleness is time-dependent so only check shape.
    stale, age = is_signals_stale("2020-01-01", None)
    assert stale is True
    assert age > 0


@pytest.mark.parametrize(
    "ref_date, expected_friday",
    [
        ("2026-09-07", date(2026, 9, 4)),  # Mon
        ("2026-09-09", date(2026, 9, 4)),  # Wed
        ("2026-09-11", date(2026, 9, 4)),  # Fri — NOT today (2026-09-11)
        ("2026-09-14", date(2026, 9, 11)),  # next Mon — rolled over
    ],
)
def test_expected_signal_friday(ref_date, expected_friday):
    assert _expected_signal_friday(date.fromisoformat(ref_date)) == expected_friday
