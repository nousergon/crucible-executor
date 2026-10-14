"""Tests for executor.alert_rationale — per-ticker lines on the order-book alerts."""

from __future__ import annotations

from executor.alert_rationale import (
    MAX_LINES_PER_SECTION,
    entry_rationale,
    order_book_rationale,
    urgent_exit_rationale,
)


class _Book:
    def __init__(self, entries=(), urgent=()):
        self._entries = list(entries)
        self._urgent = list(urgent)

    def pending_entries(self):
        return self._entries

    def pending_urgent_exits(self):
        return self._urgent


def _entry(ticker="AAPL", **kw):
    rec = {
        "ticker": ticker,
        "signal": "ENTER",
        "research_score": 74.4,
        "research_rating": "BUY",
        "research_conviction": "rising",
        "predicted_direction": "UP",
        "predicted_alpha": 0.0182,
        "prediction_confidence": 0.64,
        "position_pct": 0.041,
        "sizing_source": "portfolio_optimizer",
    }
    rec.update(kw)
    return rec


def test_entry_line_carries_score_conviction_prediction_and_sizing():
    line = entry_rationale(_entry())
    assert line == (
        "  AAPL: score 74 BUY, conviction rising, GBM UP α +1.82% (64% conf), "
        "optimizer target 4.1%"
    )


def test_urgent_exit_line_leads_with_recorded_reason():
    line = urgent_exit_rationale({
        "ticker": "XYZ",
        "signal": "EXIT",
        "reason": "optimizer_target_zero",
        "detail": "optimizer target_weight=0.0000",
        "research_score": 41,
        "research_rating": "SELL",
        "predicted_direction": "DOWN",
        "predicted_alpha": -0.009,
    })
    assert line == (
        "  XYZ EXIT: optimizer target zero (optimizer target_weight=0.0000), "
        "score 41 SELL, GBM DOWN α -0.90%"
    )


def test_cover_without_research_fields():
    line = urgent_exit_rationale({
        "ticker": "QQQ", "signal": "COVER", "reason": "short_position_cover",
        "detail": "Covering accidental short of 5 shares",
    })
    assert line == "  QQQ COVER: short position cover (Covering accidental short of 5 shares)"


def test_missing_and_malformed_fields_never_raise():
    assert entry_rationale({"ticker": "A"}) == "  A: no rationale recorded"
    assert urgent_exit_rationale({"ticker": "B"}) == "  B EXIT: no reason recorded"
    assert entry_rationale({"ticker": "C", "research_score": "n/a"}) == "  C: no rationale recorded"


def test_order_book_block_urgent_first_then_entries():
    book = _Book(
        entries=[_entry("AAPL")],
        urgent=[{"ticker": "XYZ", "signal": "EXIT", "reason": "stop_hit"}],
    )
    text = order_book_rationale(book)
    assert text.startswith("\nUrgent exits:\n  XYZ EXIT: stop hit")
    assert "\nEntries:\n  AAPL: score 74 BUY" in text
    assert text.index("Urgent exits") < text.index("Entries")


def test_empty_book_renders_nothing():
    assert order_book_rationale(_Book()) == ""


def test_large_section_is_bounded():
    book = _Book(entries=[_entry(f"T{i}") for i in range(MAX_LINES_PER_SECTION + 4)])
    text = order_book_rationale(book)
    assert text.count("\n  T") == MAX_LINES_PER_SECTION
    assert text.endswith("  +4 more")


def test_broken_book_degrades_to_empty():
    class _Bad:
        def pending_urgent_exits(self):
            raise RuntimeError("boom")

    assert order_book_rationale(_Bad()) == ""
