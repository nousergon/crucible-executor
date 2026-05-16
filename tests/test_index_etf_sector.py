"""Tests for the index/ETF sector-label helper in executor/eod_reconcile.py.

Locks the contract that broad-market ETF core positions (SPY etc., held as
the enhanced-index core since the 2026-05-13 portfolio-optimizer cutover)
resolve to "Broad Market / Index" rather than falling through the GICS
lookup chain to a bare "—"/"Unknown" that reads as missing data.
"""

from executor.eod_reconcile import (
    _INDEX_ETF_SECTOR,
    _INDEX_ETF_TICKERS,
    _index_etf_sector,
)


class TestIndexEtfSector:
    def test_spy_resolves_to_broad_market(self):
        assert _index_etf_sector("SPY") == "Broad Market / Index"

    def test_all_known_index_etfs_resolve(self):
        for ticker in _INDEX_ETF_TICKERS:
            assert _index_etf_sector(ticker) == _INDEX_ETF_SECTOR

    def test_regular_constituent_returns_none(self):
        # AAPL must fall through to the normal signals.json / entry-trade /
        # constituents lookup chain — not be short-circuited here.
        assert _index_etf_sector("AAPL") is None

    def test_cash_sentinel_returns_none(self):
        # The synthetic CASH row is handled by the renderer, not here.
        assert _index_etf_sector("CASH") is None

    def test_empty_ticker_returns_none(self):
        assert _index_etf_sector("") is None

    def test_spy_label_is_not_a_gics_sector(self):
        # Guard against someone "normalizing" the label into a real sector
        # bucket — it must stay visibly distinct from active sector picks.
        assert "Broad Market" in _INDEX_ETF_SECTOR
        assert _INDEX_ETF_SECTOR not in {
            "Technology",
            "Healthcare",
            "Financials",
            "Industrials",
            "Consumer",
            "Defensives",
            "Unknown",
        }
