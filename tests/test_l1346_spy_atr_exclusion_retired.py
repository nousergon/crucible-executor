"""Pin L1346 (c) — SPY routes to `universe` ArcticDB library, not `macro`.

Pre-fix: alpha-engine #185 excluded ALL `_MACRO_SYMBOLS` (incl. SPY) from
the ATR ticker list because macro lib was Close-only. The 2026-05-24
transition shipped a `_MACRO_SYMBOLS_NO_OHLCV = _MACRO_SYMBOLS - {"SPY"}`
defensive carve-out + a SPY-specific macro-fallback in `load_price_histories`
to bridge the cross-repo soak window after alpha-engine-data #245
(2026-05-15) lifted SPY to a full `universe` member.

Post-retirement (2026-05-28): the carve-out + defensive fallback are gone.
SPY is removed from `_MACRO_SYMBOLS` entirely — the executor reads SPY
from `universe` like any other held ticker, and the ATR-exclusion line
subtracts `_MACRO_SYMBOLS` directly (no `_NO_OHLCV` derivation).

This module pins the structural retirement so a refactor can't silently
re-introduce the dead defense.
"""
from __future__ import annotations

import inspect


def test_spy_not_in_macro_symbols():
    """`_MACRO_SYMBOLS` must NOT contain SPY post-L1346 (c) retirement."""
    from executor.price_cache import _MACRO_SYMBOLS

    assert "SPY" not in _MACRO_SYMBOLS, (
        "SPY must NOT be in `_MACRO_SYMBOLS` — universe.SPY now has full "
        "OHLCV + atr_14_pct (post L1346 #245 _UNIVERSE_EXTRA write path). "
        "Re-adding SPY here would route reads to macro (Close-only) and "
        "re-break the morning planner's ATR computation for held SPY."
    )
    # Sanity: legacy macro-only symbols (VIX, TNX, IRX, sector ETFs) still
    # belong in the routing list because they remain Close-only in macro lib.
    for sym in ("VIX", "TNX", "IRX", "XLK", "XLF"):
        assert sym in _MACRO_SYMBOLS, (
            f"{sym} must remain in `_MACRO_SYMBOLS` — it lives only in the "
            f"`macro` ArcticDB library (Close-only); ATR computation "
            f"requires OHLCV."
        )


def test_main_atr_tickers_subtracts_macro_symbols_directly():
    """White-box source-level pin: locate the atr_tickers derivation in
    `executor/main.py` and assert it subtracts `_MACRO_SYMBOLS` directly,
    NOT a `_MACRO_SYMBOLS_NO_OHLCV = _MACRO_SYMBOLS - {"SPY"}` carve-out
    (which would silently re-introduce the L1346 transitional defense).

    Mirrors how alpha-engine-backtester's tests pin SF wiring at the
    source-line level — catches a future refactor that re-adds the
    SPY-specific exclusion."""
    from executor import main
    src = inspect.getsource(main)
    assert "atr_tickers = sorted(set(atr_tickers) - _MACRO_SYMBOLS)" in src, (
        "The ATR-tickers derivation must subtract `_MACRO_SYMBOLS` "
        "directly. The pre-retirement form "
        "`atr_tickers = sorted(set(atr_tickers) - _MACRO_SYMBOLS_NO_OHLCV)` "
        "with a separate `_MACRO_SYMBOLS_NO_OHLCV = _MACRO_SYMBOLS - {'SPY'}` "
        "derivation is the dead-defense pattern L1346 (c) retired."
    )
    assert "_MACRO_SYMBOLS_NO_OHLCV" not in src, (
        "`_MACRO_SYMBOLS_NO_OHLCV` derivation must not re-appear in "
        "`executor/main.py` — it was a transitional carve-out retired "
        "post-soak when SPY became a full `universe` member."
    )


def test_price_cache_no_spy_specific_fallback():
    """`load_price_histories` must not carry a SPY-specific macro
    fallback. The defensive fallback was a transition device while
    universe.SPY's write path soaked; after retirement reads go straight
    to `universe` like any other ticker."""
    from executor import price_cache
    src = inspect.getsource(price_cache)
    assert 'if ticker == "SPY" and lib is universe' not in src, (
        "Defensive SPY-specific macro fallback must be retired — "
        "universe.SPY is the canonical source post-L1346 (c). If "
        "universe.SPY ever fails to read, the right surface is a hard "
        "failure (per feedback_no_silent_fails), not a silent macro "
        "fallback that masks the upstream gap."
    )


def test_eod_reconcile_no_spy_specific_fallback():
    """`eod_reconcile.run` must not carry a SPY-specific macro fallback.
    Same rationale as `price_cache` above."""
    from executor import eod_reconcile
    src = inspect.getsource(eod_reconcile)
    assert 'if ticker == "SPY" and lib is universe_lib' not in src, (
        "Defensive SPY-specific macro fallback must be retired from "
        "eod_reconcile — universe.SPY is the canonical source post-L1346 "
        "(c). Silent macro fallback would mask universe.SPY gaps."
    )
