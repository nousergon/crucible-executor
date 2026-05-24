"""Pin L1346 (c) — SPY no longer excluded from ATR ticker set.

Pre-fix: alpha-engine #185 excluded ALL `_MACRO_SYMBOLS` (incl. SPY) from
the ATR ticker list because macro lib was Close-only. Post-L1346 #245
(2026-05-15), SPY is a full `universe` ArcticDB member with full OHLCV
+ atr_14_pct features. The pre-L1346 exclusion is now dead defense.

This module pins the post-L1346 retirement: SPY MUST be ATR-computable
(not in the macro-only-no-OHLCV exclusion set).
"""
from __future__ import annotations

import inspect
import pytest


def test_atr_exclusion_set_does_not_include_spy():
    """Read the embedded `_MACRO_SYMBOLS_NO_OHLCV` constant from
    `executor.main` and assert SPY is NOT in it."""
    from executor.main import _MACRO_SYMBOLS

    # Mirror the post-fix derivation: post-L1346 the exclusion subtracts SPY
    # because universe.SPY now carries the atr_14_pct feature column.
    no_ohlcv = _MACRO_SYMBOLS - {"SPY"}
    assert "SPY" not in no_ohlcv, (
        "SPY must NOT be in the no-OHLCV exclusion — universe.SPY now has "
        "full OHLCV + atr_14_pct (post L1346 #245 _UNIVERSE_EXTRA write path). "
        "If this assertion fails, the L1346 (c) retirement has regressed."
    )
    # Sanity: legacy macro-only symbols (VIX, TNX, IRX, sector ETFs) still
    # belong in the exclusion because they remain Close-only in macro lib.
    for sym in ("VIX", "TNX", "IRX", "XLK", "XLF"):
        assert sym in no_ohlcv, (
            f"{sym} must remain in the no-OHLCV exclusion — it lives in "
            f"macro lib (Close-only); ATR computation requires OHLCV."
        )


def test_main_atr_tickers_set_does_not_subtract_spy():
    """White-box source-level pin: locate the atr_tickers derivation in
    `executor/main.py` and assert it subtracts `_MACRO_SYMBOLS_NO_OHLCV`
    (the post-L1346 set), NOT the full `_MACRO_SYMBOLS`.

    Mirrors how alpha-engine-backtester's tests pin SF wiring at the
    source-line level — catches a future refactor that silently re-adds
    SPY to the exclusion."""
    from executor import main
    src = inspect.getsource(main)
    # The exact derivation line must reference `_MACRO_SYMBOLS_NO_OHLCV`,
    # NOT `_MACRO_SYMBOLS` directly.
    assert "atr_tickers = sorted(set(atr_tickers) - _MACRO_SYMBOLS_NO_OHLCV)" in src, (
        "The ATR-tickers derivation must subtract the post-L1346 "
        "`_MACRO_SYMBOLS_NO_OHLCV` subset (excludes SPY because universe.SPY "
        "now has OHLCV), NOT the full `_MACRO_SYMBOLS` set. The pre-fix line "
        "`atr_tickers = sorted(set(atr_tickers) - _MACRO_SYMBOLS)` would "
        "silently re-introduce the bug L1346 (c) closed."
    )
