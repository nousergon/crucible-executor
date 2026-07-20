"""Regression tests for the 2026-04-22 ``atr_map`` / ``vwap_map`` kwarg
injection path on ``executor.main.run``.

Why these kwargs exist:

  The Saturday SF dry-run fired at 09:41 PT with ``freeze_evaluator``
  and every upstream skip flag set hit the 2h SSM ceiling still mid-
  param-sweep. py-spy pinned the hot path to
  ``load_atr_14_pct`` → ``universe.read(ticker)`` — one full ArcticDB
  frame read per ticker per simulate call, × 60 param-sweep combos ×
  2000+ dates. Same class of bottleneck as the filter that PR #49
  vectorized, but at the ArcticDB layer rather than the pandas layer.

  Fix: let the backtester precompute ATR + VWAP once per simulate
  pipeline and inject via kwargs. Executor code path is otherwise
  unchanged; live trading passes ``atr_map=None`` / ``vwap_map=None``
  and takes the existing ``load_atr_14_pct`` / ``load_daily_vwap``
  calls exactly as before.

The tests here are source-inspection: we can't easily end-to-end test
``executor.main.run`` (it expects a live IBKR client, trades.db,
signals.json, etc. — mocked existing tests for it hit all of those at
once, not this single gating branch). Instead we lock the contract
via grep-style invariants so a future refactor can't silently drop
the injection seam.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from executor import main as executor_main

_MAIN_PY = Path(__file__).parent.parent / "executor" / "main.py"


def _source() -> str:
    return _MAIN_PY.read_text()


# ── Signature contract ──────────────────────────────────────────────────────


def test_run_has_atr_map_kwarg():
    """``atr_map`` must be a keyword-accepting parameter with default
    None. When None, executor takes the existing ArcticDB path; when
    provided (dict), executor uses it as-is.

    Removing the kwarg would break the backtester's precompute path
    and silently revert to per-call ArcticDB reads (the 2h SSM
    timeout failure mode from 2026-04-22).
    """
    sig = inspect.signature(executor_main.run)
    assert "atr_map" in sig.parameters, (
        "executor.main.run must accept atr_map kwarg — removed by "
        "accident? Reintroduce to preserve backtester precompute path."
    )
    assert sig.parameters["atr_map"].default is None, (
        "atr_map must default to None — the None-case is what keeps "
        "live trading on the ArcticDB read path."
    )


def test_run_has_vwap_map_kwarg():
    sig = inspect.signature(executor_main.run)
    assert "vwap_map" in sig.parameters, (
        "executor.main.run must accept vwap_map kwarg."
    )
    assert sig.parameters["vwap_map"].default is None


def test_run_has_coverage_map_kwarg():
    """Mirrors atr_map/vwap_map — required to skip load_feature_coverage's
    per-call ArcticDB read in backtest mode. Removing this kwarg would
    reintroduce the 2026-04-22 13:27 PT timeout where the simulate loop
    hit ``universe.read(ticker)`` once per enter_ticker per combo for
    the coverage-aware sizer."""
    sig = inspect.signature(executor_main.run)
    assert "coverage_map" in sig.parameters, (
        "executor.main.run must accept coverage_map kwarg."
    )
    assert sig.parameters["coverage_map"].default is None


# ── Gating contract ─────────────────────────────────────────────────────────


def test_atr_map_injection_skips_arctic_read():
    """The ``load_atr_14_pct`` call must be nested inside an
    ``atr_map is None`` branch. If the gate is removed, every executor
    call — including backtester combos that passed a precomputed map —
    would hit ArcticDB and restore the 2h timeout.
    """
    src = _source()
    assert "if atr_map is None:" in src, (
        "Missing `if atr_map is None:` gate around load_atr_14_pct. "
        "Without it the injected map is ignored and every simulate "
        "call still reads ArcticDB."
    )

    # The load_atr_14_pct call and its preceding gate must be close
    # together in the source (same code block). Find the gate, then
    # walk forward ~15 lines looking for load_atr_14_pct(.
    lines = src.splitlines()
    gate_idx = next(
        i for i, ln in enumerate(lines) if "if atr_map is None:" in ln
    )
    block = "\n".join(lines[gate_idx:gate_idx + 15])
    assert "load_atr_14_pct(" in block, (
        "load_atr_14_pct call must live inside the `atr_map is None` "
        "branch — otherwise the kwarg injection doesn't skip the "
        "ArcticDB read."
    )


def test_vwap_map_injection_skips_arctic_read():
    src = _source()
    assert "if vwap_map is None:" in src, (
        "Missing `if vwap_map is None:` gate around load_daily_vwap."
    )

    lines = src.splitlines()
    gate_idx = next(
        i for i, ln in enumerate(lines) if "if vwap_map is None:" in ln
    )
    block = "\n".join(lines[gate_idx:gate_idx + 6])
    assert "load_daily_vwap(" in block, (
        "load_daily_vwap call must live inside the `vwap_map is None` "
        "branch."
    )


def test_coverage_map_injection_skips_arctic_read():
    """The ``load_feature_coverage`` call must be nested inside a
    ``coverage_map is None`` branch — otherwise the backtester's
    precomputed map is ignored and every simulate call re-reads
    ArcticDB (the 2026-04-22 13:27 PT timeout root cause)."""
    src = _source()
    assert "if coverage_map is None:" in src, (
        "Missing `if coverage_map is None:` gate around "
        "load_feature_coverage."
    )

    lines = src.splitlines()
    gate_idx = next(
        i for i, ln in enumerate(lines) if "if coverage_map is None:" in ln
    )
    block = "\n".join(lines[gate_idx:gate_idx + 12])
    assert "load_feature_coverage(" in block, (
        "load_feature_coverage call must live inside the "
        "`coverage_map is None` branch."
    )


def test_atr_load_not_called_outside_gate():
    """There must be exactly ONE load_atr_14_pct call site in
    executor/main.py (the gated one). A second unguarded call would
    defeat the injection.
    """
    src = _source()
    # Filter to call-site lines (exclude imports, comments, docstrings)
    lines = src.splitlines()
    call_lines = [
        i for i, ln in enumerate(lines)
        if "load_atr_14_pct(" in ln
        and not ln.strip().startswith("#")
        and not ln.strip().startswith('"""')
        and "import" not in ln
    ]
    assert len(call_lines) == 1, (
        f"Expected exactly 1 load_atr_14_pct call site in main.py; "
        f"found {len(call_lines)} at lines {[line + 1 for line in call_lines]}. "
        "A second call would bypass the atr_map kwarg gate."
    )


def test_vwap_load_not_called_outside_gate():
    src = _source()
    lines = src.splitlines()
    call_lines = [
        i for i, ln in enumerate(lines)
        if "load_daily_vwap(" in ln
        and not ln.strip().startswith("#")
        and not ln.strip().startswith('"""')
        and "import" not in ln
    ]
    assert len(call_lines) == 1, (
        f"Expected exactly 1 load_daily_vwap call site; "
        f"found {len(call_lines)}."
    )


def test_coverage_load_call_sites_all_gated():
    """``load_feature_coverage`` has two legitimate call sites in main.py:
      1. ``_read_signals`` — admission gate, already gated by
         ``if not simulate:`` so backtest skips it.
      2. The sizing-derate block in ``run`` — must be gated by
         ``if coverage_map is None:`` so the backtester kwarg injection
         can skip it.

    A third unguarded call would restore the pre-2026-04-22 13:27 PT
    timeout. The count-of-2 check is brittle against legit refactors —
    we instead assert each individual call has a guarded parent
    context by walking backward for the nearest ``if`` / ``def`` line.
    """
    src = _source()
    lines = src.splitlines()
    call_lines = [
        i for i, ln in enumerate(lines)
        if "load_feature_coverage(" in ln
        and not ln.strip().startswith("#")
        and not ln.strip().startswith('"""')
        and "import" not in ln
        and "from executor" not in ln
    ]
    assert len(call_lines) >= 1, "load_feature_coverage call site missing."

    # Each call must have a nearby `if not simulate` or
    # `if coverage_map is None` guard above it in the same function.
    for i in call_lines:
        # Walk back up to 30 lines looking for a gate.
        window = "\n".join(lines[max(0, i - 30):i + 1])
        gated = (
            "if not simulate" in window
            or "if coverage_map is None:" in window
        )
        assert gated, (
            f"load_feature_coverage call at line {i+1} is not inside any "
            f"`if not simulate` or `if coverage_map is None:` guard — "
            f"will bypass the backtester kwarg injection and restore the "
            f"2026-04-22 per-call ArcticDB regression."
        )


# ── Precedence test: injected value survives ────────────────────────────────


def test_run_accepts_injected_maps_via_kwargs():
    """Sanity: the function accepts both kwargs simultaneously via
    keyword call. A regression that turned them into positional-only
    would break the backtester's call pattern.
    """
    sig = inspect.signature(executor_main.run)
    kinds = {
        name: p.kind
        for name, p in sig.parameters.items()
        if name in ("atr_map", "vwap_map")
    }
    for name, kind in kinds.items():
        assert kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ), f"{name} must be keyword-callable, got kind={kind}"


# ── Macro-symbol exclusion from the ATR set ─────────────────────────────────


def test_atr_tickers_excludes_macro_symbols():
    """The 2026-05-15 weekday SF morning-planner failure: the
    portfolio-optimizer cutover (2026-05-13) made SPY a held core
    position, so ``current_positions`` injected SPY into ``atr_tickers``.
    SPY lives in the Close-only ``macro`` ArcticDB library (no
    atr_14_pct column) and is absent from ``universe`` →
    ``load_atr_14_pct`` hard-failed with NoSuchVersionException, taking
    down the whole planner before the daemon could start.

    Same root-cause family as eod_reconcile #181, but the correct
    remedy for ATR is *exclusion* (macro lib has no ATR feature), not
    macro-lib dispatch. Lock the ``- _MACRO_SYMBOLS`` set-difference so
    a refactor can't silently re-admit SPY/ETFs to the ATR set.
    """
    src = _source()
    assert "_MACRO_SYMBOLS" in src, (
        "main.py must import _MACRO_SYMBOLS to exclude macro-routed "
        "held positions (SPY core etc.) from the ATR lookup."
    )
    assert "set(atr_tickers) - _MACRO_SYMBOLS" in src, (
        "atr_tickers must subtract _MACRO_SYMBOLS — without it the "
        "SPY-as-held-core optimizer cutover re-breaks the morning "
        "planner via load_atr_14_pct NoSuchVersionException "
        "(2026-05-15 weekday SF failure)."
    )


def test_main_imports_macro_symbols_from_price_cache():
    """``_MACRO_SYMBOLS`` is the single source of truth for macro-routed
    tickers (defined in price_cache.py, also consumed by
    load_price_histories + eod_reconcile #181). Pin the import so the
    exclusion set never drifts from the dispatch set.

    Post-L1346 (c) (2026-05-28): SPY is excluded from `_MACRO_SYMBOLS`
    because `universe.SPY` now carries full OHLCV + atr_14_pct via
    alpha-engine-data #245's `_UNIVERSE_EXTRA` write path. SPY is read
    from `universe` like any other held ticker. VIX/VIX3M/TNX/IRX/GLD/
    USO + XL* sector ETFs remain macro-routed (no `universe` counterpart;
    Close-only).
    """
    from executor.price_cache import _MACRO_SYMBOLS

    assert "SPY" not in _MACRO_SYMBOLS, (
        "SPY must NOT be in _MACRO_SYMBOLS post-L1346 (c) retirement — "
        "universe.SPY now has full OHLCV (alpha-engine-data #245). "
        "Re-adding SPY here would re-introduce the dead-defense pattern "
        "that excluded SPY from ATR computation."
    )
    # Sanity: remaining macro-only symbols still routed to macro lib.
    for sym in ("VIX", "TNX", "IRX", "XLK", "XLF"):
        assert sym in _MACRO_SYMBOLS, (
            f"{sym} must remain in _MACRO_SYMBOLS — it lives only in the "
            f"`macro` ArcticDB library (Close-only)."
        )
    assert executor_main._MACRO_SYMBOLS is _MACRO_SYMBOLS, (
        "executor.main must reuse price_cache._MACRO_SYMBOLS, not "
        "redefine its own macro set (drift risk)."
    )
