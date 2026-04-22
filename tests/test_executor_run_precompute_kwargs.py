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
        f"found {len(call_lines)} at lines {[l+1 for l in call_lines]}. "
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
