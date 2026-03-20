"""
Strategy layer configuration.

Defaults are defined here. Override via the 'strategy' key in config/risk.yaml.
"""

from __future__ import annotations

# ── Exit Manager defaults ────────────────────────────────────────────────────
# These are conservative fallbacks — override via strategy key in risk.yaml.

# ATR trailing stop
ATR_TRAILING_ENABLED = True
ATR_PERIOD = 14               # days for ATR calculation
ATR_MULTIPLIER = 2.5          # stop = highest_high - ATR * multiplier

# Time-based exit decay
TIME_DECAY_ENABLED = True
TIME_DECAY_REDUCE_DAYS = 7    # trading days before 50% reduction
TIME_DECAY_EXIT_DAYS = 14     # trading days before full exit

# Profit-taking
PROFIT_TAKE_ENABLED = True
PROFIT_TAKE_PCT = 0.25        # 25% unrealized gain triggers REDUCE

# Sector-relative exit veto
SECTOR_RELATIVE_VETO_ENABLED = True
SECTOR_RELATIVE_OUTPERFORM_THRESHOLD = 0.05  # 5% outperformance suppresses ATR exit

# Momentum-based exit
MOMENTUM_EXIT_ENABLED = True
MOMENTUM_EXIT_THRESHOLD = -15.0  # 20d momentum % threshold
MOMENTUM_EXIT_RSI = 30           # RSI(14) threshold

# Fallback fixed-percentage stop (when ATR has insufficient price history)
FALLBACK_STOP_ENABLED = True
FALLBACK_STOP_PCT = 0.10          # 10% loss from entry triggers exit

# ── Bracket stop defaults ────────────────────────────────────────────────────
BRACKET_STOP_ENABLED = True
BRACKET_TRAIL_ATR_MULTIPLE = 2.0   # trailing stop = ATR * this value

# ── Intraday daemon defaults ────────────────────────────────────────────────
INTRADAY_ENABLED = False           # must opt-in via risk.yaml
INTRADAY_CLIENT_ID = 2             # separate from morning batch (clientId=1)
INTRADAY_POLL_INTERVAL_SEC = 60

# Entry triggers
INTRADAY_ENTRY_TRIGGERS_ENABLED = True
INTRADAY_PULLBACK_PCT = 0.02       # 2% pullback from intraday high
INTRADAY_VWAP_DISCOUNT_PCT = 0.005 # 0.5% below VWAP
INTRADAY_SUPPORT_PCT = 0.01        # within 1% of support level
INTRADAY_SUPPORT_LOOKBACK_DAYS = 20
INTRADAY_EXPIRY_TIME = "15:30"     # execute at market if no trigger by 3:30 PM ET

# Exit rules
INTRADAY_TRAILING_STOP_ATR_MULTIPLE = 2.0
INTRADAY_PROFIT_TAKE_PCT = 0.08    # 8% gain triggers 50% profit-take
INTRADAY_COLLAPSE_PCT = 0.05       # 5% intraday drop triggers full exit
INTRADAY_TIGHTEN_AFTER_DAYS = 3
INTRADAY_TIGHTEN_ATR_MULTIPLE = 1.5

# ── Graduated Drawdown defaults ──────────────────────────────────────────────

GRADUATED_DRAWDOWN_ENABLED = True
DRAWDOWN_TIERS = [
    # (threshold, sizing_multiplier, description)
    (-0.02, 1.00, "0% to -2%: full sizing"),
    (-0.04, 0.50, "-2% to -4%: half sizing"),
    (-0.06, 0.25, "-4% to -6%: quarter sizing"),
    # Beyond circuit breaker threshold: full halt
]

# Drawdown forced exits
DRAWDOWN_FORCED_EXIT_ENABLED = True
DRAWDOWN_FORCED_EXIT_TIER2_COUNT = 1
DRAWDOWN_FORCED_EXIT_TIER3_COUNT = 2


def load_strategy_config(config: dict) -> dict:
    """
    Extract strategy configuration from the main risk.yaml config.

    The 'strategy' key in risk.yaml can override any default.
    Returns a flat dict of strategy parameters.
    """
    strategy = config.get("strategy", {})

    exit_cfg = strategy.get("exit_manager", {})
    drawdown_cfg = strategy.get("graduated_drawdown", {})
    bracket_cfg = strategy.get("bracket", {})
    intraday_cfg = config.get("intraday", {})
    entry_trigger_cfg = intraday_cfg.get("entry_triggers", {})
    intraday_exit_cfg = intraday_cfg.get("exit_rules", {})

    return {
        # ATR trailing stop
        "atr_trailing_enabled": exit_cfg.get("atr_trailing_enabled", ATR_TRAILING_ENABLED),
        "atr_period": exit_cfg.get("atr_period", ATR_PERIOD),
        "atr_multiplier": exit_cfg.get("atr_multiplier", ATR_MULTIPLIER),

        # Time-based exit decay
        "time_decay_enabled": exit_cfg.get("time_decay_enabled", TIME_DECAY_ENABLED),
        "time_decay_reduce_days": exit_cfg.get("time_decay_reduce_days", TIME_DECAY_REDUCE_DAYS),
        "time_decay_exit_days": exit_cfg.get("time_decay_exit_days", TIME_DECAY_EXIT_DAYS),

        # Profit-taking
        "profit_take_enabled": exit_cfg.get("profit_take_enabled", PROFIT_TAKE_ENABLED),
        "profit_take_pct": exit_cfg.get("profit_take_pct", PROFIT_TAKE_PCT),

        # Sector-relative exit veto
        "sector_relative_veto_enabled": exit_cfg.get("sector_relative_veto_enabled", SECTOR_RELATIVE_VETO_ENABLED),
        "sector_relative_outperform_threshold": exit_cfg.get("sector_relative_outperform_threshold", SECTOR_RELATIVE_OUTPERFORM_THRESHOLD),

        # Momentum-based exit
        "momentum_exit_enabled": exit_cfg.get("momentum_exit_enabled", MOMENTUM_EXIT_ENABLED),
        "momentum_exit_threshold": exit_cfg.get("momentum_exit_threshold", MOMENTUM_EXIT_THRESHOLD),
        "momentum_exit_rsi": exit_cfg.get("momentum_exit_rsi", MOMENTUM_EXIT_RSI),

        # Fallback stop (when ATR unavailable)
        "fallback_stop_enabled": exit_cfg.get("fallback_stop_enabled", FALLBACK_STOP_ENABLED),
        "fallback_stop_pct": exit_cfg.get("fallback_stop_pct", FALLBACK_STOP_PCT),

        # Graduated drawdown
        "graduated_drawdown_enabled": drawdown_cfg.get("enabled", GRADUATED_DRAWDOWN_ENABLED),
        "drawdown_tiers": drawdown_cfg.get("tiers", DRAWDOWN_TIERS),

        # Drawdown forced exits
        "drawdown_forced_exit_enabled": drawdown_cfg.get("drawdown_forced_exit_enabled", DRAWDOWN_FORCED_EXIT_ENABLED),
        "drawdown_forced_exit_tier2_count": drawdown_cfg.get("drawdown_forced_exit_tier2_count", DRAWDOWN_FORCED_EXIT_TIER2_COUNT),
        "drawdown_forced_exit_tier3_count": drawdown_cfg.get("drawdown_forced_exit_tier3_count", DRAWDOWN_FORCED_EXIT_TIER3_COUNT),

        # Bracket stop
        "bracket_stop_enabled": bracket_cfg.get("enabled", BRACKET_STOP_ENABLED),
        "bracket_trail_atr_multiple": bracket_cfg.get("trail_atr_multiple", BRACKET_TRAIL_ATR_MULTIPLE),

        # Intraday daemon
        "intraday_enabled": intraday_cfg.get("enabled", INTRADAY_ENABLED),
        "intraday_client_id": intraday_cfg.get("client_id", INTRADAY_CLIENT_ID),
        "intraday_poll_interval_sec": intraday_cfg.get("poll_interval_sec", INTRADAY_POLL_INTERVAL_SEC),

        # Intraday entry triggers
        "intraday_entry_triggers_enabled": entry_trigger_cfg.get("enabled", INTRADAY_ENTRY_TRIGGERS_ENABLED),
        "intraday_pullback_pct": entry_trigger_cfg.get("pullback_pct", INTRADAY_PULLBACK_PCT),
        "intraday_vwap_discount_pct": entry_trigger_cfg.get("vwap_discount_pct", INTRADAY_VWAP_DISCOUNT_PCT),
        "intraday_support_pct": entry_trigger_cfg.get("support_pct", INTRADAY_SUPPORT_PCT),
        "intraday_support_lookback_days": entry_trigger_cfg.get("support_lookback_days", INTRADAY_SUPPORT_LOOKBACK_DAYS),
        "intraday_expiry_time": entry_trigger_cfg.get("expiry_time", INTRADAY_EXPIRY_TIME),

        # Intraday exit rules
        "intraday_trailing_stop_atr_multiple": intraday_exit_cfg.get("trailing_stop_atr_multiple", INTRADAY_TRAILING_STOP_ATR_MULTIPLE),
        "intraday_profit_take_pct": intraday_exit_cfg.get("profit_take_pct", INTRADAY_PROFIT_TAKE_PCT),
        "intraday_collapse_pct": intraday_exit_cfg.get("collapse_pct", INTRADAY_COLLAPSE_PCT),
        "intraday_tighten_after_days": intraday_exit_cfg.get("tighten_after_days", INTRADAY_TIGHTEN_AFTER_DAYS),
        "intraday_tighten_atr_multiple": intraday_exit_cfg.get("tighten_atr_multiple", INTRADAY_TIGHTEN_ATR_MULTIPLE),
    }
