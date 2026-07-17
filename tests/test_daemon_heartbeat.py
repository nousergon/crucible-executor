"""Flow-doctor heartbeat wiring test for the daemon entrypoint (config#646).

Verifies that ``run_daemon``'s end-of-run cleanup emits the flow-doctor
heartbeat via ``FlowDoctor.emit_heartbeat`` and — critically — targets the
RESEARCH bucket (``signals_bucket`` == ``alpha-engine-research``), the bucket
the dashboard System Health consumer reads, NOT a trades bucket.

The daemon's live loop has a deep setup chain (preflight, IB Gateway connect,
price monitor, S3 writers). Rather than exercise a full session, we patch that
chain to no-ops and drive ``run_daemon`` straight into its ``finally`` block by
tripping the shutdown flag during the pre-open wait, then assert on the
heartbeat call.
"""
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import executor.daemon as daemon


# signals_bucket is the RESEARCH bucket in this repo (risk.yaml:
# signals_bucket == alpha-engine-research); trades_bucket is a distinct
# executor/trades bucket that the heartbeat must NOT be written to.
RESEARCH_BUCKET = "alpha-engine-research"
TRADES_BUCKET = "alpha-engine-trades"


def _config():
    return {
        "signals_bucket": RESEARCH_BUCKET,
        "trades_bucket": TRADES_BUCKET,
        "db_path": ":memory:",
        "ibkr_host": "127.0.0.1",
        "ibkr_port": 4002,
        "allow_shorts": False,
    }


def test_run_daemon_emits_heartbeat_to_research_bucket():
    fd = MagicMock(name="flow_doctor")

    order_book = MagicMock(name="order_book")
    order_book.has_content.return_value = True  # skip the pre-open order-book wait
    order_book.all_tickers.return_value = []
    order_book.pending_urgent_exits.return_value = []
    order_book.pending_entries.return_value = []
    order_book.active_stops.return_value = []

    ibkr = MagicMock(name="ibkr")
    ibkr.get_positions.return_value = {}

    # is_market_hours: first (pre-open, outer no-arg) call → False so we enter
    # the wait loop; the loop's guard trips the shutdown flag so run_daemon
    # returns through the finally without ever trading.
    def _is_market_hours(now=None):
        daemon._shutdown_requested = True
        return False

    patches = [
        patch.object(daemon, "load_config", return_value=_config()),
        patch.object(daemon, "load_strategy_config",
                     return_value={"intraday_enabled": True, "intraday_client_id": 2}),
        patch("executor.preflight.ExecutorPreflight"),
        patch("nousergon_lib.logging.get_flow_doctor", return_value=fd),
        patch.object(daemon.OrderBook, "load", return_value=order_book),
        patch.object(daemon, "init_db", return_value=MagicMock(name="conn")),
        patch.object(daemon, "IBKRClient", return_value=ibkr),
        patch.object(daemon, "make_price_monitor", return_value=MagicMock()),
        patch.object(daemon, "IntradayExitManager", MagicMock()),
        patch.object(daemon, "EntryTriggerEngine", MagicMock()),
        patch.object(daemon, "compute_surveillance_universe", return_value=[]),
        patch.object(daemon, "build_conviction_map", return_value={}),
        patch.object(daemon, "IntradaySnapshotWriter", MagicMock()),
        patch.object(daemon, "OpenOrdersSnapshotWriter", MagicMock()),
        patch.object(daemon, "IntradayNavWriter", MagicMock()),
        patch.object(daemon, "IntradayNavSeriesWriter", MagicMock()),
        patch("executor.signal_reader.read_signals_with_fallback", return_value=None),
        patch("executor.data_manifest.write_data_manifest"),
        patch.object(daemon, "_get_decision_logger", MagicMock()),
        patch.object(daemon, "_cleanup_connections"),
        patch.object(daemon, "send_daemon_status"),
        patch.object(daemon, "is_market_hours", side_effect=_is_market_hours),
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        session_date = stack.enter_context(patch("nousergon_lib.dates.session_date"))
        session_date.return_value.isoformat.return_value = "2026-07-10"
        daemon._shutdown_requested = False
        try:
            daemon.run_daemon(dry_run=True)
        finally:
            daemon._shutdown_requested = False

    # End-of-run summary was logged AND the heartbeat was emitted.
    assert fd.log_summary.called
    fd.emit_heartbeat.assert_called_once_with(bucket=RESEARCH_BUCKET)

    # Guard against regressing to a trades bucket.
    _, kwargs = fd.emit_heartbeat.call_args
    assert kwargs["bucket"] != TRADES_BUCKET
