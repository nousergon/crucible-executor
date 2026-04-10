"""Tests for log_config — structured logging + flow-doctor singleton."""

import logging
import os
from unittest.mock import patch, MagicMock

import pytest

import executor.log_config as log_config
from executor.log_config import (
    JSONFormatter,
    get_flow_doctor,
    setup_logging,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure the singleton is reset between tests."""
    log_config._fd_instance = None
    yield
    log_config._fd_instance = None


class TestSetupLogging:
    def test_text_mode_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALPHA_ENGINE_JSON_LOGS", None)
            os.environ.pop("FLOW_DOCTOR_ENABLED", None)
            setup_logging("test")
            root = logging.getLogger()
            assert len(root.handlers) == 1
            assert root.level == logging.INFO

    def test_json_mode(self):
        with patch.dict(os.environ, {"ALPHA_ENGINE_JSON_LOGS": "1"}):
            os.environ.pop("FLOW_DOCTOR_ENABLED", None)
            setup_logging("test")
            root = logging.getLogger()
            assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_text_format_includes_name(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALPHA_ENGINE_JSON_LOGS", None)
            os.environ.pop("FLOW_DOCTOR_ENABLED", None)
            setup_logging("daemon")
            fmt = logging.getLogger().handlers[0].formatter._fmt
            assert "[daemon]" in fmt

    def test_clears_existing_handlers(self):
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLOW_DOCTOR_ENABLED", None)
            setup_logging("test")
        assert len(root.handlers) == 1


class TestFlowDoctorSingleton:
    def test_get_flow_doctor_returns_none_when_disabled(self):
        """get_flow_doctor() should return None if setup was not called with FD enabled."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLOW_DOCTOR_ENABLED", None)
            setup_logging("test")
            assert get_flow_doctor() is None

    def test_get_flow_doctor_returns_instance_when_enabled(self):
        """get_flow_doctor() should return the shared instance after setup."""
        mock_fd = MagicMock()
        mock_handler = MagicMock(spec=logging.Handler)
        mock_handler.level = logging.ERROR
        with patch.dict(os.environ, {"FLOW_DOCTOR_ENABLED": "1"}):
            with patch("executor.log_config.flow_doctor") as mock_module:
                mock_module.init.return_value = mock_fd
                mock_module.FlowDoctorHandler.return_value = mock_handler
                setup_logging("main")
                assert get_flow_doctor() is mock_fd

    def test_setup_loads_from_yaml_not_inline_kwargs(self):
        """Shared instance must load from flow-doctor.yaml, never inline kwargs.

        The pre-consolidation log_config.py built a second FlowDoctor instance
        from hardcoded inline kwargs that silently diverged from the YAML used
        by main.py / daemon.py / eod_reconcile.py. This test pins the new
        behavior so the regression doesn't return.
        """
        mock_fd = MagicMock()
        mock_handler = MagicMock(spec=logging.Handler)
        mock_handler.level = logging.ERROR
        with patch.dict(os.environ, {"FLOW_DOCTOR_ENABLED": "1"}):
            with patch("executor.log_config.flow_doctor") as mock_module:
                mock_module.init.return_value = mock_fd
                mock_module.FlowDoctorHandler.return_value = mock_handler
                setup_logging("main")
                mock_module.init.assert_called_once()
                call_args = mock_module.init.call_args
                # Must be called with config_path, not inline kwargs like repo/notify
                assert "config_path" in call_args.kwargs
                assert call_args.kwargs["config_path"].endswith("flow-doctor.yaml")
                # Must NOT be called with the old inline kwargs
                assert "repo" not in call_args.kwargs
                assert "notify" not in call_args.kwargs

    def test_init_failure_propagates(self):
        """flow-doctor is a hard dep — init failures should propagate, not be swallowed."""
        with patch.dict(os.environ, {"FLOW_DOCTOR_ENABLED": "1"}):
            with patch("executor.log_config.flow_doctor") as mock_module:
                mock_module.init.side_effect = RuntimeError("config error")
                with pytest.raises(RuntimeError, match="config error"):
                    setup_logging("test")

    def test_singleton_is_shared_across_call_sites(self):
        """Multiple get_flow_doctor() calls should return the same instance."""
        mock_fd = MagicMock()
        mock_handler = MagicMock(spec=logging.Handler)
        mock_handler.level = logging.ERROR
        with patch.dict(os.environ, {"FLOW_DOCTOR_ENABLED": "1"}):
            with patch("executor.log_config.flow_doctor") as mock_module:
                mock_module.init.return_value = mock_fd
                mock_module.FlowDoctorHandler.return_value = mock_handler
                setup_logging("main")
                assert get_flow_doctor() is get_flow_doctor()
                # And flow_doctor.init was called exactly once
                mock_module.init.assert_called_once()


class TestJSONFormatter:
    def test_formats_basic_record(self):
        import json
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="test.py",
            lineno=1, msg="something failed", args=(), exc_info=None,
        )
        result = json.loads(formatter.format(record))
        assert result["level"] == "ERROR"
        assert result["msg"] == "something failed"
        assert "ts" in result

    def test_includes_exception(self):
        import json
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="test.py",
                lineno=1, msg="failed", args=(), exc_info=sys.exc_info(),
            )
        result = json.loads(formatter.format(record))
        assert "ValueError" in result["exc"]

    def test_includes_ctx(self):
        import json
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="with context", args=(), exc_info=None,
        )
        record.ctx = {"ticker": "AAPL", "action": "ENTER"}
        result = json.loads(formatter.format(record))
        assert result["ctx"]["ticker"] == "AAPL"
