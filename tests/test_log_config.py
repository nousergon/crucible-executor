"""Tests for log_config — structured logging + flow-doctor integration."""

import logging
import os
from unittest.mock import patch, MagicMock

import pytest

from executor.log_config import setup_logging, _attach_flow_doctor, JSONFormatter


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
            fmt = root = logging.getLogger().handlers[0].formatter._fmt
            assert "[daemon]" in fmt

    def test_clears_existing_handlers(self):
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLOW_DOCTOR_ENABLED", None)
            setup_logging("test")
        assert len(root.handlers) == 1


class TestFlowDoctorIntegration:
    def test_not_attached_when_disabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLOW_DOCTOR_ENABLED", None)
            setup_logging("test")
            root = logging.getLogger()
            assert len(root.handlers) == 1  # only StreamHandler

    def test_attached_when_enabled(self):
        mock_fd = MagicMock()
        mock_handler = MagicMock(spec=logging.Handler)
        mock_handler.level = logging.ERROR
        with patch.dict(os.environ, {"FLOW_DOCTOR_ENABLED": "1"}):
            with patch.dict("sys.modules", {"flow_doctor": MagicMock()}) as _:
                import sys
                mock_module = sys.modules["flow_doctor"]
                mock_module.init.return_value = mock_fd
                mock_module.FlowDoctorHandler.return_value = mock_handler
                setup_logging("main")
                mock_module.init.assert_called_once()
                call_kwargs = mock_module.init.call_args[1]
                assert call_kwargs["flow_name"] == "alpha-engine-executor-main"
                assert call_kwargs["repo"] == "cipher813/alpha-engine"

    def test_graceful_when_import_fails(self):
        with patch.dict(os.environ, {"FLOW_DOCTOR_ENABLED": "1"}):
            with patch("builtins.__import__", side_effect=ImportError("no flow_doctor")):
                # Should not raise — flow-doctor is non-critical
                setup_logging("test")
                root = logging.getLogger()
                assert len(root.handlers) == 1

    def test_graceful_when_init_fails(self):
        with patch.dict(os.environ, {"FLOW_DOCTOR_ENABLED": "1"}):
            with patch.dict("sys.modules", {"flow_doctor": MagicMock()}) as _:
                import sys
                mock_module = sys.modules["flow_doctor"]
                mock_module.init.side_effect = RuntimeError("config error")
                setup_logging("test")
                root = logging.getLogger()
                assert len(root.handlers) == 1  # only StreamHandler, no flow-doctor


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
