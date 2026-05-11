"""Tests for modules/logging_utils.py — structured_log helper."""
import logging
import sys
import os
from unittest.mock import MagicMock, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.logging_utils import structured_log


class TestStructuredLog:
    def setup_method(self):
        self.mock_logger = MagicMock(spec=logging.Logger)
        # Wire the level methods so they are callable on the mock
        for level in ("debug", "info", "warning", "error", "critical"):
            setattr(self.mock_logger, level, MagicMock())

    def test_info_level_dispatches_to_logger_info(self):
        structured_log(self.mock_logger, "info", "Stage complete")
        self.mock_logger.info.assert_called_once()

    def test_warning_level_dispatches_to_logger_warning(self):
        structured_log(self.mock_logger, "warning", "Something odd")
        self.mock_logger.warning.assert_called_once()

    def test_event_alone(self):
        structured_log(self.mock_logger, "info", "hello")
        args = self.mock_logger.info.call_args[0]
        assert args[0] == "hello"

    def test_run_id_placed_first(self):
        structured_log(self.mock_logger, "info", "done", run_id="abc123", concepts=4)
        args = self.mock_logger.info.call_args[0]
        msg = args[0]
        assert msg.startswith("[run_id=abc123]")
        assert "[concepts=4]" in msg
        assert msg.endswith("done")

    def test_none_values_omitted(self):
        structured_log(self.mock_logger, "info", "msg", run_id="r1", paper_id=None)
        args = self.mock_logger.info.call_args[0]
        msg = args[0]
        assert "paper_id" not in msg
        assert "[run_id=r1]" in msg

    def test_empty_string_values_omitted(self):
        structured_log(self.mock_logger, "info", "msg", run_id="r1", extra="")
        args = self.mock_logger.info.call_args[0]
        msg = args[0]
        assert "extra" not in msg

    def test_zero_value_included(self):
        structured_log(self.mock_logger, "info", "msg", count=0)
        args = self.mock_logger.info.call_args[0]
        assert "[count=0]" in args[0]

    def test_no_fields_returns_event_only(self):
        structured_log(self.mock_logger, "info", "bare event")
        args = self.mock_logger.info.call_args[0]
        assert args[0] == "bare event"

    def test_unknown_level_falls_back_to_info(self):
        structured_log(self.mock_logger, "trace", "msg")
        self.mock_logger.info.assert_called_once()

    def test_multiple_fields_all_present(self):
        structured_log(
            self.mock_logger, "info", "Pipeline step",
            run_id="r1", stage="extraction", tokens=4096, concepts=12,
        )
        args = self.mock_logger.info.call_args[0]
        msg = args[0]
        assert "[run_id=r1]" in msg
        assert "[stage=extraction]" in msg
        assert "[tokens=4096]" in msg
        assert "[concepts=12]" in msg
        assert msg.endswith("Pipeline step")
