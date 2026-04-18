"""Tests for sentinel.core.scheduler — WS-C."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from sentinel.core.scheduler import DriftScheduler, _parse_interval


class TestParseInterval:
    def test_seconds(self) -> None:
        assert _parse_interval("30s") == 30.0

    def test_minutes(self) -> None:
        assert _parse_interval("5m") == 300.0

    def test_hours(self) -> None:
        assert _parse_interval("1h") == 3600.0

    def test_days(self) -> None:
        assert _parse_interval("7d") == 604800.0

    def test_weeks(self) -> None:
        assert _parse_interval("1w") == 604800.0

    def test_float_value(self) -> None:
        assert _parse_interval("0.5h") == 1800.0

    def test_whitespace_stripped(self) -> None:
        assert _parse_interval("  7d  ") == 604800.0

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid interval"):
            _parse_interval("bad")

    def test_missing_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid interval"):
            _parse_interval("100")

    def test_unknown_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid interval"):
            _parse_interval("10x")


class TestDriftScheduler:
    def _mock_client(self) -> MagicMock:
        client = MagicMock()
        report = MagicMock()
        report.is_drifted = False
        client.check_drift.return_value = report
        return client

    def test_start_stop_lifecycle(self) -> None:
        client = self._mock_client()
        scheduler = DriftScheduler(client, interval="30s")
        assert not scheduler.is_running
        scheduler.start()
        assert scheduler.is_running
        scheduler.stop(timeout=2.0)
        assert not scheduler.is_running

    def test_idempotent_start(self) -> None:
        client = self._mock_client()
        scheduler = DriftScheduler(client, interval="30s")
        scheduler.start()
        scheduler.start()  # second start is no-op
        assert scheduler.is_running
        scheduler.stop(timeout=2.0)

    def test_run_on_start(self) -> None:
        client = self._mock_client()
        scheduler = DriftScheduler(client, interval="999s", run_on_start=True)
        scheduler.start()
        time.sleep(0.3)  # give the thread time to execute
        scheduler.stop(timeout=2.0)
        assert client.check_drift.called
        assert scheduler.run_count >= 1

    def test_run_count_increments(self) -> None:
        client = self._mock_client()
        scheduler = DriftScheduler(client, interval="999s", run_on_start=True)
        scheduler.start()
        time.sleep(0.3)
        scheduler.stop(timeout=2.0)
        assert scheduler.run_count >= 1

    def test_error_resilience(self) -> None:
        """check_drift raising should not crash the scheduler."""
        client = self._mock_client()
        client.check_drift.side_effect = RuntimeError("kaboom")
        scheduler = DriftScheduler(client, interval="999s", run_on_start=True)
        scheduler.start()
        time.sleep(0.3)
        assert scheduler.is_running  # still alive
        scheduler.stop(timeout=2.0)
        assert scheduler.run_count >= 1

    def test_disabled_by_default(self) -> None:
        """Scheduler should not start unless explicitly started."""
        client = self._mock_client()
        scheduler = DriftScheduler(client, interval="30s")
        assert not scheduler.is_running
        assert scheduler.run_count == 0


class TestClientContextManager:
    def test_context_manager(self, minimal_config: object) -> None:
        from sentinel.core.client import SentinelClient

        with patch.object(SentinelClient, "__init__", return_value=None):
            client = SentinelClient.__new__(SentinelClient)
            client._scheduler = None
            client.audit = MagicMock()
            client.notifications = MagicMock()
            with client:
                pass
            client.audit.close.assert_called_once()
            client.notifications.close.assert_called_once()
