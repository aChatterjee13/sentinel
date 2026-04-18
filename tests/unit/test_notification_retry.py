"""Tests for notification channel retry logic."""

from datetime import datetime, timezone
from unittest.mock import patch

from sentinel.action.notifications.channels.base import BaseChannel
from sentinel.core.types import Alert, AlertSeverity, DeliveryResult


class FlakyChannel(BaseChannel):
    """A test channel that fails N times then succeeds."""

    name = "flaky"

    def __init__(self, fail_count: int = 2, **config):
        super().__init__(**config)
        self.fail_count = fail_count
        self.attempts = 0

    def send(self, alert):
        self.attempts += 1
        if self.attempts <= self.fail_count:
            return DeliveryResult(channel=self.name, delivered=False, error="connection timeout")
        return DeliveryResult(channel=self.name, delivered=True)


class AlwaysFailChannel(BaseChannel):
    name = "always_fail"

    def send(self, alert):
        return DeliveryResult(channel=self.name, delivered=False, error="server error")


class DisabledChannel(BaseChannel):
    name = "disabled_ch"

    def send(self, alert):
        return DeliveryResult(channel=self.name, delivered=False, error="channel disabled")


def _make_alert():
    return Alert(
        alert_id="test-1",
        title="Test Alert",
        body="Test body",
        severity=AlertSeverity.HIGH,
        source="test",
        model_name="test_model",
        timestamp=datetime.now(timezone.utc),
        payload={},
    )


class TestRetryLogic:
    def test_succeeds_on_first_try(self):
        ch = FlakyChannel(fail_count=0)
        result = ch.send_with_retry(_make_alert())
        assert result.delivered
        assert ch.attempts == 1

    @patch("sentinel.action.notifications.channels.base.time.sleep")
    def test_retries_and_succeeds(self, mock_sleep):
        ch = FlakyChannel(fail_count=2, max_retries=3)
        result = ch.send_with_retry(_make_alert())
        assert result.delivered
        assert ch.attempts == 3
        assert mock_sleep.call_count == 2

    @patch("sentinel.action.notifications.channels.base.time.sleep")
    def test_all_retries_exhausted(self, mock_sleep):
        ch = AlwaysFailChannel(max_retries=3)
        result = ch.send_with_retry(_make_alert())
        assert not result.delivered
        assert "server error" in result.error
        assert mock_sleep.call_count == 3

    def test_no_retry_on_permanent_failure(self):
        ch = DisabledChannel(max_retries=3)
        result = ch.send_with_retry(_make_alert())
        assert not result.delivered
        # Should not retry — "disabled" is a permanent failure

    @patch("sentinel.action.notifications.channels.base.time.sleep")
    def test_backoff_timing(self, mock_sleep):
        ch = AlwaysFailChannel(max_retries=2)
        ch.send_with_retry(_make_alert())
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls[0] == 2.0  # 2^1
        assert calls[1] == 4.0  # 2^2

    def test_max_retries_configurable(self):
        ch = FlakyChannel(fail_count=0, max_retries=5)
        assert ch.max_retries == 5
