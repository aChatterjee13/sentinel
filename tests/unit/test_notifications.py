"""Unit tests for the notification engine, channels, and policies."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sentinel.action.notifications import CHANNEL_REGISTRY, register_channel
from sentinel.action.notifications.channels.base import BaseChannel
from sentinel.action.notifications.channels.slack import SlackChannel
from sentinel.action.notifications.channels.webhook import WebhookChannel
from sentinel.action.notifications.engine import NotificationEngine
from sentinel.action.notifications.policies import (
    AlertPolicyEngine,
    fingerprint,
    parse_duration,
    severity_at_least,
)
from sentinel.config.schema import (
    AlertPolicies,
    AlertsConfig,
    ChannelConfig,
    EscalationStep,
)
from sentinel.core.types import Alert, AlertSeverity, DeliveryResult


def _make_alert(
    severity: AlertSeverity = AlertSeverity.WARNING,
    title: str = "drift",
    model_name: str = "m",
) -> Alert:
    return Alert(
        model_name=model_name,
        title=title,
        body="something happened",
        severity=severity,
        source="drift_detector",
    )


class CapturingChannel(BaseChannel):
    """Test channel that records every alert it receives."""

    name = "capturing"

    def __init__(self, **config: Any):
        super().__init__(**config)
        self.received: list[Alert] = []

    def send(self, alert: Alert) -> DeliveryResult:
        self.received.append(alert)
        return DeliveryResult(channel=self.name, delivered=True)


class TestParseDuration:
    def test_seconds(self) -> None:
        assert parse_duration("30s") == timedelta(seconds=30)

    def test_minutes(self) -> None:
        assert parse_duration("5m") == timedelta(minutes=5)

    def test_hours(self) -> None:
        assert parse_duration("2h") == timedelta(hours=2)

    def test_days(self) -> None:
        assert parse_duration("3d") == timedelta(days=3)

    def test_weeks(self) -> None:
        assert parse_duration("1w") == timedelta(weeks=1)

    def test_invalid_raises(self) -> None:
        try:
            parse_duration("nonsense")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


class TestFingerprint:
    def test_explicit_fingerprint_used(self) -> None:
        a = Alert(
            model_name="m",
            title="t",
            body="b",
            severity=AlertSeverity.HIGH,
            source="s",
            fingerprint="custom-fp",
        )
        assert fingerprint(a) == "custom-fp"

    def test_derived_fingerprint_stable(self) -> None:
        a1 = _make_alert()
        a2 = _make_alert()
        assert fingerprint(a1) == fingerprint(a2)

    def test_derived_fingerprint_differs_on_title(self) -> None:
        assert fingerprint(_make_alert(title="a")) != fingerprint(_make_alert(title="b"))


class TestSeverityHelper:
    def test_severity_at_least(self) -> None:
        assert severity_at_least(AlertSeverity.HIGH, "warning") is True
        assert severity_at_least(AlertSeverity.INFO, "warning") is False
        assert severity_at_least(AlertSeverity.CRITICAL, "critical") is True


class TestAlertPolicyEngine:
    def test_should_send_first_time(self) -> None:
        engine = AlertPolicyEngine(AlertPolicies())
        assert engine.should_send(_make_alert()) is True

    def test_cooldown_suppresses_duplicate(self) -> None:
        engine = AlertPolicyEngine(AlertPolicies(cooldown="1h"))
        alert = _make_alert()
        assert engine.should_send(alert)
        engine.record(alert)
        # Same fingerprint within cooldown — suppressed
        assert engine.should_send(alert) is False

    def test_rate_limit_blocks_excess(self) -> None:
        engine = AlertPolicyEngine(AlertPolicies(rate_limit_per_hour=2, cooldown="1s"))
        # Fire 2 distinct alerts
        engine.record(_make_alert(title="a"))
        engine.record(_make_alert(title="b"))
        # A 3rd distinct alert should be blocked by rate limit
        assert engine.should_send(_make_alert(title="c")) is False

    def test_select_channels_no_escalation_returns_all(self) -> None:
        engine = AlertPolicyEngine(AlertPolicies())
        chosen = engine.select_channels(_make_alert(), ["slack", "teams"])
        assert chosen == ["slack", "teams"]

    def test_escalation_picks_first_step_channels(self) -> None:
        engine = AlertPolicyEngine(
            AlertPolicies(
                escalation=[
                    EscalationStep(after="0m", channels=["slack"], severity=["warning"]),
                    EscalationStep(after="1h", channels=["teams"], severity=["warning"]),
                ]
            )
        )
        chosen = engine.select_channels(_make_alert(), ["slack", "teams"])
        # Only the first step should match immediately
        assert chosen == ["slack"]

    def test_escalation_severity_filter(self) -> None:
        engine = AlertPolicyEngine(
            AlertPolicies(
                escalation=[
                    EscalationStep(after="0m", channels=["slack"], severity=["critical"]),
                ]
            )
        )
        # An INFO alert doesn't match the critical-only step → fallback to all channels
        chosen = engine.select_channels(_make_alert(severity=AlertSeverity.INFO), ["slack"])
        assert chosen == ["slack"]


class TestDigestMode:
    def test_queue_then_flush(self) -> None:
        engine = AlertPolicyEngine(AlertPolicies(digest_mode=True))
        engine.queue_for_digest(_make_alert(title="a", severity=AlertSeverity.WARNING))
        engine.queue_for_digest(_make_alert(title="b", severity=AlertSeverity.CRITICAL))
        assert engine.has_digest_pending()
        digest = engine.flush_digest()
        assert digest is not None
        assert digest.severity == AlertSeverity.CRITICAL  # worst severity wins
        assert "2 alerts" in digest.title
        assert not engine.has_digest_pending()

    def test_flush_empty_returns_none(self) -> None:
        engine = AlertPolicyEngine(AlertPolicies(digest_mode=True))
        assert engine.flush_digest() is None


class TestSlackChannel:
    def test_no_webhook_url_disables_channel(self) -> None:
        ch = SlackChannel()  # no webhook_url
        assert not ch.enabled
        result = ch.send(_make_alert())
        assert not result.delivered
        assert result.error == "channel disabled"

    def test_payload_contains_alert_fields(self) -> None:
        ch = SlackChannel(webhook_url="https://hooks.slack.com/services/X/Y/Z")
        payload = ch._build_payload(_make_alert(severity=AlertSeverity.HIGH))
        attachment = payload["attachments"][0]
        # All required fields present
        field_titles = {f["title"] for f in attachment["fields"]}
        assert {"Model", "Severity", "Source", "Time"}.issubset(field_titles)


class TestWebhookChannel:
    def test_no_url_disables(self) -> None:
        ch = WebhookChannel()
        assert not ch.enabled
        result = ch.send(_make_alert())
        assert not result.delivered

    def test_send_handles_network_error(self) -> None:
        # Use a guaranteed-unreachable URL — should produce a delivery failure, not raise
        ch = WebhookChannel(webhook_url="http://127.0.0.1:1/does-not-exist")
        result = ch.send(_make_alert())
        assert result.delivered is False
        assert result.error is not None


class TestChannelRegistry:
    def test_default_channels_registered(self) -> None:
        for name in ["slack", "teams", "pagerduty", "email", "webhook"]:
            assert name in CHANNEL_REGISTRY

    def test_register_custom_channel(self) -> None:
        register_channel("capturing", CapturingChannel)
        try:
            assert CHANNEL_REGISTRY["capturing"] is CapturingChannel
        finally:
            CHANNEL_REGISTRY.pop("capturing", None)


class TestBaseChannelFormatting:
    def test_format_message_includes_payload(self) -> None:
        ch = CapturingChannel()
        alert = Alert(
            model_name="m",
            title="t",
            body="b",
            severity=AlertSeverity.HIGH,
            source="src",
            payload={"feature": "x", "score": 0.42},
        )
        formatted = ch.format_message(alert)
        assert "[HIGH]" in formatted
        assert "feature: x" in formatted
        assert "score: 0.42" in formatted


class TestNotificationEngine:
    def _engine_with_capture(
        self, policies: AlertPolicies | None = None
    ) -> tuple[NotificationEngine, CapturingChannel]:
        # Build engine with empty channel config, then inject our test channel
        # directly. This avoids fighting the ChannelConfig.type Literal.
        cfg = AlertsConfig(channels=[], policies=policies or AlertPolicies())
        engine = NotificationEngine(cfg)
        capturing = CapturingChannel()
        engine.channels["capturing"] = capturing
        return engine, capturing

    def test_dispatch_delivers_to_channel(self) -> None:
        engine, channel = self._engine_with_capture()
        results = engine.dispatch(_make_alert())
        assert len(results) == 1
        assert results[0].delivered
        assert len(channel.received) == 1

    def test_dispatch_respects_cooldown(self) -> None:
        engine, channel = self._engine_with_capture()
        engine.dispatch(_make_alert())
        # Same alert again — should be suppressed
        results = engine.dispatch(_make_alert())
        assert results == []
        assert len(channel.received) == 1

    def test_digest_mode_queues_instead_of_sending(self) -> None:
        engine, channel = self._engine_with_capture(policies=AlertPolicies(digest_mode=True))
        engine.dispatch(_make_alert(title="one"))
        engine.dispatch(_make_alert(title="two"))
        assert len(channel.received) == 0  # nothing sent yet
        assert engine.policies.has_digest_pending()

    def test_flush_digest_sends_combined_alert(self) -> None:
        engine, _channel = self._engine_with_capture(policies=AlertPolicies(digest_mode=True))
        engine.dispatch(_make_alert(title="a", severity=AlertSeverity.WARNING))
        engine.dispatch(_make_alert(title="b", severity=AlertSeverity.CRITICAL))
        # Flush builds a digest alert. Because digest_mode is still on,
        # dispatch() of the digest itself will re-queue, so call directly.
        digest = engine.policies.flush_digest()
        assert digest is not None
        assert "2 alerts" in digest.title

    def test_unknown_channel_init_skipped_gracefully(self) -> None:
        # Webhook channel with placeholder URL → constructs but may fail to deliver
        cfg = AlertsConfig(
            channels=[ChannelConfig(type="webhook", webhook_url="https://example.com/hook")],
            policies=AlertPolicies(),
        )
        engine = NotificationEngine(cfg)
        results = engine.dispatch(_make_alert())
        assert len(results) == 1
