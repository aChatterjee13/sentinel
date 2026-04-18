"""Tests for per-channel rate limiting and Jinja2 template support."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from sentinel.action.notifications.channels.base import (
    DEFAULT_TEMPLATE,
    BaseChannel,
)
from sentinel.action.notifications.engine import NotificationEngine
from sentinel.action.notifications.policies import AlertPolicyEngine
from sentinel.config.schema import AlertPolicies, AlertsConfig
from sentinel.core.types import Alert, AlertSeverity, DeliveryResult

# ── Helpers ────────────────────────────────────────────────────────


def _alert(
    severity: AlertSeverity = AlertSeverity.HIGH,
    title: str = "drift detected",
    model: str = "fraud_v1",
    source: str = "drift",
    payload: dict[str, Any] | None = None,
) -> Alert:
    return Alert(
        model_name=model,
        title=title,
        body="PSI=0.3 on feature income",
        severity=severity,
        source=source,
        payload=payload or {},
    )


class _FakeChannel(BaseChannel):
    name = "fake"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sent: list[Alert] = []

    def send(self, alert: Alert) -> DeliveryResult:
        self.sent.append(alert)
        return DeliveryResult(channel=self.name, delivered=True)


def _engine(
    *,
    channels: dict[str, _FakeChannel] | None = None,
    cooldown: str = "0s",
    rate_limit_per_hour: int = 60,
    rate_limit_window: str = "1h",
) -> NotificationEngine:
    import threading
    from unittest.mock import MagicMock

    cfg = AlertsConfig(
        channels=[],
        policies=AlertPolicies(
            cooldown=cooldown,
            rate_limit_per_hour=rate_limit_per_hour,
            rate_limit_window=rate_limit_window,
        ),
    )
    eng = NotificationEngine.__new__(NotificationEngine)
    eng.config = cfg
    eng.audit = None
    eng.policies = AlertPolicyEngine(cfg.policies)
    eng.channels = channels or {}
    eng._escalation_timer = MagicMock()
    eng._dispatch_lock = threading.Lock()
    eng._ack_lock = threading.Lock()
    eng._acknowledged = set()
    eng._digest_timer = None
    return eng


# ── Per-channel rate limit ─────────────────────────────────────────


class TestPerChannelRateLimit:
    def test_rate_limit_per_channel_independent(self) -> None:
        """Slack at limit doesn't block Teams."""
        policies = AlertPolicies(rate_limit_per_hour=2, cooldown="0s")
        engine = AlertPolicyEngine(policies)

        # Saturate slack
        now = datetime.now(timezone.utc)
        engine._sent_per_channel["slack"].append(now)
        engine._sent_per_channel["slack"].append(now)

        assert engine.should_send_to_channel("slack") is False
        assert engine.should_send_to_channel("teams") is True

    def test_rate_limit_window_expiry(self) -> None:
        """Old entries are cleaned up after window expires."""
        policies = AlertPolicies(rate_limit_per_hour=1, cooldown="0s")
        engine = AlertPolicyEngine(policies)

        old = datetime.now(timezone.utc) - timedelta(hours=2)
        engine._sent_per_channel["slack"].append(old)

        assert engine.should_send_to_channel("slack") is True

    def test_rate_limit_zero_blocks_all(self) -> None:
        """rate_limit_per_hour=0 blocks all alerts."""
        policies = AlertPolicies(rate_limit_per_hour=0, cooldown="0s")
        engine = AlertPolicyEngine(policies)

        assert engine.should_send_to_channel("slack") is False

    def test_global_rate_limit_still_works(self) -> None:
        """should_send() still checks global rate limit."""
        policies = AlertPolicies(rate_limit_per_hour=2, cooldown="0s")
        engine = AlertPolicyEngine(policies)

        engine.record(_alert(title="a"))
        engine.record(_alert(title="b"))
        assert engine.should_send(_alert(title="c")) is False

    def test_configurable_window(self) -> None:
        """rate_limit_window='30m' uses 30-minute window."""
        policies = AlertPolicies(
            rate_limit_per_hour=1,
            rate_limit_window="30m",
            cooldown="0s",
        )
        engine = AlertPolicyEngine(policies)

        # Entry 35 minutes ago — outside the 30m window
        old = datetime.now(timezone.utc) - timedelta(minutes=35)
        engine._sent_per_channel["slack"].append(old)

        assert engine.should_send_to_channel("slack") is True

        # Entry 10 minutes ago — within the 30m window
        recent = datetime.now(timezone.utc) - timedelta(minutes=10)
        engine._sent_per_channel["slack"].append(recent)

        assert engine.should_send_to_channel("slack") is False

    def test_record_channel_send(self) -> None:
        """record_channel_send populates the per-channel deque."""
        policies = AlertPolicies(cooldown="0s")
        engine = AlertPolicyEngine(policies)

        engine.record_channel_send("slack")
        assert len(engine._sent_per_channel["slack"]) == 1
        assert len(engine._sent_per_channel["teams"]) == 0

    def test_engine_dispatch_per_channel_rate_limit(self) -> None:
        """Engine skips rate-limited channels but sends to others."""
        ch_slack = _FakeChannel()
        ch_slack.name = "slack"
        ch_teams = _FakeChannel()
        ch_teams.name = "teams"
        eng = _engine(
            channels={"slack": ch_slack, "teams": ch_teams},
            rate_limit_per_hour=1,
        )

        # First dispatch goes to both
        results = eng.dispatch(_alert(title="a1"))
        assert len(results) == 2

        # Second dispatch — both channels at limit
        results2 = eng.dispatch(_alert(title="a2"))
        assert len(results2) == 0
        assert len(ch_slack.sent) == 1
        assert len(ch_teams.sent) == 1


# ── Rate limit window config ──────────────────────────────────────


class TestRateLimitWindowConfig:
    def test_rate_limit_window_validation(self) -> None:
        """Invalid rate_limit_window format rejected by schema."""
        with pytest.raises(Exception):
            AlertPolicies(rate_limit_window="invalid")

    def test_rate_limit_window_default_1h(self) -> None:
        """Default rate_limit_window is '1h'."""
        policies = AlertPolicies()
        assert policies.rate_limit_window == "1h"

    def test_rate_limit_window_30m_accepted(self) -> None:
        policies = AlertPolicies(rate_limit_window="30m")
        assert policies.rate_limit_window == "30m"


# ── Jinja2 template support ───────────────────────────────────────


class TestJinja2Templates:
    def test_custom_template_renders(self) -> None:
        """Channel with template= kwarg renders Jinja2."""
        tpl = "Alert: {{ title }} ({{ severity }})"
        ch = _FakeChannel(template=tpl)
        alert = _alert(title="drift found", severity=AlertSeverity.CRITICAL)
        msg = ch.format_message(alert)
        assert msg == "Alert: drift found (CRITICAL)"

    def test_template_has_all_alert_fields(self) -> None:
        """Template receives all alert fields."""
        tpl = (
            "{{ alert_id }}|{{ model_name }}|{{ title }}|{{ body }}"
            "|{{ severity }}|{{ source }}|{{ timestamp }}|{{ fingerprint }}"
            "|{{ payload }}"
        )
        ch = _FakeChannel(template=tpl)
        alert = _alert(payload={"k": "v"})
        msg = ch.format_message(alert)
        assert "fraud_v1" in msg
        assert "drift detected" in msg
        assert "HIGH" in msg
        assert "drift" in msg
        assert "k" in msg

    def test_fallback_when_no_template(self) -> None:
        """Without template, uses hardcoded format."""
        ch = _FakeChannel()
        alert = _alert()
        msg = ch.format_message(alert)
        assert "*[HIGH]*" in msg
        assert "*Model:*" in msg

    def test_template_render_error_falls_back(self) -> None:
        """Bad template expression falls back to hardcoded format."""
        tpl = "{{ undefined_func() }}"
        ch = _FakeChannel(template=tpl)
        alert = _alert()
        msg = ch.format_message(alert)
        # Should gracefully fall back to default
        assert "*[HIGH]*" in msg

    def test_default_template_from_config(self) -> None:
        """AlertPolicies.default_template applies to channels without explicit template."""
        policies = AlertPolicies(default_template="Global: {{ title }}")
        assert policies.default_template == "Global: {{ title }}"

    def test_default_template_constant_exists(self) -> None:
        """DEFAULT_TEMPLATE constant is available and non-empty."""
        assert isinstance(DEFAULT_TEMPLATE, str)
        assert "{{ severity }}" in DEFAULT_TEMPLATE
        assert "{{ title }}" in DEFAULT_TEMPLATE

    def test_default_template_renders(self) -> None:
        """DEFAULT_TEMPLATE produces output matching the hardcoded format."""
        ch = _FakeChannel(template=DEFAULT_TEMPLATE)
        alert = _alert(payload={"feature": "income"})
        msg = ch.format_message(alert)
        assert "[HIGH]" in msg
        assert "fraud_v1" in msg
        assert "feature: income" in msg

    def test_channel_template_overrides_default(self) -> None:
        """A channel-level template takes precedence over default_template."""
        ch = _FakeChannel(template="Custom: {{ title }}")
        msg = ch.format_message(_alert())
        assert msg.startswith("Custom: ")

    def test_format_default_preserves_backward_compat(self) -> None:
        """_format_default produces the same output as the old format_message."""
        ch = _FakeChannel()
        alert = _alert(
            payload={"feature": "x", "score": 0.42},
        )
        msg = ch._format_default(alert)
        assert "[HIGH]" in msg
        assert "feature: x" in msg
        assert "score: 0.42" in msg
