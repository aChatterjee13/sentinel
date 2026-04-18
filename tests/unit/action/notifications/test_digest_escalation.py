"""Tests for digest timer auto-fire, acknowledgement, and escalation filtering."""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

from sentinel.action.notifications.channels.base import BaseChannel
from sentinel.action.notifications.engine import DigestTimer, NotificationEngine
from sentinel.action.notifications.policies import AlertPolicyEngine, fingerprint
from sentinel.config.schema import (
    AlertPolicies,
    AlertsConfig,
    EscalationStep,
)
from sentinel.core.types import Alert, AlertSeverity, DeliveryResult

# ── Helpers ────────────────────────────────────────────────────────


def _alert(
    severity: AlertSeverity = AlertSeverity.HIGH,
    title: str = "drift detected",
    model: str = "fraud_v1",
    source: str = "drift",
    fp: str | None = None,
) -> Alert:
    return Alert(
        model_name=model,
        title=title,
        body="PSI=0.3 on feature income",
        severity=severity,
        source=source,
        fingerprint=fp,
    )


def _ok_result(channel: str = "slack") -> DeliveryResult:
    return DeliveryResult(channel=channel, delivered=True)


class _FakeChannel(BaseChannel):
    """A minimal BaseChannel substitute that records calls."""

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
    digest_mode: bool = False,
    digest_interval: str = "6h",
    escalation: list[EscalationStep] | None = None,
    cooldown: str = "0s",
    audit: MagicMock | None = None,
) -> NotificationEngine:
    """Build an engine with injected fake channels (bypassing registry lookup)."""
    cfg = AlertsConfig(
        channels=[],
        policies=AlertPolicies(
            cooldown=cooldown,
            digest_mode=digest_mode,
            digest_interval=digest_interval,
            escalation=escalation or [],
        ),
    )
    eng = NotificationEngine.__new__(NotificationEngine)
    eng.config = cfg
    eng.audit = audit
    eng.policies = AlertPolicyEngine(cfg.policies)
    eng.channels = channels or {}
    eng._dispatch_lock = threading.Lock()
    eng._ack_lock = threading.Lock()
    eng._acknowledged: set[str] = set()
    eng._digest_timer = None
    eng._escalation_timer = MagicMock()
    return eng


# ── Digest timer ──────────────────────────────────────────────────


class TestDigestTimer:
    """DigestTimer daemon thread lifecycle and auto-flush."""

    def test_digest_auto_flushes_on_interval(self) -> None:
        """When digest_mode=True, alerts are auto-flushed after digest_interval."""
        ch = _FakeChannel()
        eng = _engine(channels={"fake": ch}, digest_mode=True, digest_interval="1s")

        # Queue two alerts manually (dispatch queues when digest_mode=True)
        eng.policies.queue_for_digest(_alert(title="a1"))
        eng.policies.queue_for_digest(_alert(title="a2"))
        assert eng.policies.has_digest_pending()

        # Start a digest timer with a 1s interval
        timer = DigestTimer(interval_seconds=1.0, flush_callback=eng.flush_digest)
        timer.start()
        try:
            # Wait enough time for the timer to fire at least once
            time.sleep(2.5)
            # The timer should have called flush_digest which dispatches via _dispatch_direct
            assert len(ch.sent) >= 1
            digest = ch.sent[0]
            assert "2 alerts" in digest.title
            assert not eng.policies.has_digest_pending()
        finally:
            timer.stop(timeout=2)

    def test_digest_timer_stops_on_close(self) -> None:
        """close() stops the digest timer thread."""
        eng = _engine(digest_mode=True, digest_interval="1s")
        timer = DigestTimer(interval_seconds=1.0, flush_callback=eng.flush_digest)
        timer.start()
        eng._digest_timer = timer
        assert timer.alive
        eng.close()
        time.sleep(0.2)
        assert not timer.alive

    def test_digest_flush_bypasses_digest_mode(self) -> None:
        """flush_digest dispatches directly, not back into digest queue."""
        ch = _FakeChannel()
        eng = _engine(channels={"fake": ch}, digest_mode=True)
        eng.dispatch(_alert(title="a1"))
        eng.dispatch(_alert(title="a2"))
        assert eng.policies.has_digest_pending()

        # flush_digest should send to channel, not re-queue
        results = eng.flush_digest()
        assert results is not None
        assert len(results) == 1
        assert results[0].delivered
        assert len(ch.sent) == 1
        assert "2 alerts" in ch.sent[0].title
        assert not eng.policies.has_digest_pending()

    def test_digest_no_timer_when_digest_mode_false(self) -> None:
        """No timer thread created when digest_mode=False."""
        cfg = AlertsConfig(
            channels=[],
            policies=AlertPolicies(digest_mode=False),
        )
        eng = NotificationEngine(cfg)
        try:
            assert eng._digest_timer is None
        finally:
            eng.close()

    def test_digest_timer_created_when_digest_mode_true(self) -> None:
        """Timer is created and started when digest_mode=True."""
        cfg = AlertsConfig(
            channels=[],
            policies=AlertPolicies(digest_mode=True, digest_interval="30s"),
        )
        eng = NotificationEngine(cfg)
        try:
            assert eng._digest_timer is not None
            assert eng._digest_timer.alive
        finally:
            eng.close()


# ── Digest content enrichment ─────────────────────────────────────


class TestDigestContent:
    """Digest alert payload contains severity breakdown and fingerprints."""

    def test_digest_includes_severity_breakdown(self) -> None:
        """Digest title includes severity counts."""
        policy = AlertPolicyEngine(AlertPolicies(digest_mode=True))
        policy.queue_for_digest(_alert(title="a1", severity=AlertSeverity.WARNING))
        policy.queue_for_digest(_alert(title="a2", severity=AlertSeverity.CRITICAL))
        policy.queue_for_digest(_alert(title="a3", severity=AlertSeverity.WARNING))

        digest = policy.flush_digest()
        assert digest is not None
        assert "2 alerts" not in digest.title  # 3 alerts
        assert "3 alerts" in digest.title
        payload = digest.payload
        assert payload["severity_breakdown"]["warning"] == 2
        assert payload["severity_breakdown"]["critical"] == 1

    def test_digest_payload_has_fingerprints(self) -> None:
        """Digest payload contains fingerprints of original alerts."""
        policy = AlertPolicyEngine(AlertPolicies(digest_mode=True))
        a1 = _alert(title="drift1")
        a2 = _alert(title="drift2")
        policy.queue_for_digest(a1)
        policy.queue_for_digest(a2)

        digest = policy.flush_digest()
        assert digest is not None
        fps = digest.payload["fingerprints"]
        assert len(fps) == 2
        assert fingerprint(a1) in fps
        assert fingerprint(a2) in fps

    def test_digest_payload_has_model_names(self) -> None:
        """Digest payload contains unique model names."""
        policy = AlertPolicyEngine(AlertPolicies(digest_mode=True))
        policy.queue_for_digest(_alert(model="model_a"))
        policy.queue_for_digest(_alert(model="model_b"))
        policy.queue_for_digest(_alert(model="model_a"))

        digest = policy.flush_digest()
        assert digest is not None
        models = digest.payload["models"]
        assert set(models) == {"model_a", "model_b"}

    def test_digest_body_includes_model_names(self) -> None:
        """Each line in the digest body shows the model name."""
        policy = AlertPolicyEngine(AlertPolicies(digest_mode=True))
        policy.queue_for_digest(_alert(title="x", model="claims_v2"))

        digest = policy.flush_digest()
        assert digest is not None
        assert "model=claims_v2" in digest.body


# ── Acknowledgement ───────────────────────────────────────────────


class TestAcknowledge:
    """acknowledge() cancels escalations and suppresses re-dispatch."""

    def test_acknowledge_cancels_escalation(self) -> None:
        """acknowledge() calls escalation_timer.cancel()."""
        eng = _engine()
        alert = _alert(fp="fp-cancel")
        eng.acknowledge(alert)
        eng._escalation_timer.cancel.assert_called_once_with("fp-cancel")

    def test_acknowledge_with_alert_object(self) -> None:
        """Can pass Alert object to acknowledge()."""
        eng = _engine()
        alert = _alert(fp="fp-obj")
        eng.acknowledge(alert)
        eng._escalation_timer.cancel.assert_called_once_with("fp-obj")

    def test_acknowledge_with_fingerprint_string(self) -> None:
        """Can pass fingerprint string to acknowledge()."""
        eng = _engine()
        eng.acknowledge("fp-string")
        eng._escalation_timer.cancel.assert_called_once_with("fp-string")

    def test_acknowledge_logs_to_audit(self) -> None:
        """acknowledge() logs to audit trail."""
        audit = MagicMock()
        eng = _engine(audit=audit)
        alert = _alert(fp="fp-audit")
        eng.acknowledge(alert)
        audit.log.assert_called_once()
        call_kwargs = audit.log.call_args.kwargs
        assert call_kwargs["event_type"] == "alert_acknowledged"
        assert call_kwargs["fingerprint"] == "fp-audit"
        assert call_kwargs["model_name"] == "fraud_v1"

    def test_acknowledge_string_logs_none_model(self) -> None:
        """When acknowledging by fingerprint string, model_name is None."""
        audit = MagicMock()
        eng = _engine(audit=audit)
        eng.acknowledge("bare-fp")
        call_kwargs = audit.log.call_args.kwargs
        assert call_kwargs["model_name"] is None

    def test_acknowledged_alert_suppressed_on_redispatch(self) -> None:
        """After acknowledgment, same alert is suppressed."""
        ch = _FakeChannel()
        eng = _engine(channels={"fake": ch})
        alert = _alert(fp="suppress-me")
        # First dispatch works
        results1 = eng.dispatch(alert)
        assert len(results1) == 1
        assert len(ch.sent) == 1

        # Acknowledge
        eng.acknowledge(alert)

        # Second dispatch is suppressed
        results2 = eng.dispatch(alert)
        assert results2 == []
        assert len(ch.sent) == 1  # no new sends

    def test_acknowledged_escalation_skipped(self) -> None:
        """Escalation callback skips acknowledged alerts."""
        ch = _FakeChannel()
        eng = _engine(channels={"slack": ch})
        alert = _alert(fp="ack-esc")
        step = EscalationStep(after="30m", channels=["slack"])

        eng.acknowledge(alert)
        eng._on_escalation(alert, step)
        # Channel should NOT have received the escalation
        assert len(ch.sent) == 0


# ── Escalation filtering ─────────────────────────────────────────


class TestEscalationFiltering:
    """Remaining escalation steps filter out unconfigured channels."""

    def test_remaining_steps_filters_missing_channels(self) -> None:
        """Steps with non-existent channels are filtered out."""
        steps = [
            EscalationStep(after="0m", channels=["slack"], severity=["high"]),
            EscalationStep(after="30m", channels=["pagerduty"], severity=["high"]),
            EscalationStep(after="2h", channels=["slack"], severity=["high"]),
        ]
        policy = AlertPolicyEngine(AlertPolicies(escalation=steps))
        alert = _alert()
        # Only "slack" is available — "pagerduty" step should be filtered
        remaining = policy.remaining_escalation_steps(
            alert, available_channels=["slack"],
        )
        # step[0] already elapsed, step[1] filtered (pagerduty), step[2] kept
        assert len(remaining) == 1
        assert remaining[0].channels == ["slack"]
        assert remaining[0].after == "2h"

    def test_remaining_steps_no_filter_when_none(self) -> None:
        """When available_channels=None, no filtering is applied."""
        steps = [
            EscalationStep(after="30m", channels=["nonexistent"], severity=["high"]),
        ]
        policy = AlertPolicyEngine(AlertPolicies(escalation=steps))
        alert = _alert()
        remaining = policy.remaining_escalation_steps(alert, available_channels=None)
        assert len(remaining) == 1

    def test_remaining_steps_all_filtered_returns_empty(self) -> None:
        """If all future steps reference missing channels, result is empty."""
        steps = [
            EscalationStep(after="0m", channels=["slack"], severity=["high"]),
            EscalationStep(after="30m", channels=["pagerduty"], severity=["high"]),
        ]
        policy = AlertPolicyEngine(AlertPolicies(escalation=steps))
        alert = _alert()
        remaining = policy.remaining_escalation_steps(
            alert, available_channels=["slack"],
        )
        assert remaining == []

    def test_engine_passes_available_channels(self) -> None:
        """Engine.dispatch passes available_channels to remaining_escalation_steps."""
        steps = [
            EscalationStep(after="0m", channels=["fake"], severity=["high"]),
            EscalationStep(after="30m", channels=["nonexistent"], severity=["high"]),
        ]
        ch = _FakeChannel()
        eng = _engine(channels={"fake": ch}, escalation=steps)
        eng.dispatch(_alert(severity=AlertSeverity.HIGH))
        # The "nonexistent" step should be filtered, so schedule is NOT called
        eng._escalation_timer.schedule.assert_not_called()
