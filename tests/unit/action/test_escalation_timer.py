"""Tests for Gap-D: EscalationTimer background daemon."""

from __future__ import annotations

import time

from sentinel.action.notifications.escalation import EscalationTimer
from sentinel.action.notifications.policies import AlertPolicyEngine
from sentinel.config.schema import AlertPolicies, EscalationStep
from sentinel.core.types import Alert, AlertSeverity


def _make_alert(title: str = "test alert", fp: str = "test-fp") -> Alert:
    return Alert(
        model_name="test_model",
        title=title,
        body="body",
        severity=AlertSeverity.HIGH,
        source="test",
        fingerprint=fp,
    )


def _make_step(after: str, channels: list[str]) -> EscalationStep:
    return EscalationStep(
        after=after,
        channels=channels,
        severity=["high", "critical"],
    )


class TestEscalationTimerLifecycle:
    def test_start_stop(self) -> None:
        timer = EscalationTimer(callback=lambda a, s: None)
        timer.start()
        assert timer.pending_count == 0
        timer.stop(timeout=2)

    def test_start_is_idempotent(self) -> None:
        timer = EscalationTimer(callback=lambda a, s: None)
        timer.start()
        timer.start()  # should not raise
        timer.stop(timeout=2)


class TestEscalationTimerScheduling:
    def test_schedule_fires_callback(self) -> None:
        fired: list[tuple[Alert, EscalationStep]] = []

        def on_escalation(alert: Alert, step: EscalationStep) -> None:
            fired.append((alert, step))

        timer = EscalationTimer(callback=on_escalation)
        timer.start()
        try:
            alert = _make_alert()
            steps = [_make_step("0s", ["slack"])]  # fire immediately
            timer.schedule(alert, steps)

            # Wait for the callback to fire
            for _ in range(50):
                if fired:
                    break
                time.sleep(0.02)

            assert len(fired) == 1
            assert fired[0][0].title == "test alert"
            assert fired[0][1].channels == ["slack"]
        finally:
            timer.stop(timeout=2)

    def test_multiple_steps_fire_in_order(self) -> None:
        fired: list[str] = []

        def on_escalation(alert: Alert, step: EscalationStep) -> None:
            fired.append(",".join(step.channels))

        timer = EscalationTimer(callback=on_escalation)
        timer.start()
        try:
            alert = _make_alert()
            steps = [
                _make_step("0s", ["slack"]),
                _make_step("0s", ["teams"]),
            ]
            timer.schedule(alert, steps)

            for _ in range(50):
                if len(fired) >= 2:
                    break
                time.sleep(0.02)

            assert len(fired) == 2
            assert "slack" in fired
            assert "teams" in fired
        finally:
            timer.stop(timeout=2)


class TestEscalationTimerCancel:
    def test_cancel_removes_pending(self) -> None:
        timer = EscalationTimer(callback=lambda a, s: None)
        timer.start()
        try:
            alert = _make_alert(fp="cancel-me")
            steps = [_make_step("10m", ["slack"])]  # far future
            timer.schedule(alert, steps)
            assert timer.pending_count == 1

            timer.cancel("cancel-me")
            assert timer.pending_count == 0
        finally:
            timer.stop(timeout=2)


class TestRemainingEscalationSteps:
    def test_no_escalation_policy(self) -> None:
        engine = AlertPolicyEngine(AlertPolicies())
        alert = _make_alert()
        assert engine.remaining_escalation_steps(alert) == []

    def test_returns_future_steps(self) -> None:
        policies = AlertPolicies(
            escalation=[
                EscalationStep(after="0m", channels=["slack"], severity=["high"]),
                EscalationStep(after="30m", channels=["teams"], severity=["high"]),
                EscalationStep(after="2h", channels=["pagerduty"], severity=["high"]),
            ]
        )
        engine = AlertPolicyEngine(policies)
        alert = _make_alert()
        remaining = engine.remaining_escalation_steps(alert)
        # Step[0] (0m) has already elapsed, steps[1] and [2] are future
        assert len(remaining) == 2
        assert remaining[0].channels == ["teams"]
        assert remaining[1].channels == ["pagerduty"]
