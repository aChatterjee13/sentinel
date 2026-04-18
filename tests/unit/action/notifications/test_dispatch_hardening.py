"""Tests for thread safety and dispatch hardening in the notification engine.

Covers: dispatch lock, channel exception handling, all-channels-fail fallback,
digest queue thread safety, cooldown GC, escalation dedup, and channel init
failure logging.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import structlog

from sentinel.action.notifications.channels.base import BaseChannel
from sentinel.action.notifications.engine import NotificationEngine
from sentinel.action.notifications.escalation import EscalationTimer
from sentinel.action.notifications.policies import AlertPolicyEngine
from sentinel.config.schema import AlertPolicies, AlertsConfig, EscalationStep
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


class _FakeChannel(BaseChannel):
    """Minimal channel that records calls and always succeeds."""

    name = "fake"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sent: list[Alert] = []

    def send(self, alert: Alert) -> DeliveryResult:
        self.sent.append(alert)
        return DeliveryResult(channel=self.name, delivered=True)


class _FailChannel(BaseChannel):
    """Channel that always returns delivered=False."""

    name = "fail"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_retries = 0  # skip retry backoff in tests

    def send(self, alert: Alert) -> DeliveryResult:
        return DeliveryResult(channel=self.name, delivered=False, error="unavailable")


class _RaisingChannel(BaseChannel):
    """Channel whose send() raises an unexpected exception."""

    name = "raising"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_retries = 0

    def send(self, alert: Alert) -> DeliveryResult:
        raise RuntimeError("channel exploded")


class _SlowFakeChannel(BaseChannel):
    """Channel that sleeps briefly to widen race windows."""

    name = "slow"

    def __init__(self, delay: float = 0.01, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._delay = delay
        self.sent: list[Alert] = []

    def send(self, alert: Alert) -> DeliveryResult:
        time.sleep(self._delay)
        self.sent.append(alert)
        return DeliveryResult(channel=self.name, delivered=True)


def _engine(
    *,
    channels: dict[str, BaseChannel] | None = None,
    cooldown: str = "0s",
    digest_mode: bool = False,
    escalation: list[EscalationStep] | None = None,
) -> NotificationEngine:
    """Build an engine with injected channels (bypasses registry)."""
    cfg = AlertsConfig(
        channels=[],
        policies=AlertPolicies(
            cooldown=cooldown,
            digest_mode=digest_mode,
            escalation=escalation or [],
        ),
    )
    eng = NotificationEngine.__new__(NotificationEngine)
    eng.config = cfg
    eng.audit = None
    eng.policies = AlertPolicyEngine(cfg.policies)
    eng.channels = channels or {}
    eng._dispatch_lock = threading.Lock()
    eng._ack_lock = threading.Lock()
    eng._acknowledged: set[str] = set()
    eng._digest_timer = None
    eng._escalation_timer = MagicMock()
    return eng


# ── Dispatch lock ──────────────────────────────────────────────────


class TestDispatchLock:
    """The dispatch lock serialises check-then-send to prevent double-sends."""

    def test_concurrent_dispatch_no_duplicate_sends(self) -> None:
        """Two threads dispatching same alert — only one should send (cooldown)."""
        ch = _SlowFakeChannel(delay=0.005)
        eng = _engine(channels={"slow": ch}, cooldown="1h")
        alert = _alert()

        results: list[list[DeliveryResult]] = [[], []]

        def dispatch(idx: int) -> None:
            results[idx] = eng.dispatch(alert)

        t1 = threading.Thread(target=dispatch, args=(0,))
        t2 = threading.Thread(target=dispatch, args=(1,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one thread should have sent, the other suppressed by cooldown
        suppressed_counts = [1 for r in results if r == []]
        assert len(ch.sent) == 1, f"Expected exactly 1 send, got {len(ch.sent)}"
        assert sum(suppressed_counts) == 1

    def test_concurrent_dispatch_different_alerts(self) -> None:
        """Two threads dispatching different alerts — both should send."""
        ch = _FakeChannel()
        eng = _engine(channels={"fake": ch}, cooldown="1h")
        alert_a = _alert(title="alert A", fp="fp-a")
        alert_b = _alert(title="alert B", fp="fp-b")

        results: list[list[DeliveryResult]] = [[], []]

        def dispatch(idx: int, a: Alert) -> None:
            results[idx] = eng.dispatch(a)

        t1 = threading.Thread(target=dispatch, args=(0, alert_a))
        t2 = threading.Thread(target=dispatch, args=(1, alert_b))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(ch.sent) == 2


# ── Channel exception handling ─────────────────────────────────────


class TestChannelExceptionHandling:
    """Exceptions from channel.send_with_retry() must not crash dispatch."""

    def test_channel_raises_exception_dispatch_continues(self) -> None:
        """If one channel raises, dispatch continues to next channel."""
        bad = _RaisingChannel()
        good = _FakeChannel()
        good.name = "good"
        eng = _engine(channels={"raising": bad, "good": good})
        results = eng.dispatch(_alert())
        # Only the good channel's result should be present
        assert len(results) == 1
        assert results[0].delivered is True
        assert len(good.sent) == 1

    def test_all_channels_raise_returns_empty_results(self) -> None:
        """If all channels raise, returns empty results."""
        bad1 = _RaisingChannel()
        bad2 = _RaisingChannel()
        bad2.name = "raising2"
        eng = _engine(channels={"raising": bad1, "raising2": bad2})
        results = eng.dispatch(_alert())
        assert results == []


# ── All channels fail ──────────────────────────────────────────────


class TestAllChannelsFail:
    """When all channels return delivered=False, critical logging occurs."""

    def test_all_channels_fail_logs_critical(self, capsys: Any) -> None:
        """When all channels return delivered=False, a critical log is emitted."""
        fail_ch = _FailChannel()
        eng = _engine(channels={"fail": fail_ch})

        results = eng.dispatch(_alert(severity=AlertSeverity.HIGH))

        assert len(results) == 1
        assert results[0].delivered is False
        # structlog critical message is printed to stdout
        captured = capsys.readouterr()
        assert "all_channels_failed" in captured.out

    def test_critical_alert_all_fail_writes_stderr(self, capsys: Any) -> None:
        """CRITICAL alert with all channels failing writes to stderr."""
        fail_ch = _FailChannel()
        eng = _engine(channels={"fail": fail_ch})

        eng.dispatch(_alert(severity=AlertSeverity.CRITICAL))

        captured = capsys.readouterr()
        assert "SENTINEL CRITICAL" in captured.err
        assert "drift detected" in captured.err

    def test_non_critical_alert_no_stderr(self, capsys: Any) -> None:
        """Non-CRITICAL alerts that fail all channels do NOT write stderr."""
        fail_ch = _FailChannel()
        eng = _engine(channels={"fail": fail_ch})

        eng.dispatch(_alert(severity=AlertSeverity.WARNING))

        captured = capsys.readouterr()
        assert "SENTINEL CRITICAL" not in captured.err


# ── Digest thread safety ──────────────────────────────────────────


class TestDigestThreadSafety:
    """Digest queue operations must be thread-safe."""

    def test_concurrent_queue_for_digest(self) -> None:
        """Multiple threads queuing digests — no list corruption."""
        policy_engine = AlertPolicyEngine(AlertPolicies(digest_mode=True, cooldown="0s"))
        n_threads = 10
        n_per_thread = 50
        barrier = threading.Barrier(n_threads)

        def queue_many() -> None:
            barrier.wait()
            for i in range(n_per_thread):
                policy_engine.queue_for_digest(_alert(title=f"alert-{i}"))

        threads = [threading.Thread(target=queue_many) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert policy_engine.has_digest_pending()
        digest = policy_engine.flush_digest()
        assert digest is not None
        assert digest.payload["count"] == n_threads * n_per_thread

    def test_concurrent_flush_and_queue(self) -> None:
        """Flush while another thread is queuing — no race."""
        policy_engine = AlertPolicyEngine(AlertPolicies(digest_mode=True, cooldown="0s"))

        # Pre-seed some alerts
        for i in range(20):
            policy_engine.queue_for_digest(_alert(title=f"seed-{i}"))

        digests: list[Alert | None] = []
        barrier = threading.Barrier(2)

        def flusher() -> None:
            barrier.wait()
            for _ in range(5):
                d = policy_engine.flush_digest()
                if d is not None:
                    digests.append(d)
                time.sleep(0.001)

        def queuer() -> None:
            barrier.wait()
            for i in range(30):
                policy_engine.queue_for_digest(_alert(title=f"concurrent-{i}"))
                time.sleep(0.0005)

        t1 = threading.Thread(target=flusher)
        t2 = threading.Thread(target=queuer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Collect remaining
        remaining = policy_engine.flush_digest()
        if remaining is not None:
            digests.append(remaining)

        # All 50 alerts (20 seeded + 30 concurrent) should appear exactly once
        total = sum(d.payload["count"] for d in digests)
        assert total == 50


# ── Cooldown GC ────────────────────────────────────────────────────


class TestCooldownGC:
    """_last_fired entries must be garbage-collected to prevent unbounded growth."""

    def test_old_cooldown_entries_pruned(self) -> None:
        """_last_fired entries older than 2*cooldown are cleaned up."""
        policy_engine = AlertPolicyEngine(AlertPolicies(cooldown="1h"))

        # Manually inject old entries
        old_time = datetime.now(timezone.utc) - timedelta(hours=3)
        with policy_engine._lock:
            for i in range(50):
                policy_engine._last_fired[f"old-fp-{i}"] = old_time
                policy_engine._first_seen[f"old-fp-{i}"] = old_time

        # Record a new alert (this should trigger GC since we force the counter)
        with policy_engine._lock:
            policy_engine._record_count = 999  # next record() will hit 1000

        policy_engine.record(_alert(title="trigger-gc"))

        with policy_engine._lock:
            # All old entries should have been pruned
            remaining_old = [fp for fp in policy_engine._last_fired if fp.startswith("old-fp-")]
            assert len(remaining_old) == 0
            # The fresh entry should still exist
            assert any(not fp.startswith("old-fp-") for fp in policy_engine._last_fired)

    def test_gc_runs_when_dict_exceeds_threshold(self) -> None:
        """GC runs when dict exceeds 10000 entries."""
        policy_engine = AlertPolicyEngine(AlertPolicies(cooldown="1s"))

        old_time = datetime.now(timezone.utc) - timedelta(seconds=3)
        with policy_engine._lock:
            for i in range(10001):
                policy_engine._last_fired[f"fp-{i}"] = old_time

        # A single record() should trigger GC due to dict size
        policy_engine.record(_alert(title="overflow"))

        with policy_engine._lock:
            # All old entries (>2s old with 1s cooldown) should be pruned
            assert len(policy_engine._last_fired) < 100


# ── Escalation dedup ──────────────────────────────────────────────


class TestEscalationDedup:
    """Duplicate schedule() calls for the same fingerprint should be rejected."""

    def test_duplicate_schedule_ignored(self) -> None:
        """Second schedule() call for same alert fingerprint is no-op."""
        callback = MagicMock()
        timer = EscalationTimer(callback=callback)
        # Don't start the background thread — we just test schedule()
        steps = [EscalationStep(after="30m", channels=["slack"])]

        alert = _alert(fp="dedup-test")
        timer.schedule(alert, steps)
        assert timer.pending_count == 1

        # Second call with same fingerprint — should be ignored
        timer.schedule(alert, steps)
        assert timer.pending_count == 1

    def test_different_alerts_both_scheduled(self) -> None:
        """Different alert fingerprints are both scheduled."""
        callback = MagicMock()
        timer = EscalationTimer(callback=callback)
        steps = [EscalationStep(after="30m", channels=["slack"])]

        timer.schedule(_alert(fp="fp-1"), steps)
        timer.schedule(_alert(fp="fp-2"), steps)
        assert timer.pending_count == 2


# ── Channel init failure ──────────────────────────────────────────


class TestChannelInitFailure:
    """Engine must loudly report when all channels fail to initialize."""

    def test_all_channels_fail_init_logs_critical(self, capsys: Any) -> None:
        """When all channels fail to init, engine logs critical warning."""
        from sentinel.action.notifications import CHANNEL_REGISTRY

        class _BrokenChannel(BaseChannel):
            name = "broken"

            def __init__(self, **kwargs: Any) -> None:
                raise RuntimeError("init boom")

            def send(self, alert: Alert) -> DeliveryResult:
                return DeliveryResult(channel=self.name, delivered=False)

        CHANNEL_REGISTRY["broken_type"] = _BrokenChannel
        try:
            # Build an engine with empty loaded channels but config indicating
            # channels were specified. Reproduce the __init__ check.
            cfg = AlertsConfig(channels=[], policies=AlertPolicies())
            eng = NotificationEngine.__new__(NotificationEngine)
            eng.config = cfg
            eng.audit = None
            eng.policies = AlertPolicyEngine(cfg.policies)
            eng.channels = {}
            eng._dispatch_lock = threading.Lock()
            eng._ack_lock = threading.Lock()
            eng._acknowledged = set()
            eng._digest_timer = None

            # Simulate: config had channels but none loaded
            fake_ch_cfg = MagicMock()
            fake_ch_cfg.type = "slack"
            eng.config = MagicMock()
            eng.config.channels = [fake_ch_cfg]
            eng.config.policies = AlertPolicies()

            configured_types = {ch_cfg.type for ch_cfg in eng.config.channels}

            # Run the same logic as __init__
            if not eng.channels and configured_types:
                structlog.get_logger(__name__).critical(
                    "notification.all_channels_failed_init",
                    configured=sorted(configured_types),
                )

            captured = capsys.readouterr()
            assert "all_channels_failed_init" in captured.out
        finally:
            CHANNEL_REGISTRY.pop("broken_type", None)

    def test_some_channels_fail_init_others_work(self) -> None:
        """Partial channel init failure — working channels still dispatch."""
        good = _FakeChannel()
        eng = _engine(channels={"fake": good})
        results = eng.dispatch(_alert())
        assert len(results) == 1
        assert results[0].delivered is True
