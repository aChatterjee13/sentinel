"""Alert policies — cooldown, rate limiting, escalation, digest mode."""

from __future__ import annotations

import hashlib
import re
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from sentinel.config.schema import AlertPolicies
from sentinel.core.types import Alert, AlertSeverity

log = structlog.get_logger(__name__)

_DURATION_RE = re.compile(r"^(\d+)([smhdw])$")


def parse_duration(value: str) -> timedelta:
    """Parse a duration string like ``1h``, ``30m``, ``7d``."""
    if not isinstance(value, str):
        raise ValueError(f"duration must be a string, got {type(value).__name__}")
    match = _DURATION_RE.match(value.strip())
    if not match:
        raise ValueError(f"invalid duration: {value!r}")
    n = int(match.group(1))
    unit = match.group(2)
    return {
        "s": timedelta(seconds=n),
        "m": timedelta(minutes=n),
        "h": timedelta(hours=n),
        "d": timedelta(days=n),
        "w": timedelta(weeks=n),
    }[unit]


def fingerprint(alert: Alert) -> str:
    """Stable identifier used for cooldown deduplication."""
    if alert.fingerprint:
        return alert.fingerprint
    raw = f"{alert.model_name}|{alert.source}|{alert.title}|{alert.severity.value}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class AlertPolicyEngine:
    """Enforces cooldown, rate limiting, and escalation policies.

    Returns the channels that an alert should be dispatched to *right now*,
    based on the configured policies.
    """

    def __init__(self, policies: AlertPolicies):
        self.policies = policies
        self.cooldown = parse_duration(policies.cooldown)
        self._rate_window = parse_duration(policies.rate_limit_window)
        self._lock = threading.Lock()
        self._last_fired: dict[str, datetime] = {}
        self._sent_per_hour: deque[datetime] = deque()
        self._sent_per_channel: dict[str, deque[datetime]] = defaultdict(deque)
        self._pending_digest: list[Alert] = []
        self._first_seen: dict[str, datetime] = {}
        self._escalated: dict[str, int] = defaultdict(int)

    # ── Decision ──────────────────────────────────────────────────

    def should_send(self, alert: Alert) -> bool:
        """True if the alert is not in cooldown and we are below the rate limit."""
        now = datetime.now(timezone.utc)
        fp = fingerprint(alert)
        with self._lock:
            last = self._last_fired.get(fp)
            if last and (now - last) < self.cooldown:
                return False
            self._gc_rate_window(now)
            if len(self._sent_per_hour) >= self.policies.rate_limit_per_hour:
                return False
        return True

    def select_channels(
        self,
        alert: Alert,
        all_channels: list[str],
    ) -> list[str]:
        """Pick the channels to dispatch to based on escalation policy."""
        if not self.policies.escalation:
            return all_channels

        fp = fingerprint(alert)
        first_seen = self._first_seen.setdefault(fp, datetime.now(timezone.utc))
        elapsed = datetime.now(timezone.utc) - first_seen

        chosen: set[str] = set()
        for step in self.policies.escalation:
            after = parse_duration(step.after)
            if elapsed >= after and (not step.severity or alert.severity.value in step.severity):
                chosen.update(step.channels)
        return [c for c in all_channels if c in chosen] or self._fallback_all(alert, all_channels)

    def _fallback_all(self, alert: Alert, all_channels: list[str]) -> list[str]:
        log.warning(
            "notification.no_channels_matched_using_all",
            severity=alert.severity.value,
        )
        return all_channels

    def record(self, alert: Alert) -> None:
        """Mark an alert as sent so cooldown / rate limiting take effect."""
        now = datetime.now(timezone.utc)
        fp = fingerprint(alert)
        with self._lock:
            self._last_fired[fp] = now
            self._sent_per_hour.append(now)
            # Record first-seen for escalation timing
            self._first_seen.setdefault(fp, now)
            # Periodic garbage collection of stale cooldown entries
            self._record_count = getattr(self, "_record_count", 0) + 1
            if self._record_count % 1000 == 0 or len(self._last_fired) > 10000:
                self._gc_cooldown(now)

    def _gc_cooldown(self, now: datetime) -> None:
        """Remove ``_last_fired`` entries older than 2x cooldown.

        Must be called while ``self._lock`` is held.
        """
        cutoff = now - (self.cooldown * 2)
        stale = [fp for fp, ts in self._last_fired.items() if ts < cutoff]
        for fp in stale:
            del self._last_fired[fp]
            self._first_seen.pop(fp, None)
        if stale:
            log.debug("notification.cooldown_gc", pruned=len(stale))

    def _gc_rate_window(self, now: datetime) -> None:
        cutoff = now - self._rate_window
        while self._sent_per_hour and self._sent_per_hour[0] < cutoff:
            self._sent_per_hour.popleft()
        for ch_deque in self._sent_per_channel.values():
            while ch_deque and ch_deque[0] < cutoff:
                ch_deque.popleft()

    def should_send_to_channel(self, channel_name: str) -> bool:
        """True if the per-channel rate limit has not been exceeded."""
        if self.policies.rate_limit_per_hour <= 0:
            return False
        now = datetime.now(timezone.utc)
        with self._lock:
            self._gc_rate_window(now)
            ch_deque = self._sent_per_channel.get(channel_name)
            if ch_deque and len(ch_deque) >= self.policies.rate_limit_per_hour:
                return False
        return True

    def record_channel_send(self, channel_name: str) -> None:
        """Record a send to a specific channel for per-channel rate limiting."""
        now = datetime.now(timezone.utc)
        with self._lock:
            self._sent_per_channel[channel_name].append(now)

    # ── Digest mode ───────────────────────────────────────────────

    def queue_for_digest(self, alert: Alert) -> None:
        """Add an alert to the digest queue (thread-safe)."""
        with self._lock:
            self._pending_digest.append(alert)

    def flush_digest(self) -> Alert | None:
        """Build a single combined Alert representing the queue, then reset.

        Thread-safe: acquires ``_lock`` to snapshot and clear the queue
        atomically so the digest timer and manual calls don't race.

        Returns:
            A combined ``Alert`` summarising all queued alerts, or ``None``
            if the queue was empty.
        """
        from collections import Counter

        with self._lock:
            if not self._pending_digest:
                return None
            alerts = list(self._pending_digest)
            self._pending_digest.clear()

        worst = max(alerts, key=lambda a: list(AlertSeverity).index(a.severity))
        body_lines = [
            f"- [{a.severity.value}] {a.title} (model={a.model_name})" for a in alerts
        ]

        severity_counts = Counter(a.severity.value for a in alerts)
        summary = ", ".join(f"{count} {sev}" for sev, count in severity_counts.items())

        digest = Alert(
            model_name=worst.model_name,
            title=f"Sentinel digest: {len(alerts)} alerts ({summary})",
            body="\n".join(body_lines),
            severity=worst.severity,
            source="digest",
            payload={
                "count": len(alerts),
                "severity_breakdown": dict(severity_counts),
                "fingerprints": [fingerprint(a) for a in alerts],
                "models": list({a.model_name for a in alerts}),
            },
        )
        return digest

    def has_digest_pending(self) -> bool:
        with self._lock:
            return bool(self._pending_digest)

    def remaining_escalation_steps(
        self,
        alert: Alert,
        available_channels: list[str] | None = None,
    ) -> list[Any]:
        """Return escalation steps whose ``after`` has not yet elapsed.

        Skips the first step (already dispatched by the engine). Used by
        :class:`~sentinel.action.notifications.escalation.EscalationTimer`
        to schedule future escalations.

        Args:
            alert: The alert to compute remaining steps for.
            available_channels: Channel names that are actually configured.
                Steps referencing only unconfigured channels are filtered
                out to avoid scheduling escalations that fire into the void.
        """
        if not self.policies.escalation:
            return []

        fp = fingerprint(alert)
        with self._lock:
            first_seen = self._first_seen.get(fp)
        if first_seen is None:
            first_seen = datetime.now(timezone.utc)

        elapsed = datetime.now(timezone.utc) - first_seen

        remaining = []
        for step in self.policies.escalation:
            after = parse_duration(step.after)
            if after > elapsed and (not step.severity or alert.severity.value in step.severity):
                if available_channels is not None:
                    usable = [c for c in step.channels if c in available_channels]
                    if not usable:
                        continue
                remaining.append(step)
        return remaining


def severity_at_least(actual: AlertSeverity, minimum: str) -> bool:
    """Helper for filter rules."""
    levels = [s.value for s in AlertSeverity]
    return levels.index(actual.value) >= levels.index(minimum)


# Re-export for convenience
__all__ = ["AlertPolicyEngine", "fingerprint", "parse_duration", "severity_at_least"]


def _ensure_dict(x: Any) -> dict[str, Any]:
    return dict(x) if isinstance(x, dict) else {}
