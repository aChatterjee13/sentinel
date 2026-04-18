"""Notification engine — alert routing, cooldown, escalation, digest, audit."""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING

import structlog

from sentinel.action.notifications.channels.base import BaseChannel
from sentinel.action.notifications.escalation import EscalationTimer
from sentinel.action.notifications.policies import (
    AlertPolicyEngine,
    fingerprint,
    parse_duration,
)
from sentinel.config.schema import AlertsConfig, EscalationStep
from sentinel.core.types import Alert, AlertSeverity, DeliveryResult

if TYPE_CHECKING:
    from sentinel.foundation.audit.trail import AuditTrail

log = structlog.get_logger(__name__)


class DigestTimer:
    """Daemon thread that auto-flushes the digest queue on a fixed interval.

    Uses :class:`threading.Event` for clean shutdown — the same pattern
    as :class:`~sentinel.action.notifications.escalation.EscalationTimer`.

    Args:
        interval_seconds: How often to flush, in seconds.
        flush_callback: Called with no args each time the timer fires.

    Example:
        >>> timer = DigestTimer(interval_seconds=3600, flush_callback=engine.flush_digest)
        >>> timer.start()
        >>> timer.stop()
    """

    def __init__(self, interval_seconds: float, flush_callback: object) -> None:
        self._interval = interval_seconds
        self._callback = flush_callback
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background worker (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="sentinel-digest-timer",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal stop and wait for the worker to finish."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._interval)
            if self._stop_event.is_set():
                return
            try:
                self._callback()  # type: ignore[operator]
            except Exception:
                log.exception("digest_timer.flush_failed")

    @property
    def alive(self) -> bool:
        """True if the worker thread is running."""
        return self._thread is not None and self._thread.is_alive()


class NotificationEngine:
    """Coordinates channels, policies, and the audit trail.

    Example:
        >>> engine = NotificationEngine(config.alerts, audit_trail)
        >>> engine.dispatch(alert)
    """

    def __init__(
        self,
        config: AlertsConfig,
        audit_trail: AuditTrail | None = None,
    ):
        self.config = config
        self.audit = audit_trail
        self.policies = AlertPolicyEngine(config.policies)
        self.channels: dict[str, BaseChannel] = {}
        self._dispatch_lock = threading.Lock()
        self._build_channels()
        configured_types = {ch_cfg.type for ch_cfg in self.config.channels}
        loaded_types = set(self.channels.keys())
        missing = configured_types - loaded_types
        if missing:
            log.error(
                "notification.channels_unavailable",
                missing=sorted(missing),
                available=sorted(loaded_types),
                message="Some notification channels failed to initialize. Alerts to these channels will be lost.",
            )
        if not self.channels and configured_types:
            log.critical(
                "notification.all_channels_failed_init",
                configured=sorted(configured_types),
                message="ALL configured notification channels failed to initialize. "
                "The engine will not be able to deliver any alerts.",
            )
        self._escalation_timer = EscalationTimer(callback=self._on_escalation)
        self._escalation_timer.start()

        self._acknowledged: set[str] = set()
        self._ack_lock = threading.Lock()

        # Digest timer — only started when digest_mode is active.
        self._digest_timer: DigestTimer | None = None
        if config.policies.digest_mode:
            interval_s = parse_duration(config.policies.digest_interval).total_seconds()
            self._digest_timer = DigestTimer(
                interval_seconds=interval_s,
                flush_callback=self.flush_digest,
            )
            self._digest_timer.start()

    def _build_channels(self) -> None:
        from sentinel.action.notifications import CHANNEL_REGISTRY

        default_tpl = self.config.policies.default_template
        for ch_cfg in self.config.channels:
            cls = CHANNEL_REGISTRY.get(ch_cfg.type)
            if cls is None:
                log.warning("notification.unknown_channel", type=ch_cfg.type)
                continue
            cfg_dict = ch_cfg.model_dump(exclude={"type"})
            # Apply default_template when the channel has no explicit template
            if cfg_dict.get("template") is None and default_tpl is not None:
                cfg_dict["template"] = default_tpl
            try:
                channel = cls(**cfg_dict)
                self.channels[ch_cfg.type] = channel
            except Exception as e:
                log.error("notification.channel_init_failed", type=ch_cfg.type, error=str(e))

    # ── Public dispatch ───────────────────────────────────────────

    def dispatch(self, alert: Alert) -> list[DeliveryResult]:
        """Send an alert through all eligible channels.

        In digest mode the alert is queued for a later combined dispatch.
        Otherwise it is sent immediately via :meth:`_dispatch_direct`.
        """
        with self._dispatch_lock:
            if self.config.policies.digest_mode:
                self.policies.queue_for_digest(alert)
                log.info("alert.queued_for_digest", title=alert.title, severity=alert.severity.value)
                return []
            return self._dispatch_direct_unlocked(alert)

    def _dispatch_direct(self, alert: Alert) -> list[DeliveryResult]:
        """Send an alert bypassing digest-mode queueing.

        Acquires ``_dispatch_lock`` for thread safety. Used by
        :meth:`flush_digest` so the combined digest alert goes to
        channels instead of being re-queued.
        """
        with self._dispatch_lock:
            return self._dispatch_direct_unlocked(alert)

    def _dispatch_direct_unlocked(self, alert: Alert) -> list[DeliveryResult]:
        """Actual send logic, must be called under ``_dispatch_lock``."""
        fp = fingerprint(alert)

        with self._ack_lock:
            if fp in self._acknowledged:
                log.debug("alert.suppressed_acknowledged", fp=fp)
                self._log_audit(alert, suppressed=True, results=[])
                return []

        if not self.policies.should_send(alert):
            log.debug("alert.suppressed", title=alert.title, fp=fingerprint(alert))
            self._log_audit(alert, suppressed=True, results=[])
            return []

        target_channels = self.policies.select_channels(alert, list(self.channels))
        results: list[DeliveryResult] = []
        for ch_name in target_channels:
            channel = self.channels.get(ch_name)
            if channel is None:
                continue
            if not self.policies.should_send_to_channel(ch_name):
                log.debug(
                    "alert.channel_rate_limited",
                    channel=ch_name,
                    title=alert.title,
                )
                continue
            try:
                result = channel.send_with_retry(alert)
            except Exception:
                log.exception(
                    "alert.channel_send_error",
                    channel=ch_name,
                    model=alert.model_name,
                )
                continue
            results.append(result)
            if result.delivered:
                self.policies.record_channel_send(ch_name)
            log.info(
                "alert.dispatched",
                channel=ch_name,
                delivered=result.delivered,
                model=alert.model_name,
                severity=alert.severity.value,
            )

        # Check if all channels failed to deliver
        if target_channels and not any(r.delivered for r in results):
            log.critical(
                "alert.all_channels_failed",
                title=alert.title,
                severity=alert.severity.value,
                model=alert.model_name,
                channels=target_channels,
                message="All notification channels failed to deliver this alert.",
            )
            if alert.severity == AlertSeverity.CRITICAL:
                print(
                    f"[SENTINEL CRITICAL] All channels failed for critical alert: "
                    f"{alert.title} (model={alert.model_name})",
                    file=sys.stderr,
                    flush=True,
                )

        self.policies.record(alert)

        # Schedule future escalation steps (filter out unconfigured channels)
        remaining = self.policies.remaining_escalation_steps(
            alert, available_channels=list(self.channels),
        )
        if remaining:
            self._escalation_timer.schedule(alert, remaining)

        self._log_audit(alert, suppressed=False, results=results)
        return results

    def flush_digest(self) -> list[DeliveryResult] | None:
        """Send any queued digest alerts.

        Dispatches via :meth:`_dispatch_direct` so the combined digest
        alert is sent to channels rather than being re-queued into the
        digest queue.
        """
        digest = self.policies.flush_digest()
        if digest is None:
            return None
        return self._dispatch_direct(digest)

    def _on_escalation(self, alert: Alert, step: EscalationStep) -> None:
        """Called by the escalation timer when a step fires."""
        fp = fingerprint(alert)
        with self._ack_lock:
            if fp in self._acknowledged:
                log.info("alert.escalation_skipped_acknowledged", fingerprint=fp)
                return

        for ch_name in step.channels:
            channel = self.channels.get(ch_name)
            if channel is None:
                continue
            try:
                result = channel.send_with_retry(alert)
                log.info("alert.escalated", channel=ch_name, delivered=result.delivered, severity=alert.severity.value)
            except Exception as e:
                log.error("alert.escalation_failed", channel=ch_name, error=str(e))
        self._log_audit(alert, suppressed=False, results=[])

    # ── Acknowledgement ───────────────────────────────────────────

    def acknowledge(self, alert_or_fingerprint: Alert | str) -> None:
        """Mark an alert as resolved, cancelling pending escalations.

        Args:
            alert_or_fingerprint: Either an :class:`Alert` object or a
                fingerprint string previously returned by
                :func:`~sentinel.action.notifications.policies.fingerprint`.

        The fingerprint is added to an internal acknowledged set so that
        future dispatches of the same fingerprint are suppressed until the
        cooldown window naturally expires.
        """
        if isinstance(alert_or_fingerprint, Alert):
            fp = fingerprint(alert_or_fingerprint)
            model_name = alert_or_fingerprint.model_name
        else:
            fp = alert_or_fingerprint
            model_name = None

        self._escalation_timer.cancel(fp)

        with self._ack_lock:
            self._acknowledged.add(fp)

        log.info("alert.acknowledged", fingerprint=fp)
        if self.audit is not None:
            self.audit.log(
                event_type="alert_acknowledged",
                model_name=model_name,
                fingerprint=fp,
            )

    def close(self) -> None:
        """Stop timers, flush remaining digest, and release resources."""
        if self._digest_timer is not None:
            self._digest_timer.stop()
        try:
            self.flush_digest()
        except Exception:
            log.debug("notification.digest_flush_on_close_failed")
        self._escalation_timer.stop()

    def _log_audit(
        self,
        alert: Alert,
        suppressed: bool,
        results: list[DeliveryResult],
    ) -> None:
        if self.audit is None:
            return
        self.audit.log(
            event_type="alert_dispatched" if not suppressed else "alert_suppressed",
            model_name=alert.model_name,
            alert=alert.model_dump(mode="json"),
            results=[r.model_dump(mode="json") for r in results],
        )
