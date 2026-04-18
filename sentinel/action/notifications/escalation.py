"""Escalation timer — background daemon that fires escalation callbacks on schedule.

Follows the same daemon-thread + Event pattern as
:class:`~sentinel.core.scheduler.DriftScheduler`. Uses a heap-based
priority queue of ``(fire_at, alert, step)`` tuples. A single worker
thread sleeps until the next fire time.
"""

from __future__ import annotations

import heapq
import threading
import time
from collections.abc import Callable

import structlog

from sentinel.config.schema import EscalationStep
from sentinel.core.types import Alert

log = structlog.get_logger(__name__)


class EscalationTimer:
    """Background daemon that fires escalation callbacks on schedule.

    Args:
        callback: Invoked as ``callback(alert, step)`` when an
            escalation step's ``after`` duration elapses.

    Example:
        >>> timer = EscalationTimer(callback=engine._on_escalation)
        >>> timer.start()
        >>> timer.schedule(alert, steps)
        >>> timer.stop()
    """

    def __init__(self, callback: Callable[[Alert, EscalationStep], None]) -> None:
        self._callback = callback
        # Heap of (fire_at_monotonic, counter, alert, step)
        self._queue: list[tuple[float, int, Alert, EscalationStep]] = []
        self._counter = 0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background worker (idempotent)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="sentinel-escalation-timer",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Signal stop and wait for the worker to drain."""
        self._stop_event.set()
        self._wake.set()  # unblock the sleeper
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def schedule(self, alert: Alert, steps: list[EscalationStep]) -> None:
        """Enqueue future escalation steps.

        Each step fires at ``now + step.after``. Step[0] is typically
        already dispatched by the engine, so callers should pass only
        the *remaining* steps.

        Duplicate scheduling for the same alert fingerprint is rejected
        to prevent race conditions from enqueuing double escalations.
        """
        from sentinel.action.notifications.policies import fingerprint as fp_fn
        from sentinel.action.notifications.policies import parse_duration

        self._cleanup_stale()
        fp = fp_fn(alert)
        now = time.monotonic()
        with self._lock:
            # Deduplicate: skip if this fingerprint already has pending entries
            existing_fps = {fp_fn(entry[2]) for entry in self._queue}
            if fp in existing_fps:
                log.debug("escalation.already_scheduled", fingerprint=fp)
                return
            for step in steps:
                delay_s = parse_duration(step.after).total_seconds()
                fire_at = now + delay_s
                self._counter += 1
                heapq.heappush(self._queue, (fire_at, self._counter, alert, step))
        self._wake.set()  # wake the worker to recalculate sleep

    def cancel(self, fingerprint: str) -> None:
        """Cancel pending escalations for an acknowledged alert."""
        from sentinel.action.notifications.policies import fingerprint as fp_fn

        with self._lock:
            self._queue = [entry for entry in self._queue if fp_fn(entry[2]) != fingerprint]
            heapq.heapify(self._queue)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                if not self._queue:
                    next_fire: float | None = None
                else:
                    next_fire = self._queue[0][0]

            if next_fire is None:
                # Nothing queued — wait indefinitely until woken.
                self._wake.wait()
                self._wake.clear()
                continue

            delay = next_fire - time.monotonic()
            if delay > 0:
                # Sleep until the next fire time or until woken.
                self._wake.wait(timeout=delay)
                self._wake.clear()
                if self._stop_event.is_set():
                    return
                continue

            # Pop and fire
            with self._lock:
                if not self._queue:
                    continue
                _, _, alert, step = heapq.heappop(self._queue)

            try:
                self._callback(alert, step)
            except Exception:
                log.exception(
                    "escalation.callback_failed",
                    alert_title=alert.title,
                )

    @property
    def pending_count(self) -> int:
        """Number of pending escalation entries."""
        with self._lock:
            return len(self._queue)

    def _cleanup_stale(self) -> None:
        """Remove entries older than 24 hours to prevent memory leaks."""
        cutoff = time.monotonic() - 86400  # 24 hours
        with self._lock:
            before = len(self._queue)
            self._queue = [entry for entry in self._queue if entry[0] > cutoff]
            if len(self._queue) != before:
                heapq.heapify(self._queue)
                log.info("escalation.stale_cleanup", removed=before - len(self._queue))
