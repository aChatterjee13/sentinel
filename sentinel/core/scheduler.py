"""Background drift-detection scheduler — daemon thread with Event-based stop."""

from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from sentinel.core.client import SentinelClient

log = structlog.get_logger(__name__)

_INTERVAL_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(s|m|h|d|w)$")
_MULTIPLIERS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}


def _parse_interval(value: str) -> float:
    """Convert a human-friendly interval like ``"7d"`` to seconds.

    Supported suffixes: ``s`` (seconds), ``m`` (minutes), ``h`` (hours),
    ``d`` (days), ``w`` (weeks).

    Raises:
        ValueError: If *value* does not match the expected pattern.
    """
    match = _INTERVAL_RE.match(value.strip())
    if not match:
        raise ValueError(f"invalid interval {value!r} — expected <number><s|m|h|d|w>, e.g. '7d'")
    amount = float(match.group(1))
    unit = match.group(2)
    return amount * _MULTIPLIERS[unit]


class DriftScheduler:
    """Runs ``client.check_drift()`` on a background daemon thread.

    The scheduler follows the same pattern as
    :class:`~sentinel.foundation.audit.shipper.ThreadedShipper`: a daemon
    thread, ``threading.Event``-based stop, and structlog throughout.

    Args:
        client: A live :class:`SentinelClient`.
        interval: Human-readable interval (e.g. ``"7d"``).
        run_on_start: If True, execute a drift check immediately upon start.

    Example:
        >>> scheduler = DriftScheduler(client, interval="7d")
        >>> scheduler.start()
        >>> scheduler.stop()
    """

    def __init__(
        self,
        client: SentinelClient,
        interval: str = "7d",
        run_on_start: bool = False,
    ) -> None:
        self._client = client
        self._interval_s = _parse_interval(interval)
        self._run_on_start = run_on_start
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._run_count = 0
        self._lock = threading.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background scheduler (idempotent)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="sentinel-drift-scheduler",
                daemon=True,
            )
            self._thread.start()
            log.info(
                "scheduler.started",
                interval_s=self._interval_s,
                run_on_start=self._run_on_start,
            )

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the scheduler to stop and wait up to *timeout* seconds."""
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            log.info("scheduler.stopped", run_count=self._run_count)

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def run_count(self) -> int:
        return self._run_count

    # ── Internal ──────────────────────────────────────────────────

    def _run_loop(self) -> None:
        if self._run_on_start:
            self._execute_check()
        while not self._stop_event.wait(self._interval_s):
            self._execute_check()

    def _execute_check(self) -> None:
        try:
            report = self._client.check_drift()
            self._run_count += 1
            log.info(
                "scheduler.drift_check_completed",
                run_count=self._run_count,
                is_drifted=report.is_drifted,
            )
        except Exception:
            self._run_count += 1
            log.exception("scheduler.drift_check_failed", run_count=self._run_count)
