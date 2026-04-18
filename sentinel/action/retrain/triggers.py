"""Trigger evaluators — decide when to launch a retrain."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

try:
    from croniter import croniter  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    croniter = None  # type: ignore[assignment]

from sentinel.core.types import DriftReport


@dataclass
class RetrainTrigger:
    """A reason a retrain was launched — recorded in the audit trail."""

    trigger_type: str
    reason: str
    timestamp: datetime
    payload: dict[str, Any]


class TriggerEvaluator:
    """Evaluates trigger conditions: drift_confirmed | scheduled | manual."""

    def __init__(
        self,
        trigger_mode: str = "drift_confirmed",
        schedule: str | None = None,
        min_consecutive_drift: int = 2,
    ):
        self.trigger_mode = trigger_mode
        self.schedule = schedule
        self.min_consecutive_drift = min_consecutive_drift
        self._consecutive_drift = 0
        self._last_run: datetime | None = None

    # ── Drift trigger ─────────────────────────────────────────────

    def on_drift(self, report: DriftReport) -> RetrainTrigger | None:
        if self.trigger_mode != "drift_confirmed":
            return None
        if not report.is_drifted:
            self._consecutive_drift = 0
            return None
        self._consecutive_drift += 1
        if self._consecutive_drift >= self.min_consecutive_drift:
            self._consecutive_drift = 0
            return RetrainTrigger(
                trigger_type="drift_confirmed",
                reason=f"{report.method} drift confirmed for {len(report.drifted_features)} features",
                timestamp=datetime.now(timezone.utc),
                payload={"report_id": report.report_id, "severity": report.severity.value},
            )
        return None

    # ── Schedule trigger ──────────────────────────────────────────

    def on_tick(self, now: datetime | None = None) -> RetrainTrigger | None:
        if self.trigger_mode != "scheduled" or self.schedule is None or croniter is None:
            return None
        now = now or datetime.now(timezone.utc)
        last = self._last_run or now
        cron = croniter(self.schedule, last)
        next_run = cron.get_next(datetime)
        if now >= next_run:
            self._last_run = now
            return RetrainTrigger(
                trigger_type="scheduled",
                reason=f"cron {self.schedule}",
                timestamp=now,
                payload={"schedule": self.schedule},
            )
        return None

    # ── Manual ────────────────────────────────────────────────────

    def manual(self, reason: str = "manual trigger", **payload: Any) -> RetrainTrigger:
        return RetrainTrigger(
            trigger_type="manual",
            reason=reason,
            timestamp=datetime.now(timezone.utc),
            payload=payload,
        )
