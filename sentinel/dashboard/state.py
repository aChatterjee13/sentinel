"""DashboardState — thin wrapper around a SentinelClient.

The state object holds the live :class:`~sentinel.SentinelClient` plus any
view-time caches the dashboard needs (e.g. the most recent
:class:`DriftReport` that was produced through the UI). All view functions
take a ``DashboardState`` so they remain trivially testable without spinning
up a FastAPI server.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sentinel.config.schema import DashboardConfig
from sentinel.core.types import DriftReport

if TYPE_CHECKING:
    from sentinel.core.client import SentinelClient


class DashboardState:
    """Holds the SentinelClient and per-process dashboard state.

    Attributes:
        client: The live SentinelClient the dashboard is bound to.
        config: The dashboard configuration block.
        started_at: When the dashboard process booted (UTC).
    """

    def __init__(
        self,
        client: SentinelClient,
        config: DashboardConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or client.config.dashboard
        self.started_at = datetime.now(timezone.utc)
        self._lock = threading.RLock()
        self._recent_drift: deque[DriftReport] = deque(maxlen=200)

    # ── Drift cache ───────────────────────────────────────────────

    def record_drift_report(self, report: DriftReport) -> None:
        """Cache a drift report so the UI can show its details on demand."""
        with self._lock:
            self._recent_drift.append(report)

    def recent_drift_reports(self, limit: int = 50) -> list[DriftReport]:
        with self._lock:
            return list(self._recent_drift)[-limit:]

    def find_drift_report(self, report_id: str) -> DriftReport | None:
        with self._lock:
            for report in reversed(self._recent_drift):
                if report.report_id == report_id:
                    return report
        return None
