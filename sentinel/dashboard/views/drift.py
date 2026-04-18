"""Drift history view — reads the audit trail and the in-process cache."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sentinel.dashboard.state import DashboardState


def build(state: DashboardState, limit: int = 200) -> dict[str, Any]:
    """Build the drift page payload."""
    client = state.client
    try:
        events = list(client.audit.query(event_type="drift_checked", limit=limit))
    except Exception:
        events = []

    rows: list[dict[str, Any]] = []
    for ev in events:
        payload = ev.payload or {}
        rows.append(
            {
                "timestamp": ev.timestamp.isoformat(),
                "model_name": ev.model_name,
                "method": payload.get("method"),
                "is_drifted": bool(payload.get("is_drifted")),
                "severity": payload.get("severity"),
                "n_drifted": payload.get("n_drifted", 0),
            }
        )

    cached_reports = [r.model_dump(mode="json") for r in state.recent_drift_reports(limit=20)]
    return {
        "events": rows,
        "cached_reports": cached_reports,
        "model_name": client.model_name,
        "config": {
            "method": client.config.drift.data.method,
            "threshold": client.config.drift.data.threshold,
            "window": client.config.drift.data.window,
        },
    }


def timeseries(state: DashboardState, window: str = "all") -> dict[str, Any]:
    """Return a chart-ready timeseries of drift test statistics."""
    client = state.client
    timestamps: list[str] = []
    statistics: list[float] = []
    severities: list[str] = []
    try:
        events = list(client.audit.query(event_type="drift_checked", limit=1000))
    except Exception:
        events = []
    for ev in events:
        payload = ev.payload or {}
        timestamps.append(ev.timestamp.isoformat())
        statistics.append(float(payload.get("n_drifted", 0)))
        severities.append(str(payload.get("severity", "info")))

    cached = state.recent_drift_reports(limit=200)
    for report in cached:
        timestamps.append(report.timestamp.isoformat())
        statistics.append(float(report.test_statistic))
        severities.append(report.severity.value)

    return {
        "timestamps": timestamps,
        "statistics": statistics,
        "severities": severities,
        "threshold": client.config.drift.data.threshold,
        "window": window,
    }


def detail(state: DashboardState, report_id: str) -> dict[str, Any] | None:
    """Look up a single drift report from the in-process cache."""
    report = state.find_drift_report(report_id)
    if report is None:
        return None
    payload = report.model_dump(mode="json")
    payload["feature_scores_sorted"] = sorted(
        report.feature_scores.items(), key=lambda kv: kv[1], reverse=True
    )
    return payload


def parse_iso(value: str | None) -> datetime | None:
    """Best-effort parser for filter query params."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
