"""Compliance frameworks summary view."""

from __future__ import annotations

from typing import Any

from sentinel.dashboard.state import DashboardState


def build(state: DashboardState) -> dict[str, Any]:
    """Build the compliance summary payload."""
    client = state.client
    frameworks = list(client.config.audit.compliance_frameworks)

    try:
        all_events = list(client.audit.query(limit=5000))
    except Exception:
        all_events = []

    event_count_by_type: dict[str, int] = {}
    for ev in all_events:
        event_count_by_type[ev.event_type] = event_count_by_type.get(ev.event_type, 0) + 1

    event_rows = [
        {
            "timestamp": ev.timestamp.isoformat(),
            "event_type": ev.event_type,
            "model_name": ev.model_name,
            "actor": ev.actor,
        }
        for ev in all_events[:200]
    ]

    return {
        "frameworks": frameworks,
        "retention_days": client.config.audit.retention_days,
        "log_predictions": client.config.audit.log_predictions,
        "log_explanations": client.config.audit.log_explanations,
        "total_events": len(all_events),
        "event_counts": event_count_by_type,
        "events": event_rows,
    }


def chart_data(state: DashboardState) -> dict[str, Any]:
    """Return chart-ready event type breakdown for the donut chart."""
    data = build(state)
    counts = data.get("event_counts", {})
    labels = list(counts.keys())
    values = list(counts.values())
    return {"labels": labels, "values": values}
