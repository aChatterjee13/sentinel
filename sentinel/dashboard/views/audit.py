"""Audit trail view — query the JSONL audit log with filters."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sentinel.dashboard.state import DashboardState


def build(
    state: DashboardState,
    *,
    event_type: str | None = None,
    model_name: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Build the audit query page payload."""
    client = state.client
    try:
        events = list(
            client.audit.query(
                event_type=event_type or None,
                model_name=model_name or None,
                since=since,
                until=until,
                limit=limit,
            )
        )
    except Exception:
        events = []

    rows = [
        {
            "event_id": ev.event_id,
            "timestamp": ev.timestamp.isoformat(),
            "event_type": ev.event_type,
            "model_name": ev.model_name,
            "model_version": ev.model_version,
            "actor": ev.actor,
            "payload": ev.payload,
        }
        for ev in events
    ]

    distinct_event_types = sorted({r["event_type"] for r in rows if r["event_type"]})
    return {
        "events": rows,
        "filters": {
            "event_type": event_type,
            "model_name": model_name,
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
            "limit": limit,
        },
        "distinct_event_types": distinct_event_types,
        "total": len(rows),
    }
