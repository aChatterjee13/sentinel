"""Deployments dashboard view."""

from __future__ import annotations

from typing import Any

import structlog

from sentinel.dashboard.state import DashboardState

log = structlog.get_logger(__name__)


def build(state: DashboardState) -> dict[str, Any]:
    """Build the deployments page payload."""
    client = state.client
    try:
        active = [d.model_dump(mode="json") for d in client.deployment_manager.list_active()]
    except Exception as e:
        log.warning("dashboard.deployments.list_active_failed", error=str(e))
        active = []

    history: list[dict[str, Any]] = []
    try:
        events = list(client.audit.query(event_type=None, limit=200))
        for ev in events:
            if not ev.event_type.startswith("deployment"):
                continue
            history.append(
                {
                    "timestamp": ev.timestamp.isoformat(),
                    "event_type": ev.event_type,
                    "model_name": ev.model_name,
                    "model_version": ev.model_version,
                    "payload": ev.payload,
                }
            )
    except Exception as e:
        log.warning("dashboard.deployments.query_failed", error=str(e))
        history = []

    return {
        "active": active,
        "history": history,
        "default_strategy": client.config.deployment.strategy,
    }
