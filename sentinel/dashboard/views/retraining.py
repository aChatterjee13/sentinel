"""Retraining dashboard view — drift status, pending approvals, trigger button."""

from __future__ import annotations

from typing import Any

from sentinel.dashboard.state import DashboardState


def build(state: DashboardState) -> dict[str, Any]:
    """Build the retraining page payload."""
    client = state.client

    # Latest drift reports
    drift_cache = state.recent_drift_reports(limit=5)
    latest_drift = drift_cache[-1].model_dump(mode="json") if drift_cache else None

    # Retrain config summary
    retrain_cfg = client.config.retraining
    retrain_config_info: dict[str, Any] = {
        "trigger_mode": retrain_cfg.trigger,
        "pipeline": retrain_cfg.pipeline,
        "approval_mode": retrain_cfg.approval.mode,
        "has_pipeline_runner": client.retrain._pipeline_runner is not None,
    }

    # Pending approvals
    pending_approvals = [
        req.model_dump(mode="json") for req in client.retrain.approval.list_pending()
    ]

    # Recent retrain events from audit trail
    retrain_events: list[dict[str, Any]] = []
    try:
        all_events = list(client.audit.query(limit=100))
        retrain_events = [
            e.model_dump(mode="json")
            for e in all_events
            if e.event_type.startswith("retrain_") or e.event_type.startswith("approval_")
        ][:20]
    except Exception:
        pass

    return {
        "latest_drift": latest_drift,
        "retrain_config": retrain_config_info,
        "pending_approvals": pending_approvals,
        "retrain_events": retrain_events,
    }
