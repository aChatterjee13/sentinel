"""Overview homepage — top-level health snapshot."""

from __future__ import annotations

from typing import Any

from sentinel.dashboard.state import DashboardState


def build(state: DashboardState) -> dict[str, Any]:
    """Aggregate the homepage payload from the live SentinelClient."""
    client = state.client
    status = client.status()

    try:
        recent_audit = [e.model_dump(mode="json") for e in client.audit.latest(20)]
    except Exception:
        recent_audit = []

    try:
        active_deployments = [
            d.model_dump(mode="json") for d in client.deployment_manager.list_active()
        ]
    except Exception:
        active_deployments = []

    drift_cache = state.recent_drift_reports(limit=5)
    latest_drift = drift_cache[-1].model_dump(mode="json") if drift_cache else None

    # KPI linkage (WS-E) — include in overview for quick glance
    latest_metrics: dict[str, float] = {}
    try:
        latest_mv = client.registry.get_latest(client.model_name)
        latest_metrics = latest_mv.metrics
    except Exception:
        pass
    kpi_report = client.kpi_linker.report(latest_metrics)

    return {
        "status": status,
        "recent_audit": recent_audit,
        "active_deployments": active_deployments,
        "latest_drift": latest_drift,
        "compliance_frameworks": list(client.config.audit.compliance_frameworks),
        "llmops_enabled": client.config.llmops.enabled,
        "agentops_enabled": client.config.agentops.enabled,
        "domain": client.config.model.domain,
        "kpi_report": kpi_report,
    }
