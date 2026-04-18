"""Intelligence layer dashboard view — model graph, KPI links, explainability."""

from __future__ import annotations

from typing import Any

from sentinel.dashboard.state import DashboardState


def build(state: DashboardState) -> dict[str, Any]:
    """Build the intelligence page payload."""
    client = state.client

    # Model dependency graph
    model_graph = client.model_graph.to_dict()

    # Cascade impact for the current model
    cascade_impact = client.model_graph.cascade_impact(client.model_name)

    # KPI linkage report — uses cached/latest metrics from the registry
    latest_metrics: dict[str, float] = {}
    try:
        latest_mv = client.registry.get_latest(client.model_name)
        latest_metrics = latest_mv.metrics
    except Exception:
        pass
    kpi_report = client.kpi_linker.report(latest_metrics)

    # Explainability status
    explainability_info: dict[str, Any] = {
        "configured": client.explainability_engine is not None,
        "method": "shap" if client.explainability_engine is not None else "not configured",
    }

    return {
        "model_graph": model_graph,
        "cascade_impact": cascade_impact,
        "kpi_report": kpi_report,
        "explainability": explainability_info,
    }
