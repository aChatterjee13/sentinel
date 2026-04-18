"""Explainability view — global + cohort feature importance."""

from __future__ import annotations

from typing import Any

from sentinel.dashboard.state import DashboardState


def build(state: DashboardState) -> dict[str, Any]:
    """Build the explainability page payload."""
    client = state.client
    payload: dict[str, Any] = {
        "model_name": client.model_name,
        "available": client._explainability is not None,
        "global_importance": {},
        "cohort_importances": {},
        "summary": "no model set for explanations",
    }

    if client._explainability is None:
        return payload

    # Try global explanation from buffered predictions
    try:
        data = client._buffered_features()
        if data is not None and len(data) > 0:
            global_imp = client.explain_global(data)
            payload["global_importance"] = global_imp
            payload["summary"] = f"global importance for {len(global_imp)} features"
    except Exception as e:
        payload["error"] = str(e)

    return payload


def chart_data(state: DashboardState) -> dict[str, Any]:
    """Return chart-ready global feature importance data."""
    data = build(state)
    imp = data.get("global_importance", {})
    # Convert to sorted list for charting
    sorted_features = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:15]
    return {
        "features": [{"name": n, "importance": v} for n, v in sorted_features],
    }
