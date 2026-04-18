"""Feature health view."""

from __future__ import annotations

from typing import Any

from sentinel.dashboard.state import DashboardState


def build(state: DashboardState) -> dict[str, Any]:
    """Build the feature health page payload.

    Returns an empty payload (rather than raising) when there are not yet
    any predictions buffered — the page renders an empty state instead of
    a 500.
    """
    client = state.client
    payload: dict[str, Any] = {
        "model_name": client.model_name,
        "buffer_size": client.buffer_size(),
        "features": [],
        "top_n_drifted": [],
        "summary": "no buffered predictions",
        "available": False,
    }

    if client.buffer_size() == 0:
        return payload

    try:
        report = client.get_feature_health()
    except Exception as e:
        payload["error"] = str(e)
        return payload

    payload["available"] = True
    payload["summary"] = report.summary
    payload["top_n_drifted"] = list(report.top_n_drifted)
    payload["features"] = [
        {
            "name": f.name,
            "importance": f.importance,
            "drift_score": f.drift_score,
            "null_rate": f.null_rate,
            "is_drifted": f.is_drifted,
            "severity": f.severity.value,
        }
        for f in report.features
    ]
    return payload


def chart_data(state: DashboardState) -> dict[str, Any]:
    """Return chart-ready feature importance + drift data."""
    data = build(state)
    features = data.get("features", [])
    # Sort by importance descending for the chart
    sorted_feats = sorted(features, key=lambda f: f.get("importance", 0), reverse=True)
    return {"features": sorted_feats[:15]}
