"""Cohort analysis view."""

from __future__ import annotations

from typing import Any

from sentinel.dashboard.state import DashboardState


def build(state: DashboardState) -> dict[str, Any]:
    """Build the cohort comparison page payload."""
    client = state.client
    payload: dict[str, Any] = {
        "model_name": client.model_name,
        "enabled": client.cohort_analyzer is not None,
        "cohorts": [],
        "disparity_flags": [],
        "global_accuracy": None,
        "summary": "cohort analysis disabled",
    }

    if client.cohort_analyzer is None:
        return payload

    report = client.compare_cohorts()
    if report is None:
        return payload

    payload["summary"] = report.summary
    payload["disparity_flags"] = list(report.disparity_flags)
    payload["global_accuracy"] = report.global_accuracy
    payload["cohorts"] = [
        {
            "cohort_id": m.cohort_id,
            "count": m.count,
            "mean_prediction": m.mean_prediction,
            "mean_actual": m.mean_actual,
            "accuracy": m.accuracy,
            "drift_score": m.drift_score,
        }
        for m in report.cohorts
    ]
    return payload


def detail(state: DashboardState, cohort_id: str) -> dict[str, Any] | None:
    """Build the single cohort detail payload."""
    client = state.client
    if client.cohort_analyzer is None:
        return None

    report = client.get_cohort_report(cohort_id)
    if report is None:
        return None

    m = report.metrics
    return {
        "cohort_id": m.cohort_id,
        "count": m.count,
        "mean_prediction": m.mean_prediction,
        "mean_actual": m.mean_actual,
        "accuracy": m.accuracy,
        "drift_score": m.drift_score,
        "model_name": report.model_name,
    }


def chart_data(state: DashboardState) -> dict[str, Any]:
    """Return chart-ready cohort comparison data."""
    data = build(state)
    cohorts = data.get("cohorts", [])
    return {
        "cohorts": cohorts,
        "disparity_flags": data.get("disparity_flags", []),
    }
