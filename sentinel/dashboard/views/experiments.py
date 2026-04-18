"""Experiment tracking views — list experiments, runs, and run detail."""

from __future__ import annotations

from typing import Any

import structlog

from sentinel.dashboard.state import DashboardState

log = structlog.get_logger(__name__)


def build_list(state: DashboardState) -> dict[str, Any]:
    """Build the experiment list page payload.

    Returns:
        Dict with ``experiments`` list (each with run count and latest
        metrics summary) and ``total`` count.
    """
    client = state.client
    try:
        tracker = client.experiments
        experiments = tracker.list_experiments()
    except Exception:
        experiments = []

    items: list[dict[str, Any]] = []
    for exp in experiments:
        try:
            runs = tracker.list_runs(experiment_name=exp.name)
        except Exception:
            runs = []

        latest_metrics: dict[str, float] = {}
        if runs:
            last = max(runs, key=lambda r: r.started_at)
            latest_metrics = dict(last.metrics)

        items.append(
            {
                "name": exp.name,
                "description": exp.description,
                "tags": exp.tags,
                "created_at": exp.created_at,
                "run_count": len(runs),
                "latest_metrics": latest_metrics,
            }
        )

    return {"experiments": items, "total": len(items)}


def build_detail(state: DashboardState, experiment_name: str) -> dict[str, Any] | None:
    """Build the experiment detail page payload (runs table).

    Args:
        state: Dashboard state.
        experiment_name: Experiment to display.

    Returns:
        Dict with ``experiment`` metadata and ``runs`` list, or ``None``
        if the experiment does not exist.
    """
    client = state.client
    try:
        tracker = client.experiments
        exp = tracker.get_experiment(experiment_name)
    except (KeyError, Exception):
        return None

    try:
        runs = tracker.list_runs(experiment_name=experiment_name)
    except Exception:
        runs = []

    run_rows = [
        {
            "run_id": r.run_id,
            "name": r.name,
            "status": r.status,
            "parameters": r.parameters,
            "metrics": r.metrics,
            "tags": r.tags,
            "started_at": r.started_at,
            "ended_at": r.ended_at,
            "parent_run_id": r.parent_run_id,
            "promoted_to": r.promoted_to,
        }
        for r in runs
    ]

    return {
        "experiment": {
            "name": exp.name,
            "description": exp.description,
            "tags": exp.tags,
            "created_at": exp.created_at,
        },
        "runs": run_rows,
        "total_runs": len(run_rows),
    }


def build_run_detail(state: DashboardState, run_id: str) -> dict[str, Any] | None:
    """Build the single-run detail page payload.

    Args:
        state: Dashboard state.
        run_id: Run identifier.

    Returns:
        Dict with full run data plus metric history, or ``None`` if the
        run does not exist.
    """
    client = state.client
    try:
        tracker = client.experiments
        run = tracker.get_run(run_id)
    except (KeyError, Exception):
        return None

    history: dict[str, list[dict[str, Any]]] = {}
    for key in run.metric_history:
        history[key] = [
            {"value": e.value, "step": e.step, "timestamp": e.timestamp}
            for e in run.metric_history[key]
        ]

    return {
        "run": {
            "run_id": run.run_id,
            "experiment_name": run.experiment_name,
            "name": run.name,
            "status": run.status,
            "parent_run_id": run.parent_run_id,
            "parameters": run.parameters,
            "metrics": run.metrics,
            "artifacts": run.artifacts,
            "tags": run.tags,
            "started_at": run.started_at,
            "ended_at": run.ended_at,
            "promoted_to": run.promoted_to,
            "dataset_refs": run.dataset_refs,
            "metadata": run.metadata,
        },
        "metric_history": history,
    }


def metrics_json(state: DashboardState, experiment_name: str) -> dict[str, Any]:
    """Return metric history for all runs in an experiment (JSON for charts).

    Args:
        state: Dashboard state.
        experiment_name: Experiment name.

    Returns:
        Dict mapping run_id → metric_name → list of ``{value, step, timestamp}``.
    """
    client = state.client
    try:
        tracker = client.experiments
        runs = tracker.list_runs(experiment_name=experiment_name)
    except Exception:
        runs = []

    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for run in runs:
        run_data: dict[str, list[dict[str, Any]]] = {}
        for key, entries in run.metric_history.items():
            run_data[key] = [
                {"value": e.value, "step": e.step, "timestamp": e.timestamp} for e in entries
            ]
        result[run.run_id] = run_data

    return {"experiment": experiment_name, "runs": result}
