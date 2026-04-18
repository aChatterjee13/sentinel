"""LLMOps dashboard views — prompts, guardrails, token economics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sentinel.dashboard.state import DashboardState


def _llmops(state: DashboardState) -> Any | None:
    client = state.client
    if not client.config.llmops.enabled:
        return None
    try:
        return client.llmops
    except Exception:
        return None


def prompts(state: DashboardState) -> dict[str, Any]:
    """List registered prompts and their versions."""
    llmops = _llmops(state)
    if llmops is None:
        return {"enabled": False, "prompts": []}
    rows: list[dict[str, Any]] = []
    try:
        names = llmops.prompts.list_prompts()
    except Exception:
        names = []
    for name in names:
        try:
            versions = llmops.prompts.list_versions(name)
        except Exception:
            versions = []
        version_rows: list[dict[str, Any]] = []
        for v in versions:
            try:
                pv = llmops.prompts.get(name, v)
                version_rows.append(pv.to_dict())
            except Exception:
                continue
        rows.append({"name": name, "versions": version_rows})
    return {"enabled": True, "prompts": rows}


def guardrails(state: DashboardState) -> dict[str, Any]:
    """Aggregate recent guardrail violations from the audit trail."""
    client = state.client
    if not client.config.llmops.enabled:
        return {"enabled": False, "violations": [], "config": None}
    rows: list[dict[str, Any]] = []
    try:
        events = list(client.audit.query(event_type="llmops.guardrail_violation", limit=200))
        for ev in events:
            payload = ev.payload or {}
            rows.append(
                {
                    "timestamp": ev.timestamp.isoformat(),
                    "model_name": ev.model_name,
                    "rule": payload.get("rule"),
                    "severity": payload.get("severity"),
                    "action": payload.get("action"),
                    "payload": payload,
                }
            )
    except Exception:
        rows = []

    cfg = client.config.llmops.guardrails
    return {
        "enabled": True,
        "violations": rows,
        "config": {
            "input": [r.model_dump(mode="json") for r in cfg.input],
            "output": [r.model_dump(mode="json") for r in cfg.output],
        },
    }


def tokens(state: DashboardState) -> dict[str, Any]:
    """Build the token economics page payload."""
    llmops = _llmops(state)
    if llmops is None:
        return {"enabled": False, "totals": {}, "daily": [], "trend": 0.0}
    try:
        totals = llmops.token_tracker.totals()
    except Exception:
        totals = {}
    try:
        trend = llmops.token_tracker.trend()
    except Exception:
        trend = 0.0
    daily = _daily_series(llmops.token_tracker, days=14)
    budgets = state.client.config.llmops.token_economics.budgets
    return {
        "enabled": True,
        "totals": totals,
        "trend": trend,
        "daily": daily,
        "budgets": budgets,
    }


def guardrails_trend(state: DashboardState, days: int = 14) -> dict[str, Any]:
    """Return chart-ready daily guardrail violation counts."""
    client = state.client
    if not client.config.llmops.enabled:
        return {"days": []}
    try:
        events = list(client.audit.query(event_type="llmops.guardrail_violation", limit=5000))
    except Exception:
        events = []

    today = datetime.now(timezone.utc).date()
    counts: dict[str, int] = {}
    for i in range(days - 1, -1, -1):
        day = today - timedelta(days=i)
        counts[day.strftime("%Y-%m-%d")] = 0
    for ev in events:
        key = ev.timestamp.strftime("%Y-%m-%d")
        if key in counts:
            counts[key] += 1
    return {"days": [{"day": k, "count": v} for k, v in counts.items()]}


def tokens_by_model(state: DashboardState) -> dict[str, Any]:
    """Return chart-ready cost breakdown by model for the donut chart."""
    llmops = _llmops(state)
    if llmops is None:
        return {"models": []}
    try:
        by_model = llmops.token_tracker.totals_by_dimension("model")
    except Exception:
        by_model = {}

    rows = [
        {"model": model, "cost_usd": stats.get("cost_usd", 0.0)}
        for model, stats in by_model.items()
        if stats.get("cost_usd", 0.0) > 0
    ]
    rows.sort(key=lambda r: r["cost_usd"], reverse=True)
    return {"models": rows[:10]}


def _daily_series(tracker: Any, days: int = 14) -> list[dict[str, Any]]:
    """Build a chart-ready daily cost series for the last N days."""
    today = datetime.now(timezone.utc).date()
    out: list[dict[str, Any]] = []
    for i in range(days - 1, -1, -1):
        day = today - timedelta(days=i)
        key = day.strftime("%Y-%m-%d")
        try:
            cost = float(tracker.daily_total(key))
        except Exception:
            cost = 0.0
        out.append({"day": key, "cost_usd": cost})
    return out
