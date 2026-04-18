"""AgentOps dashboard views — traces, tools, agents."""

from __future__ import annotations

from typing import Any

from sentinel.agentops.trace.visualiser import TraceVisualiser
from sentinel.dashboard.state import DashboardState


def _agentops(state: DashboardState) -> Any | None:
    client = state.client
    if not client.config.agentops.enabled:
        return None
    try:
        return client.agentops
    except Exception:
        return None


def traces(state: DashboardState, limit: int = 100) -> dict[str, Any]:
    """List recent agent traces."""
    agentops = _agentops(state)
    if agentops is None:
        return {"enabled": False, "traces": []}
    try:
        recent = agentops.tracer.get_recent(limit)
    except Exception:
        recent = []
    visualiser = TraceVisualiser()
    rows = [visualiser.summary(t) for t in recent]
    return {"enabled": True, "traces": rows}


def trace_detail(state: DashboardState, trace_id: str) -> dict[str, Any] | None:
    """Build a single trace detail payload."""
    agentops = _agentops(state)
    if agentops is None:
        return None
    try:
        all_traces = agentops.tracer.all_traces()
    except Exception:
        return None
    trace = next((t for t in all_traces if t.trace_id == trace_id), None)
    if trace is None:
        return None
    visualiser = TraceVisualiser()
    return {
        "summary": visualiser.summary(trace),
        "timeline": visualiser.timeline(trace),
        "tree": visualiser.tree(trace),
        "spans": [
            {
                "span_id": s.span_id,
                "parent_id": s.parent_id,
                "name": s.name,
                "kind": s.kind,
                "status": s.status.value,
                "start_time": s.start_time.isoformat(),
                "end_time": s.end_time.isoformat() if s.end_time else None,
                "duration_ms": s.duration_ms,
                "attributes": s.attributes,
            }
            for s in trace.spans
        ],
    }


def tools(state: DashboardState) -> dict[str, Any]:
    """Aggregate per-tool stats."""
    agentops = _agentops(state)
    if agentops is None:
        return {"enabled": False, "tools": {}}
    try:
        stats = agentops.tool_monitor.all_stats()
    except Exception:
        stats = {}
    return {"enabled": True, "tools": stats}


def tools_chart(state: DashboardState) -> dict[str, Any]:
    """Return chart-ready tool success/failure data."""
    data = tools(state)
    tool_stats = data.get("tools", {})
    rows: list[dict[str, Any]] = []
    for name, s in tool_stats.items():
        rows.append(
            {
                "name": name,
                "successes": s.get("successes", 0),
                "failures": s.get("failures", 0),
                "total": s.get("total_calls", s.get("calls", 0)),
                "avg_latency_ms": s.get("avg_latency_ms", 0),
            }
        )
    return {"tools": rows}


def trace_gantt(state: DashboardState, trace_id: str) -> dict[str, Any] | None:
    """Return chart-ready Gantt data for a single trace."""
    detail = trace_detail(state, trace_id)
    if detail is None:
        return None
    spans = detail.get("spans", [])
    if not spans:
        return {"spans": []}
    rows: list[dict[str, Any]] = []
    for i, s in enumerate(spans):
        rows.append(
            {
                "name": s.get("name", f"span-{i}"),
                "kind": s.get("kind", "internal"),
                "status": s.get("status", "ok"),
                "duration_ms": s.get("duration_ms", 0) or 0,
                "offset_ms": i * 50,  # approximate offset for visual ordering
            }
        )
    return {"spans": rows}


def agents(state: DashboardState) -> dict[str, Any]:
    """List registered agents and their versions."""
    agentops = _agentops(state)
    if agentops is None:
        return {"enabled": False, "agents": []}
    rows: list[dict[str, Any]] = []
    try:
        names = agentops.registry.list_agents()
    except Exception:
        names = []
    for name in names:
        try:
            versions = agentops.registry.list_versions(name)
        except Exception:
            versions = []
        version_rows = []
        for v in versions:
            try:
                spec = agentops.registry.get(name, v)
                version_rows.append(spec.to_dict())
            except Exception:
                continue
        rows.append({"name": name, "versions": version_rows})
    return {"enabled": True, "agents": rows}
