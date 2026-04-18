"""JSON API routes for the dashboard.

These mirror the page routes and return raw JSON ready for HTMX partial
updates and Plotly chart payloads.

NOTE: This module must NOT use ``from __future__ import annotations``
because FastAPI needs concrete ``Request`` type resolution for
dependency injection (see MEMORY.md).
"""

import csv
import io
import json
import re
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse

from sentinel.dashboard.deps import get_state
from sentinel.dashboard.routes.models import AuditChartData, HealthResponse
from sentinel.dashboard.security.rbac import AUTHENTICATED, require_permission
from sentinel.dashboard.state import DashboardState
from sentinel.dashboard.views import (
    agentops as agentops_view,
)
from sentinel.dashboard.views import (
    audit as audit_view,
)
from sentinel.dashboard.views import (
    cohorts as cohorts_view,
)
from sentinel.dashboard.views import (
    compliance as compliance_view,
)
from sentinel.dashboard.views import (
    datasets as datasets_view,
)
from sentinel.dashboard.views import (
    deployments as deployments_view,
)
from sentinel.dashboard.views import (
    drift as drift_view,
)
from sentinel.dashboard.views import (
    experiments as experiments_view,
)
from sentinel.dashboard.views import (
    explanations as explanations_view,
)
from sentinel.dashboard.views import (
    features as features_view,
)
from sentinel.dashboard.views import (
    intelligence as intelligence_view,
)
from sentinel.dashboard.views import (
    llmops as llmops_view,
)
from sentinel.dashboard.views import (
    overview as overview_view,
)
from sentinel.dashboard.views import (
    registry as registry_view,
)
from sentinel.dashboard.views import (
    retraining as retraining_view,
)

log = structlog.get_logger(__name__)


def build_api_router() -> APIRouter:
    """Construct the FastAPI router that returns JSON."""
    router = APIRouter(prefix="/api")

    # ── API root redirect ─────────────────────────────────────────

    @router.get("", include_in_schema=False)
    def api_root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    # ── Overview / Health ─────────────────────────────────────────

    @router.get(
        "/overview",
        tags=["Overview"],
        summary="System overview",
        description=(
            "Returns a comprehensive summary of the monitored model including "
            "current drift status, recent audit events, feature health snapshot, "
            "and active deployment information. Use this as the entry-point to "
            "build a status dashboard."
        ),
        response_description="Overview payload with status, drift summary, recent audit events, and module health.",
    )
    def api_overview(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission(AUTHENTICATED)),
    ) -> dict[str, Any]:
        log.info("dashboard.api", path="/api/overview")
        return overview_view.build(state)

    @router.get(
        "/health",
        tags=["Overview"],
        summary="Service health check",
        description=(
            "Lightweight liveness probe. Returns the model name, current version, "
            "and the timestamp when the dashboard process started. Suitable for "
            "load-balancer health checks and uptime monitors."
        ),
        response_model=HealthResponse,
        response_description="Service status with model identity and uptime.",
    )
    def api_health(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission(AUTHENTICATED)),
    ) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": state.client.model_name,
            "version": state.client.current_version,
            "started_at": state.started_at.isoformat(),
        }

    @router.get("/health/live", tags=["Overview"], include_in_schema=True)
    def liveness_probe() -> dict[str, str]:
        """Unauthenticated liveness probe for load balancers."""
        return {"status": "ok"}

    # ── Drift Detection ───────────────────────────────────────────

    @router.get(
        "/drift",
        tags=["Drift Detection"],
        summary="Drift summary",
        description=(
            "Returns an aggregate view of all drift reports: total count, "
            "number of reports showing drift, per-feature scores, and the "
            "latest check timestamp. The underlying detection method (PSI, KS, "
            "etc.) is determined by the model's YAML config."
        ),
        response_description="Drift summary with report counts, feature-level scores, and severity breakdown.",
    )
    def api_drift(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("drift.read")),
    ) -> dict[str, Any]:
        log.info("dashboard.api", path="/api/drift")
        return drift_view.build(state)

    @router.get(
        "/drift/timeseries",
        tags=["Drift Detection"],
        summary="Drift time-series",
        description=(
            "Returns drift scores plotted over time for Plotly chart rendering. "
            "Use the `window` parameter to control the lookback period. "
            "Supported windows: `all`, `7d`, `30d`, `90d`."
        ),
        response_description="Time-indexed drift scores suitable for line-chart visualisation.",
    )
    def api_drift_timeseries(
        window: str = "all",
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("drift.read")),
    ) -> dict[str, Any]:
        if window not in ("all", "7d", "30d", "90d"):
            raise HTTPException(status_code=400, detail=f"invalid window: {window}")
        log.info("dashboard.api", path="/api/drift/timeseries")
        return drift_view.timeseries(state, window=window)

    @router.get(
        "/drift/{report_id}",
        tags=["Drift Detection"],
        summary="Drift report detail",
        description=(
            "Fetch a single drift report by its unique identifier. Returns "
            "per-feature drift scores, the statistical test used, p-values, "
            "severity classification, and the detection timestamp."
        ),
        response_description="Full drift report with per-feature breakdowns.",
    )
    def api_drift_detail(
        report_id: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("drift.read")),
    ) -> dict[str, Any]:
        if not re.match(r"^[a-zA-Z0-9_-]+$", report_id):
            raise HTTPException(status_code=400, detail="invalid report_id format")
        report = drift_view.detail(state, report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="drift report not found")
        return report

    # ── Feature Health ────────────────────────────────────────────

    @router.get(
        "/features",
        tags=["Feature Health"],
        summary="Feature health summary",
        description=(
            "Returns per-feature monitoring data including drift scores, "
            "importance rankings, missing-value rates, and distribution "
            "statistics. Use this to identify which features are degrading."
        ),
        response_description="Feature-level health metrics with drift and importance scores.",
    )
    def api_features(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("features.read")),
    ) -> dict[str, Any]:
        log.info("dashboard.api", path="/api/features")
        return features_view.build(state)

    @router.get(
        "/features/ranked",
        tags=["Feature Health"],
        summary="Features ranked by drift",
        description=(
            "Returns the same feature data as `/api/features` but sorted "
            "descending by drift score. Useful for quickly spotting the most "
            "drifted features without client-side sorting."
        ),
        response_description="Feature list sorted by drift score (highest first).",
    )
    def api_features_ranked(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("features.read")),
    ) -> list[dict[str, Any]]:
        data = features_view.build(state)
        rows: list[dict[str, Any]] = list(data.get("features", []))
        rows.sort(key=lambda f: f.get("drift_score", 0.0), reverse=True)
        return rows

    @router.get(
        "/features/chart",
        tags=["Feature Health"],
        summary="Feature chart data",
        description=(
            "Returns feature drift and importance data pre-formatted for "
            "Plotly bar/scatter chart rendering."
        ),
        response_description="Chart-ready feature arrays (names, drift scores, importance values).",
    )
    def api_features_chart(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("features.read")),
    ) -> dict[str, Any]:
        return features_view.chart_data(state)

    # ── Model Registry ────────────────────────────────────────────

    @router.get(
        "/registry",
        tags=["Model Registry"],
        summary="List registered models",
        description=(
            "Returns all model versions in the registry with their metadata, "
            "performance baselines, and registration timestamps."
        ),
        response_description="List of registered model versions with metadata.",
    )
    def api_registry(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("registry.read")),
    ) -> dict[str, Any]:
        log.info("dashboard.api", path="/api/registry")
        return registry_view.list_models(state)

    @router.get(
        "/registry/{model}/{version}",
        tags=["Model Registry"],
        summary="Model version detail",
        description=(
            "Fetch detailed metadata for a specific model version including "
            "hyperparameters, training data reference, performance baselines, "
            "and feature schema."
        ),
        response_description="Full model version metadata and baselines.",
    )
    def api_registry_detail(
        model: str,
        version: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("registry.read")),
    ) -> dict[str, Any]:
        data = registry_view.detail(state, model, version)
        if data is None:
            raise HTTPException(status_code=404, detail="model version not found")
        return data

    @router.get(
        "/registry/{model}/compare",
        tags=["Model Registry"],
        summary="Compare model versions",
        description=(
            "Side-by-side comparison of two model versions. Returns parameter "
            "diffs, metric deltas, and schema changes between versions `v1` "
            "and `v2`."
        ),
        response_description="Comparison payload with parameter and metric diffs.",
    )
    def api_registry_compare(
        model: str,
        v1: str,
        v2: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("registry.read")),
    ) -> dict[str, Any]:
        data = registry_view.compare(state, model, v1, v2)
        if data is None:
            raise HTTPException(status_code=404, detail="comparison failed")
        return data

    # ── Audit Trail ───────────────────────────────────────────────

    @router.get(
        "/audit",
        tags=["Audit Trail"],
        summary="Query audit events",
        description=(
            "Returns audit trail entries with optional filters. Every model "
            "lifecycle action (drift detected, model registered, deployment "
            "changed, alert fired) is logged here immutably. Supports "
            "filtering by event type, model name, and time range."
        ),
        response_description="Paginated audit events matching the filter criteria.",
    )
    def api_audit(
        event_type: str | None = None,
        model_name: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("audit.read")),
    ) -> dict[str, Any]:
        log.info("dashboard.api", path="/api/audit")
        return audit_view.build(
            state,
            event_type=event_type,
            model_name=model_name,
            since=drift_view.parse_iso(since),
            until=drift_view.parse_iso(until),
            limit=limit,
        )

    @router.get(
        "/audit/chart",
        tags=["Audit Trail"],
        summary="Audit event chart data",
        description=(
            "Aggregates the last 200 audit events by type and returns "
            "label/value arrays suitable for bar or pie chart rendering."
        ),
        response_model=AuditChartData,
        response_description="Event type labels and their counts.",
    )
    def api_audit_chart(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("audit.read")),
    ) -> dict[str, Any]:
        """Return audit event counts by type for charting."""
        log.info("dashboard.api", path="/api/audit/chart")
        data = audit_view.build(state, limit=200)
        type_counts: dict[str, int] = {}
        for ev in data.get("events", []):
            et = (
                ev.get("event_type", "unknown")
                if isinstance(ev, dict)
                else getattr(ev, "event_type", "unknown")
            )
            type_counts[et] = type_counts.get(et, 0) + 1
        return {"labels": list(type_counts.keys()), "values": list(type_counts.values())}

    @router.get(
        "/audit/stream",
        tags=["Audit Trail"],
        summary="Recent audit stream",
        description=(
            "Returns the most recent audit events, optionally filtered to "
            "those newer than the `since` timestamp. Designed for HTMX "
            "polling to build a live event feed."
        ),
        response_description="Recent audit events in reverse-chronological order.",
    )
    def api_audit_stream(
        since: str | None = None,
        limit: int = 50,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("audit.read")),
    ) -> dict[str, Any]:
        return audit_view.build(
            state,
            since=drift_view.parse_iso(since),
            limit=limit,
        )

    # ── LLMOps ────────────────────────────────────────────────────

    @router.get(
        "/llmops/prompts",
        tags=["LLMOps"],
        summary="Prompt registry",
        description=(
            "Lists all registered prompt versions with their A/B test "
            "assignments, quality scores, and token usage statistics."
        ),
        response_description="Prompt versions with performance metadata.",
    )
    def api_llmops_prompts(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("llmops.read")),
    ) -> dict[str, Any]:
        return llmops_view.prompts(state)

    @router.get(
        "/llmops/guardrails",
        tags=["LLMOps"],
        summary="Guardrail statistics",
        description=(
            "Returns pass/warn/block counts for each configured guardrail "
            "(PII detection, jailbreak, toxicity, groundedness, etc.). "
            "Use this to monitor guardrail activation rates."
        ),
        response_description="Per-guardrail violation counts and action breakdown.",
    )
    def api_llmops_guardrails(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("llmops.read")),
    ) -> dict[str, Any]:
        return llmops_view.guardrails(state)

    @router.get(
        "/llmops/tokens",
        tags=["LLMOps"],
        summary="Token economics overview",
        description=(
            "Returns token usage, cost breakdowns, and budget utilisation. "
            "Includes per-model, per-prompt-version, and per-day aggregations."
        ),
        response_description="Token and cost metrics with budget status.",
    )
    def api_llmops_tokens(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("llmops.read")),
    ) -> dict[str, Any]:
        return llmops_view.tokens(state)

    @router.get(
        "/tokens/daily",
        tags=["LLMOps"],
        summary="Daily token usage",
        description=(
            "Returns per-day token usage and cost for the last `days` days. "
            "Defaults to a 14-day window. Useful for cost trend charts."
        ),
        response_description="Daily token counts and associated costs.",
    )
    def api_tokens_daily(
        days: int = 14,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("llmops.read")),
    ) -> dict[str, Any]:
        data = llmops_view.tokens(state)
        # Trim to requested window
        if data.get("daily"):
            data["daily"] = data["daily"][-days:]
        return data

    @router.get(
        "/llmops/guardrails/trend",
        tags=["LLMOps"],
        summary="Guardrail violation trend",
        description=(
            "Returns guardrail violation rates over time, suitable for "
            "trend-line chart rendering. Helps detect if guardrail "
            "activation is increasing (possible prompt regression)."
        ),
        response_description="Time-indexed guardrail violation rates.",
    )
    def api_guardrails_trend(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("llmops.read")),
    ) -> dict[str, Any]:
        return llmops_view.guardrails_trend(state)

    @router.get(
        "/tokens/by-model",
        tags=["LLMOps"],
        summary="Token usage by model",
        description=(
            "Breaks down token usage and cost per LLM model (e.g., "
            "gpt-4o vs gpt-4o-mini). Use this to understand cost "
            "distribution across model tiers."
        ),
        response_description="Per-model token and cost aggregations.",
    )
    def api_tokens_by_model(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("llmops.read")),
    ) -> dict[str, Any]:
        return llmops_view.tokens_by_model(state)

    # ── AgentOps ──────────────────────────────────────────────────

    @router.get(
        "/agentops/traces",
        tags=["AgentOps"],
        summary="List agent traces",
        description=(
            "Returns recent agent execution traces with step counts, "
            "total tokens, cost, tool call counts, and completion status. "
            "Each trace represents one end-to-end agent run."
        ),
        response_description="Agent traces with summary statistics.",
    )
    def api_traces(
        limit: int = 100,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("agentops.read")),
    ) -> dict[str, Any]:
        return agentops_view.traces(state, limit=limit)

    @router.get(
        "/traces/recent",
        tags=["AgentOps"],
        summary="Recent agent traces",
        description=(
            "Convenience endpoint returning the most recent agent traces "
            "with a smaller default limit (50). Designed for HTMX live-feed "
            "updates."
        ),
        response_description="Recent agent traces (default limit: 50).",
    )
    def api_traces_recent(
        limit: int = 50,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("agentops.read")),
    ) -> dict[str, Any]:
        return agentops_view.traces(state, limit=limit)

    @router.get(
        "/traces/{trace_id}",
        tags=["AgentOps"],
        summary="Trace detail",
        description=(
            "Returns the full span tree for a single agent trace — "
            "every reasoning step, tool call, delegation, and the "
            "final output. Includes latency and token usage per span."
        ),
        response_description="Complete trace span tree with per-span metrics.",
    )
    def api_trace_detail(
        trace_id: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("agentops.read")),
    ) -> dict[str, Any]:
        data = agentops_view.trace_detail(state, trace_id)
        if data is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return data

    @router.get(
        "/agentops/tools",
        tags=["AgentOps"],
        summary="Tool audit summary",
        description=(
            "Returns per-tool success/failure rates, average latency, "
            "and call counts across all agents. Use this to identify "
            "flaky or slow tools."
        ),
        response_description="Per-tool audit metrics (success rate, latency, call count).",
    )
    def api_tools(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("agentops.read")),
    ) -> dict[str, Any]:
        return agentops_view.tools(state)

    @router.get(
        "/agentops/tools/chart",
        tags=["AgentOps"],
        summary="Tool usage chart data",
        description=(
            "Returns tool call frequency and success rates formatted for Plotly chart rendering."
        ),
        response_description="Chart-ready tool usage arrays.",
    )
    def api_tools_chart(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("agentops.read")),
    ) -> dict[str, Any]:
        return agentops_view.tools_chart(state)

    @router.get(
        "/traces/{trace_id}/gantt",
        tags=["AgentOps"],
        summary="Trace Gantt chart data",
        description=(
            "Returns span start/end times for a trace formatted as a Gantt "
            "chart payload. Visualises the timeline of agent reasoning "
            "steps and tool calls."
        ),
        response_description="Gantt-formatted span timing data.",
    )
    def api_trace_gantt(
        trace_id: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("agentops.read")),
    ) -> dict[str, Any]:
        data = agentops_view.trace_gantt(state, trace_id)
        if data is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return data

    @router.get(
        "/agentops/agents",
        tags=["AgentOps"],
        summary="Agent registry",
        description=(
            "Lists all registered agents with their capabilities, health "
            "status, budget limits, and recent performance baselines."
        ),
        response_description="Registered agents with capability manifests and health status.",
    )
    def api_agents(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("agentops.read")),
    ) -> dict[str, Any]:
        return agentops_view.agents(state)

    # ── Experiments ────────────────────────────────────────────────

    @router.get(
        "/experiments",
        tags=["Experiments"],
        summary="List experiments",
        description=(
            "Returns all experiments with run counts, latest metrics, "
            "and tags. Each experiment groups multiple training runs."
        ),
        response_description="Experiment list with run counts and latest metrics.",
    )
    def api_experiments(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("experiments.read")),
    ) -> dict[str, Any]:
        return experiments_view.build_list(state)

    @router.get(
        "/experiments/{name}/metrics",
        tags=["Experiments"],
        summary="Experiment metric history",
        description=(
            "Returns metric time-series for all runs in the named experiment. "
            "Useful for plotting training curves and comparing run performance."
        ),
        response_description="Per-run metric history arrays.",
    )
    def api_experiment_metrics(
        name: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("experiments.read")),
    ) -> dict[str, Any]:
        return experiments_view.metrics_json(state, name)

    # ── Deployments ───────────────────────────────────────────────

    @router.get(
        "/deployments",
        tags=["Deployments"],
        summary="Deployment status",
        description=(
            "Returns current deployment state including active strategy "
            "(canary, shadow, blue-green), traffic split percentages, "
            "champion/challenger versions, and rollback eligibility."
        ),
        response_description="Active deployment details with traffic split and strategy info.",
    )
    def api_deployments(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("deployments.read")),
    ) -> dict[str, Any]:
        return deployments_view.build(state)

    # ── Compliance ────────────────────────────────────────────────

    @router.get(
        "/compliance",
        tags=["Compliance"],
        summary="Compliance summary",
        description=(
            "Returns compliance status for configured regulatory frameworks "
            "(FCA Consumer Duty, EU AI Act, etc.). Includes check results, "
            "open findings, and report generation timestamps."
        ),
        response_description="Compliance check results per framework.",
    )
    def api_compliance(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("compliance.read")),
    ) -> dict[str, Any]:
        return compliance_view.build(state)

    @router.get(
        "/compliance/chart",
        tags=["Compliance"],
        summary="Compliance chart data",
        description=(
            "Returns compliance scores and finding counts pre-formatted for chart visualisation."
        ),
        response_description="Chart-ready compliance score arrays.",
    )
    def api_compliance_chart(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("compliance.read")),
    ) -> dict[str, Any]:
        return compliance_view.chart_data(state)

    # ── Intelligence ──────────────────────────────────────────────

    @router.get(
        "/intelligence",
        tags=["Intelligence"],
        summary="Model graph and KPI links",
        description=(
            "Returns the multi-model dependency DAG and business KPI "
            "mappings. Shows upstream/downstream relationships and "
            "cascade alert paths."
        ),
        response_description="Model dependency graph and business KPI link definitions.",
    )
    def api_intelligence(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission(AUTHENTICATED)),
    ) -> dict[str, Any]:
        return intelligence_view.build(state)

    # ── Retraining ────────────────────────────────────────────────

    @router.get(
        "/retraining",
        tags=["Retraining"],
        summary="Retraining status",
        description=(
            "Returns current retrain orchestration state: pending triggers, "
            "active pipeline runs, approval requests awaiting review, and "
            "recently completed retrain cycles."
        ),
        response_description="Retrain pipeline status and pending approval requests.",
    )
    def api_retraining(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("retrain.trigger")),
    ) -> dict[str, Any]:
        return retraining_view.build(state)

    @router.post(
        "/retrain/trigger",
        tags=["Retraining"],
        summary="Trigger retraining",
        description=(
            "Manually trigger a retrain pipeline run. Requires a configured "
            "pipeline runner (`retraining.pipeline` in config). The trigger "
            "is attributed to the authenticated principal."
        ),
        response_description="Trigger result with pipeline run ID and status.",
    )
    def api_retrain_trigger(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("retrain.trigger")),
    ) -> dict[str, Any]:
        log.info("dashboard.api", path="/api/retrain/trigger")
        client = state.client
        orchestrator = client.retrain

        if orchestrator._pipeline_runner is None:
            raise HTTPException(
                status_code=400,
                detail="Pipeline runner not configured. Set retraining.pipeline in config.",
            )

        principal = getattr(request.state, "principal", None) or "anonymous"
        trigger = orchestrator.evaluator.manual(f"dashboard trigger by {principal}")
        result = orchestrator.run(
            model_name=client.model_name,
            trigger=trigger,
            context={"triggered_by": principal, "source": "dashboard"},
        )
        return result

    @router.post(
        "/retrain/approve/{request_id}",
        tags=["Retraining"],
        summary="Approve retrain request",
        description=(
            "Approve a pending human-in-the-loop retrain request. The "
            "request_id comes from the retraining status endpoint. "
            "Approval is logged in the audit trail."
        ),
        response_description="Approval result with updated pipeline status.",
    )
    def api_retrain_approve(
        request: Request,
        request_id: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("retrain.trigger")),
    ) -> dict[str, Any]:
        log.info("dashboard.api", path=f"/api/retrain/approve/{request_id}")
        principal = getattr(request.state, "principal", None) or "anonymous"
        result = state.client.retrain.approve(request_id, by=principal)
        return result

    @router.post(
        "/retrain/reject/{request_id}",
        tags=["Retraining"],
        summary="Reject retrain request",
        description=(
            "Reject a pending human-in-the-loop retrain request. The "
            "rejection reason is attributed to the authenticated principal "
            "and logged in the audit trail."
        ),
        response_description="Rejection result with updated request status.",
    )
    def api_retrain_reject(
        request: Request,
        request_id: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("retrain.trigger")),
    ) -> dict[str, Any]:
        log.info("dashboard.api", path=f"/api/retrain/reject/{request_id}")
        principal = getattr(request.state, "principal", None) or "anonymous"
        result = state.client.retrain.reject(request_id, by=principal)
        return result

    # ── Cohort analysis ───────────────────────────────────────────

    @router.get(
        "/cohorts/summary",
        tags=["Cohorts"],
        summary="Cohort summary",
        description=(
            "Returns a summary of all defined cohorts with per-cohort "
            "sample counts, drift indicators, and performance metrics. "
            "Use cohorts to monitor model fairness across data segments."
        ),
        response_description="Per-cohort summary with sample counts and performance.",
    )
    def api_cohorts_summary(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("features.read")),
    ) -> dict[str, Any]:
        log.info("dashboard.api", path="/api/cohorts/summary")
        return cohorts_view.build(state)

    @router.get(
        "/cohorts/chart",
        tags=["Cohorts"],
        summary="Cohort chart data",
        description=(
            "Returns per-cohort metric distributions formatted for "
            "Plotly grouped bar or box chart rendering."
        ),
        response_description="Chart-ready cohort metric arrays.",
    )
    def api_cohorts_chart(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("features.read")),
    ) -> dict[str, Any]:
        log.info("dashboard.api", path="/api/cohorts/chart")
        return cohorts_view.chart_data(state)

    @router.get(
        "/cohorts/{cohort_id}",
        tags=["Cohorts"],
        summary="Cohort detail",
        description=(
            "Fetch detailed metrics for a single cohort including "
            "feature distributions, performance breakdown, and drift "
            "scores relative to the global baseline."
        ),
        response_description="Full cohort detail with per-feature and per-metric breakdowns.",
    )
    def api_cohort_detail(
        cohort_id: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("features.read")),
    ) -> dict[str, Any]:
        data = cohorts_view.detail(state, cohort_id)
        if data is None:
            raise HTTPException(status_code=404, detail="cohort not found")
        return data

    # ── Explanations ──────────────────────────────────────────────

    @router.get(
        "/explanations/global",
        tags=["Explanations"],
        summary="Global feature explanations",
        description=(
            "Returns global SHAP or LIME feature importance values "
            "formatted for chart rendering. Shows which features "
            "have the greatest impact on model predictions overall."
        ),
        response_description="Global feature importance values for chart visualisation.",
    )
    def api_explanations_global(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("features.read")),
    ) -> dict[str, Any]:
        log.info("dashboard.api", path="/api/explanations/global")
        return explanations_view.chart_data(state)

    # ── Datasets ──────────────────────────────────────────────────

    @router.get(
        "/datasets",
        tags=["Datasets"],
        summary="List datasets",
        description=(
            "Returns all registered datasets grouped by name, with "
            "version history, format, row/feature counts, and lineage "
            "references."
        ),
        response_description="Dataset list grouped by name with version summaries.",
    )
    def api_datasets(
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("datasets.read")),
    ) -> dict[str, Any]:
        log.info("dashboard.api", path="/api/datasets")
        return datasets_view.list_datasets(state)

    @router.get(
        "/datasets/{name}/{version}",
        tags=["Datasets"],
        summary="Dataset version detail",
        description=(
            "Fetch detailed metadata for a specific dataset version "
            "including schema, row count, storage location, creation "
            "timestamp, and lineage (which experiments consumed it)."
        ),
        response_description="Full dataset version metadata and lineage.",
    )
    def api_dataset_detail(
        name: str,
        version: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("datasets.read")),
    ) -> dict[str, Any]:
        data = datasets_view.detail(state, name, version)
        if data is None:
            raise HTTPException(status_code=404, detail="dataset version not found")
        return data

    # ── CSV Exports ──────────────────────────────────────────────

    @router.get(
        "/export/audit.csv",
        tags=["Export"],
        summary="Export audit events as CSV",
        description=(
            "Streams audit trail events as a downloadable CSV file. "
            "Optionally filter by event type. Use the `limit` parameter "
            "to control the maximum number of rows."
        ),
    )
    def export_audit_csv(
        event_type: str | None = None,
        limit: int = 10000,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("audit.read")),
    ) -> StreamingResponse:
        """Export audit events as CSV."""
        events = list(state.client.audit.query(event_type=event_type, limit=limit))

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            ["timestamp", "event_type", "model_name", "model_version", "actor", "payload"]
        )
        for event in events:
            writer.writerow([
                event.timestamp.isoformat(),
                event.event_type,
                event.model_name or "",
                event.model_version or "",
                event.actor or "",
                json.dumps(event.payload) if event.payload else "",
            ])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=audit_export.csv"},
        )

    @router.get(
        "/export/drift.csv",
        tags=["Export"],
        summary="Export drift history as CSV",
        description=(
            "Streams drift detection events as a downloadable CSV. "
            "Pulls from the audit trail filtered to `drift_checked` events."
        ),
    )
    def export_drift_csv(
        limit: int = 1000,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("drift.read")),
    ) -> StreamingResponse:
        """Export drift detection history as CSV."""
        events = list(state.client.audit.query(event_type="drift_checked", limit=limit))

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            ["timestamp", "model_name", "method", "is_drifted", "severity", "test_statistic"]
        )
        for event in events:
            payload = event.payload or {}
            writer.writerow([
                event.timestamp.isoformat(),
                event.model_name or "",
                payload.get("method", ""),
                payload.get("is_drifted", ""),
                payload.get("severity", ""),
                payload.get("test_statistic", ""),
            ])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=drift_export.csv"},
        )

    @router.get(
        "/export/metrics.csv",
        tags=["Export"],
        summary="Export model metrics as CSV",
        description=(
            "Streams prediction-logged events as a downloadable CSV. "
            "Use this for bulk analysis of prediction metadata."
        ),
    )
    def export_metrics_csv(
        limit: int = 1000,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("registry.read")),
    ) -> StreamingResponse:
        """Export model metrics history as CSV."""
        events = list(
            state.client.audit.query(event_type="prediction_logged", limit=limit)
        )

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp", "model_name", "model_version", "payload"])
        for event in events:
            writer.writerow([
                event.timestamp.isoformat(),
                event.model_name or "",
                event.model_version or "",
                json.dumps(event.payload) if event.payload else "",
            ])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=metrics_export.csv"},
        )

    # ── Deployment Rollback ──────────────────────────────────────

    @router.post(
        "/deployments/rollback",
        tags=["Deployments"],
        summary="Roll back deployment",
        description=(
            "Roll back to a previous model version using the direct "
            "deployment strategy. Optionally specify the target version; "
            "if omitted the system uses the previously active version."
        ),
    )
    def rollback_deployment(
        request: Request,
        version: str | None = None,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("deployments.write")),
    ) -> dict[str, Any]:
        """Roll back to a previous deployment version."""
        client = state.client
        try:
            deploy_version = version or client.current_version or "unknown"
            deploy_state = client.deploy(version=deploy_version, strategy="direct")
            return {
                "status": "rolled_back",
                "version": deploy_version,
                "state": deploy_state.model_dump(mode="json"),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    return router
