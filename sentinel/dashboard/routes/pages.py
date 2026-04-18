"""HTML page routes for the dashboard.

Each handler is a thin wrapper around a view function from
:mod:`sentinel.dashboard.views`. Templates render the result.
"""

import re
import time
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from sentinel.core.exceptions import DashboardError
from sentinel.dashboard.deps import get_state
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

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

log = structlog.get_logger(__name__)


def build_pages_router(templates: "Jinja2Templates") -> APIRouter:
    """Construct the FastAPI router that renders HTML pages."""
    router = APIRouter(default_response_class=HTMLResponse)

    def _render(
        request: Request,
        template: str,
        context: dict[str, Any],
        state: DashboardState,
    ) -> HTMLResponse:
        start = time.perf_counter()
        ctx = {
            "request": request,
            "ui": state.config.ui,
            "client": state.client,
            "model_name": state.client.model_name,
            "model_version": state.client.current_version,
            "domain": state.client.config.model.domain,
            "llmops_enabled": state.client.config.llmops.enabled,
            "agentops_enabled": state.client.config.agentops.enabled,
            "show_modules": state.config.ui.show_modules,
            **context,
        }
        try:
            response = templates.TemplateResponse(request, template, ctx)
        except Exception as e:
            log.error("dashboard.error", template=template, error=str(e))
            raise DashboardError(f"failed to render {template}: {e}") from e
        log.info(
            "dashboard.render",
            template=template,
            duration_ms=(time.perf_counter() - start) * 1000.0,
        )
        return response

    # ── Pages ─────────────────────────────────────────────────────
    #
    # Each route declares its required permission via
    # ``Depends(require_permission(...))``. When RBAC is disabled the
    # check is a no-op (every authenticated request gets the wildcard
    # permission). When RBAC is enabled, the auth middleware sets
    # ``request.state.principal`` and the dependency raises 403 when
    # the principal lacks the named permission.

    @router.get("/", include_in_schema=False)
    def overview_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/")
        data = overview_view.build(state)
        return _render(request, "overview.html", {"data": data}, state)

    @router.get("/drift", include_in_schema=False)
    def drift_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("drift.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/drift")
        data = drift_view.build(state)
        return _render(request, "drift.html", {"data": data}, state)

    @router.get("/drift/{report_id}", include_in_schema=False)
    def drift_detail_page(
        request: Request,
        report_id: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("drift.read")),
    ) -> HTMLResponse:
        if not re.match(r"^[a-zA-Z0-9_-]+$", report_id):
            raise HTTPException(status_code=400, detail="invalid report_id format")
        log.info("dashboard.request", path=f"/drift/{report_id}")
        report = drift_view.detail(state, report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="drift report not found")
        return _render(request, "drift_detail.html", {"report": report}, state)

    @router.get("/features", include_in_schema=False)
    def features_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("features.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/features")
        data = features_view.build(state)
        return _render(request, "features.html", {"data": data}, state)

    @router.get("/registry", include_in_schema=False)
    def registry_list_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("registry.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/registry")
        data = registry_view.list_models(state)
        return _render(request, "registry_list.html", {"data": data}, state)

    @router.get("/registry/{model}/{version}", include_in_schema=False)
    def registry_detail_page(
        request: Request,
        model: str,
        version: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("registry.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path=f"/registry/{model}/{version}")
        data = registry_view.detail(state, model, version)
        if data is None:
            raise HTTPException(status_code=404, detail="model version not found")
        return _render(
            request, "registry_detail.html", {"data": data, "model_name_arg": model}, state
        )

    @router.get("/audit", include_in_schema=False)
    def audit_page(
        request: Request,
        event_type: str | None = None,
        model_name: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("audit.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/audit")
        data = audit_view.build(
            state,
            event_type=event_type,
            model_name=model_name,
            since=drift_view.parse_iso(since),
            until=drift_view.parse_iso(until),
            limit=limit,
        )
        return _render(request, "audit.html", {"data": data}, state)

    @router.get("/llmops/prompts", include_in_schema=False)
    def llmops_prompts_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("llmops.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/llmops/prompts")
        data = llmops_view.prompts(state)
        return _render(request, "llmops_prompts.html", {"data": data}, state)

    @router.get("/llmops/guardrails", include_in_schema=False)
    def llmops_guardrails_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("llmops.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/llmops/guardrails")
        data = llmops_view.guardrails(state)
        return _render(request, "llmops_guardrails.html", {"data": data}, state)

    @router.get("/llmops/tokens", include_in_schema=False)
    def llmops_tokens_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("llmops.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/llmops/tokens")
        data = llmops_view.tokens(state)
        return _render(request, "llmops_tokens.html", {"data": data}, state)

    @router.get("/agentops/traces", include_in_schema=False)
    def agentops_traces_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("agentops.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/agentops/traces")
        data = agentops_view.traces(state)
        return _render(request, "agentops_traces.html", {"data": data}, state)

    @router.get("/agentops/traces/{trace_id}", include_in_schema=False)
    def agentops_trace_detail_page(
        request: Request,
        trace_id: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("agentops.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path=f"/agentops/traces/{trace_id}")
        data = agentops_view.trace_detail(state, trace_id)
        if data is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return _render(request, "agentops_trace_detail.html", {"data": data}, state)

    @router.get("/agentops/tools", include_in_schema=False)
    def agentops_tools_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("agentops.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/agentops/tools")
        data = agentops_view.tools(state)
        return _render(request, "agentops_tools.html", {"data": data}, state)

    @router.get("/agentops/agents", include_in_schema=False)
    def agentops_agents_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("agentops.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/agentops/agents")
        data = agentops_view.agents(state)
        return _render(request, "agentops_agents.html", {"data": data}, state)

    # ── Experiments ───────────────────────────────────────────────

    @router.get("/experiments", include_in_schema=False)
    def experiments_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("experiments.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/experiments")
        data = experiments_view.build_list(state)
        return _render(request, "experiments.html", {"data": data}, state)

    @router.get("/experiments/runs/{run_id}", include_in_schema=False)
    def experiment_run_detail_page(
        request: Request,
        run_id: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("experiments.read")),
    ) -> HTMLResponse:
        if not re.match(r"^[a-zA-Z0-9_-]+$", run_id):
            raise HTTPException(status_code=400, detail="invalid run_id format")
        log.info("dashboard.request", path=f"/experiments/runs/{run_id}")
        data = experiments_view.build_run_detail(state, run_id)
        if data is None:
            raise HTTPException(status_code=404, detail="run not found")
        return _render(request, "run_detail.html", {"data": data}, state)

    @router.get("/experiments/{name}", include_in_schema=False)
    def experiment_detail_page(
        request: Request,
        name: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("experiments.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path=f"/experiments/{name}")
        data = experiments_view.build_detail(state, name)
        if data is None:
            raise HTTPException(status_code=404, detail="experiment not found")
        return _render(request, "experiment_detail.html", {"data": data}, state)

    @router.get("/deployments", include_in_schema=False)
    def deployments_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("deployments.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/deployments")
        data = deployments_view.build(state)
        return _render(request, "deployments.html", {"data": data}, state)

    @router.get("/compliance", include_in_schema=False)
    def compliance_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("compliance.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/compliance")
        data = compliance_view.build(state)
        return _render(request, "compliance.html", {"data": data}, state)

    @router.get("/retraining", include_in_schema=False)
    def retraining_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("retrain.trigger")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/retraining")
        data = retraining_view.build(state)
        return _render(request, "retraining.html", {"data": data}, state)

    @router.get("/intelligence", include_in_schema=False)
    def intelligence_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission(AUTHENTICATED)),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/intelligence")
        data = intelligence_view.build(state)
        return _render(request, "intelligence.html", {"data": data}, state)

    @router.get("/cohorts", include_in_schema=False)
    def cohorts_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("features.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/cohorts")
        data = cohorts_view.build(state)
        return _render(request, "cohorts.html", {"data": data}, state)

    @router.get("/cohorts/{cohort_id}", include_in_schema=False)
    def cohort_detail_page(
        request: Request,
        cohort_id: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("features.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path=f"/cohorts/{cohort_id}")
        data = cohorts_view.detail(state, cohort_id)
        if data is None:
            raise HTTPException(status_code=404, detail="cohort not found")
        return _render(request, "cohort_detail.html", {"data": data}, state)

    @router.get("/explanations", include_in_schema=False)
    def explanations_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("features.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/explanations")
        data = explanations_view.build(state)
        return _render(request, "explanations.html", {"data": data}, state)

    # ── Datasets ──────────────────────────────────────────────────

    @router.get("/datasets", include_in_schema=False)
    def datasets_list_page(
        request: Request,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("datasets.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path="/datasets")
        data = datasets_view.list_datasets(state)
        return _render(request, "datasets.html", {"data": data}, state)

    @router.get("/datasets/{name}/{version}", include_in_schema=False)
    def datasets_detail_page(
        request: Request,
        name: str,
        version: str,
        state: DashboardState = Depends(get_state),
        _: None = Depends(require_permission("datasets.read")),
    ) -> HTMLResponse:
        log.info("dashboard.request", path=f"/datasets/{name}/{version}")
        data = datasets_view.detail(state, name, version)
        if data is None:
            raise HTTPException(status_code=404, detail="dataset version not found")
        return _render(
            request, "dataset_detail.html", {"data": data, "dataset_name_arg": name}, state
        )

    return router
