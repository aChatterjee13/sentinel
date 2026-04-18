"""FastAPI app factory + uvicorn launcher for the Sentinel dashboard."""

from __future__ import annotations

import base64
import secrets
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from sentinel.config.schema import DashboardConfig, DashboardServerConfig
from sentinel.config.secrets import unwrap
from sentinel.core.exceptions import (
    DashboardError,
    DashboardNotInstalledError,
)
from sentinel.dashboard.routes.api import build_api_router
from sentinel.dashboard.routes.pages import build_pages_router
from sentinel.dashboard.security.auth import build_auth_dependency
from sentinel.dashboard.security.csrf import CSRFMiddleware
from sentinel.dashboard.security.headers import SecurityHeadersMiddleware
from sentinel.dashboard.security.rate_limit import RateLimitMiddleware
from sentinel.dashboard.security.rbac import RBACPolicy
from sentinel.dashboard.state import DashboardState

if TYPE_CHECKING:
    from sentinel.core.client import SentinelClient

log = structlog.get_logger(__name__)


_HERE = Path(__file__).resolve().parent
_TEMPLATE_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"

# ── OpenAPI metadata ─────────────────────────────────────────────────

OPENAPI_DESCRIPTION = """\
**Sentinel MLOps Dashboard** — a unified monitoring and governance API for
production ML models, LLM applications, and autonomous agent systems.

## What this API provides

| Area | Endpoints |
|------|-----------|
| **Monitoring** | Drift detection, feature health, data quality |
| **Model management** | Model registry, deployment status, retraining |
| **LLMOps** | Prompt versions, guardrail stats, token economics |
| **AgentOps** | Agent traces, tool audit, safety metrics |
| **Governance** | Audit trail, compliance reports, explainability |
| **Experimentation** | Experiment tracking, run comparison |
| **Data** | Dataset registry, versioning, lineage |

## Authentication

Authentication is configured per-deployment:

- **None** — open access (development only).
- **HTTP Basic** — set `dashboard.server.auth: basic` with username/password.
- **Bearer JWT** — set `dashboard.server.auth: bearer` with a JWKS URL.

When auth is enabled, pass credentials via the **Authorize** button above.

## Alternative documentation

- [ReDoc](/redoc) — a clean, printable API reference.
- [OpenAPI JSON](/openapi.json) — raw schema for code generators.
"""

OPENAPI_TAGS: list[dict[str, str]] = [
    {
        "name": "Overview",
        "description": "System health check and summary metrics across all modules.",
    },
    {
        "name": "Drift Detection",
        "description": (
            "Data drift (PSI, KS, JS divergence), concept drift (DDM, ADWIN), "
            "and model performance drift monitoring."
        ),
    },
    {
        "name": "Feature Health",
        "description": "Per-feature drift scores, importance ranking, and distribution charts.",
    },
    {
        "name": "Model Registry",
        "description": "Model version management, metadata lookup, and version comparison.",
    },
    {
        "name": "Audit Trail",
        "description": "Immutable event log covering every model lifecycle action for compliance.",
    },
    {
        "name": "LLMOps",
        "description": (
            "Prompt version registry, guardrail violation trends, and token usage / cost economics."
        ),
    },
    {
        "name": "AgentOps",
        "description": (
            "Agent reasoning traces, tool call audit, safety metrics, and agent registry health."
        ),
    },
    {
        "name": "Experiments",
        "description": "Experiment tracking, run management, metric history, and run comparison.",
    },
    {
        "name": "Datasets",
        "description": "Dataset registry, versioning, row/feature counts, and lineage.",
    },
    {
        "name": "Deployments",
        "description": "Deployment status, canary/shadow/blue-green progress, and promotion.",
    },
    {
        "name": "Compliance",
        "description": "Regulatory compliance reports (FCA, EU AI Act) and audit summaries.",
    },
    {
        "name": "Intelligence",
        "description": "Multi-model dependency graph and business KPI linking.",
    },
    {
        "name": "Retraining",
        "description": "Retrain orchestration status, manual triggers, and approval workflows.",
    },
    {
        "name": "Cohorts",
        "description": "Cohort-based analysis of model performance across data segments.",
    },
    {
        "name": "Explanations",
        "description": "Global and per-prediction model explainability (SHAP / LIME).",
    },
]


def _require_fastapi() -> None:
    try:
        import fastapi  # noqa: F401
        import jinja2  # noqa: F401
    except ImportError as e:
        raise DashboardNotInstalledError(
            "Dashboard extras are missing. Install with `pip install sentinel-mlops[dashboard]`."
        ) from e


def build_basic_auth_dependency(server_cfg: DashboardServerConfig) -> Callable[..., Any]:
    """Build a FastAPI dependency that enforces HTTP Basic auth.

    The dependency is a no-op when ``server_cfg.auth != "basic"``. When
    enabled, it validates the ``Authorization: Basic ...`` header
    against the configured username and password using
    :func:`secrets.compare_digest` for constant-time comparison.

    Args:
        server_cfg: The :class:`DashboardServerConfig` block.

    Returns:
        A callable suitable for use with FastAPI's
        ``Depends(...)`` mechanism.

    Notes:
        Basic auth is the *only* dashboard auth mechanism in scope for
        workstream #1. OIDC, session cookies, RBAC, CSRF, and rate
        limiting all live in the security workstream that follows.
    """
    _require_fastapi()
    from fastapi import Header, HTTPException, status

    if server_cfg.auth != "basic":

        async def _no_auth() -> None:  # pragma: no cover - trivial
            return None

        return _no_auth

    expected_user = server_cfg.basic_auth_username or ""
    expected_pass = unwrap(server_cfg.basic_auth_password) or ""

    if not expected_user or not expected_pass:
        log.warning(
            "dashboard.basic_auth.missing_credentials",
            message=(
                "auth=basic but basic_auth_username/basic_auth_password are unset; "
                "the dashboard will reject every request"
            ),
        )

    unauthorised = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": 'Basic realm="Sentinel"'},
    )

    # NOTE: We read the Authorization header via FastAPI's ``Header``
    # parameter rather than ``request: Request`` because the latter
    # interacts badly with ``from __future__ import annotations`` —
    # the string forward-ref makes FastAPI fall back to treating the
    # parameter as a query field, returning 422 instead of 401.
    async def _basic_auth(
        authorization: str | None = Header(default=None),
    ) -> None:
        if not authorization or not authorization.lower().startswith("basic "):
            raise unauthorised
        try:
            encoded = authorization.split(" ", 1)[1].strip()
            decoded = base64.b64decode(encoded.encode("ascii")).decode("utf-8")
            user, _, password = decoded.partition(":")
        except (ValueError, UnicodeDecodeError) as e:
            raise unauthorised from e
        # Constant-time comparison guards against timing oracles.
        user_ok = secrets.compare_digest(user.encode("utf-8"), expected_user.encode("utf-8"))
        pass_ok = secrets.compare_digest(password.encode("utf-8"), expected_pass.encode("utf-8"))
        if not (user_ok and pass_ok and expected_user and expected_pass):
            raise unauthorised
        return None

    return _basic_auth


def create_dashboard_app(
    client: SentinelClient,
    config: DashboardConfig | None = None,
) -> Any:
    """Create the FastAPI application bound to a live SentinelClient.

    Args:
        client: A live :class:`~sentinel.SentinelClient`. The dashboard
            never owns the client lifecycle — it only reads from it.
        config: Optional dashboard config override. Defaults to
            ``client.config.dashboard``.

    Returns:
        A FastAPI application instance.

    Raises:
        DashboardNotInstalledError: When the optional ``[dashboard]``
            extras are not installed.

    Example:
        >>> from sentinel import SentinelClient
        >>> from sentinel.dashboard import create_dashboard_app
        >>> client = SentinelClient.from_config("sentinel.yaml")
        >>> app = create_dashboard_app(client)
    """
    _require_fastapi()

    from fastapi import Depends, FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    cfg = config or client.config.dashboard
    state = DashboardState(client=client, config=cfg)
    rbac_policy = RBACPolicy(cfg.server.rbac)

    app_kwargs: dict[str, Any] = {
        "title": cfg.ui.title or "Sentinel MLOps Dashboard",
        "description": OPENAPI_DESCRIPTION,
        "version": getattr(client, "__version__", "0.1.0"),
        "root_path": cfg.server.root_path,
        "openapi_tags": OPENAPI_TAGS,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json",
    }
    # Apply the auth dispatcher at app construction time so every
    # router mounted below it inherits the dependency. The dispatcher
    # always populates ``request.state.principal`` so the per-route
    # ``require_permission`` checks have something to read regardless
    # of which auth mode (none / basic / bearer) is configured.
    app_kwargs["dependencies"] = [
        Depends(build_auth_dependency(cfg.server, rbac_policy)),
    ]
    log.info(
        "dashboard.auth.configured",
        mode=cfg.server.auth,
        rbac_enabled=cfg.server.rbac.enabled,
    )

    app = FastAPI(**app_kwargs)
    app.state.dashboard_state = state
    app.state.rbac_policy = rbac_policy

    # ── Middleware stack (outermost first) ─────────────────────────
    #
    # Starlette runs middlewares in reverse order of ``add_middleware``
    # calls — the *last* one added is the *outermost* wrapper. We want
    # the order from outermost to innermost to be:
    #
    #   1. SecurityHeadersMiddleware  (always cheap)
    #   2. RateLimitMiddleware        (block flood before doing real work)
    #   3. CSRFMiddleware             (block forged writes early)
    #
    # Auth dispatch and RBAC checks live in the per-route Depends
    # chain, not as middlewares.
    app.add_middleware(CSRFMiddleware, cfg=cfg.server.csrf)
    app.add_middleware(RateLimitMiddleware, cfg=cfg.server.rate_limit)
    app.add_middleware(SecurityHeadersMiddleware, csp=cfg.server.csp)

    templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
    templates.env.globals["title"] = cfg.ui.title
    templates.env.globals["theme"] = cfg.ui.theme

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    app.include_router(build_pages_router(templates))
    app.include_router(build_api_router())

    @app.exception_handler(DashboardError)
    async def _dashboard_error_handler(_: Request, exc: DashboardError) -> JSONResponse:
        log.error("dashboard.error", error=str(exc))
        return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.exception_handler(Exception)
    async def _generic_error_handler(_: Request, exc: Exception) -> JSONResponse:
        log.error("dashboard.unhandled_error", error=str(exc), type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )

    log.info(
        "dashboard.app_initialised",
        title=cfg.ui.title,
        modules=cfg.ui.show_modules,
    )
    return app


class SentinelDashboardRouter:
    """An embeddable dashboard mountable inside an existing FastAPI app.

    Use this when you want to add Sentinel pages to a customer's
    internal tooling without spinning up a separate process:

    Example:
        >>> from fastapi import FastAPI
        >>> from sentinel import SentinelClient
        >>> from sentinel.dashboard import SentinelDashboardRouter
        >>> client = SentinelClient.from_config("sentinel.yaml")
        >>> app = FastAPI()
        >>> dash = SentinelDashboardRouter(client)
        >>> app.include_router(dash.pages, prefix="/sentinel")
        >>> app.include_router(dash.api, prefix="/sentinel")
    """

    def __init__(
        self,
        client: SentinelClient,
        config: DashboardConfig | None = None,
    ) -> None:
        _require_fastapi()
        from fastapi.templating import Jinja2Templates

        self.client = client
        self.config = config or client.config.dashboard
        self.state = DashboardState(client=client, config=self.config)
        self.templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
        self.pages = build_pages_router(self.templates)
        self.api = build_api_router()

    def attach(self, app: Any, prefix: str = "/sentinel") -> None:
        """Convenience: mount both routers and bind the dashboard state."""
        app.state.dashboard_state = self.state
        app.include_router(self.pages, prefix=prefix)
        app.include_router(self.api, prefix=prefix)


def run(
    client: SentinelClient,
    host: str | None = None,
    port: int | None = None,
    reload: bool = False,
) -> None:
    """Boot the dashboard with uvicorn (blocking).

    Args:
        client: A live SentinelClient.
        host: Optional override for ``dashboard.server.host``.
        port: Optional override for ``dashboard.server.port``.
        reload: Enable uvicorn auto-reload (dev mode).

    Raises:
        DashboardNotInstalledError: When extras are missing.
    """
    _require_fastapi()
    try:
        import uvicorn
    except ImportError as e:
        raise DashboardNotInstalledError(
            "uvicorn is required to run the dashboard. Install with `pip install sentinel-mlops[dashboard]`."
        ) from e

    cfg = client.config.dashboard
    app = create_dashboard_app(client, cfg)
    uvicorn.run(
        app,
        host=host or cfg.server.host,
        port=port or cfg.server.port,
        reload=reload,
        log_level="info",
    )
