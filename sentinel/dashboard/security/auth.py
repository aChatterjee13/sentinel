"""Auth-mode dispatch for the dashboard.

This module owns the single FastAPI dependency that runs on every
request. Its job is to:

1. Decide *how* to authenticate the caller based on
   ``DashboardServerConfig.auth`` (``none``, ``basic``, ``bearer``).
2. Resolve the caller to a :class:`Principal` via the configured
   :class:`RBACPolicy`.
3. Stash the principal on ``request.state.principal`` so the
   per-route ``require_permission`` dependencies have something to
   read.

The dependency is intentionally a single closure rather than a
Starlette middleware. FastAPI's dependency-resolution surface is
already aware of ``Request``, lets us short-circuit with
``HTTPException`` cleanly, and integrates with the route-level
``Depends`` chain — middlewares wrap the entire stack and would force
us to translate exceptions back to JSON manually.

This module deliberately does **not** use ``from __future__ import
annotations`` — see :file:`MEMORY.md` for the FastAPI forward-ref
gotcha that bites any module mixing future annotations with
``Request``-typed dependency parameters.
"""

import base64
import secrets
from collections.abc import Callable
from typing import Any

import structlog
from fastapi import HTTPException, Request, status

from sentinel.config.schema import DashboardServerConfig
from sentinel.config.secrets import unwrap
from sentinel.dashboard.security.principal import Principal
from sentinel.dashboard.security.rbac import RBACPolicy

log = structlog.get_logger(__name__)


def build_auth_dependency(
    server_cfg: DashboardServerConfig,
    rbac_policy: RBACPolicy,
) -> Callable[..., Any]:
    """Build the FastAPI dependency that authenticates the caller.

    Returns a closure suitable for ``FastAPI(dependencies=[...])``
    that runs once per request, validates whichever credential the
    server config asks for, resolves it to a :class:`Principal`, and
    stashes the result on ``request.state.principal`` for the per-route
    permission checks to read.

    Args:
        server_cfg: The :class:`DashboardServerConfig` block.
        rbac_policy: The shared :class:`RBACPolicy` — used to map a
            username (and optional roles claim) to a fully-resolved
            principal.

    Returns:
        A FastAPI-compatible dependency callable. The callable's
        signature uses ``Header(default=None)`` rather than
        ``Request`` to dodge the future-annotations forward-ref bug.
    """
    auth_mode = server_cfg.auth

    if auth_mode == "none":
        return _build_none_dependency(rbac_policy)
    if auth_mode == "basic":
        return _build_basic_dependency(server_cfg, rbac_policy)
    if auth_mode == "bearer":
        return _build_bearer_dependency(server_cfg, rbac_policy)

    raise ValueError(f"unknown dashboard auth mode: {auth_mode!r}")


# ── auth: none ────────────────────────────────────────────────────────


def _build_none_dependency(rbac_policy: RBACPolicy) -> Callable[..., Any]:
    """Inject an anonymous principal that holds the wildcard permission.

    When auth is disabled, RBAC is also a no-op (the policy returns
    a wildcard principal even if RBAC is on but no username is
    available). This keeps the local-first dev experience identical
    to pre-workstream-#3 behaviour.
    """
    # Pre-build the principal once. When RBAC is disabled this is the
    # wildcard principal; when RBAC is enabled with auth=none we still
    # return the anonymous principal so per-route checks fall through
    # to whichever permissions the operator granted to the
    # ``default_role`` — but in practice nobody runs auth=none with
    # rbac.enabled=true, so we keep the wildcard fallback.
    principal = rbac_policy.resolve_principal(username=None, auth_mode="none")
    if principal.is_anonymous and not rbac_policy.config.enabled:
        # When RBAC is off, give the anonymous principal the wildcard
        # so empty-string and explicit permissions both pass.
        principal = Principal(
            username="anonymous",
            roles=frozenset(),
            permissions=frozenset({"*"}),
            auth_mode="none",
        )

    async def _no_auth(request: Request) -> None:
        request.state.principal = principal

    return _no_auth


# ── auth: basic ───────────────────────────────────────────────────────


def _build_basic_dependency(
    server_cfg: DashboardServerConfig,
    rbac_policy: RBACPolicy,
) -> Callable[..., Any]:
    """HTTP Basic auth dependency.

    Validates the ``Authorization: Basic`` header against the
    configured username/password, then resolves the username to a
    principal via the RBAC policy. Uses
    :func:`secrets.compare_digest` for constant-time comparison.
    """
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

    async def _basic_auth(request: Request) -> None:
        authorization = request.headers.get("authorization")
        if not authorization or not authorization.lower().startswith("basic "):
            raise unauthorised
        try:
            encoded = authorization.split(" ", 1)[1].strip()
            decoded = base64.b64decode(encoded.encode("ascii")).decode("utf-8")
            user, _, password = decoded.partition(":")
        except (ValueError, UnicodeDecodeError) as e:
            raise unauthorised from e
        user_ok = secrets.compare_digest(user.encode("utf-8"), expected_user.encode("utf-8"))
        pass_ok = secrets.compare_digest(password.encode("utf-8"), expected_pass.encode("utf-8"))
        if not (user_ok and pass_ok and expected_user and expected_pass):
            raise unauthorised

        principal = rbac_policy.resolve_principal(username=user, auth_mode="basic")
        request.state.principal = principal

    return _basic_auth


# ── auth: bearer ──────────────────────────────────────────────────────


def _build_bearer_dependency(
    server_cfg: DashboardServerConfig,
    rbac_policy: RBACPolicy,
) -> Callable[..., Any]:
    """Bearer JWT auth dependency.

    The actual JWT validation lives in
    :mod:`sentinel.dashboard.security.bearer` and is imported lazily
    so that ``auth: none`` and ``auth: basic`` deployments don't
    need :mod:`PyJWT` installed.
    """
    from sentinel.dashboard.security.bearer import (
        BearerTokenError,
        validate_bearer_token,
    )

    bearer_cfg = server_cfg.bearer

    unauthorised = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": 'Bearer realm="Sentinel"'},
    )

    async def _bearer_auth(request: Request) -> None:
        authorization = request.headers.get("authorization")
        if not authorization or not authorization.lower().startswith("bearer "):
            raise unauthorised
        token = authorization.split(" ", 1)[1].strip()
        if not token:
            raise unauthorised
        try:
            principal = validate_bearer_token(token, bearer_cfg, rbac_policy)
        except BearerTokenError as exc:
            log.warning(
                "dashboard.bearer.invalid",
                reason=str(exc),
                path=request.url.path,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid bearer token",
                headers={
                    "WWW-Authenticate": ('Bearer realm="Sentinel", error="invalid_token"'),
                },
            ) from exc
        request.state.principal = principal

    return _bearer_auth


__all__ = ["build_auth_dependency"]
