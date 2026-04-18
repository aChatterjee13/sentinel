"""Dashboard security primitives — authn, authz, CSRF, rate limit, headers.

This sub-package was added in workstream #3 to harden the dashboard
beyond the basic-auth-only surface that shipped in workstream #1. The
public surface is intentionally small:

* :class:`Principal` and :data:`ANONYMOUS_PRINCIPAL` describe the
  authenticated caller and live on ``request.state.principal``.
* :class:`RBACPolicy` resolves usernames to principals and answers
  permission checks. :func:`require_permission` is a FastAPI
  dependency factory built around it.
* :func:`build_auth_middleware` returns a single FastAPI dependency
  that dispatches on ``DashboardServerConfig.auth`` (none, basic,
  bearer) and populates ``request.state.principal``.
* :class:`CSRFMiddleware`, :class:`RateLimitMiddleware`,
  :class:`SecurityHeadersMiddleware` are the Starlette middlewares
  that round out the stack.
"""

from sentinel.dashboard.security.principal import (
    ANONYMOUS_PRINCIPAL,
    Principal,
)
from sentinel.dashboard.security.rbac import (
    AUTHENTICATED,
    RBACPolicy,
    require_permission,
)
from sentinel.dashboard.security.route_perms import (
    permission_for_path,
)

__all__ = [
    "ANONYMOUS_PRINCIPAL",
    "AUTHENTICATED",
    "Principal",
    "RBACPolicy",
    "permission_for_path",
    "require_permission",
]
