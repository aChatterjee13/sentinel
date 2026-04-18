"""Role-based access control for the dashboard.

The :class:`RBACPolicy` is constructed once at app startup from the
``dashboard.server.rbac`` config block. It owns three pieces of state:

* the role → permission mapping (e.g. ``viewer → {drift.read, ...}``)
* the role hierarchy (``[viewer, operator, admin]`` by default)
* the username → role mapping (used by basic auth resolution)

Permission checks happen in :func:`require_permission`, a FastAPI
dependency factory that reads ``request.state.principal`` and raises
HTTP 403 when the permission isn't held. When RBAC is disabled the
factory still returns a working dependency — it just always allows
the request through, so route definitions stay identical regardless
of whether the operator has flipped RBAC on.

This module deliberately does **not** use ``from __future__ import
annotations``: see :file:`MEMORY.md` for why that combination breaks
FastAPI dependency injection when ``Request`` is a parameter type.
"""

from collections.abc import Callable

import structlog
from fastapi import HTTPException, Request, status

from sentinel.config.schema import RBACConfig
from sentinel.dashboard.security.principal import ANONYMOUS_PRINCIPAL, Principal

log = structlog.get_logger(__name__)

AUTHENTICATED = "__authenticated__"
"""Sentinel value for :func:`require_permission` meaning "any authenticated
(non-anonymous) principal" — no specific permission is checked."""


class RBACPolicy:
    """Resolves principals and answers permission questions."""

    def __init__(self, config: RBACConfig) -> None:
        self.config = config
        self._user_roles: dict[str, frozenset[str]] = {
            binding.username: frozenset(binding.roles) for binding in config.users
        }
        self._role_perms: dict[str, frozenset[str]] = self._expand_role_permissions(
            config.role_permissions,
            config.role_hierarchy,
        )

    @staticmethod
    def _expand_role_permissions(
        role_permissions: dict[str, list[str]],
        role_hierarchy: list[str],
    ) -> dict[str, frozenset[str]]:
        """Compute the transitive closure of role → permissions.

        Roles inherit permissions from every role appearing earlier in
        ``role_hierarchy``. So with the default
        ``[viewer, operator, admin]`` ordering, ``operator`` gains all
        viewer perms in addition to its own, and ``admin`` gains
        everything below it.
        """
        expanded: dict[str, set[str]] = {}
        for role in role_hierarchy:
            inherited: set[str] = set()
            # Pick up everything every role below this one already
            # accumulated. Walking ``role_hierarchy`` in order means
            # ``expanded`` already contains the lower roles.
            for lower in role_hierarchy:
                if lower == role:
                    break
                inherited |= expanded.get(lower, set())
            inherited |= set(role_permissions.get(role, []))
            expanded[role] = inherited
        # Roles that exist in role_permissions but aren't part of the
        # hierarchy still get their direct perms (no inheritance).
        for role, perms in role_permissions.items():
            if role not in expanded:
                expanded[role] = set(perms)
        return {role: frozenset(perms) for role, perms in expanded.items()}

    def permissions_for_roles(self, roles: frozenset[str]) -> frozenset[str]:
        """Union of permissions held by every role in ``roles``."""
        out: set[str] = set()
        for role in roles:
            out |= self._role_perms.get(role, frozenset())
        return frozenset(out)

    def resolve_principal(
        self,
        username: str | None,
        roles: list[str] | None = None,
        auth_mode: str = "basic",
    ) -> Principal:
        """Build a fully resolved :class:`Principal` from raw inputs.

        Args:
            username: Caller identifier from basic auth or a JWT
                claim. ``None`` falls through to the anonymous
                principal when RBAC is disabled.
            roles: Optional explicit role list — used by Bearer auth
                where the JWT carries the roles claim. When omitted
                we look the username up in the configured user
                bindings.
            auth_mode: ``"basic"`` or ``"bearer"`` (or ``"none"``).

        Returns:
            A populated :class:`Principal`. When RBAC is disabled
            and no username is provided, returns
            :data:`ANONYMOUS_PRINCIPAL`.
        """
        if not self.config.enabled:
            # When RBAC is off the dashboard still wants a principal
            # so middlewares can stash it on request.state, but every
            # permission check is a no-op via the wildcard permission.
            if username is None:
                return ANONYMOUS_PRINCIPAL
            return Principal(
                username=username,
                roles=frozenset(),
                permissions=frozenset({"*"}),
                auth_mode=auth_mode,
            )

        if username is None:
            return ANONYMOUS_PRINCIPAL

        if roles is None:
            assigned = self._user_roles.get(username)
            if assigned is None:
                assigned = frozenset({self.config.default_role})
        else:
            assigned = frozenset(roles)

        perms = self.permissions_for_roles(assigned)
        return Principal(
            username=username,
            roles=assigned,
            permissions=perms,
            auth_mode=auth_mode,
        )


def require_permission(permission: str) -> Callable[[Request], None]:
    """FastAPI dependency factory enforcing a permission check.

    Reads ``request.state.principal`` (set by the auth middleware)
    and raises ``HTTPException(403)`` when the principal lacks the
    required permission. When RBAC is disabled the principal will
    have the ``*`` wildcard permission and the check passes.

    The empty-string permission is treated as "no check at all" —
    the request is unconditionally allowed. **Avoid using ``""`` for
    routes that should require authentication**; use
    :data:`AUTHENTICATED` instead, which verifies the caller is not
    anonymous but does not require a specific permission.

    Example:
        >>> from fastapi import Depends
        >>> @router.get("/drift")
        ... def drift(_: None = Depends(require_permission("drift.read"))):
        ...     ...
    """

    def _checker(request: Request) -> None:
        if not permission:
            return None
        principal: Principal = getattr(
            request.state,
            "principal",
            ANONYMOUS_PRINCIPAL,
        )
        # AUTHENTICATED: only require a non-anonymous principal.
        # Also pass when the principal holds wildcard "*" (RBAC disabled).
        if permission == AUTHENTICATED:
            if not principal.is_anonymous or principal.has_permission("*"):
                return None
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="authentication required",
            )
        if principal.has_permission(permission):
            return None
        log.warning(
            "dashboard.rbac.denied",
            username=principal.username,
            required=permission,
            held=sorted(principal.permissions),
            path=request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"missing required permission: {permission}",
        )

    return _checker
