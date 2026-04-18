"""Tests for the dashboard RBAC dependency stack.

These cover the three layers introduced in workstream #3:

* :class:`RBACPolicy` — role hierarchy expansion and principal
  resolution from raw usernames + role lists.
* :func:`require_permission` — the FastAPI dependency factory that
  reads ``request.state.principal`` and decides 200/403.
* The end-to-end wiring through ``create_dashboard_app`` — verifies
  that flipping ``rbac.enabled`` actually gates routes for users
  who lack the required permission.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from sentinel import SentinelClient
from sentinel.config.schema import (
    AlertsConfig,
    AuditConfig,
    DashboardConfig,
    DashboardServerConfig,
    DataDriftConfig,
    DriftConfig,
    ModelConfig,
    RBACConfig,
    RBACUserBinding,
    SentinelConfig,
)
from sentinel.dashboard.security.principal import ANONYMOUS_PRINCIPAL, Principal
from sentinel.dashboard.security.rbac import AUTHENTICATED, RBACPolicy, require_permission
from sentinel.dashboard.security.route_perms import permission_for_path

pytest.importorskip("fastapi")
pytest.importorskip("jinja2")

from fastapi import HTTPException
from fastapi.testclient import TestClient

from sentinel.dashboard.server import create_dashboard_app

# ── Helpers ───────────────────────────────────────────────────────────


def _basic(user: str, password: str) -> dict[str, str]:
    raw = f"{user}:{password}".encode()
    return {"Authorization": "Basic " + base64.b64encode(raw).decode()}


def _build_config(
    tmp_path: Path,
    *,
    rbac_enabled: bool,
    auth: str = "basic",
    users: list[RBACUserBinding] | None = None,
) -> SentinelConfig:
    return SentinelConfig(
        model=ModelConfig(name="rbac_test_model", domain="tabular"),
        drift=DriftConfig(
            data=DataDriftConfig(method="psi", threshold=0.2, window="7d"),
        ),
        alerts=AlertsConfig(),
        audit=AuditConfig(storage="local", path=str(tmp_path / "audit")),
        dashboard=DashboardConfig(
            enabled=True,
            server=DashboardServerConfig(
                auth=auth,  # type: ignore[arg-type]
                basic_auth_username="alice" if auth == "basic" else None,
                basic_auth_password="hunter2" if auth == "basic" else None,  # type: ignore[arg-type]
                rbac=RBACConfig(
                    enabled=rbac_enabled,
                    default_role="viewer",
                    users=users or [],
                ),
            ),
        ),
    )


# ── RBACPolicy unit tests ────────────────────────────────────────────


class TestRBACPolicy:
    def test_role_hierarchy_transitive_closure(self) -> None:
        cfg = RBACConfig(
            enabled=True,
            role_permissions={
                "viewer": ["drift.read"],
                "operator": ["deployments.promote"],
                "admin": ["*"],
            },
            role_hierarchy=["viewer", "operator", "admin"],
        )
        policy = RBACPolicy(cfg)

        viewer_perms = policy.permissions_for_roles(frozenset({"viewer"}))
        operator_perms = policy.permissions_for_roles(frozenset({"operator"}))
        admin_perms = policy.permissions_for_roles(frozenset({"admin"}))

        assert "drift.read" in viewer_perms
        assert "deployments.promote" not in viewer_perms

        # Operator inherits viewer permissions through the hierarchy.
        assert "drift.read" in operator_perms
        assert "deployments.promote" in operator_perms

        # Admin inherits everything plus the wildcard.
        assert "*" in admin_perms
        assert "drift.read" in admin_perms
        assert "deployments.promote" in admin_perms

    def test_resolve_principal_with_explicit_roles(self) -> None:
        policy = RBACPolicy(RBACConfig(enabled=True))
        principal = policy.resolve_principal(
            username="bob",
            roles=["operator"],
            auth_mode="bearer",
        )
        assert principal.username == "bob"
        assert principal.roles == frozenset({"operator"})
        assert principal.has_permission("drift.read")
        assert principal.has_permission("deployments.promote")
        assert not principal.has_permission("nonexistent.thing")

    def test_resolve_principal_with_user_table(self) -> None:
        policy = RBACPolicy(
            RBACConfig(
                enabled=True,
                users=[RBACUserBinding(username="carol", roles=["admin"])],
            )
        )
        principal = policy.resolve_principal(username="carol", auth_mode="basic")
        assert principal.has_permission("anything.you.can.imagine")  # wildcard

    def test_resolve_principal_unknown_user_uses_default_role(self) -> None:
        policy = RBACPolicy(RBACConfig(enabled=True, default_role="viewer"))
        principal = policy.resolve_principal(username="stranger", auth_mode="basic")
        assert principal.roles == frozenset({"viewer"})
        assert principal.has_permission("drift.read")
        assert not principal.has_permission("deployments.promote")

    def test_resolve_principal_when_disabled_yields_wildcard(self) -> None:
        policy = RBACPolicy(RBACConfig(enabled=False))
        principal = policy.resolve_principal(username="anyone", auth_mode="basic")
        assert "*" in principal.permissions

    def test_resolve_principal_when_disabled_no_username_is_anonymous(self) -> None:
        policy = RBACPolicy(RBACConfig(enabled=False))
        principal = policy.resolve_principal(username=None, auth_mode="none")
        assert principal == ANONYMOUS_PRINCIPAL


# ── require_permission unit tests ────────────────────────────────────


class _RequestStub:
    """Tiny shim for the bits of ``Request`` we exercise."""

    def __init__(self, principal: Principal | None) -> None:
        class _State:
            pass

        self.state = _State()
        if principal is not None:
            self.state.principal = principal

        class _URL:
            path = "/test"

        self.url = _URL()


class TestRequirePermission:
    def test_empty_string_permission_always_passes(self) -> None:
        checker = require_permission("")
        # Even with no principal at all, empty perm short-circuits.
        checker(_RequestStub(None))  # type: ignore[arg-type]

    def test_authenticated_passes_for_non_anonymous(self) -> None:
        principal = Principal(
            username="alice",
            permissions=frozenset(),
            auth_mode="basic",
        )
        checker = require_permission(AUTHENTICATED)
        checker(_RequestStub(principal))  # type: ignore[arg-type]

    def test_authenticated_rejects_anonymous(self) -> None:
        checker = require_permission(AUTHENTICATED)
        with pytest.raises(HTTPException) as exc_info:
            checker(_RequestStub(ANONYMOUS_PRINCIPAL))  # type: ignore[arg-type]
        assert exc_info.value.status_code == 401

    def test_authenticated_rejects_missing_principal(self) -> None:
        checker = require_permission(AUTHENTICATED)
        with pytest.raises(HTTPException) as exc_info:
            checker(_RequestStub(None))  # type: ignore[arg-type]
        assert exc_info.value.status_code == 401

    def test_wildcard_principal_passes(self) -> None:
        principal = Principal(
            username="root",
            permissions=frozenset({"*"}),
            auth_mode="basic",
        )
        require_permission("anything.read")(_RequestStub(principal))  # type: ignore[arg-type]

    def test_held_permission_passes(self) -> None:
        principal = Principal(
            username="bob",
            permissions=frozenset({"drift.read"}),
        )
        require_permission("drift.read")(_RequestStub(principal))  # type: ignore[arg-type]

    def test_missing_permission_raises_403(self) -> None:
        principal = Principal(
            username="bob",
            permissions=frozenset({"drift.read"}),
        )
        with pytest.raises(HTTPException) as exc_info:
            require_permission("deployments.promote")(_RequestStub(principal))  # type: ignore[arg-type]
        assert exc_info.value.status_code == 403
        assert "deployments.promote" in str(exc_info.value.detail)

    def test_missing_principal_falls_back_to_anonymous(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            require_permission("drift.read")(_RequestStub(None))  # type: ignore[arg-type]
        assert exc_info.value.status_code == 403


# ── route_perms lookup tests ─────────────────────────────────────────


class TestPermissionForPath:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("/", ""),
            ("/api/health", ""),
            ("/api/overview", ""),
            ("/drift", "drift.read"),
            ("/drift/abc", "drift.read"),
            ("/api/drift", "drift.read"),
            ("/api/drift/timeseries", "drift.read"),
            ("/features", "features.read"),
            ("/registry/foo/1.0", "registry.read"),
            ("/api/audit", "audit.read"),
            ("/api/audit/stream", "audit.read"),
            ("/llmops/prompts", "llmops.read"),
            ("/api/tokens/daily", "llmops.read"),
            ("/agentops/traces", "agentops.read"),
            ("/api/traces/recent", "agentops.read"),
            ("/deployments", "deployments.read"),
            ("/api/compliance", "compliance.read"),
        ],
    )
    def test_lookup(self, path: str, expected: str) -> None:
        assert permission_for_path(path) == expected

    def test_unknown_path_is_unprivileged(self) -> None:
        assert permission_for_path("/totally/unknown") == ""


# ── End-to-end through the FastAPI app ───────────────────────────────


class TestRBACEndToEnd:
    def test_rbac_disabled_lets_authenticated_user_through(self, tmp_path: Path) -> None:
        cfg = _build_config(tmp_path, rbac_enabled=False)
        client = SentinelClient(cfg)
        app = create_dashboard_app(client)
        with TestClient(app) as test_client:
            resp = test_client.get("/drift", headers=_basic("alice", "hunter2"))
            assert resp.status_code == 200

    def test_rbac_enabled_admin_can_read(self, tmp_path: Path) -> None:
        cfg = _build_config(
            tmp_path,
            rbac_enabled=True,
            users=[RBACUserBinding(username="alice", roles=["admin"])],
        )
        client = SentinelClient(cfg)
        app = create_dashboard_app(client)
        with TestClient(app) as test_client:
            resp = test_client.get("/drift", headers=_basic("alice", "hunter2"))
            assert resp.status_code == 200

    def test_rbac_enabled_viewer_can_read_drift(self, tmp_path: Path) -> None:
        cfg = _build_config(
            tmp_path,
            rbac_enabled=True,
            users=[RBACUserBinding(username="alice", roles=["viewer"])],
        )
        client = SentinelClient(cfg)
        app = create_dashboard_app(client)
        with TestClient(app) as test_client:
            resp = test_client.get("/api/drift", headers=_basic("alice", "hunter2"))
            assert resp.status_code == 200

    def test_rbac_enabled_unknown_role_cannot_read(self, tmp_path: Path) -> None:
        # Configure a user whose role isn't in the default permissions
        # mapping at all → empty permission set → 403 on any guarded route.
        cfg = SentinelConfig(
            model=ModelConfig(name="rbac_test_model", domain="tabular"),
            drift=DriftConfig(
                data=DataDriftConfig(method="psi", threshold=0.2, window="7d"),
            ),
            alerts=AlertsConfig(),
            audit=AuditConfig(storage="local", path=str(tmp_path / "audit")),
            dashboard=DashboardConfig(
                enabled=True,
                server=DashboardServerConfig(
                    auth="basic",
                    basic_auth_username="alice",
                    basic_auth_password="hunter2",  # type: ignore[arg-type]
                    rbac=RBACConfig(
                        enabled=True,
                        default_role="nobody",
                        users=[RBACUserBinding(username="alice", roles=["nobody"])],
                        role_permissions={
                            "viewer": ["drift.read"],
                            "nobody": [],
                        },
                        role_hierarchy=["nobody", "viewer", "admin"],
                    ),
                ),
            ),
        )
        client = SentinelClient(cfg)
        app = create_dashboard_app(client)
        with TestClient(app) as test_client:
            resp = test_client.get("/api/drift", headers=_basic("alice", "hunter2"))
            assert resp.status_code == 403
            # Free routes still pass.
            assert test_client.get("/", headers=_basic("alice", "hunter2")).status_code == 200
            assert (
                test_client.get("/api/health", headers=_basic("alice", "hunter2")).status_code
                == 200
            )

    def test_role_inheritance_through_app(self, tmp_path: Path) -> None:
        cfg = _build_config(
            tmp_path,
            rbac_enabled=True,
            users=[RBACUserBinding(username="alice", roles=["operator"])],
        )
        client = SentinelClient(cfg)
        app = create_dashboard_app(client)
        with TestClient(app) as test_client:
            # Operator inherits viewer perms → drift.read should pass.
            resp = test_client.get("/api/drift", headers=_basic("alice", "hunter2"))
            assert resp.status_code == 200
