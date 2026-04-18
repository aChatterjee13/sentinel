"""Tests for the CSRF double-submit cookie middleware."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel import SentinelClient
from sentinel.config.schema import (
    AlertsConfig,
    AuditConfig,
    CSRFConfig,
    DashboardConfig,
    DashboardServerConfig,
    DataDriftConfig,
    DriftConfig,
    ModelConfig,
    SentinelConfig,
)

pytest.importorskip("fastapi")
pytest.importorskip("jinja2")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sentinel.dashboard.security.csrf import CSRFMiddleware
from sentinel.dashboard.server import create_dashboard_app


def _build_app(*, csrf_enabled: bool = True) -> FastAPI:
    """A standalone FastAPI app with just the CSRF middleware mounted.

    We use a minimal app rather than the full dashboard so the test
    surface stays focused on the CSRF middleware itself.
    """
    app = FastAPI()
    app.add_middleware(CSRFMiddleware, cfg=CSRFConfig(enabled=csrf_enabled))

    @app.get("/")
    def get_root() -> dict[str, str]:
        return {"ok": "yes"}

    @app.post("/write")
    def post_write() -> dict[str, str]:
        return {"ok": "wrote"}

    return app


def _build_full_dashboard(tmp_path: Path) -> FastAPI:
    cfg = SentinelConfig(
        model=ModelConfig(name="csrf_test_model", domain="tabular"),
        drift=DriftConfig(
            data=DataDriftConfig(method="psi", threshold=0.2, window="7d"),
        ),
        alerts=AlertsConfig(),
        audit=AuditConfig(storage="local", path=str(tmp_path / "audit")),
        dashboard=DashboardConfig(
            enabled=True,
            server=DashboardServerConfig(
                csrf=CSRFConfig(enabled=True),
            ),
        ),
    )
    client = SentinelClient(cfg)
    return create_dashboard_app(client)


class TestCSRFMiddleware:
    def test_get_sets_cookie(self) -> None:
        client = TestClient(_build_app())
        resp = client.get("/")
        assert resp.status_code == 200
        assert "sentinel_csrf" in resp.cookies

    def test_get_reuses_existing_cookie(self) -> None:
        client = TestClient(_build_app())
        first = client.get("/")
        first_token = first.cookies.get("sentinel_csrf")
        # The next request should not rotate the existing token.
        second = client.get("/")
        assert second.cookies.get("sentinel_csrf") in (None, first_token)

    def test_post_without_token_is_blocked(self) -> None:
        client = TestClient(_build_app())
        resp = client.post("/write")
        assert resp.status_code == 403
        assert "CSRF" in resp.json()["error"]

    def test_post_with_matching_token_succeeds(self) -> None:
        client = TestClient(_build_app())
        get_resp = client.get("/")
        token = get_resp.cookies.get("sentinel_csrf")
        assert token is not None
        resp = client.post("/write", headers={"X-CSRF-Token": token})
        assert resp.status_code == 200

    def test_post_with_mismatched_token_is_blocked(self) -> None:
        client = TestClient(_build_app())
        client.get("/")  # set the cookie
        resp = client.post("/write", headers={"X-CSRF-Token": "totally-wrong"})
        assert resp.status_code == 403
        assert "mismatch" in resp.json()["error"]

    def test_bearer_request_bypasses_csrf(self) -> None:
        client = TestClient(_build_app())
        # No cookie, no header, but Authorization: Bearer set.
        resp = client.post("/write", headers={"Authorization": "Bearer token"})
        assert resp.status_code == 200

    def test_disabled_middleware_is_passthrough(self) -> None:
        client = TestClient(_build_app(csrf_enabled=False))
        resp = client.post("/write")
        assert resp.status_code == 200
        # No cookie should be set when CSRF is off.
        assert "sentinel_csrf" not in resp.cookies

    def test_safe_methods_dont_require_token(self) -> None:
        client = TestClient(_build_app())
        for method in ("get", "head", "options"):
            resp = getattr(client, method)("/")
            assert resp.status_code in {200, 405}

    def test_full_dashboard_get_sets_cookie(self, tmp_path: Path) -> None:
        app = _build_full_dashboard(tmp_path)
        with TestClient(app) as client:
            resp = client.get("/api/health")
            assert resp.status_code == 200
            assert "sentinel_csrf" in resp.cookies
