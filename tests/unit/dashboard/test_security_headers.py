"""Tests for the dashboard security headers middleware."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel import SentinelClient
from sentinel.config.schema import (
    AlertsConfig,
    AuditConfig,
    CSPConfig,
    DashboardConfig,
    DashboardServerConfig,
    DataDriftConfig,
    DriftConfig,
    ModelConfig,
    SentinelConfig,
)
from sentinel.dashboard.security.headers import (
    DEFAULT_CSP,
    SecurityHeadersMiddleware,
)

pytest.importorskip("fastapi")
pytest.importorskip("jinja2")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sentinel.dashboard.server import create_dashboard_app


def _build_minimal_app(csp: CSPConfig | None = None) -> FastAPI:
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware, csp=csp)

    @app.get("/")
    def root() -> dict[str, str]:
        return {"ok": "yes"}

    return app


class TestSecurityHeadersMiddleware:
    def test_baseline_headers_present(self) -> None:
        client = TestClient(_build_minimal_app())
        resp = client.get("/")
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "Permissions-Policy" in resp.headers

    def test_default_csp_applied(self) -> None:
        client = TestClient(_build_minimal_app())
        resp = client.get("/")
        assert "Content-Security-Policy" in resp.headers
        assert resp.headers["Content-Security-Policy"] == DEFAULT_CSP

    def test_custom_csp_applied(self) -> None:
        custom = "default-src 'none'; script-src 'self'"
        client = TestClient(_build_minimal_app(csp=CSPConfig(enabled=True, policy=custom)))
        resp = client.get("/")
        assert resp.headers["Content-Security-Policy"] == custom

    def test_disabled_csp_omits_header(self) -> None:
        client = TestClient(_build_minimal_app(csp=CSPConfig(enabled=False)))
        resp = client.get("/")
        assert "Content-Security-Policy" not in resp.headers

    def test_hsts_only_on_https(self) -> None:
        client = TestClient(_build_minimal_app())
        # Plain HTTP request → no HSTS.
        resp = client.get("/")
        assert "Strict-Transport-Security" not in resp.headers

    def test_dashboard_app_emits_security_headers(self, tmp_path: Path) -> None:
        cfg = SentinelConfig(
            model=ModelConfig(name="hdr_test_model", domain="tabular"),
            drift=DriftConfig(
                data=DataDriftConfig(method="psi", threshold=0.2, window="7d"),
            ),
            alerts=AlertsConfig(),
            audit=AuditConfig(storage="local", path=str(tmp_path / "audit")),
            dashboard=DashboardConfig(
                enabled=True,
                server=DashboardServerConfig(),
            ),
        )
        client = SentinelClient(cfg)
        app = create_dashboard_app(client)
        with TestClient(app) as test_client:
            resp = test_client.get("/api/health")
            assert resp.headers.get("X-Frame-Options") == "DENY"
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"
            assert "Content-Security-Policy" in resp.headers
