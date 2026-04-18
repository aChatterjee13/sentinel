"""Tests for the dashboard's HTTP Basic auth dependency."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from sentinel import SentinelClient
from sentinel.config.schema import (
    AlertsConfig,
    AuditConfig,
    DashboardConfig,
    DashboardServerConfig,
    DataDriftConfig,
    DriftConfig,
    ModelConfig,
    SentinelConfig,
)
from sentinel.dashboard.server import build_basic_auth_dependency, create_dashboard_app


def _basic_header(user: str, password: str) -> str:
    raw = f"{user}:{password}".encode()
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _make_config(
    *, auth: str = "basic", username: str = "admin", password: str = "hunter2"
) -> SentinelConfig:
    return SentinelConfig(
        model=ModelConfig(name="auth_test", domain="tabular"),
        drift=DriftConfig(
            data=DataDriftConfig(method="psi", threshold=0.2, window="7d"),
        ),
        alerts=AlertsConfig(),
        audit=AuditConfig(storage="local"),
        dashboard=DashboardConfig(
            enabled=True,
            server=DashboardServerConfig(
                auth=auth,  # type: ignore[arg-type]
                basic_auth_username=username,
                basic_auth_password=password,
            ),
        ),
    )


@pytest.fixture
def basic_auth_client(tmp_path: Path) -> TestClient:
    cfg = _make_config()
    cfg.audit.path = str(tmp_path / "audit")
    client = SentinelClient(cfg)
    app = create_dashboard_app(client)
    return TestClient(app)


@pytest.fixture
def no_auth_client(tmp_path: Path) -> TestClient:
    cfg = _make_config(auth="none")
    cfg.audit.path = str(tmp_path / "audit")
    client = SentinelClient(cfg)
    app = create_dashboard_app(client)
    return TestClient(app)


class TestBasicAuthEnabled:
    def test_missing_header_returns_401(self, basic_auth_client: TestClient) -> None:
        r = basic_auth_client.get("/")
        assert r.status_code == 401
        assert r.headers.get("www-authenticate", "").lower().startswith("basic")

    def test_wrong_password_returns_401(self, basic_auth_client: TestClient) -> None:
        r = basic_auth_client.get("/", headers={"Authorization": _basic_header("admin", "wrong")})
        assert r.status_code == 401

    def test_wrong_username_returns_401(self, basic_auth_client: TestClient) -> None:
        r = basic_auth_client.get(
            "/", headers={"Authorization": _basic_header("intruder", "hunter2")}
        )
        assert r.status_code == 401

    def test_correct_credentials_accepted(self, basic_auth_client: TestClient) -> None:
        r = basic_auth_client.get("/", headers={"Authorization": _basic_header("admin", "hunter2")})
        # 200 (page rendered) or any non-401 result is acceptable —
        # we're testing the guard, not the page templates.
        assert r.status_code != 401

    def test_malformed_header_returns_401(self, basic_auth_client: TestClient) -> None:
        r = basic_auth_client.get("/", headers={"Authorization": "Bearer abc.def.ghi"})
        assert r.status_code == 401

    def test_garbage_basic_payload_returns_401(self, basic_auth_client: TestClient) -> None:
        r = basic_auth_client.get("/", headers={"Authorization": "Basic !!!not-base64!!!"})
        assert r.status_code == 401


class TestNoAuth:
    def test_no_auth_bypasses_guard(self, no_auth_client: TestClient) -> None:
        r = no_auth_client.get("/")
        # We don't pin the exact status — the page may 200 or redirect —
        # but it must not be 401.
        assert r.status_code != 401


class TestBuildDependency:
    def test_no_auth_returns_noop(self) -> None:
        cfg = DashboardServerConfig(auth="none")
        dep = build_basic_auth_dependency(cfg)
        assert dep is not None

    def test_basic_auth_returns_callable(self) -> None:
        cfg = DashboardServerConfig(
            auth="basic",
            basic_auth_username="u",
            basic_auth_password="p",
        )
        dep = build_basic_auth_dependency(cfg)
        assert callable(dep)

    def test_missing_credentials_raises_validation_error(self) -> None:
        # When auth=basic but no creds are provided, validation rejects it.
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="basic_auth_username and basic_auth_password"):
            DashboardServerConfig(auth="basic")
