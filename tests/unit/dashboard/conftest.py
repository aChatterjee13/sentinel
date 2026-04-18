"""Shared fixtures for dashboard route + view tests.

Every test in this directory boots a fresh :class:`SentinelClient` against
a temporary audit directory and a fresh ``DashboardState``. Routes are
exercised via FastAPI's :class:`TestClient` (synchronous, no real port).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from sentinel import SentinelClient
from sentinel.config.schema import (
    AlertsConfig,
    AuditConfig,
    DataDriftConfig,
    DriftConfig,
    ModelConfig,
    SentinelConfig,
)

if TYPE_CHECKING:
    pass

pytest.importorskip("fastapi")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient

from sentinel.dashboard.server import create_dashboard_app
from sentinel.dashboard.state import DashboardState


@pytest.fixture
def dashboard_config(tmp_path: Path) -> SentinelConfig:
    """A minimal SentinelConfig with a writable temp audit directory."""
    return SentinelConfig(
        model=ModelConfig(name="dash_test_model", domain="tabular"),
        drift=DriftConfig(
            data=DataDriftConfig(method="psi", threshold=0.2, window="7d"),
        ),
        alerts=AlertsConfig(),
        audit=AuditConfig(storage="local", path=str(tmp_path / "audit")),
    )


@pytest.fixture
def seeded_client(dashboard_config: SentinelConfig) -> SentinelClient:
    """A SentinelClient with a few synthetic audit events on disk."""
    client = SentinelClient(dashboard_config)
    client.audit.log(
        "drift_checked",
        model_name=client.model_name,
        method="psi",
        is_drifted=True,
        severity="high",
        n_drifted=3,
    )
    client.audit.log(
        "drift_checked",
        model_name=client.model_name,
        method="psi",
        is_drifted=False,
        severity="info",
        n_drifted=0,
    )
    client.audit.log(
        "model_registered",
        model_name=client.model_name,
        model_version="1.0",
    )
    return client


@pytest.fixture
def dashboard_app(seeded_client: SentinelClient):
    """A FastAPI app bound to a seeded SentinelClient."""
    return create_dashboard_app(seeded_client)


@pytest.fixture
def dashboard_state(seeded_client: SentinelClient) -> DashboardState:
    return DashboardState(client=seeded_client)


@pytest.fixture
def client(dashboard_app):
    """Synchronous FastAPI TestClient."""
    return TestClient(dashboard_app)
