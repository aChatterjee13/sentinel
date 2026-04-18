"""AgentOps page + API tests — inject a synthetic trace."""

from __future__ import annotations

import pytest

from sentinel import SentinelClient
from sentinel.config.schema import (
    AgentOpsConfig,
    AlertsConfig,
    AuditConfig,
    DataDriftConfig,
    DriftConfig,
    ModelConfig,
    SentinelConfig,
)


@pytest.fixture
def agentops_client(tmp_path):
    """A SentinelClient with AgentOps enabled and one trace recorded."""
    cfg = SentinelConfig(
        model=ModelConfig(name="agent_test", domain="tabular"),
        drift=DriftConfig(
            data=DataDriftConfig(method="psi", threshold=0.2, window="7d"),
        ),
        alerts=AlertsConfig(),
        audit=AuditConfig(storage="local", path=str(tmp_path / "audit")),
        agentops=AgentOpsConfig(enabled=True),
    )
    client = SentinelClient(cfg)
    tracer = client.agentops.tracer
    with tracer.trace("planner_agent"):
        with tracer.span("plan"):
            pass
        with tracer.span("tool_call: search", kind="tool_call"):
            pass
        with tracer.span("synthesise"):
            pass
    return client


@pytest.fixture
def agentops_test_client(agentops_client):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from sentinel.dashboard.server import create_dashboard_app

    return TestClient(create_dashboard_app(agentops_client))


class TestAgentOpsRoutes:
    def test_traces_page_renders(self, agentops_test_client) -> None:
        resp = agentops_test_client.get("/agentops/traces")
        assert resp.status_code == 200
        assert "Recent agent traces" in resp.text

    def test_api_traces_returns_recent(self, agentops_test_client) -> None:
        resp = agentops_test_client.get("/api/agentops/traces")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is True
        assert len(body["traces"]) >= 1

    def test_trace_detail_renders(
        self, agentops_test_client, agentops_client: SentinelClient
    ) -> None:
        last = agentops_client.agentops.tracer.get_last_trace()
        assert last is not None
        resp = agentops_test_client.get(f"/agentops/traces/{last.trace_id}")
        assert resp.status_code == 200
        assert last.trace_id in resp.text

    def test_trace_detail_404(self, agentops_test_client) -> None:
        resp = agentops_test_client.get("/agentops/traces/no-such-trace")
        assert resp.status_code == 404

    def test_api_trace_detail_404(self, agentops_test_client) -> None:
        resp = agentops_test_client.get("/api/traces/no-such-trace")
        assert resp.status_code == 404

    def test_agents_page_renders(self, agentops_test_client) -> None:
        resp = agentops_test_client.get("/agentops/agents")
        assert resp.status_code == 200
        assert "Agent registry" in resp.text

    def test_tools_page_renders(self, agentops_test_client) -> None:
        resp = agentops_test_client.get("/agentops/tools")
        assert resp.status_code == 200
        assert "Tool audit" in resp.text
