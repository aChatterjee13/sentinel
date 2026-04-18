"""Overview homepage smoke tests — page render + JSON parity."""

from __future__ import annotations

from sentinel.dashboard.state import DashboardState
from sentinel.dashboard.views import overview as overview_view


class TestOverviewView:
    def test_build_returns_status_block(self, dashboard_state: DashboardState) -> None:
        data = overview_view.build(dashboard_state)
        assert "status" in data
        assert data["status"]["model"] == "dash_test_model"
        assert data["domain"] == "tabular"

    def test_build_includes_recent_audit(self, dashboard_state: DashboardState) -> None:
        data = overview_view.build(dashboard_state)
        # seeded_client logged 3 audit events
        assert len(data["recent_audit"]) >= 3
        types = {ev["event_type"] for ev in data["recent_audit"]}
        assert "drift_checked" in types
        assert "model_registered" in types

    def test_compliance_frameworks_default_empty(self, dashboard_state: DashboardState) -> None:
        data = overview_view.build(dashboard_state)
        assert data["compliance_frameworks"] == []


class TestOverviewRoutes:
    def test_overview_page_renders(self, client) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "dash_test_model" in resp.text
        assert "Overview" in resp.text

    def test_api_overview_returns_json(self, client) -> None:
        resp = client.get("/api/overview")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"]["model"] == "dash_test_model"
        assert isinstance(body["recent_audit"], list)

    def test_api_health_endpoint(self, client) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model"] == "dash_test_model"
