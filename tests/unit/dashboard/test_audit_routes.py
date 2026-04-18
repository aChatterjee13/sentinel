"""Audit page + API tests — verify filter passthrough."""

from __future__ import annotations

from sentinel.dashboard.state import DashboardState
from sentinel.dashboard.views import audit as audit_view


class TestAuditView:
    def test_build_returns_seeded_events(self, dashboard_state: DashboardState) -> None:
        data = audit_view.build(dashboard_state)
        assert data["total"] >= 3
        assert "drift_checked" in data["distinct_event_types"]

    def test_filter_by_event_type(self, dashboard_state: DashboardState) -> None:
        data = audit_view.build(dashboard_state, event_type="drift_checked")
        assert all(ev["event_type"] == "drift_checked" for ev in data["events"])
        assert data["filters"]["event_type"] == "drift_checked"

    def test_limit_caps_results(self, dashboard_state: DashboardState) -> None:
        data = audit_view.build(dashboard_state, limit=1)
        assert len(data["events"]) <= 1


class TestAuditRoutes:
    def test_audit_page_renders(self, client) -> None:
        resp = client.get("/audit")
        assert resp.status_code == 200
        assert "Audit trail" in resp.text

    def test_api_audit_returns_events(self, client) -> None:
        resp = client.get("/api/audit")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 3

    def test_api_audit_filter_event_type(self, client) -> None:
        resp = client.get("/api/audit?event_type=drift_checked")
        assert resp.status_code == 200
        body = resp.json()
        assert all(ev["event_type"] == "drift_checked" for ev in body["events"])
