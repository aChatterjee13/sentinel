"""Drift page + API tests."""

from __future__ import annotations

from sentinel.dashboard.state import DashboardState
from sentinel.dashboard.views import drift as drift_view


class TestDriftView:
    def test_build_returns_seeded_events(self, dashboard_state: DashboardState) -> None:
        data = drift_view.build(dashboard_state)
        assert "events" in data
        assert len(data["events"]) >= 2
        assert data["config"]["method"] == "psi"

    def test_timeseries_returns_aligned_arrays(self, dashboard_state: DashboardState) -> None:
        ts = drift_view.timeseries(dashboard_state)
        assert "timestamps" in ts
        assert "statistics" in ts
        assert "severities" in ts
        assert len(ts["timestamps"]) == len(ts["statistics"]) == len(ts["severities"])
        assert ts["threshold"] == 0.2

    def test_detail_returns_none_for_unknown_id(self, dashboard_state: DashboardState) -> None:
        assert drift_view.detail(dashboard_state, "no-such-report") is None

    def test_parse_iso_handles_invalid(self) -> None:
        assert drift_view.parse_iso(None) is None
        assert drift_view.parse_iso("") is None
        assert drift_view.parse_iso("not-a-date") is None
        assert drift_view.parse_iso("2026-01-15T00:00:00") is not None


class TestDriftRoutes:
    def test_drift_page_renders(self, client) -> None:
        resp = client.get("/drift")
        assert resp.status_code == 200
        assert "Drift events" in resp.text

    def test_api_drift_returns_events(self, client) -> None:
        resp = client.get("/api/drift")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["events"]) >= 2

    def test_api_drift_timeseries(self, client) -> None:
        resp = client.get("/api/drift/timeseries")
        assert resp.status_code == 200
        body = resp.json()
        assert "timestamps" in body
        assert "statistics" in body

    def test_api_drift_detail_404(self, client) -> None:
        resp = client.get("/api/drift/no-such-report")
        assert resp.status_code == 404
