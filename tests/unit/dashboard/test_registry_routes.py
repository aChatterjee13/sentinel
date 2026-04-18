"""Registry page + API tests."""

from __future__ import annotations

from sentinel import SentinelClient
from sentinel.dashboard.state import DashboardState
from sentinel.dashboard.views import registry as registry_view


class TestRegistryView:
    def test_list_models_empty(self, dashboard_state: DashboardState) -> None:
        data = registry_view.list_models(dashboard_state)
        assert "models" in data
        assert data["active_model"] == "dash_test_model"

    def test_list_models_after_register(
        self, dashboard_state: DashboardState, seeded_client: SentinelClient
    ) -> None:
        seeded_client.registry.register(
            seeded_client.model_name,
            "1.0",
            framework="sklearn",
        )
        data = registry_view.list_models(dashboard_state)
        names = [m["name"] for m in data["models"]]
        assert "dash_test_model" in names
        model_entry = next(m for m in data["models"] if m["name"] == "dash_test_model")
        assert "1.0" in model_entry["versions"]

    def test_detail_returns_none_for_unknown(self, dashboard_state: DashboardState) -> None:
        assert registry_view.detail(dashboard_state, "ghost", "1.0") is None


class TestRegistryRoutes:
    def test_registry_page_renders(self, client) -> None:
        resp = client.get("/registry")
        assert resp.status_code == 200
        assert "Model registry" in resp.text

    def test_api_registry_returns_models(self, client) -> None:
        resp = client.get("/api/registry")
        assert resp.status_code == 200
        body = resp.json()
        assert "models" in body

    def test_api_registry_detail_404(self, client) -> None:
        resp = client.get("/api/registry/ghost/1.0")
        assert resp.status_code == 404

    def test_registry_detail_after_register(self, client, seeded_client: SentinelClient) -> None:
        seeded_client.registry.register(
            seeded_client.model_name,
            "2.1",
            framework="xgboost",
        )
        resp = client.get(f"/registry/{seeded_client.model_name}/2.1")
        assert resp.status_code == 200
        assert "2.1" in resp.text
