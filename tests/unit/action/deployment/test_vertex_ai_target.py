"""Tests for ``VertexAIEndpointTarget`` with a fake ``google.cloud.aiplatform``."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from sentinel.core.exceptions import DeploymentError


def _install_fake_gcp(
    monkeypatch: pytest.MonkeyPatch,
    *,
    endpoint: MagicMock | None = None,
    endpoint_list_returns: list[Any] | None = None,
) -> MagicMock:
    """Install fake ``google.cloud.aiplatform`` and return the mock module."""
    aiplatform = MagicMock(name="aiplatform")

    if endpoint_list_returns is not None:
        aiplatform.Endpoint.list.return_value = endpoint_list_returns
    elif endpoint is not None:
        aiplatform.Endpoint.list.return_value = [endpoint]
    else:
        ep = MagicMock(name="Endpoint")
        ep.gca_resource.deployed_models = []
        ep.gca_resource.traffic_split = {}
        aiplatform.Endpoint.list.return_value = [ep]

    google_mod = types.ModuleType("google")
    google_cloud_mod = types.ModuleType("google.cloud")
    google_cloud_mod.aiplatform = aiplatform  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.cloud", google_cloud_mod)
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", aiplatform)

    return aiplatform


def _fresh_module() -> Any:
    mod_key = "sentinel.action.deployment.targets.vertex_ai"
    if mod_key in sys.modules:
        del sys.modules[mod_key]
    import sentinel.action.deployment.targets.vertex_ai as mod

    return mod


def _make_target(mod: Any) -> Any:
    return mod.VertexAIEndpointTarget(
        endpoint_name="test-ep",
        project="test-project",
    )


# ── Construction ──────────────────────────────────────────────────


class TestConstruction:
    def test_missing_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", None)
        monkeypatch.setitem(sys.modules, "google.cloud", None)
        mod = _fresh_module()
        with pytest.raises(DeploymentError, match="google-cloud-aiplatform"):
            mod.VertexAIEndpointTarget(
                endpoint_name="e", project="p"
            )

    def test_stores_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gcp(monkeypatch)
        mod = _fresh_module()
        target = mod.VertexAIEndpointTarget(
            endpoint_name="my-ep",
            project="my-proj",
            location="europe-west4",
        )
        assert target.name == "vertex_ai_endpoint"
        assert target._endpoint_name == "my-ep"
        assert target._project == "my-proj"
        assert target._location == "europe-west4"


# ── Traffic split ─────────────────────────────────────────────────


class TestSetTrafficSplit:
    def test_weights_must_sum_to_100(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_gcp(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        with pytest.raises(DeploymentError, match="must sum to 100"):
            target.set_traffic_split("fraud", {"1.0.0": 50})

    def test_updates_traffic_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ep = MagicMock(name="Endpoint")
        dm1 = MagicMock()
        dm1.id = "dm-1"
        dm1.display_name = "fraud-1.0.0"
        dm2 = MagicMock()
        dm2.id = "dm-2"
        dm2.display_name = "fraud-2.0.0"
        ep.gca_resource.deployed_models = [dm1, dm2]
        ep.gca_resource.traffic_split = {"dm-1": 100}

        _install_fake_gcp(monkeypatch, endpoint=ep)
        mod = _fresh_module()
        target = _make_target(mod)
        target.set_traffic_split("fraud", {"1.0.0": 50, "2.0.0": 50})
        ep.update.assert_called_once()
        call_kwargs = ep.update.call_args.kwargs
        assert call_kwargs["traffic_split"] == {"dm-1": 50, "dm-2": 50}

    def test_api_error_wraps_to_deployment_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ep = MagicMock(name="Endpoint")
        ep.gca_resource.deployed_models = []
        ep.update.side_effect = RuntimeError("quota")
        _install_fake_gcp(monkeypatch, endpoint=ep)
        mod = _fresh_module()
        target = _make_target(mod)
        # No deployed models means no update call — so trigger via
        # a different mechanism: make list fail after the sum-check.
        aip = sys.modules["google.cloud.aiplatform"]
        aip.Endpoint.list.side_effect = RuntimeError("API down")
        with pytest.raises(DeploymentError, match="set_traffic_split failed"):
            target.set_traffic_split("fraud", {"1.0.0": 100})


# ── Health check ──────────────────────────────────────────────────


class TestHealthCheck:
    def test_healthy_when_version_deployed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ep = MagicMock(name="Endpoint")
        dm = MagicMock()
        dm.display_name = "fraud-2.3.1"
        ep.gca_resource.deployed_models = [dm]
        _install_fake_gcp(monkeypatch, endpoint=ep)
        mod = _fresh_module()
        target = _make_target(mod)
        assert target.health_check("fraud", "2.3.1") is True

    def test_unhealthy_when_version_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ep = MagicMock(name="Endpoint")
        ep.gca_resource.deployed_models = []
        _install_fake_gcp(monkeypatch, endpoint=ep)
        mod = _fresh_module()
        target = _make_target(mod)
        assert target.health_check("fraud", "2.3.1") is False

    def test_unhealthy_on_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_gcp(monkeypatch)
        aip = sys.modules["google.cloud.aiplatform"]
        aip.Endpoint.list.side_effect = RuntimeError("network")
        mod = _fresh_module()
        target = _make_target(mod)
        assert target.health_check("fraud", "2.3.1") is False


# ── Rollback ──────────────────────────────────────────────────────


class TestRollback:
    def test_rollback_sets_full_traffic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ep = MagicMock(name="Endpoint")
        dm = MagicMock()
        dm.id = "dm-1"
        dm.display_name = "fraud-2.3.0"
        ep.gca_resource.deployed_models = [dm]
        ep.gca_resource.traffic_split = {"dm-1": 100}
        _install_fake_gcp(monkeypatch, endpoint=ep)
        mod = _fresh_module()
        target = _make_target(mod)
        target.rollback_to("fraud", "2.3.0")
        ep.update.assert_called_once()
        call_kwargs = ep.update.call_args.kwargs
        assert call_kwargs["traffic_split"] == {"dm-1": 100}


# ── Describe ──────────────────────────────────────────────────────


class TestDescribe:
    def test_describe_returns_deployed_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ep = MagicMock(name="Endpoint")
        dm = MagicMock()
        dm.id = "dm-1"
        dm.display_name = "fraud-1.0.0"
        ep.gca_resource.deployed_models = [dm]
        ep.gca_resource.traffic_split = {"dm-1": 100}
        _install_fake_gcp(monkeypatch, endpoint=ep)
        mod = _fresh_module()
        target = _make_target(mod)
        info = target.describe("fraud")
        assert info["target"] == "vertex_ai_endpoint"
        assert info["endpoint"] == "test-ep"
        assert info["deployed_models"] == {"fraud-1.0.0": 100}

    def test_describe_returns_error_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_gcp(monkeypatch)
        aip = sys.modules["google.cloud.aiplatform"]
        aip.Endpoint.list.side_effect = RuntimeError("gone")
        mod = _fresh_module()
        target = _make_target(mod)
        info = target.describe("fraud")
        assert "error" in info
        assert info["target"] == "vertex_ai_endpoint"
