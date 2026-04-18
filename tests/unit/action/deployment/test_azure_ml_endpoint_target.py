"""Tests for ``AzureMLEndpointTarget`` with a fake ``azure.ai.ml``."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from sentinel.core.exceptions import DeploymentError


def _install_fake_azure_ml(
    monkeypatch: pytest.MonkeyPatch,
    *,
    endpoint: MagicMock | None = None,
    deployment: MagicMock | None = None,
    deployment_get_raises: Exception | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Install fake ``azure.ai.ml`` + ``azure.identity`` modules."""
    if endpoint is None:
        endpoint = MagicMock(name="OnlineEndpoint")
        endpoint.traffic = {}
    if deployment is None:
        deployment = MagicMock(name="OnlineDeployment")
        deployment.provisioning_state = "Succeeded"

    online_endpoints = MagicMock(name="OnlineEndpointsOps")
    online_endpoints.get.return_value = endpoint
    poller = MagicMock(name="Poller")
    poller.result.return_value = endpoint
    online_endpoints.begin_create_or_update.return_value = poller

    online_deployments = MagicMock(name="OnlineDeploymentsOps")
    if deployment_get_raises is not None:
        online_deployments.get.side_effect = deployment_get_raises
    else:
        online_deployments.get.return_value = deployment

    ml_client = MagicMock(name="MLClient")
    ml_client.online_endpoints = online_endpoints
    ml_client.online_deployments = online_deployments

    ml_mod = types.ModuleType("azure.ai.ml")
    ml_mod.MLClient = MagicMock(return_value=ml_client)  # type: ignore[attr-defined]

    ai_mod = types.ModuleType("azure.ai")
    azure_mod = types.ModuleType("azure")
    identity_mod = types.ModuleType("azure.identity")
    identity_mod.DefaultAzureCredential = MagicMock(  # type: ignore[attr-defined]
        name="DefaultAzureCredential"
    )

    monkeypatch.setitem(sys.modules, "azure", azure_mod)
    monkeypatch.setitem(sys.modules, "azure.ai", ai_mod)
    monkeypatch.setitem(sys.modules, "azure.ai.ml", ml_mod)
    monkeypatch.setitem(sys.modules, "azure.identity", identity_mod)

    return ml_client, endpoint


def _fresh_module() -> Any:
    if "sentinel.action.deployment.targets.azure_ml_endpoint" in sys.modules:
        del sys.modules["sentinel.action.deployment.targets.azure_ml_endpoint"]
    import sentinel.action.deployment.targets.azure_ml_endpoint as mod

    return mod


def _make_target(mod: Any) -> Any:
    return mod.AzureMLEndpointTarget(
        endpoint_name="fraud-endpoint",
        subscription_id="sub",
        resource_group="rg",
        workspace_name="ws",
    )


class TestConstruction:
    def test_missing_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "azure.ai.ml", None)
        monkeypatch.setitem(sys.modules, "azure.identity", None)
        mod = _fresh_module()
        with pytest.raises(DeploymentError, match="azure extra"):
            mod.AzureMLEndpointTarget(
                endpoint_name="e",
                subscription_id="s",
                resource_group="r",
                workspace_name="w",
            )


class TestSetTrafficSplit:
    def test_updates_traffic_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ml_client, endpoint = _install_fake_azure_ml(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        target.set_traffic_split("fraud", {"2.3.0": 50, "2.3.1": 50})
        ml_client.online_endpoints.begin_create_or_update.assert_called_once()
        assert endpoint.traffic == {"fraud-2.3.0": 50, "fraud-2.3.1": 50}

    def test_weights_must_sum_to_100(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_ml(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        with pytest.raises(DeploymentError, match="must sum to 100"):
            target.set_traffic_split("fraud", {"v1": 30, "v2": 30})

    def test_sdk_error_wraps_to_deployment_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ml_client, _ = _install_fake_azure_ml(monkeypatch)
        ml_client.online_endpoints.begin_create_or_update.side_effect = RuntimeError("quota")
        mod = _fresh_module()
        target = _make_target(mod)
        with pytest.raises(DeploymentError, match="set_traffic_split failed"):
            target.set_traffic_split("fraud", {"v1": 100})


class TestHealthCheck:
    def test_succeeded_is_healthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_ml(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        assert target.health_check("fraud", "2.3.1") is True

    def test_provisioning_failed_is_unhealthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad_deployment = MagicMock(name="BadDeployment")
        bad_deployment.provisioning_state = "Failed"
        _install_fake_azure_ml(monkeypatch, deployment=bad_deployment)
        mod = _fresh_module()
        target = _make_target(mod)
        assert target.health_check("fraud", "2.3.1") is False

    def test_get_raises_is_unhealthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_ml(monkeypatch, deployment_get_raises=RuntimeError("not found"))
        mod = _fresh_module()
        target = _make_target(mod)
        assert target.health_check("fraud", "2.3.1") is False


class TestRollback:
    def test_rollback_sets_full_traffic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ml_client, endpoint = _install_fake_azure_ml(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        target.rollback_to("fraud", "2.3.0")
        assert endpoint.traffic == {"fraud-2.3.0": 100}


class TestDescribe:
    def test_describe_returns_traffic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, endpoint = _install_fake_azure_ml(monkeypatch)
        endpoint.traffic = {"fraud-2.3.0": 100}
        mod = _fresh_module()
        target = _make_target(mod)
        info = target.describe("fraud")
        assert info["target"] == "azure_ml_endpoint"
        assert info["endpoint"] == "fraud-endpoint"
        assert info["traffic"] == {"fraud-2.3.0": 100}
