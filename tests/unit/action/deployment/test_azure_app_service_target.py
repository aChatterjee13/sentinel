"""Tests for ``AzureAppServiceTarget`` with a fake ``azure.mgmt.web``."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from sentinel.core.exceptions import DeploymentError


def _install_fake_azure_web(
    monkeypatch: pytest.MonkeyPatch,
    *,
    swap_raises: Exception | None = None,
) -> MagicMock:
    """Install fake ``azure.mgmt.web`` + ``azure.identity`` modules."""
    web_apps = MagicMock(name="WebAppsOps")
    poller = MagicMock(name="Poller")
    poller.result.return_value = None
    if swap_raises is not None:
        web_apps.begin_swap_slot.side_effect = swap_raises
    else:
        web_apps.begin_swap_slot.return_value = poller

    mgmt_client = MagicMock(name="WebSiteManagementClient")
    mgmt_client.web_apps = web_apps

    web_mod = types.ModuleType("azure.mgmt.web")
    web_mod.WebSiteManagementClient = MagicMock(  # type: ignore[attr-defined]
        return_value=mgmt_client
    )
    mgmt_mod = types.ModuleType("azure.mgmt")
    azure_mod = types.ModuleType("azure")
    identity_mod = types.ModuleType("azure.identity")
    identity_mod.DefaultAzureCredential = MagicMock(  # type: ignore[attr-defined]
        name="DefaultAzureCredential"
    )

    monkeypatch.setitem(sys.modules, "azure", azure_mod)
    monkeypatch.setitem(sys.modules, "azure.mgmt", mgmt_mod)
    monkeypatch.setitem(sys.modules, "azure.mgmt.web", web_mod)
    monkeypatch.setitem(sys.modules, "azure.identity", identity_mod)

    return mgmt_client


def _fresh_module() -> Any:
    if "sentinel.action.deployment.targets.azure_app_service" in sys.modules:
        del sys.modules["sentinel.action.deployment.targets.azure_app_service"]
    import sentinel.action.deployment.targets.azure_app_service as mod

    return mod


def _make_target(mod: Any) -> Any:
    return mod.AzureAppServiceTarget(
        subscription_id="sub",
        resource_group="rg",
        site_name="fraud-app",
    )


class TestConstruction:
    def test_missing_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "azure.mgmt.web", None)
        monkeypatch.setitem(sys.modules, "azure.identity", None)
        mod = _fresh_module()
        with pytest.raises(DeploymentError, match="azure extra"):
            mod.AzureAppServiceTarget(
                subscription_id="s",
                resource_group="r",
                site_name="x",
            )


class TestSetTrafficSplit:
    def test_swaps_slots_on_atomic_100(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgmt = _install_fake_azure_web(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        target.set_traffic_split("fraud", {"2.3.1": 100, "2.3.0": 0})
        mgmt.web_apps.begin_swap_slot.assert_called_once()
        kwargs = mgmt.web_apps.begin_swap_slot.call_args.kwargs
        assert kwargs["slot"] == "staging"
        assert kwargs["target_slot"] == "production"

    def test_partial_split_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_web(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        with pytest.raises(DeploymentError, match="atomic 0/100"):
            target.set_traffic_split("fraud", {"2.3.0": 75, "2.3.1": 25})

    def test_weights_must_sum_to_100(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_web(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        with pytest.raises(DeploymentError, match="must sum to 100"):
            target.set_traffic_split("fraud", {"v1": 50, "v2": 40})

    def test_sdk_error_wraps_to_deployment_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_web(monkeypatch, swap_raises=RuntimeError("boom"))
        mod = _fresh_module()
        target = _make_target(mod)
        with pytest.raises(DeploymentError, match="swap_slot failed"):
            target.set_traffic_split("fraud", {"v1": 100})


class TestHealthCheck:
    def test_2xx_is_healthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_web(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        fake_resp = MagicMock(name="Resp")
        fake_resp.status_code = 200
        monkeypatch.setattr(mod.httpx, "get", lambda *a, **k: fake_resp)
        assert target.health_check("fraud", "2.3.1") is True

    def test_5xx_is_unhealthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_web(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        fake_resp = MagicMock(name="Resp")
        fake_resp.status_code = 503
        monkeypatch.setattr(mod.httpx, "get", lambda *a, **k: fake_resp)
        assert target.health_check("fraud", "2.3.1") is False

    def test_network_error_is_unhealthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_web(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)

        def _raise(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("timeout")

        monkeypatch.setattr(mod.httpx, "get", _raise)
        assert target.health_check("fraud", "2.3.1") is False


class TestRollback:
    def test_rollback_swaps_slots_reverse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mgmt = _install_fake_azure_web(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        target.rollback_to("fraud", "2.3.0")
        kwargs = mgmt.web_apps.begin_swap_slot.call_args.kwargs
        assert kwargs["slot"] == "production"
        assert kwargs["target_slot"] == "staging"
