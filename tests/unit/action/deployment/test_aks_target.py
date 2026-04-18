"""Tests for ``AKSDeploymentTarget`` with a fake ``kubernetes`` client."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from sentinel.core.exceptions import DeploymentError


def _install_fake_k8s(
    monkeypatch: pytest.MonkeyPatch,
    *,
    patch_raises: Exception | None = None,
    status_raises: Exception | None = None,
    deployment_status: MagicMock | None = None,
) -> MagicMock:
    """Install fake ``kubernetes`` module with ``client`` and ``config``."""
    apps_v1 = MagicMock(name="AppsV1Api")
    if patch_raises is not None:
        apps_v1.patch_namespaced_deployment_scale.side_effect = patch_raises

    if status_raises is not None:
        apps_v1.read_namespaced_deployment_status.side_effect = status_raises
    else:
        if deployment_status is None:
            deployment_status = MagicMock(name="DeploymentStatus")
            deployment_status.spec = MagicMock(replicas=5)
            deployment_status.status = MagicMock(ready_replicas=5)
        apps_v1.read_namespaced_deployment_status.return_value = deployment_status

    client_mod = types.ModuleType("kubernetes.client")
    client_mod.AppsV1Api = MagicMock(return_value=apps_v1)  # type: ignore[attr-defined]

    config_mod = types.ModuleType("kubernetes.config")
    config_mod.load_kube_config = MagicMock()  # type: ignore[attr-defined]
    config_mod.load_incluster_config = MagicMock()  # type: ignore[attr-defined]

    kubernetes_mod = types.ModuleType("kubernetes")
    kubernetes_mod.client = client_mod  # type: ignore[attr-defined]
    kubernetes_mod.config = config_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "kubernetes", kubernetes_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.client", client_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.config", config_mod)

    return apps_v1


def _fresh_module() -> Any:
    if "sentinel.action.deployment.targets.aks" in sys.modules:
        del sys.modules["sentinel.action.deployment.targets.aks"]
    import sentinel.action.deployment.targets.aks as mod

    return mod


def _make_target(mod: Any, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "namespace": "ml",
        "service_name": "fraud-svc",
        "replicas_total": 10,
    }
    kwargs.update(overrides)
    return mod.AKSDeploymentTarget(**kwargs)


class TestConstruction:
    def test_missing_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "kubernetes", None)
        monkeypatch.setitem(sys.modules, "kubernetes.client", None)
        monkeypatch.setitem(sys.modules, "kubernetes.config", None)
        mod = _fresh_module()
        with pytest.raises(DeploymentError, match="k8s extra"):
            mod.AKSDeploymentTarget(namespace="ml", service_name="s")


class TestComputeReplicas:
    def test_exact_split(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_k8s(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        replicas = target._compute_replicas({"v1": 50, "v2": 50})
        assert replicas == {"v1": 5, "v2": 5}

    def test_uneven_uses_largest_remainder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_k8s(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        replicas = target._compute_replicas({"v1": 33, "v2": 33, "v3": 34})
        # 10 pods distributed over 33/33/34 — must sum to exactly 10.
        assert sum(replicas.values()) == 10

    def test_total_always_matches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_k8s(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod, replicas_total=20)
        replicas = target._compute_replicas({"v1": 25, "v2": 75})
        assert sum(replicas.values()) == 20
        assert replicas == {"v1": 5, "v2": 15}


class TestSetTrafficSplit:
    def test_patches_each_deployment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        apps = _install_fake_k8s(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        target.set_traffic_split("fraud", {"2.3.0": 50, "2.3.1": 50})
        assert apps.patch_namespaced_deployment_scale.call_count == 2

    def test_weights_must_sum_to_100(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_k8s(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        with pytest.raises(DeploymentError, match="must sum to 100"):
            target.set_traffic_split("fraud", {"v1": 40, "v2": 40})

    def test_patch_error_wraps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_k8s(monkeypatch, patch_raises=RuntimeError("rbac"))
        mod = _fresh_module()
        target = _make_target(mod)
        with pytest.raises(DeploymentError, match="aks set_traffic_split"):
            target.set_traffic_split("fraud", {"v1": 100})


class TestHealthCheck:
    def test_ready_equals_desired(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_k8s(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        assert target.health_check("fraud", "2.3.1") is True

    def test_ready_less_than_desired(self, monkeypatch: pytest.MonkeyPatch) -> None:
        status = MagicMock(name="DeploymentStatus")
        status.spec = MagicMock(replicas=5)
        status.status = MagicMock(ready_replicas=3)
        _install_fake_k8s(monkeypatch, deployment_status=status)
        mod = _fresh_module()
        target = _make_target(mod)
        assert target.health_check("fraud", "2.3.1") is False

    def test_zero_desired_is_healthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        status = MagicMock(name="DeploymentStatus")
        status.spec = MagicMock(replicas=0)
        status.status = MagicMock(ready_replicas=0)
        _install_fake_k8s(monkeypatch, deployment_status=status)
        mod = _fresh_module()
        target = _make_target(mod)
        assert target.health_check("fraud", "2.3.0") is True

    def test_api_error_is_unhealthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_k8s(monkeypatch, status_raises=RuntimeError("not found"))
        mod = _fresh_module()
        target = _make_target(mod)
        assert target.health_check("fraud", "2.3.1") is False


class TestRollback:
    def test_rollback_sets_full_replicas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        apps = _install_fake_k8s(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        target.rollback_to("fraud", "2.3.0")
        # Single call because weights are {'2.3.0': 100}
        apps.patch_namespaced_deployment_scale.assert_called_once()
        kwargs = apps.patch_namespaced_deployment_scale.call_args.kwargs
        assert kwargs["name"] == "fraud-2.3.0"
        assert kwargs["body"]["spec"]["replicas"] == 10
