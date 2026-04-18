"""Tests for the deployment target registry and LocalDeploymentTarget."""

from __future__ import annotations

import pytest

from sentinel.action.deployment.targets import (
    TARGET_REGISTRY,
    BaseDeploymentTarget,
    LocalDeploymentTarget,
    register_target,
    resolve_target,
)
from sentinel.core.exceptions import DeploymentError


class TestLocalDeploymentTarget:
    def test_name(self) -> None:
        t = LocalDeploymentTarget()
        assert t.name == "local"

    def test_set_traffic_split_is_noop(self) -> None:
        t = LocalDeploymentTarget()
        # Should not raise, should not touch anything.
        t.set_traffic_split("model", {"v1": 50, "v2": 50})

    def test_health_check_always_true(self) -> None:
        t = LocalDeploymentTarget()
        assert t.health_check("model", "v1") is True

    def test_rollback_is_noop(self) -> None:
        t = LocalDeploymentTarget()
        t.rollback_to("model", "v1")  # must not raise

    def test_describe_returns_dict(self) -> None:
        t = LocalDeploymentTarget()
        info = t.describe("model")
        assert info["target"] == "local"
        assert info["model"] == "model"


class TestTargetRegistry:
    def test_local_resolves(self) -> None:
        t = resolve_target("local")
        assert isinstance(t, LocalDeploymentTarget)

    def test_unknown_raises(self) -> None:
        with pytest.raises(DeploymentError, match="unknown deployment target"):
            resolve_target("turbo_encabulator")

    def test_registered_targets(self) -> None:
        assert set(TARGET_REGISTRY) >= {
            "local",
            "azure_ml_endpoint",
            "azure_app_service",
            "aks",
        }

    def test_register_custom(self) -> None:
        class _Fake(BaseDeploymentTarget):
            name = "fake"

            def set_traffic_split(self, model_name: str, weights: dict[str, int]) -> None:
                pass

            def health_check(self, model_name: str, version: str) -> bool:
                return True

            def rollback_to(self, model_name: str, version: str) -> None:
                pass

        register_target("fake", lambda **kwargs: _Fake())
        try:
            t = resolve_target("fake")
            assert isinstance(t, _Fake)
        finally:
            TARGET_REGISTRY.pop("fake", None)

    def test_close_is_optional(self) -> None:
        t = LocalDeploymentTarget()
        t.close()  # default no-op — should not raise
