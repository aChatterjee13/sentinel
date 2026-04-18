"""Tests for ``SageMakerEndpointTarget`` — all boto3 calls mocked.

Uses the same ``sys.modules`` patching technique as the Azure ML
endpoint target tests.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from sentinel.core.exceptions import DeploymentError


def _install_fake_boto3(
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    """Inject a fake ``boto3`` + ``botocore`` into *sys.modules* and return the SM client."""
    sm_client = MagicMock(name="SageMakerClient")

    def client_factory(service: str, **kwargs: Any) -> MagicMock:
        if service == "sagemaker":
            return sm_client
        return MagicMock()  # pragma: no cover

    boto3_mod = types.ModuleType("boto3")
    boto3_mod.client = client_factory  # type: ignore[attr-defined]

    # botocore.config.Config is needed by the target constructor
    botocore_config_mod = types.ModuleType("botocore.config")
    botocore_config_mod.Config = MagicMock(name="BotoConfig")  # type: ignore[attr-defined]
    botocore_mod = types.ModuleType("botocore")
    botocore_mod.config = botocore_config_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)
    monkeypatch.setitem(sys.modules, "botocore", botocore_mod)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config_mod)

    return sm_client


def _fresh_module() -> Any:
    key = "sentinel.action.deployment.targets.sagemaker"
    if key in sys.modules:
        del sys.modules[key]
    import sentinel.action.deployment.targets.sagemaker as mod

    return mod


def _make_target(mod: Any) -> Any:
    return mod.SageMakerEndpointTarget(endpoint_name="test-ep")


# ── Construction ──────────────────────────────────────────────────


class TestConstruction:
    def test_creates_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        assert target.name == "sagemaker_endpoint"
        assert target._endpoint_name == "test-ep"

    def test_missing_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "boto3", None)
        mod = _fresh_module()
        with pytest.raises(DeploymentError, match="aws extra"):
            mod.SageMakerEndpointTarget(endpoint_name="ep")

    def test_region_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(monkeypatch)
        mod = _fresh_module()
        target = mod.SageMakerEndpointTarget(
            endpoint_name="ep", region_name="eu-west-1"
        )
        assert target._sm is not None


# ── Variant naming ────────────────────────────────────────────────


class TestVariantName:
    def test_dots_replaced_with_dashes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_boto3(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        assert target._variant_name("fraud", "2.3.1") == "fraud-2-3-1"


# ── set_traffic_split ─────────────────────────────────────────────


class TestSetTrafficSplit:
    def test_updates_weights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sm = _install_fake_boto3(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        target.set_traffic_split("fraud", {"1.0.0": 90, "2.0.0": 10})
        sm.update_endpoint_weights_and_capacities.assert_called_once()

        call_kwargs = (
            sm.update_endpoint_weights_and_capacities.call_args.kwargs
        )
        assert call_kwargs["EndpointName"] == "test-ep"
        variants = {
            w["VariantName"]: w["DesiredWeight"]
            for w in call_kwargs["DesiredWeightsAndCapacities"]
        }
        assert variants["fraud-1-0-0"] == pytest.approx(0.9)
        assert variants["fraud-2-0-0"] == pytest.approx(0.1)

    def test_weights_must_sum_to_100(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_boto3(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        with pytest.raises(DeploymentError, match="must sum to 100"):
            target.set_traffic_split("fraud", {"v1": 50})

    def test_sdk_error_wraps_to_deployment_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sm = _install_fake_boto3(monkeypatch)
        sm.update_endpoint_weights_and_capacities.side_effect = RuntimeError(
            "throttled"
        )
        mod = _fresh_module()
        target = _make_target(mod)
        with pytest.raises(DeploymentError, match="set_traffic_split failed"):
            target.set_traffic_split("fraud", {"v1": 100})


# ── health_check ──────────────────────────────────────────────────


class TestHealthCheck:
    def test_healthy_variant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sm = _install_fake_boto3(monkeypatch)
        sm.describe_endpoint.return_value = {
            "EndpointStatus": "InService",
            "ProductionVariants": [
                {
                    "VariantName": "fraud-1-0-0",
                    "CurrentInstanceCount": 2,
                    "CurrentWeight": 1.0,
                },
            ],
        }
        mod = _fresh_module()
        target = _make_target(mod)
        assert target.health_check("fraud", "1.0.0") is True

    def test_endpoint_not_in_service(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sm = _install_fake_boto3(monkeypatch)
        sm.describe_endpoint.return_value = {
            "EndpointStatus": "Creating",
            "ProductionVariants": [],
        }
        mod = _fresh_module()
        target = _make_target(mod)
        assert target.health_check("fraud", "1.0.0") is False

    def test_variant_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sm = _install_fake_boto3(monkeypatch)
        sm.describe_endpoint.return_value = {
            "EndpointStatus": "InService",
            "ProductionVariants": [
                {"VariantName": "other-variant", "CurrentInstanceCount": 1},
            ],
        }
        mod = _fresh_module()
        target = _make_target(mod)
        assert target.health_check("fraud", "1.0.0") is False

    def test_zero_instances(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sm = _install_fake_boto3(monkeypatch)
        sm.describe_endpoint.return_value = {
            "EndpointStatus": "InService",
            "ProductionVariants": [
                {
                    "VariantName": "fraud-1-0-0",
                    "CurrentInstanceCount": 0,
                },
            ],
        }
        mod = _fresh_module()
        target = _make_target(mod)
        assert target.health_check("fraud", "1.0.0") is False

    def test_api_error_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sm = _install_fake_boto3(monkeypatch)
        sm.describe_endpoint.side_effect = RuntimeError("oops")
        mod = _fresh_module()
        target = _make_target(mod)
        assert target.health_check("fraud", "1.0.0") is False


# ── rollback_to ───────────────────────────────────────────────────


class TestRollback:
    def test_rollback_sets_full_traffic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sm = _install_fake_boto3(monkeypatch)
        mod = _fresh_module()
        target = _make_target(mod)
        target.rollback_to("fraud", "2.3.0")
        sm.update_endpoint_weights_and_capacities.assert_called_once()
        call_kwargs = (
            sm.update_endpoint_weights_and_capacities.call_args.kwargs
        )
        weights = call_kwargs["DesiredWeightsAndCapacities"]
        assert len(weights) == 1
        assert weights[0]["DesiredWeight"] == pytest.approx(1.0)


# ── describe ──────────────────────────────────────────────────────


class TestDescribe:
    def test_describe_returns_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sm = _install_fake_boto3(monkeypatch)
        sm.describe_endpoint.return_value = {
            "EndpointStatus": "InService",
            "ProductionVariants": [
                {"VariantName": "fraud-1-0-0", "CurrentWeight": 1.0},
            ],
        }
        mod = _fresh_module()
        target = _make_target(mod)
        info = target.describe("fraud")

        assert info["target"] == "sagemaker_endpoint"
        assert info["endpoint"] == "test-ep"
        assert info["status"] == "InService"
        assert info["variants"] == {"fraud-1-0-0": 1.0}

    def test_describe_on_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sm = _install_fake_boto3(monkeypatch)
        sm.describe_endpoint.side_effect = RuntimeError("gone")
        mod = _fresh_module()
        target = _make_target(mod)
        info = target.describe("fraud")
        assert "error" in info
        assert info["target"] == "sagemaker_endpoint"
