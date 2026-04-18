"""Tests for DeploymentConfig cross-field validation (target + strategy)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sentinel.config.schema import (
    AKSDeploymentTargetConfig,
    AzureAppServiceTargetConfig,
    AzureMLEndpointTargetConfig,
    DeploymentConfig,
)


class TestTargetRequiresSubConfig:
    def test_azure_ml_endpoint_requires_sub_config(self) -> None:
        with pytest.raises(ValidationError, match="azure_ml_endpoint"):
            DeploymentConfig(strategy="blue_green", target="azure_ml_endpoint")

    def test_azure_app_service_requires_sub_config(self) -> None:
        with pytest.raises(ValidationError, match="azure_app_service"):
            DeploymentConfig(strategy="blue_green", target="azure_app_service")

    def test_aks_requires_sub_config(self) -> None:
        with pytest.raises(ValidationError, match="aks"):
            DeploymentConfig(strategy="blue_green", target="aks")

    def test_local_needs_no_sub_config(self) -> None:
        cfg = DeploymentConfig(strategy="canary", target="local")
        assert cfg.target == "local"


class TestStrategyTargetCompatibility:
    def test_canary_app_service_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not compatible"):
            DeploymentConfig(
                strategy="canary",
                target="azure_app_service",
                azure_app_service=AzureAppServiceTargetConfig(
                    subscription_id="s",
                    resource_group="r",
                    site_name="x",
                ),
            )

    def test_blue_green_app_service_ok(self) -> None:
        cfg = DeploymentConfig(
            strategy="blue_green",
            target="azure_app_service",
            azure_app_service=AzureAppServiceTargetConfig(
                subscription_id="s",
                resource_group="r",
                site_name="x",
            ),
        )
        assert cfg.target == "azure_app_service"

    def test_canary_azure_ml_endpoint_ok(self) -> None:
        cfg = DeploymentConfig(
            strategy="canary",
            target="azure_ml_endpoint",
            azure_ml_endpoint=AzureMLEndpointTargetConfig(
                endpoint_name="e",
                subscription_id="s",
                resource_group="r",
                workspace_name="w",
            ),
        )
        assert cfg.target == "azure_ml_endpoint"

    def test_canary_aks_ok(self) -> None:
        cfg = DeploymentConfig(
            strategy="canary",
            target="aks",
            aks=AKSDeploymentTargetConfig(
                namespace="ml",
                service_name="svc",
            ),
        )
        assert cfg.target == "aks"

    def test_shadow_any_target_ok(self) -> None:
        cfg = DeploymentConfig(
            strategy="shadow",
            target="azure_app_service",
            azure_app_service=AzureAppServiceTargetConfig(
                subscription_id="s",
                resource_group="r",
                site_name="x",
            ),
        )
        assert cfg.strategy == "shadow"
