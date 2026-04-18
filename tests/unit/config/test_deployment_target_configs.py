"""Tests for SageMaker and Vertex AI deployment target config models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sentinel.config.schema import (
    DeploymentConfig,
    SageMakerEndpointTargetConfig,
    VertexAIEndpointTargetConfig,
)


class TestSageMakerEndpointTargetConfig:
    def test_required_fields(self) -> None:
        cfg = SageMakerEndpointTargetConfig(endpoint_name="my-ep")
        assert cfg.endpoint_name == "my-ep"
        assert cfg.region_name is None
        assert cfg.variant_name_pattern == "{model_name}-{version}"

    def test_missing_endpoint_name_raises(self) -> None:
        with pytest.raises(ValidationError, match="endpoint_name"):
            SageMakerEndpointTargetConfig()  # type: ignore[call-arg]

    def test_custom_values(self) -> None:
        cfg = SageMakerEndpointTargetConfig(
            endpoint_name="prod-ep",
            region_name="eu-west-1",
            variant_name_pattern="v-{version}",
        )
        assert cfg.region_name == "eu-west-1"
        assert cfg.variant_name_pattern == "v-{version}"


class TestVertexAIEndpointTargetConfig:
    def test_required_fields(self) -> None:
        cfg = VertexAIEndpointTargetConfig(endpoint_name="my-ep", project="proj")
        assert cfg.endpoint_name == "my-ep"
        assert cfg.project == "proj"
        assert cfg.location == "us-central1"

    def test_missing_endpoint_name_raises(self) -> None:
        with pytest.raises(ValidationError, match="endpoint_name"):
            VertexAIEndpointTargetConfig(project="proj")  # type: ignore[call-arg]

    def test_missing_project_raises(self) -> None:
        with pytest.raises(ValidationError, match="project"):
            VertexAIEndpointTargetConfig(endpoint_name="ep")  # type: ignore[call-arg]

    def test_custom_location(self) -> None:
        cfg = VertexAIEndpointTargetConfig(
            endpoint_name="ep", project="p", location="asia-east1"
        )
        assert cfg.location == "asia-east1"


class TestDeploymentConfigValidatesSageMaker:
    def test_sagemaker_target_without_config_raises(self) -> None:
        with pytest.raises(ValidationError, match="sagemaker_endpoint"):
            DeploymentConfig(strategy="canary", target="sagemaker_endpoint")

    def test_sagemaker_target_with_config_ok(self) -> None:
        cfg = DeploymentConfig(
            strategy="canary",
            target="sagemaker_endpoint",
            sagemaker_endpoint=SageMakerEndpointTargetConfig(endpoint_name="ep"),
        )
        assert cfg.target == "sagemaker_endpoint"


class TestDeploymentConfigValidatesVertexAI:
    def test_vertex_ai_target_without_config_raises(self) -> None:
        with pytest.raises(ValidationError, match="vertex_ai_endpoint"):
            DeploymentConfig(strategy="canary", target="vertex_ai_endpoint")

    def test_vertex_ai_target_with_config_ok(self) -> None:
        cfg = DeploymentConfig(
            strategy="canary",
            target="vertex_ai_endpoint",
            vertex_ai_endpoint=VertexAIEndpointTargetConfig(
                endpoint_name="ep", project="proj"
            ),
        )
        assert cfg.target == "vertex_ai_endpoint"
