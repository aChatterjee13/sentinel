"""Tests for sentinel.integrations.azure.pipeline_runner — WS-B."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from sentinel.config.schema import (
    ModelConfig,
    RegistryConfig,
    RetrainingConfig,
    SentinelConfig,
)


class TestAzureMLPipelineRunner:
    def test_callable_protocol(self) -> None:
        """Runner must be callable(pipeline_uri, context) -> dict."""
        from sentinel.integrations.azure.pipeline_runner import AzureMLPipelineRunner

        runner = AzureMLPipelineRunner("sub", "rg", "ws")
        assert callable(runner)

    @patch("sentinel.integrations.azure.pipeline_runner.AzureMLPipelineRunner._get_client")
    def test_call_returns_dict(self, mock_get: MagicMock) -> None:
        import sys

        # Mock azure.ai.ml modules so the lazy imports succeed
        azure_ai_ml_mock = MagicMock()
        sys.modules["azure.ai.ml"] = azure_ai_ml_mock
        sys.modules["azure.ai.ml.dsl"] = azure_ai_ml_mock.dsl

        try:
            from sentinel.integrations.azure.pipeline_runner import AzureMLPipelineRunner

            mock_client = MagicMock()
            mock_job = MagicMock()
            mock_job.name = "test-job"
            mock_job.status = "Completed"
            mock_job.outputs = {"version": "2.0.0", "metrics": {"f1": 0.95}}
            mock_client.jobs.create_or_update.return_value = mock_job
            mock_client.jobs.stream.return_value = mock_job
            mock_get.return_value = mock_client

            runner = AzureMLPipelineRunner("sub", "rg", "ws")
            result = runner("azureml://pipelines/retrain_fraud", {"trigger": "drift"})

            assert result["version"] == "2.0.0"
            assert result["metrics"] == {"f1": 0.95}
            assert "job_name" in result
        finally:
            sys.modules.pop("azure.ai.ml", None)
            sys.modules.pop("azure.ai.ml.dsl", None)


class TestAutoWirePipelineRunner:
    def test_auto_wire_when_azureml_pipeline(self, tmp_path: Path) -> None:
        """Client should auto-wire runner when pipeline starts with azureml://."""
        from sentinel.core.client import SentinelClient

        config = SentinelConfig(
            model=ModelConfig(name="test"),
            retraining=RetrainingConfig(pipeline="azureml://pipelines/retrain"),
            registry=RegistryConfig(
                backend="azure_ml",
                subscription_id="sub",
                resource_group="rg",
                workspace_name="ws",
                path=str(tmp_path / "reg"),
            ),
        )

        with (
            patch("sentinel.core.client.SentinelClient._build_registry_backend") as mock_backend,
            patch(
                "sentinel.integrations.azure.pipeline_runner.AzureMLPipelineRunner"
            ) as mock_runner_cls,
        ):
            mock_backend.return_value = MagicMock()
            mock_runner_instance = MagicMock()
            mock_runner_cls.return_value = mock_runner_instance

            client = SentinelClient(config)
            assert client.retrain._pipeline_runner is mock_runner_instance

    def test_no_auto_wire_when_local_pipeline(self, tmp_path: Path) -> None:
        """Client should NOT auto-wire runner when pipeline is local."""
        from sentinel.core.client import SentinelClient

        config = SentinelConfig(
            model=ModelConfig(name="test"),
            retraining=RetrainingConfig(pipeline="scripts/retrain.py"),
            registry=RegistryConfig(path=str(tmp_path / "reg")),
        )
        client = SentinelClient(config)
        assert client.retrain._pipeline_runner is None

    def test_no_auto_wire_when_no_pipeline(self, tmp_path: Path) -> None:
        """Client should NOT auto-wire when pipeline is None."""
        from sentinel.core.client import SentinelClient

        config = SentinelConfig(
            model=ModelConfig(name="test"),
            registry=RegistryConfig(path=str(tmp_path / "reg")),
        )
        client = SentinelClient(config)
        assert client.retrain._pipeline_runner is None
