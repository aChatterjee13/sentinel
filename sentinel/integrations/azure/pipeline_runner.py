"""Azure ML pipeline runner — submits and polls PipelineJob instances."""

from __future__ import annotations

import time
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_DEFAULT_TIMEOUT_S = 7200  # 2 hours


class AzureMLPipelineRunner:
    """Implements the ``PipelineRunner`` callable protocol for Azure ML.

    When called, it submits a ``PipelineJob`` to the configured Azure ML
    workspace, polls until completion, and returns a result payload
    containing the candidate model version and metrics.

    Args:
        subscription_id: Azure subscription ID.
        resource_group: Azure resource group name.
        workspace_name: Azure ML workspace name.
        timeout_seconds: Maximum time to wait for pipeline completion.
            Defaults to 7200 (2 hours). Set to 0 to disable.

    Example:
        >>> runner = AzureMLPipelineRunner(sub, rg, ws)
        >>> result = runner("azureml://pipelines/retrain_fraud", {"trigger": "drift"})
    """

    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        workspace_name: str,
        timeout_seconds: int = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._subscription_id = subscription_id
        self._resource_group = resource_group
        self._workspace_name = workspace_name
        self._timeout = timeout_seconds

    def _get_client(self) -> Any:
        """Lazy-import and construct an MLClient."""
        from azure.ai.ml import MLClient  # type: ignore[import-untyped]
        from azure.identity import DefaultAzureCredential

        return MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group,
            workspace_name=self._workspace_name,
        )

    def __call__(self, pipeline_uri: str, context: dict[str, Any]) -> dict[str, Any]:
        """Submit the pipeline and return the result payload.

        Args:
            pipeline_uri: ``azureml://pipelines/<name>`` reference.
            context: Trigger context forwarded as pipeline parameters.

        Returns:
            A dict with ``version``, ``metrics``, ``framework``, and
            ``description`` keys — compatible with
            :meth:`RetrainOrchestrator.run`.
        """
        ml_client = self._get_client()

        pipeline_name = pipeline_uri.replace("azureml://pipelines/", "").split("/")[0]
        log.info(
            "azure_pipeline.submitting",
            pipeline=pipeline_name,
            context_keys=list(context.keys()),
        )

        # Build a simple pipeline job from the URI and context
        job = ml_client.jobs.create_or_update(
            {
                "type": "pipeline",
                "display_name": f"sentinel-retrain-{pipeline_name}",
                "settings": {"default_compute": "cpu-cluster"},
                "inputs": context,
            }
        )
        log.info("azure_pipeline.submitted", job_name=job.name)

        # Poll until terminal state, with timeout
        start = time.monotonic()
        finished_job = ml_client.jobs.stream(job.name)
        if finished_job is None:
            finished_job = ml_client.jobs.get(job.name)

        elapsed = time.monotonic() - start
        if self._timeout > 0 and elapsed > self._timeout:
            log.error(
                "azure_pipeline.timeout",
                job_name=job.name,
                timeout=self._timeout,
                elapsed=elapsed,
            )
            raise TimeoutError(
                f"Azure ML pipeline job {job.name} did not complete "
                f"within {self._timeout}s (elapsed: {elapsed:.0f}s)"
            )

        status = getattr(finished_job, "status", "Unknown")
        if status not in ("Completed", "Finished"):
            raise RuntimeError(f"Azure ML pipeline job {job.name} ended with status: {status}")

        # Extract outputs — the pipeline should produce these as outputs
        outputs = getattr(finished_job, "outputs", {}) or {}
        result: dict[str, Any] = {
            "version": outputs.get("version", "unknown"),
            "metrics": outputs.get("metrics", {}),
            "framework": outputs.get("framework"),
            "description": f"Retrained via Azure ML pipeline {pipeline_name}",
            "job_name": job.name,
        }
        log.info("azure_pipeline.completed", job_name=job.name, result_keys=list(result.keys()))
        return result
