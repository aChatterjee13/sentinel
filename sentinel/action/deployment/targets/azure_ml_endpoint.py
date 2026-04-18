"""Azure ML managed online endpoint as a deployment target.

Talks to ``azure.ai.ml.MLClient`` to update endpoint traffic, probe
deployment health, and roll back. SDK is imported lazily in
``__init__`` — ``import sentinel`` stays Azure-free.

The mapping between Sentinel's ``version`` concept and Azure ML's
``deployment`` concept is governed by ``deployment_name_pattern``:
``"{model_name}-{version}"`` means version ``"2.3.1"`` of a model
called ``"fraud"`` lives in a deployment called ``"fraud-2.3.1"``.
Customers with a different naming convention can override the
pattern in ``sentinel.yaml``.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from sentinel.action.deployment.targets.base import BaseDeploymentTarget
from sentinel.core.exceptions import DeploymentError

log = structlog.get_logger(__name__)


class AzureMLEndpointTarget(BaseDeploymentTarget):
    """Azure ML Managed Online Endpoint deployment target.

    Requires the ``azure`` extra: ``pip install sentinel-mlops[azure]``.
    Uses :class:`azure.identity.DefaultAzureCredential` — no per-target
    credentials live in the schema.
    """

    name = "azure_ml_endpoint"

    def __init__(
        self,
        *,
        endpoint_name: str,
        subscription_id: str,
        resource_group: str,
        workspace_name: str,
        deployment_name_pattern: str = "{model_name}-{version}",
        credential: Any = None,
        timeout_seconds: int = 300,
    ) -> None:
        try:
            from azure.ai.ml import MLClient  # type: ignore[import-untyped]
            from azure.identity import DefaultAzureCredential
        except ImportError as e:
            raise DeploymentError(
                "azure extra not installed — `pip install sentinel-mlops[azure]`"
            ) from e

        self._endpoint_name = endpoint_name
        self._deployment_name_pattern = deployment_name_pattern
        self._timeout = timeout_seconds
        self._client = MLClient(
            credential=credential or DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )

    def _deployment_name(self, model_name: str, version: str) -> str:
        return self._deployment_name_pattern.format(model_name=model_name, version=version)

    def set_traffic_split(self, model_name: str, weights: dict[str, int]) -> None:
        """Update the endpoint's traffic dict and wait for completion."""
        if sum(weights.values()) != 100:
            raise DeploymentError(f"traffic weights must sum to 100, got {sum(weights.values())}")
        try:
            endpoint = self._client.online_endpoints.get(name=self._endpoint_name)
            endpoint.traffic = {
                self._deployment_name(model_name, version): weight
                for version, weight in weights.items()
            }
            poller = self._client.online_endpoints.begin_create_or_update(endpoint)
            poller.result(timeout=self._timeout)
        except Exception as e:
            raise DeploymentError(
                f"azure_ml_endpoint set_traffic_split failed for {self._endpoint_name}: {e}"
            ) from e
        log.info(
            "deployment.azure_ml_endpoint.traffic_updated",
            endpoint=self._endpoint_name,
            model=model_name,
            weights=weights,
        )

    def health_check(self, model_name: str, version: str) -> bool:
        """Check the online deployment's provisioning state."""
        deployment_name = self._deployment_name(model_name, version)
        try:
            deployment = self._client.online_deployments.get(
                name=deployment_name,
                endpoint_name=self._endpoint_name,
            )
        except Exception as e:
            log.warning(
                "deployment.azure_ml_endpoint.health_check_failed",
                deployment=deployment_name,
                error=str(e),
            )
            return False
        state = getattr(deployment, "provisioning_state", None)
        return state == "Succeeded"

    def rollback_to(self, model_name: str, version: str) -> None:
        """Route 100% traffic to the named deployment."""
        self.set_traffic_split(model_name, {version: 100})

    def describe(self, model_name: str) -> dict[str, Any]:
        try:
            endpoint = self._client.online_endpoints.get(name=self._endpoint_name)
            return {
                "target": self.name,
                "endpoint": self._endpoint_name,
                "traffic": dict(getattr(endpoint, "traffic", {}) or {}),
            }
        except Exception as e:  # pragma: no cover — defensive
            log.warning("deployment.azure_ml_endpoint.describe_failed", error=str(e))
            return {"target": self.name, "endpoint": self._endpoint_name, "error": str(e)}

    @staticmethod
    def _sleep(seconds: float) -> None:  # pragma: no cover — test hook
        time.sleep(seconds)
