"""Google Cloud Vertex AI Endpoint as a deployment target.

Uses ``google-cloud-aiplatform`` to manage endpoint traffic splits.
SDK is imported lazily in ``__init__`` — ``import sentinel`` stays
GCP-free.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from sentinel.action.deployment.targets.base import BaseDeploymentTarget
from sentinel.core.exceptions import DeploymentError

log = structlog.get_logger(__name__)


class VertexAIEndpointTarget(BaseDeploymentTarget):
    """Google Cloud Vertex AI Endpoint deployment target.

    Requires: ``pip install google-cloud-aiplatform``.

    Maps Sentinel model versions to Vertex AI deployed models on an
    endpoint.  Traffic splitting is achieved via ``traffic_split`` on
    the endpoint resource.

    Args:
        endpoint_name: Vertex AI endpoint resource name or display name.
        project: GCP project ID.
        location: GCP region (e.g. ``'us-central1'``).
        timeout_seconds: Timeout in seconds for API operations such as
            ``endpoint.update()``.  Defaults to 300.
    """

    name = "vertex_ai_endpoint"

    def __init__(
        self,
        *,
        endpoint_name: str,
        project: str,
        location: str = "us-central1",
        timeout_seconds: int = 300,
    ) -> None:
        try:
            from google.cloud import aiplatform  # type: ignore[import-not-found]
        except ImportError as e:
            raise DeploymentError(
                "google-cloud-aiplatform not installed — "
                "`pip install google-cloud-aiplatform`"
            ) from e
        aiplatform.init(project=project, location=location)
        self._aiplatform = aiplatform
        self._endpoint_name = endpoint_name
        self._project = project
        self._location = location
        self._timeout = timeout_seconds

    # ── version matching ─────────────────────────────────────────

    @staticmethod
    def _version_matches(display_name: str, version: str) -> bool:
        """Check if *display_name* contains *version* as an exact segment.

        Handles common patterns like ``"model-1.2.3"``,
        ``"model_v1.2.3"``, ``"fraud-classifier-2.0.0"``.  Prevents
        ``"1.1"`` from matching ``"1.11"``.

        Args:
            display_name: Deployed model display name.
            version: Version string to match.

        Returns:
            True if *version* appears as a complete delimited segment.
        """
        if not display_name or not version:
            return False
        if display_name == version:
            return True
        escaped = re.escape(version)
        # Version may be preceded by start-of-string, a delimiter, or
        # a "v"/"V" prefix after a delimiter.  It must be followed by
        # end-of-string or a delimiter.
        pattern = rf"(?:^|[-_/\s])v?({escaped})(?:$|[-_/\s])"
        return bool(re.search(pattern, display_name))

    # ── helpers ────────────────────────────────────────────────────

    def _get_endpoint(self) -> Any:
        """Resolve endpoint by display name, falling back to resource name.

        Returns:
            A ``google.cloud.aiplatform.Endpoint`` instance.

        Raises:
            DeploymentError: If the endpoint cannot be found.
        """
        endpoints = self._aiplatform.Endpoint.list(
            filter=f'display_name="{self._endpoint_name}"'
        )
        if endpoints:
            return endpoints[0]
        try:
            return self._aiplatform.Endpoint(self._endpoint_name)
        except Exception as e:
            raise DeploymentError(
                f"endpoint {self._endpoint_name} not found: {e}"
            ) from e

    # ── BaseDeploymentTarget interface ────────────────────────────

    def set_traffic_split(
        self, model_name: str, weights: dict[str, int]
    ) -> None:
        """Update Vertex AI endpoint traffic split.

        Args:
            model_name: Logical model name.
            weights: ``{version: percentage}`` mapping that must sum
                to 100.

        Raises:
            DeploymentError: If weights don't sum to 100 or the API
                call fails.
        """
        if sum(weights.values()) != 100:
            raise DeploymentError(
                f"traffic weights must sum to 100, got {sum(weights.values())}"
            )
        try:
            endpoint = self._get_endpoint()
            traffic_split: dict[str, int] = {}
            for deployed_model in endpoint.gca_resource.deployed_models:
                dm_id = deployed_model.id
                display = deployed_model.display_name or ""
                for version, weight in weights.items():
                    if self._version_matches(display, version):
                        traffic_split[dm_id] = weight
            if traffic_split:
                endpoint.update(
                    traffic_split=traffic_split,
                    timeout=self._timeout,
                )
            log.info(
                "deployment.vertex_ai.traffic_updated",
                endpoint=self._endpoint_name,
                model=model_name,
                weights=weights,
            )
        except DeploymentError:
            raise
        except Exception as e:
            raise DeploymentError(
                f"Vertex AI set_traffic_split failed: {e}"
            ) from e

    def health_check(self, model_name: str, version: str) -> bool:
        """Check whether *version* is deployed on the endpoint.

        Args:
            model_name: Logical model name.
            version: Semantic version string.

        Returns:
            ``True`` if a deployed model whose display name contains
            *version* is found on the endpoint.
        """
        try:
            endpoint = self._get_endpoint()
            for dm in endpoint.gca_resource.deployed_models:
                display = dm.display_name or ""
                if self._version_matches(display, version):
                    return True
            return False
        except Exception as e:
            log.warning(
                "deployment.vertex_ai.health_check_failed", error=str(e)
            )
            return False

    def rollback_to(self, model_name: str, version: str) -> None:
        """Route 100 % traffic to a single model version.

        Args:
            model_name: Logical model name.
            version: Semantic version string to receive all traffic.
        """
        self.set_traffic_split(model_name, {version: 100})

    def describe(self, model_name: str) -> dict[str, Any]:
        """Return Vertex AI endpoint status.

        Args:
            model_name: Logical model name.

        Returns:
            Dict with ``target``, ``endpoint``, and either
            ``deployed_models`` or ``error``.
        """
        try:
            endpoint = self._get_endpoint()
            deployed = {
                dm.display_name: endpoint.gca_resource.traffic_split.get(
                    dm.id, 0
                )
                for dm in endpoint.gca_resource.deployed_models
            }
            return {
                "target": self.name,
                "endpoint": self._endpoint_name,
                "deployed_models": deployed,
            }
        except Exception as e:
            return {
                "target": self.name,
                "endpoint": self._endpoint_name,
                "error": str(e),
            }
