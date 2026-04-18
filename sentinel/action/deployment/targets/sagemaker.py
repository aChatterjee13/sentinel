"""AWS SageMaker real-time endpoint as a deployment target.

Uses boto3 SageMaker runtime to manage endpoint traffic weights via
production variants.  SDK is imported lazily in ``__init__``.
"""

from __future__ import annotations

from typing import Any

import structlog

from sentinel.action.deployment.targets.base import BaseDeploymentTarget
from sentinel.core.exceptions import DeploymentError

log = structlog.get_logger(__name__)


class SageMakerEndpointTarget(BaseDeploymentTarget):
    """AWS SageMaker Real-Time Endpoint deployment target.

    Requires the ``aws`` extra: ``pip install sentinel-mlops[aws]``.

    Maps Sentinel model versions to SageMaker production variants.
    Traffic splitting is achieved via variant weights on the endpoint.

    Args:
        endpoint_name: Name of the SageMaker endpoint.
        region_name: AWS region.  Uses boto3 default if *None*.
        variant_name_pattern: Pattern for variant names.
            Default: ``"{model_name}-{version}"``.
    """

    name = "sagemaker_endpoint"

    def __init__(
        self,
        *,
        endpoint_name: str,
        region_name: str | None = None,
        variant_name_pattern: str = "{model_name}-{version}",
        timeout_seconds: int = 300,
    ) -> None:
        try:
            import boto3  # type: ignore[import-not-found]
            from botocore.config import Config as BotoConfig  # type: ignore[import-not-found]
        except ImportError as e:
            raise DeploymentError(
                "aws extra not installed — `pip install sentinel-mlops[aws]`"
            ) from e

        self._timeout = timeout_seconds
        kwargs: dict[str, Any] = {
            "config": BotoConfig(
                read_timeout=timeout_seconds,
                connect_timeout=min(30, timeout_seconds),
            ),
        }
        if region_name:
            kwargs["region_name"] = region_name

        self._sm = boto3.client("sagemaker", **kwargs)
        self._endpoint_name = endpoint_name
        self._variant_pattern = variant_name_pattern

    # ── Helpers ────────────────────────────────────────────────────

    def _variant_name(self, model_name: str, version: str) -> str:
        """Build variant name from the configured pattern."""
        return self._variant_pattern.format(
            model_name=model_name, version=version
        ).replace(".", "-")

    # ── Required interface ─────────────────────────────────────────

    def set_traffic_split(
        self, model_name: str, weights: dict[str, int]
    ) -> None:
        """Update SageMaker endpoint variant weights.

        Args:
            model_name: Logical model name.
            weights: Maps ``version -> percentage``.  Must sum to 100.

        Raises:
            DeploymentError: If weights are invalid or the API call fails.
        """
        if sum(weights.values()) != 100:
            raise DeploymentError(
                f"traffic weights must sum to 100, got {sum(weights.values())}"
            )
        try:
            desired_weights = [
                {
                    "VariantName": self._variant_name(model_name, version),
                    "DesiredWeight": float(weight) / 100.0,
                }
                for version, weight in weights.items()
            ]
            self._sm.update_endpoint_weights_and_capacities(
                EndpointName=self._endpoint_name,
                DesiredWeightsAndCapacities=desired_weights,
            )
            log.info(
                "deployment.sagemaker.traffic_updated",
                endpoint=self._endpoint_name,
                model=model_name,
                weights=weights,
            )
        except DeploymentError:
            raise
        except Exception as e:
            raise DeploymentError(
                f"SageMaker set_traffic_split failed for "
                f"{self._endpoint_name}: {e}"
            ) from e

    def health_check(self, model_name: str, version: str) -> bool:
        """Check SageMaker endpoint variant health.

        Returns ``True`` when the endpoint is ``InService`` and the
        variant has at least one running instance.
        """
        variant = self._variant_name(model_name, version)
        try:
            resp = self._sm.describe_endpoint(
                EndpointName=self._endpoint_name,
            )
            status = resp.get("EndpointStatus", "")
            if status != "InService":
                return False
            for v in resp.get("ProductionVariants", []):
                if v.get("VariantName") == variant:
                    return v.get("CurrentInstanceCount", 0) > 0
            return False
        except Exception as e:
            log.warning(
                "deployment.sagemaker.health_check_failed",
                variant=variant,
                error=str(e),
            )
            return False

    def rollback_to(self, model_name: str, version: str) -> None:
        """Route 100 %% traffic to a single variant."""
        self.set_traffic_split(model_name, {version: 100})

    def describe(self, model_name: str) -> dict[str, Any]:
        """Return SageMaker endpoint status and variant weights."""
        try:
            resp = self._sm.describe_endpoint(
                EndpointName=self._endpoint_name,
            )
            variants = {
                v["VariantName"]: v.get("CurrentWeight", 0)
                for v in resp.get("ProductionVariants", [])
            }
            return {
                "target": self.name,
                "endpoint": self._endpoint_name,
                "status": resp.get("EndpointStatus"),
                "variants": variants,
            }
        except Exception as e:
            return {
                "target": self.name,
                "endpoint": self._endpoint_name,
                "error": str(e),
            }
