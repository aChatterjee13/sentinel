"""Azure Kubernetes Service deployment target — replica scaling.

Approximates traffic splits by scaling each version's Deployment to
``round(replicas_total * weight / 100)`` pods. This is coarse
(10 replicas = 10% granularity) and assumes a Service routes traffic
round-robin across all matching pods. For finer control, customers
should use a service mesh target (not in WS#2 scope).

The mapping from Sentinel ``version`` to Kubernetes Deployment name
uses ``deployment_name_pattern`` (default ``"{model_name}-{version}"``).
"""

from __future__ import annotations

from typing import Any

import structlog

from sentinel.action.deployment.targets.base import BaseDeploymentTarget
from sentinel.core.exceptions import DeploymentError

log = structlog.get_logger(__name__)


class AKSDeploymentTarget(BaseDeploymentTarget):
    """Kubernetes Deployment scaling for approximate traffic splits.

    Requires the ``k8s`` extra: ``pip install sentinel-mlops[k8s]``.
    """

    name = "aks"

    def __init__(
        self,
        *,
        namespace: str,
        service_name: str,
        deployment_name_pattern: str = "{model_name}-{version}",
        replicas_total: int = 10,
        kubeconfig_path: str | None = None,
        timeout_seconds: int = 300,
    ) -> None:
        try:
            from kubernetes import client as k8s_client  # type: ignore[import-not-found]
            from kubernetes import config as k8s_config
        except ImportError as e:
            raise DeploymentError(
                "k8s extra not installed — `pip install sentinel-mlops[k8s]`"
            ) from e

        # Loading order: explicit kubeconfig → in-cluster → default
        # kubeconfig. Wrap each step so a missing file does not leak
        # raw FileNotFoundError up to callers.
        try:
            if kubeconfig_path is not None:
                k8s_config.load_kube_config(config_file=kubeconfig_path)
            else:
                try:
                    k8s_config.load_incluster_config()
                except Exception:
                    k8s_config.load_kube_config()
        except Exception as e:
            raise DeploymentError(f"failed to load kubernetes config: {e}") from e

        self._namespace = namespace
        self._service_name = service_name
        self._deployment_name_pattern = deployment_name_pattern
        self._replicas_total = replicas_total
        self._timeout = timeout_seconds
        self._apps = k8s_client.AppsV1Api()

    def _deployment_name(self, model_name: str, version: str) -> str:
        return self._deployment_name_pattern.format(model_name=model_name, version=version)

    def _compute_replicas(self, weights: dict[str, int]) -> dict[str, int]:
        """Distribute ``replicas_total`` across versions by weight.

        Uses the largest-remainder method so the totals always match
        ``replicas_total`` exactly and no active version ever drops to
        zero replicas unless its weight is explicitly 0.
        """
        raw = {v: self._replicas_total * w / 100.0 for v, w in weights.items()}
        floors = {v: int(r) for v, r in raw.items()}
        remainder = self._replicas_total - sum(floors.values())
        fractions = sorted(
            ((raw[v] - floors[v], v) for v in raw),
            reverse=True,
        )
        for _, version in fractions[:remainder]:
            floors[version] += 1
        return floors

    def set_traffic_split(self, model_name: str, weights: dict[str, int]) -> None:
        if sum(weights.values()) != 100:
            raise DeploymentError(f"traffic weights must sum to 100, got {sum(weights.values())}")
        replicas_by_version = self._compute_replicas(weights)
        for version, replicas in replicas_by_version.items():
            deployment_name = self._deployment_name(model_name, version)
            try:
                self._apps.patch_namespaced_deployment_scale(
                    name=deployment_name,
                    namespace=self._namespace,
                    body={"spec": {"replicas": replicas}},
                    _request_timeout=self._timeout,
                )
            except Exception as e:
                raise DeploymentError(
                    f"aks set_traffic_split failed on {deployment_name}: {e}"
                ) from e
        log.info(
            "deployment.aks.scaled",
            namespace=self._namespace,
            model=model_name,
            replicas=replicas_by_version,
        )

    def health_check(self, model_name: str, version: str) -> bool:
        deployment_name = self._deployment_name(model_name, version)
        try:
            status = self._apps.read_namespaced_deployment_status(
                name=deployment_name,
                namespace=self._namespace,
                _request_timeout=self._timeout,
            )
        except Exception as e:
            log.warning(
                "deployment.aks.health_check_failed",
                deployment=deployment_name,
                error=str(e),
            )
            return False
        desired = getattr(status.spec, "replicas", 0) or 0
        ready = getattr(status.status, "ready_replicas", 0) or 0
        # Deployments with 0 desired replicas are "healthy" in the
        # sense that they are at their target — they just aren't
        # serving. An explicit 0 desired means this version is parked
        # on purpose, so that counts as healthy.
        if desired == 0:
            return True
        return ready == desired

    def rollback_to(self, model_name: str, version: str) -> None:
        self.set_traffic_split(model_name, {version: 100})

    def describe(self, model_name: str) -> dict[str, Any]:
        return {
            "target": self.name,
            "namespace": self._namespace,
            "service": self._service_name,
            "replicas_total": self._replicas_total,
        }
