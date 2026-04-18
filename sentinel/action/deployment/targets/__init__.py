"""Deployment targets — pluggable backends that the strategy layer talks to.

A *strategy* (canary, blue_green, shadow) decides *what* to do next;
a *target* (local, azure_ml_endpoint, azure_app_service, aks) knows
*how* to actually do it on the underlying platform.

Cloud-specific targets lazy-import their SDKs inside ``__init__``, so
``import sentinel.action.deployment.targets`` is free of Azure /
Kubernetes imports. Only the constructor for a specific target pulls
in its dependencies.
"""

from __future__ import annotations

from typing import Any

from sentinel.action.deployment.targets.base import BaseDeploymentTarget
from sentinel.action.deployment.targets.local import LocalDeploymentTarget
from sentinel.core.exceptions import DeploymentError


def _make_azure_ml_endpoint(**kwargs: Any) -> BaseDeploymentTarget:
    from sentinel.action.deployment.targets.azure_ml_endpoint import AzureMLEndpointTarget

    return AzureMLEndpointTarget(**kwargs)


def _make_azure_app_service(**kwargs: Any) -> BaseDeploymentTarget:
    from sentinel.action.deployment.targets.azure_app_service import AzureAppServiceTarget

    return AzureAppServiceTarget(**kwargs)


def _make_aks(**kwargs: Any) -> BaseDeploymentTarget:
    from sentinel.action.deployment.targets.aks import AKSDeploymentTarget

    return AKSDeploymentTarget(**kwargs)


def _make_sagemaker_endpoint(**kwargs: Any) -> BaseDeploymentTarget:
    from sentinel.action.deployment.targets.sagemaker import SageMakerEndpointTarget

    return SageMakerEndpointTarget(**kwargs)


def _make_vertex_ai_endpoint(**kwargs: Any) -> BaseDeploymentTarget:
    from sentinel.action.deployment.targets.vertex_ai import VertexAIEndpointTarget

    return VertexAIEndpointTarget(**kwargs)


TARGET_REGISTRY: dict[str, Any] = {
    "local": lambda **kwargs: LocalDeploymentTarget(**kwargs),
    "azure_ml_endpoint": _make_azure_ml_endpoint,
    "azure_app_service": _make_azure_app_service,
    "aks": _make_aks,
    "sagemaker_endpoint": _make_sagemaker_endpoint,
    "vertex_ai_endpoint": _make_vertex_ai_endpoint,
}


def register_target(name: str, factory: Any) -> None:
    """Plug-in API: register a custom deployment target factory."""
    TARGET_REGISTRY[name] = factory


def resolve_target(name: str, **kwargs: Any) -> BaseDeploymentTarget:
    """Instantiate the deployment target registered under ``name``."""
    factory = TARGET_REGISTRY.get(name)
    if factory is None:
        raise DeploymentError(
            f"unknown deployment target: {name!r}. Known targets: {sorted(TARGET_REGISTRY)}"
        )
    return factory(**kwargs)  # type: ignore[no-any-return]


__all__ = [
    "TARGET_REGISTRY",
    "BaseDeploymentTarget",
    "LocalDeploymentTarget",
    "register_target",
    "resolve_target",
]
