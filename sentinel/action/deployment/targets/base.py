"""Abstract deployment target — the thing a strategy actually talks to.

A *strategy* decides "advance canary from 25% to 50%". A *target*
knows how to actually *do* that — whether that means flipping slot
traffic on Azure App Service, updating an Azure ML online endpoint,
or scaling Kubernetes replicas.

The split keeps the strategy layer pure Python (trivially unit-testable)
and isolates the cloud SDKs behind this tiny, stable interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDeploymentTarget(ABC):
    """Interface every deployment target must implement.

    Implementations are responsible for lazy-importing their cloud SDK
    inside ``__init__`` so that ``import sentinel`` never pulls the
    full Azure / Kubernetes stack into memory.
    """

    name: str = "base"

    @abstractmethod
    def set_traffic_split(self, model_name: str, weights: dict[str, int]) -> None:
        """Route traffic across one or more versions of ``model_name``.

        ``weights`` maps ``version -> percentage`` and must sum to 100.
        Implementations should raise a :class:`DeploymentError` on
        failure rather than swallowing exceptions — the strategy layer
        relies on failure signals to drive rollback.
        """

    @abstractmethod
    def health_check(self, model_name: str, version: str) -> bool:
        """Return ``True`` if ``version`` of ``model_name`` is healthy.

        Healthy means: deployed, reachable, and responding with
        acceptable status. The definition is target-specific — App
        Service targets probe a health URL, AKS targets read the
        Deployment's ``ready_replicas``, Azure ML targets read the
        online deployment's provisioning state.
        """

    @abstractmethod
    def rollback_to(self, model_name: str, version: str) -> None:
        """Route 100% traffic back to ``version`` of ``model_name``."""

    def describe(self, model_name: str) -> dict[str, Any]:
        """Return a target-specific snapshot of the current state.

        Default is an empty dict. Used by ``sentinel cloud test`` to
        print a human-readable status line per backend.
        """
        return {}

    def close(self) -> None:  # noqa: B027 — optional hook
        """Release any long-lived clients the target is holding."""
