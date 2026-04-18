"""Abstract deployment strategy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from sentinel.action.deployment.targets.base import BaseDeploymentTarget


class DeploymentPhase(str, Enum):
    """Lifecycle states of a deployment."""

    PENDING = "pending"
    RUNNING = "running"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class DeploymentState(BaseModel):
    """Persistent state of an in-flight deployment."""

    model_config = ConfigDict(extra="allow")

    deployment_id: str
    model_name: str
    from_version: str | None
    to_version: str
    strategy: str
    phase: DeploymentPhase = DeploymentPhase.PENDING
    traffic_pct: int = 0
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class BaseDeploymentStrategy(ABC):
    """Strategy interface — start, advance, rollback, status."""

    name: str = "base"

    def __init__(
        self,
        model_name: str,
        *,
        target: BaseDeploymentTarget | None = None,
        **config: Any,
    ):
        # Avoid a cycle at import time by deferring the local target
        # import until the strategy actually needs a default. In the
        # common case a caller passes the concrete target from
        # DeploymentManager._build_strategy().
        if target is None:
            from sentinel.action.deployment.targets.local import LocalDeploymentTarget

            target = LocalDeploymentTarget()
        self.model_name = model_name
        self.target = target
        self.config = config

    @abstractmethod
    def start(self, from_version: str | None, to_version: str) -> DeploymentState:
        """Begin the deployment."""

    @abstractmethod
    def advance(
        self, state: DeploymentState, observed_metrics: dict[str, float]
    ) -> DeploymentState:
        """Move to the next phase, possibly increasing traffic %."""

    @abstractmethod
    def rollback(self, state: DeploymentState, reason: str) -> DeploymentState:
        """Abort and roll back to the previous version."""

    @abstractmethod
    def promote(self, state: DeploymentState) -> DeploymentState:
        """Promote to 100% traffic / production."""
