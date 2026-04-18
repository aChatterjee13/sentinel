"""Direct in-place deployment — no progressive rollout."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sentinel.action.deployment.strategies.base import (
    BaseDeploymentStrategy,
    DeploymentPhase,
    DeploymentState,
)


class DirectStrategy(BaseDeploymentStrategy):
    """Replace the model in-place. Use only for non-critical or sandbox cases."""

    name = "direct"

    def start(self, from_version: str | None, to_version: str) -> DeploymentState:
        return DeploymentState(
            deployment_id=uuid4().hex[:12],
            model_name=self.model_name,
            from_version=from_version,
            to_version=to_version,
            strategy=self.name,
            phase=DeploymentPhase.PROMOTED,
            traffic_pct=100,
        )

    def advance(
        self, state: DeploymentState, observed_metrics: dict[str, float]
    ) -> DeploymentState:
        return state

    def rollback(self, state: DeploymentState, reason: str) -> DeploymentState:
        return state.model_copy(
            update={
                "phase": DeploymentPhase.ROLLED_BACK,
                "error": reason,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def promote(self, state: DeploymentState) -> DeploymentState:
        return state
