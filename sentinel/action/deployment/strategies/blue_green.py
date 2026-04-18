"""Blue-green deployment — atomic environment switch with instant rollback."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sentinel.action.deployment.strategies.base import (
    BaseDeploymentStrategy,
    DeploymentPhase,
    DeploymentState,
)


class BlueGreenStrategy(BaseDeploymentStrategy):
    """Two identical environments — switch atomically."""

    name = "blue_green"

    def __init__(
        self,
        model_name: str,
        health_check_url: str | None = None,
        warmup_seconds: int = 30,
        **config: Any,
    ):
        super().__init__(model_name, **config)
        self.health_check_url = health_check_url
        self.warmup_seconds = warmup_seconds

    def start(self, from_version: str | None, to_version: str) -> DeploymentState:
        # Stand up the green environment, traffic still 0% until promote()
        return DeploymentState(
            deployment_id=uuid4().hex[:12],
            model_name=self.model_name,
            from_version=from_version,
            to_version=to_version,
            strategy=self.name,
            phase=DeploymentPhase.RUNNING,
            traffic_pct=0,
            metrics={"warming_up": float(self.warmup_seconds)},
        )

    def advance(
        self,
        state: DeploymentState,
        observed_metrics: dict[str, float],
    ) -> DeploymentState:
        # Synthetic override wins — kept for backward compat with the
        # existing in-memory strategy tests that pass the pass/fail
        # signal directly as a metric.
        if observed_metrics.get("health_check_passed", 1.0) < 1.0:
            return self.rollback(state, reason="health check failed")
        # Real target health check — a LocalDeploymentTarget always
        # returns True so nothing changes for the default case.
        try:
            if not self.target.health_check(self.model_name, state.to_version):
                return self.rollback(state, reason="target health check failed")
        except Exception as e:
            return self.rollback(state, reason=f"health check error: {e}")
        # Atomic switch — route all traffic to the new version.
        try:
            self.target.set_traffic_split(self.model_name, {state.to_version: 100})
        except Exception as e:
            # Try to restore traffic to old version
            try:
                if state.from_version:
                    self.target.set_traffic_split(self.model_name, {state.from_version: 100})
            except Exception:
                pass  # Best-effort rollback
            return self.rollback(state, reason=f"target swap failed: {e}")
        return self.promote(state)

    def rollback(self, state: DeploymentState, reason: str) -> DeploymentState:
        return state.model_copy(
            update={
                "phase": DeploymentPhase.ROLLED_BACK,
                "traffic_pct": 0,
                "error": reason,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def promote(self, state: DeploymentState) -> DeploymentState:
        return state.model_copy(
            update={
                "phase": DeploymentPhase.PROMOTED,
                "traffic_pct": 100,
                "updated_at": datetime.now(timezone.utc),
            }
        )
