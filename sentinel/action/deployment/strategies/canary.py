"""Canary deployment strategy with auto-rollback on metric regression."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog

from sentinel.action.deployment.strategies.base import (
    BaseDeploymentStrategy,
    DeploymentPhase,
    DeploymentState,
)
from sentinel.core.exceptions import DeploymentError

log = structlog.get_logger(__name__)


class CanaryStrategy(BaseDeploymentStrategy):
    """Ramp traffic from 5% → 25% → 50% → 100% with auto-rollback."""

    name = "canary"

    def __init__(
        self,
        model_name: str,
        ramp_steps: list[int] | None = None,
        rollback_on: dict[str, float] | None = None,
        **config: Any,
    ):
        super().__init__(model_name, **config)
        self.ramp_steps = ramp_steps or [5, 25, 50, 100]
        self.rollback_on = rollback_on or {}

    def start(self, from_version: str | None, to_version: str) -> DeploymentState:
        return DeploymentState(
            deployment_id=uuid4().hex[:12],
            model_name=self.model_name,
            from_version=from_version,
            to_version=to_version,
            strategy=self.name,
            phase=DeploymentPhase.RUNNING,
            traffic_pct=self.ramp_steps[0],
        )

    def advance(
        self,
        state: DeploymentState,
        observed_metrics: dict[str, float],
        baseline_metrics: dict[str, float] | None = None,
    ) -> DeploymentState:
        # Check rollback conditions before advancing
        for metric, max_increase in self.rollback_on.items():
            observed = observed_metrics.get(metric, 0.0)
            if baseline_metrics and metric in baseline_metrics:
                threshold = baseline_metrics[metric] * (1 + max_increase)
                if observed > threshold:
                    return self.rollback(
                        state,
                        reason=f"{metric}={observed:.4f} exceeded baseline*{1 + max_increase:.2f}={threshold:.4f}",
                    )
            elif observed > max_increase:
                return self.rollback(state, reason=f"{metric} exceeded {max_increase}")

        try:
            current_idx = self.ramp_steps.index(state.traffic_pct)
        except ValueError:
            current_idx = 0

        if current_idx + 1 >= len(self.ramp_steps):
            return self.promote(state)

        next_pct = self.ramp_steps[current_idx + 1]

        # Delegate the actual traffic split to the target. A local
        # (no-op) target makes this a logged stub so existing unit
        # tests for CanaryStrategy keep passing.
        if state.from_version is not None:
            weights = {state.to_version: next_pct, state.from_version: 100 - next_pct}
        else:
            weights = {state.to_version: next_pct}
        try:
            self.target.set_traffic_split(self.model_name, weights)
        except DeploymentError as e:
            log.error(
                "deployment.canary.target_failed",
                model=self.model_name,
                error=str(e),
            )
            return self.rollback(state, reason=f"target error: {e}")

        return state.model_copy(
            update={
                "traffic_pct": next_pct,
                "updated_at": datetime.now(timezone.utc),
                "metrics": {**state.metrics, **observed_metrics},
                "phase": DeploymentPhase.PROMOTED if next_pct == 100 else DeploymentPhase.RUNNING,
            }
        )

    def rollback(self, state: DeploymentState, reason: str) -> DeploymentState:
        # Best-effort rollback on the underlying target. Swallow target
        # errors here — the deployment is already being marked
        # ROLLED_BACK and double-raising would hide the original reason.
        if state.from_version is not None:
            try:
                self.target.rollback_to(self.model_name, state.from_version)
            except DeploymentError as e:
                log.warning(
                    "deployment.canary.rollback_target_failed",
                    model=self.model_name,
                    error=str(e),
                )
        return state.model_copy(
            update={
                "phase": DeploymentPhase.ROLLED_BACK,
                "traffic_pct": 0,
                "updated_at": datetime.now(timezone.utc),
                "error": reason,
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
