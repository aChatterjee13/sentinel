"""Shadow deployment — deploy candidate alongside champion for evaluation."""

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


class ShadowStrategy(BaseDeploymentStrategy):
    """Run a candidate model alongside the champion for evaluation.

    The shadow model is deployed on the target but receives 0% live
    traffic.  Mirrored requests are used for offline comparison.
    After the configured duration, the shadow can be promoted to
    serve live traffic or rolled back.

    Args:
        model_name: Logical model name.
        duration: Human-readable duration string (e.g. ``"24h"``).
        **config: Passed to :class:`BaseDeploymentStrategy`.
    """

    name = "shadow"

    def __init__(self, model_name: str, duration: str = "24h", **config: Any):
        super().__init__(model_name, **config)
        self.duration = duration

    def start(self, from_version: str | None, to_version: str) -> DeploymentState:
        """Deploy the candidate model at 0% traffic for shadow evaluation.

        Args:
            from_version: Current production version (may be None).
            to_version: Candidate version to shadow-deploy.

        Returns:
            A :class:`DeploymentState` with ``traffic_pct=0`` and
            phase ``RUNNING``.
        """
        if from_version is not None:
            weights = {from_version: 100, to_version: 0}
            try:
                self.target.set_traffic_split(self.model_name, weights)
            except DeploymentError as exc:
                log.warning(
                    "deployment.shadow.start_target_failed",
                    model=self.model_name,
                    error=str(exc),
                )
        else:
            log.warning(
                "deployment.shadow.no_champion",
                model=self.model_name,
                to_version=to_version,
                detail="Skipping target call — shadow without a champion runs locally only",
            )

        return DeploymentState(
            deployment_id=uuid4().hex[:12],
            model_name=self.model_name,
            from_version=from_version,
            to_version=to_version,
            strategy=self.name,
            phase=DeploymentPhase.RUNNING,
            traffic_pct=0,
            metrics={"shadow": 1.0},
        )

    def advance(
        self,
        state: DeploymentState,
        observed_metrics: dict[str, float],
    ) -> DeploymentState:
        """Record shadow metrics and check candidate health.

        Args:
            state: Current deployment state.
            observed_metrics: Metrics from shadow evaluation.

        Returns:
            Updated :class:`DeploymentState` with merged metrics.
        """
        try:
            healthy = self.target.health_check(self.model_name, state.to_version)
            if not healthy:
                log.warning(
                    "deployment.shadow.unhealthy",
                    model=self.model_name,
                    version=state.to_version,
                )
        except DeploymentError as exc:
            log.warning(
                "deployment.shadow.health_check_failed",
                model=self.model_name,
                error=str(exc),
            )

        return state.model_copy(
            update={
                "metrics": {**state.metrics, **observed_metrics},
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def rollback(self, state: DeploymentState, reason: str) -> DeploymentState:
        """Remove the shadow model and restore champion-only traffic.

        Args:
            state: Current deployment state.
            reason: Why the shadow is being rolled back.

        Returns:
            Updated :class:`DeploymentState` with ROLLED_BACK phase.
        """
        if state.from_version is not None:
            try:
                self.target.rollback_to(self.model_name, state.from_version)
            except DeploymentError as exc:
                log.warning(
                    "deployment.shadow.rollback_target_failed",
                    model=self.model_name,
                    error=str(exc),
                )

        return state.model_copy(
            update={
                "phase": DeploymentPhase.ROLLED_BACK,
                "traffic_pct": 0,
                "error": reason,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def promote(self, state: DeploymentState) -> DeploymentState:
        """Promote the shadow candidate to receive 100% live traffic.

        Args:
            state: Current deployment state.

        Returns:
            Updated :class:`DeploymentState` with PROMOTED phase.
        """
        try:
            self.target.set_traffic_split(
                self.model_name, {state.to_version: 100}
            )
        except DeploymentError as exc:
            log.error(
                "deployment.shadow.promote_target_failed",
                model=self.model_name,
                error=str(exc),
            )

        return state.model_copy(
            update={
                "phase": DeploymentPhase.PROMOTED,
                "traffic_pct": 100,
                "updated_at": datetime.now(timezone.utc),
            }
        )
