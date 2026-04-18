"""Local (no-op) deployment target — the default.

Used when ``deployment.target = "local"``. Every operation is a
logged no-op, which preserves the pre-WS#2 behaviour of the strategy
classes: a canary "advance" moves traffic from 25% to 50% in the
strategy's internal state, but no external system is touched.

Tests for the existing strategy classes should keep passing with
this target installed.
"""

from __future__ import annotations

from typing import Any

import structlog

from sentinel.action.deployment.targets.base import BaseDeploymentTarget

log = structlog.get_logger(__name__)


class LocalDeploymentTarget(BaseDeploymentTarget):
    """Logs traffic split / rollback calls without touching anything real."""

    name = "local"

    def set_traffic_split(self, model_name: str, weights: dict[str, int]) -> None:
        log.info("deployment.local.set_traffic_split", model=model_name, weights=weights)

    def health_check(self, model_name: str, version: str) -> bool:
        # Local target has no real health to check — always report
        # healthy so the existing strategy fallbacks keep working.
        log.info("deployment.local.health_check", model=model_name, version=version)
        return True

    def rollback_to(self, model_name: str, version: str) -> None:
        log.info("deployment.local.rollback_to", model=model_name, version=version)

    def describe(self, model_name: str) -> dict[str, Any]:
        return {"target": "local", "model": model_name}
