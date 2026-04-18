"""Deployment automation — shadow, canary, blue-green, direct."""

from sentinel.action.deployment.manager import DeploymentManager
from sentinel.action.deployment.promotion import PromotionPolicy
from sentinel.action.deployment.strategies.base import BaseDeploymentStrategy, DeploymentState
from sentinel.action.deployment.strategies.blue_green import BlueGreenStrategy
from sentinel.action.deployment.strategies.canary import CanaryStrategy
from sentinel.action.deployment.strategies.direct import DirectStrategy
from sentinel.action.deployment.strategies.shadow import ShadowStrategy

STRATEGY_REGISTRY: dict[str, type[BaseDeploymentStrategy]] = {
    "shadow": ShadowStrategy,
    "canary": CanaryStrategy,
    "blue_green": BlueGreenStrategy,
    "direct": DirectStrategy,
}


def register_strategy(name: str, cls: type[BaseDeploymentStrategy]) -> None:
    """Plug-in API: register a custom deployment strategy."""
    STRATEGY_REGISTRY[name] = cls


__all__ = [
    "STRATEGY_REGISTRY",
    "BaseDeploymentStrategy",
    "BlueGreenStrategy",
    "CanaryStrategy",
    "DeploymentManager",
    "DeploymentState",
    "DirectStrategy",
    "PromotionPolicy",
    "ShadowStrategy",
    "register_strategy",
]
