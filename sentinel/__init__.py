"""Project Sentinel — Unified MLOps + LLMOps + AgentOps SDK.

A single pip-installable SDK that lets ML teams monitor, govern, and operate
production machine learning models, LLM applications, and autonomous agent
systems through one config-driven interface.

Example:
    >>> from sentinel import SentinelClient
    >>> client = SentinelClient.from_config("sentinel.yaml")
    >>> client.log_prediction(features=X, prediction=y_pred)
    >>> drift = client.check_drift()
    >>> if drift.is_drifted:
    ...     print(f"Drift detected: {drift.severity}")
"""

from typing import TYPE_CHECKING, Any

from sentinel.core.client import SentinelClient
from sentinel.core.exceptions import (
    AgentError,
    AuditError,
    BudgetExceededError,
    ConfigError,
    DashboardError,
    DashboardNotInstalledError,
    DeploymentError,
    DriftDetectionError,
    GuardrailError,
    LoopDetectedError,
    RegistryError,
    SentinelError,
)
from sentinel.core.types import (
    Alert,
    AlertSeverity,
    DriftReport,
    FeatureHealthReport,
    PredictionRecord,
    QualityReport,
)

if TYPE_CHECKING:
    from sentinel.dashboard.server import (
        SentinelDashboardRouter,
        create_dashboard_app,
    )

__version__ = "0.1.0"

__all__ = [
    "AgentError",
    "Alert",
    "AlertSeverity",
    "AuditError",
    "BudgetExceededError",
    "ConfigError",
    "DashboardError",
    "DashboardNotInstalledError",
    "DeploymentError",
    "DriftDetectionError",
    "DriftReport",
    "FeatureHealthReport",
    "GuardrailError",
    "LoopDetectedError",
    "PredictionRecord",
    "QualityReport",
    "RegistryError",
    "SentinelClient",
    "SentinelDashboardRouter",
    "SentinelError",
    "__version__",
    "create_dashboard_app",
]


def __getattr__(name: str) -> Any:
    """Lazy import dashboard symbols so missing extras don't break ``import sentinel``."""
    if name in ("create_dashboard_app", "SentinelDashboardRouter"):
        try:
            from sentinel.dashboard.server import (
                SentinelDashboardRouter,
                create_dashboard_app,
            )
        except ImportError as e:
            raise DashboardNotInstalledError(
                "Dashboard extras are missing. "
                "Install with `pip install sentinel-mlops[dashboard]`."
            ) from e
        return {
            "create_dashboard_app": create_dashboard_app,
            "SentinelDashboardRouter": SentinelDashboardRouter,
        }[name]
    raise AttributeError(f"module 'sentinel' has no attribute {name!r}")
