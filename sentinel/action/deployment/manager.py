"""High-level deployment manager — strategy resolution + state persistence."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import structlog

from sentinel.action.deployment.strategies.base import (
    BaseDeploymentStrategy,
    DeploymentPhase,
    DeploymentState,
)
from sentinel.action.deployment.targets.base import BaseDeploymentTarget
from sentinel.config.schema import DeploymentConfig
from sentinel.core.exceptions import DeploymentError

if TYPE_CHECKING:
    from sentinel.foundation.audit.trail import AuditTrail
    from sentinel.foundation.registry.model_registry import ModelRegistry

log = structlog.get_logger(__name__)


class DeploymentManager:
    """Coordinates deployments across strategies and integrates with the registry.

    Example:
        >>> dm = DeploymentManager(config.deployment, registry, audit)
        >>> state = dm.start(model_name="fraud", to_version="2.3.1")
        >>> state = dm.advance(state, observed_metrics={"error_rate_increase": 0.001})
    """

    def __init__(
        self,
        config: DeploymentConfig,
        registry: ModelRegistry | None = None,
        audit: AuditTrail | None = None,
        target: BaseDeploymentTarget | None = None,
    ):
        self.config = config
        self.registry = registry
        self.audit = audit
        # Build once per manager — targets may hold long-lived cloud
        # clients. Tests can inject a custom target via the kwarg.
        self._target: BaseDeploymentTarget = target or self._build_target(config)
        self._active: dict[str, DeploymentState] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _build_target(config: DeploymentConfig) -> BaseDeploymentTarget:
        from sentinel.action.deployment.targets import resolve_target

        if config.target == "local":
            return resolve_target("local")
        if config.target == "azure_ml_endpoint":
            assert config.azure_ml_endpoint is not None
            return resolve_target(
                "azure_ml_endpoint",
                endpoint_name=config.azure_ml_endpoint.endpoint_name,
                subscription_id=config.azure_ml_endpoint.subscription_id,
                resource_group=config.azure_ml_endpoint.resource_group,
                workspace_name=config.azure_ml_endpoint.workspace_name,
                deployment_name_pattern=config.azure_ml_endpoint.deployment_name_pattern,
            )
        if config.target == "azure_app_service":
            assert config.azure_app_service is not None
            return resolve_target(
                "azure_app_service",
                subscription_id=config.azure_app_service.subscription_id,
                resource_group=config.azure_app_service.resource_group,
                site_name=config.azure_app_service.site_name,
                production_slot=config.azure_app_service.production_slot,
                staging_slot=config.azure_app_service.staging_slot,
                health_check_path=config.azure_app_service.health_check_path,
            )
        if config.target == "aks":
            assert config.aks is not None
            return resolve_target(
                "aks",
                namespace=config.aks.namespace,
                service_name=config.aks.service_name,
                deployment_name_pattern=config.aks.deployment_name_pattern,
                replicas_total=config.aks.replicas_total,
                kubeconfig_path=config.aks.kubeconfig_path,
            )
        if config.target == "sagemaker_endpoint":
            assert config.sagemaker_endpoint is not None
            return resolve_target(
                "sagemaker_endpoint",
                endpoint_name=config.sagemaker_endpoint.endpoint_name,
                region_name=config.sagemaker_endpoint.region_name,
                variant_name_pattern=config.sagemaker_endpoint.variant_name_pattern,
            )
        if config.target == "vertex_ai_endpoint":
            assert config.vertex_ai_endpoint is not None
            return resolve_target(
                "vertex_ai_endpoint",
                endpoint_name=config.vertex_ai_endpoint.endpoint_name,
                project=config.vertex_ai_endpoint.project,
                location=config.vertex_ai_endpoint.location,
            )
        raise DeploymentError(f"unknown deployment target: {config.target}")

    def _build_strategy(
        self, model_name: str, override: str | None = None
    ) -> BaseDeploymentStrategy:
        from sentinel.action.deployment import STRATEGY_REGISTRY

        strategy_name = override or self.config.strategy
        cls = STRATEGY_REGISTRY.get(strategy_name)
        if cls is None:
            raise DeploymentError(f"unknown deployment strategy: {strategy_name}")

        if strategy_name == "canary":
            return cls(
                model_name=model_name,
                ramp_steps=self.config.canary.ramp_steps,
                rollback_on=self.config.canary.rollback_on,
                target=self._target,
            )
        if strategy_name == "shadow":
            return cls(
                model_name=model_name,
                duration=self.config.shadow.duration,
                target=self._target,
            )
        if strategy_name == "blue_green":
            return cls(
                model_name=model_name,
                health_check_url=self.config.blue_green.health_check_url,
                warmup_seconds=self.config.blue_green.warmup_seconds,
                target=self._target,
            )
        return cls(model_name=model_name, target=self._target)

    def start(
        self,
        model_name: str,
        to_version: str,
        from_version: str | None = None,
        strategy_override: str | None = None,
    ) -> DeploymentState:
        """Begin a new deployment.

        Args:
            model_name: Logical model name.
            to_version: Target version to deploy.
            from_version: Current production version.  Auto-resolved from
                the registry when *None*.
            strategy_override: Use a different strategy for this deploy.

        Returns:
            Initial :class:`DeploymentState`.

        Raises:
            DeploymentError: If *to_version* is not in the registry.
        """
        # Prerequisite validation
        if self.registry is not None and not self.registry.backend.exists(model_name, to_version):
                raise DeploymentError(
                    f"model {model_name} version {to_version} not found in registry"
                )

        if from_version is None and self.registry is not None:
            existing_prod = self.registry.list_by_status(model_name, "production")
            from_version = existing_prod[-1].version if existing_prod else None

        strategy = self._build_strategy(model_name, override=strategy_override)
        state = strategy.start(from_version=from_version, to_version=to_version)
        with self._lock:
            self._active[state.deployment_id] = state
        self._persist_state(state)
        self._log_event("deployment_started", state)
        return state

    def advance(
        self,
        state: DeploymentState,
        observed_metrics: dict[str, float],
    ) -> DeploymentState:
        """Advance an in-flight deployment one step."""
        strategy = self._build_strategy(state.model_name, override=state.strategy)

        # Fetch baseline metrics from registry for canary rollback decisions
        baseline_metrics: dict[str, float] | None = None
        if self.registry is not None:
            try:
                entry = self.registry.get_latest(state.model_name)
                baseline_metrics = getattr(entry, "metrics", None) if entry else None
            except Exception:
                baseline_metrics = None

        if state.strategy == "canary" and baseline_metrics:
            new_state = strategy.advance(state, observed_metrics, baseline_metrics=baseline_metrics)
        else:
            new_state = strategy.advance(state, observed_metrics)

        with self._lock:
            self._active[new_state.deployment_id] = new_state
        self._persist_state(new_state)
        self._log_event("deployment_advanced", new_state)

        if new_state.phase == DeploymentPhase.PROMOTED and self.registry is not None:
            self.registry.promote(state.model_name, state.to_version, status="production")
        if new_state.phase == DeploymentPhase.ROLLED_BACK:
            self._log_event("deployment_rolled_back", new_state)
        return new_state

    def rollback(self, state: DeploymentState, reason: str) -> DeploymentState:
        """Manually roll back a deployment."""
        strategy = self._build_strategy(state.model_name, override=state.strategy)
        new_state = strategy.rollback(state, reason)
        with self._lock:
            self._active[new_state.deployment_id] = new_state
        self._persist_state(new_state)
        self._log_event("deployment_rolled_back", new_state)
        return new_state

    def get(self, deployment_id: str) -> DeploymentState:
        """Retrieve a deployment state by ID.

        Args:
            deployment_id: Unique deployment identifier.

        Returns:
            The matching :class:`DeploymentState`.

        Raises:
            DeploymentError: If the deployment is not found.
        """
        with self._lock:
            if deployment_id not in self._active:
                raise DeploymentError(f"deployment {deployment_id} not found")
            return self._active[deployment_id]

    def list_active(self) -> list[DeploymentState]:
        """Return all deployments currently in RUNNING phase."""
        with self._lock:
            return [s for s in self._active.values() if s.phase == DeploymentPhase.RUNNING]

    def detect_stalled(self, max_age_seconds: float = 3600) -> list[DeploymentState]:
        """Return deployments stuck in RUNNING longer than *max_age_seconds*.

        Args:
            max_age_seconds: Maximum age in seconds before a deployment
                is considered stalled.  Defaults to 1 hour.

        Returns:
            List of stalled :class:`DeploymentState` objects.
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        stalled: list[DeploymentState] = []
        with self._lock:
            for state in self._active.values():
                if state.phase == DeploymentPhase.RUNNING:
                    age = (now - state.updated_at).total_seconds()
                    if age > max_age_seconds:
                        stalled.append(state)
        return stalled

    def _log_event(self, event_type: str, state: DeploymentState) -> None:
        log.info(
            event_type,
            model=state.model_name,
            version=state.to_version,
            phase=state.phase.value,
            traffic=state.traffic_pct,
        )
        if self.audit is not None:
            self.audit.log(
                event_type=event_type,
                model_name=state.model_name,
                model_version=state.to_version,
                deployment=state.model_dump(mode="json"),
            )

    def _persist_state(self, state: DeploymentState) -> None:
        """Write deployment state to audit trail so it survives crashes."""
        if self.audit is not None:
            self.audit.log(
                event_type="deployment.state_change",
                deployment_id=state.deployment_id,
                model_name=state.model_name,
                phase=state.phase.value,
                traffic_pct=state.traffic_pct,
            )
