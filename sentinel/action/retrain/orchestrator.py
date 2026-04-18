"""Retrain orchestrator — drift → trigger → pipeline → validate → approve → promote."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog

from sentinel.action.retrain.approval import ApprovalGate, ApprovalStatus
from sentinel.action.retrain.triggers import RetrainTrigger, TriggerEvaluator
from sentinel.config.schema import RetrainingConfig
from sentinel.core.exceptions import RetrainError
from sentinel.core.types import DriftReport

if TYPE_CHECKING:
    from sentinel.action.deployment.manager import DeploymentManager
    from sentinel.foundation.audit.trail import AuditTrail
    from sentinel.foundation.registry.model_registry import ModelRegistry

log = structlog.get_logger(__name__)

PipelineRunner = Callable[[str, dict[str, Any]], dict[str, Any]]
"""Callable signature: ``(pipeline_uri, context) -> result_payload``."""


class RetrainOrchestrator:
    """End-to-end retrain coordination.

    Example:
        >>> orch = RetrainOrchestrator(config.retraining, registry, audit)
        >>> orch.set_pipeline_runner(my_runner)
        >>> trigger = orch.evaluator.manual("ad-hoc retrain")
        >>> result = orch.run(model_name="fraud", trigger=trigger)
    """

    def __init__(
        self,
        config: RetrainingConfig,
        registry: ModelRegistry | None = None,
        audit: AuditTrail | None = None,
        deployment_manager: DeploymentManager | None = None,
    ):
        self.config = config
        self.registry = registry
        self.audit = audit
        self._deployment_manager = deployment_manager
        self.evaluator = TriggerEvaluator(
            trigger_mode=config.trigger,
            schedule=config.schedule,
        )
        self.approval = ApprovalGate(config.approval)
        self._pipeline_runner: PipelineRunner | None = None

    def set_pipeline_runner(self, runner: PipelineRunner) -> None:
        """Inject the function that actually runs the retrain pipeline."""
        self._pipeline_runner = runner

    # ── Drift / scheduled hooks ───────────────────────────────────

    def on_drift(self, report: DriftReport) -> RetrainTrigger | None:
        return self.evaluator.on_drift(report)

    # ── End-to-end run ────────────────────────────────────────────

    def run(
        self,
        model_name: str,
        trigger: RetrainTrigger,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the full retrain → validate → approve → promote pipeline."""
        if self._pipeline_runner is None:
            raise RetrainError("pipeline runner not configured — call set_pipeline_runner()")

        self._audit("retrain_started", model_name, trigger=trigger.__dict__)

        try:
            result = self._pipeline_runner(self.config.pipeline or "", context or {})
        except Exception as e:
            self._audit("retrain_failed", model_name, error=str(e))
            raise RetrainError(f"pipeline execution failed: {e}") from e

        candidate_version = result.get("version") or "unknown"
        challenger_metrics = result.get("metrics", {})

        # Validate against floors
        for k, floor in self.config.validation.min_performance.items():
            if k not in challenger_metrics:
                self._audit(
                    "retrain_validation_failed",
                    model_name,
                    metric=k,
                    value=None,
                )
                raise RetrainError(
                    f"validation failed: required metric '{k}' missing from challenger results"
                )
            if challenger_metrics[k] < floor:
                self._audit(
                    "retrain_validation_failed",
                    model_name,
                    metric=k,
                    value=challenger_metrics.get(k),
                )
                raise RetrainError(f"validation failed: {k}={challenger_metrics.get(k)} < {floor}")

        # Champion metrics from registry
        champion_metrics: dict[str, float] = {}
        if self.registry is not None:
            try:
                champion_metrics = self.registry.get_latest(model_name).metrics
            except Exception:
                champion_metrics = {}

        # Approval
        request = self.approval.request(
            model_name=model_name,
            candidate_version=candidate_version,
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
        )
        self._audit("approval_requested", model_name, request_id=request.request_id)

        # Auto-approved → register & promote immediately
        if request.status in (ApprovalStatus.AUTO_APPROVED, ApprovalStatus.APPROVED):
            if self.registry is not None:
                self.registry.register_if_new(
                    model_name,
                    candidate_version,
                    metrics=challenger_metrics,
                    framework=result.get("framework"),
                    description=result.get("description"),
                )
                self.registry.promote(model_name, candidate_version, status="production")
            self._audit("retrain_completed", model_name, version=candidate_version, auto=True)
            return self._maybe_deploy(
                model_name,
                candidate_version,
                challenger_metrics,
                request.request_id,
            )

        # Pending — caller must call orchestrator.approve(...) later
        return {
            "status": "pending_approval",
            "version": candidate_version,
            "metrics": challenger_metrics,
            "request_id": request.request_id,
        }

    def approve(self, request_id: str, by: str, comment: str | None = None) -> dict[str, Any]:
        """Approve a pending request and promote the candidate."""
        req = self.approval.approve(request_id, by=by, comment=comment)
        if self.registry is not None:
            self.registry.register_if_new(
                req.model_name,
                req.candidate_version,
                metrics=req.challenger_metrics,
            )
            self.registry.promote(req.model_name, req.candidate_version, status="production")
        self._audit(
            "approval_decision",
            req.model_name,
            decision="approved",
            by=by,
            request_id=request_id,
        )
        return self._maybe_deploy(
            req.model_name,
            req.candidate_version,
            req.challenger_metrics,
            request_id,
        )

    def reject(self, request_id: str, by: str, comment: str | None = None) -> dict[str, Any]:
        req = self.approval.reject(request_id, by=by, comment=comment)
        self._audit(
            "approval_decision",
            req.model_name,
            decision="rejected",
            by=by,
            request_id=request_id,
        )
        return {"status": "rejected"}

    def _maybe_deploy(
        self,
        model_name: str,
        version: str,
        metrics: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        """Trigger a deployment if ``deploy_on_promote`` is set."""
        result: dict[str, Any] = {
            "status": "promoted",
            "version": version,
            "metrics": metrics,
            "request_id": request_id,
        }
        if self.config.deploy_on_promote and self._deployment_manager is not None:
            deploy_state = self._deployment_manager.start(
                model_name=model_name,
                to_version=version,
            )
            self._audit("deploy_triggered", model_name, version=version)
            result["deployment"] = deploy_state.model_dump(mode="json")
        return result

    def _audit(self, event_type: str, model_name: str, **payload: Any) -> None:
        log.info(event_type, model=model_name, **payload)
        if self.audit is not None:
            self.audit.log(event_type=event_type, model_name=model_name, **payload)
