"""Unit tests for sentinel.action.retrain.orchestrator.RetrainOrchestrator."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from sentinel.action.retrain.orchestrator import RetrainOrchestrator
from sentinel.config.schema import ApprovalConfig, RetrainingConfig, ValidationConfig
from sentinel.core.exceptions import RetrainError
from sentinel.core.types import AlertSeverity, DriftReport

# ── Helpers ────────────────────────────────────────────────────────


def _cfg(
    *,
    mode: str = "auto",
    trigger: str = "manual",
    min_perf: dict[str, float] | None = None,
    deploy_on_promote: bool = False,
) -> RetrainingConfig:
    return RetrainingConfig(
        trigger=trigger,
        pipeline="test://pipe",
        approval=ApprovalConfig(mode=mode),
        validation=ValidationConfig(min_performance=min_perf or {}),
        deploy_on_promote=deploy_on_promote,
    )


def _runner(
    version: str = "2.0.0",
    metrics: dict[str, float] | None = None,
    **extra: Any,
) -> MagicMock:
    result: dict[str, Any] = {
        "version": version,
        "metrics": metrics or {"accuracy": 0.92, "f1": 0.88},
        **extra,
    }
    return MagicMock(return_value=result)


def _drift_report(*, drifted: bool = True) -> DriftReport:
    return DriftReport(
        model_name="m",
        method="psi",
        is_drifted=drifted,
        severity=AlertSeverity.HIGH if drifted else AlertSeverity.INFO,
        test_statistic=0.25 if drifted else 0.05,
        drifted_features=["f1"] if drifted else [],
    )


# ── Tests ──────────────────────────────────────────────────────────


class TestRunBasicPaths:
    """Core run() happy-path and error-path tests."""

    def test_auto_approve_returns_promoted(self) -> None:
        """Auto-mode should skip the human gate and promote immediately."""
        orch = RetrainOrchestrator(_cfg(), audit=MagicMock())
        orch.set_pipeline_runner(_runner())
        trigger = orch.evaluator.manual("test")
        result = orch.run("fraud_v1", trigger)
        assert result["status"] == "promoted"
        assert result["version"] == "2.0.0"

    def test_run_without_pipeline_runner_raises(self) -> None:
        """Calling run() before set_pipeline_runner() must raise."""
        orch = RetrainOrchestrator(_cfg())
        trigger = orch.evaluator.manual("test")
        with pytest.raises(RetrainError, match="pipeline runner not configured"):
            orch.run("model", trigger)

    def test_pipeline_exception_wraps_in_retrain_error(self) -> None:
        """Pipeline failures must be wrapped in RetrainError."""
        orch = RetrainOrchestrator(_cfg(), audit=MagicMock())
        boom = MagicMock(side_effect=RuntimeError("GPU OOM"))
        orch.set_pipeline_runner(boom)
        trigger = orch.evaluator.manual("test")
        with pytest.raises(RetrainError, match="pipeline execution failed"):
            orch.run("model", trigger)


class TestValidationFloors:
    """min_performance validation checks."""

    def test_validation_failure_raises(self) -> None:
        """Challenger below the floor must be rejected."""
        cfg = _cfg(min_perf={"accuracy": 0.95})
        orch = RetrainOrchestrator(cfg, audit=MagicMock())
        orch.set_pipeline_runner(_runner(metrics={"accuracy": 0.80}))
        trigger = orch.evaluator.manual("test")
        with pytest.raises(RetrainError, match="validation failed"):
            orch.run("model", trigger)

    def test_validation_passes_when_above_floor(self) -> None:
        """Challenger that meets the floor should proceed."""
        cfg = _cfg(min_perf={"accuracy": 0.70})
        orch = RetrainOrchestrator(cfg, audit=MagicMock())
        orch.set_pipeline_runner(_runner(metrics={"accuracy": 0.90}))
        trigger = orch.evaluator.manual("test")
        result = orch.run("model", trigger)
        assert result["status"] == "promoted"

    def test_missing_metric_treated_as_zero(self) -> None:
        """A metric absent from challenger defaults to 0.0 and fails floor."""
        cfg = _cfg(min_perf={"f1": 0.5})
        orch = RetrainOrchestrator(cfg, audit=MagicMock())
        orch.set_pipeline_runner(_runner(metrics={"accuracy": 0.9}))
        trigger = orch.evaluator.manual("test")
        with pytest.raises(RetrainError, match="validation failed.*f1"):
            orch.run("model", trigger)


class TestHumanInLoopApproval:
    """Pending → approve / reject flow."""

    def test_human_in_loop_returns_pending(self) -> None:
        """human_in_loop mode should return pending_approval."""
        cfg = _cfg(mode="human_in_loop")
        orch = RetrainOrchestrator(cfg, audit=MagicMock())
        orch.set_pipeline_runner(_runner())
        trigger = orch.evaluator.manual("test")
        result = orch.run("model", trigger)
        assert result["status"] == "pending_approval"
        assert "request_id" in result

    def test_approve_promotes_pending_request(self) -> None:
        """Approving a pending request must promote it."""
        cfg = _cfg(mode="human_in_loop")
        registry = MagicMock()
        registry.get_latest.side_effect = Exception("none")
        orch = RetrainOrchestrator(cfg, registry=registry, audit=MagicMock())
        orch.set_pipeline_runner(_runner(version="3.0.0"))
        trigger = orch.evaluator.manual("test")
        pending = orch.run("model", trigger)
        result = orch.approve(pending["request_id"], by="alice")
        assert result["status"] == "promoted"
        assert result["version"] == "3.0.0"
        registry.promote.assert_called_once()

    def test_reject_returns_rejected(self) -> None:
        """Rejecting must return rejected status."""
        cfg = _cfg(mode="human_in_loop")
        orch = RetrainOrchestrator(cfg, audit=MagicMock())
        orch.set_pipeline_runner(_runner())
        trigger = orch.evaluator.manual("test")
        pending = orch.run("model", trigger)
        result = orch.reject(pending["request_id"], by="bob", comment="not ready")
        assert result["status"] == "rejected"


class TestOnDriftHook:
    """on_drift passes through to the evaluator."""

    def test_on_drift_returns_none_for_non_drift_trigger(self) -> None:
        """Manual trigger mode ignores drift reports."""
        orch = RetrainOrchestrator(_cfg(trigger="manual"))
        assert orch.on_drift(_drift_report(drifted=True)) is None

    def test_on_drift_forwards_to_evaluator(self) -> None:
        """drift_confirmed mode should delegate to the evaluator."""
        cfg = _cfg(trigger="drift_confirmed")
        orch = RetrainOrchestrator(cfg)
        # First drift — evaluator needs min 2 consecutive by default
        assert orch.on_drift(_drift_report()) is None
        # Second drift — triggers
        trigger = orch.on_drift(_drift_report())
        assert trigger is not None
        assert trigger.trigger_type == "drift_confirmed"


class TestAuditLogging:
    """Verify audit trail is called at key lifecycle points."""

    def test_audit_trail_called_on_run(self) -> None:
        """Audit should receive retrain_started and retrain_completed."""
        audit = MagicMock()
        orch = RetrainOrchestrator(_cfg(), audit=audit)
        orch.set_pipeline_runner(_runner())
        trigger = orch.evaluator.manual("test")
        orch.run("model", trigger)
        event_types = [c.kwargs["event_type"] for c in audit.log.call_args_list]
        assert "retrain_started" in event_types
        assert "retrain_completed" in event_types or "approval_requested" in event_types
