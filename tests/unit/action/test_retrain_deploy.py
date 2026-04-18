"""Tests for Gap-A: retrain → deploy wiring."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from sentinel.action.retrain.orchestrator import RetrainOrchestrator
from sentinel.config.schema import ApprovalConfig, RetrainingConfig, ValidationConfig


def _make_config(*, deploy_on_promote: bool = False) -> RetrainingConfig:
    return RetrainingConfig(
        trigger="manual",
        pipeline="test://pipeline",
        approval=ApprovalConfig(mode="auto"),
        validation=ValidationConfig(min_performance={"accuracy": 0.5}),
        deploy_on_promote=deploy_on_promote,
    )


def _make_runner(version: str = "2.0.0", metrics: dict[str, float] | None = None) -> Any:
    """Return a pipeline runner that produces a canned result."""
    result = {
        "version": version,
        "metrics": metrics or {"accuracy": 0.9, "f1": 0.85},
    }
    return MagicMock(return_value=result)


class TestDeployOnPromoteDisabled:
    def test_no_deployment_when_flag_off(self) -> None:
        config = _make_config(deploy_on_promote=False)
        dm = MagicMock(name="DeploymentManager")
        registry = MagicMock()
        registry.get_latest.side_effect = Exception("none")
        orch = RetrainOrchestrator(config, registry=registry, audit=None, deployment_manager=dm)
        orch.set_pipeline_runner(_make_runner())
        trigger = orch.evaluator.manual("test")
        result = orch.run("my_model", trigger)
        assert result["status"] == "promoted"
        assert "deployment" not in result
        dm.start.assert_not_called()


class TestDeployOnPromoteEnabled:
    def test_deployment_triggered_on_auto_approve(self) -> None:
        config = _make_config(deploy_on_promote=True)
        dm = MagicMock(name="DeploymentManager")
        deploy_state = MagicMock()
        deploy_state.model_dump.return_value = {"phase": "running", "traffic_pct": 5}
        dm.start.return_value = deploy_state
        registry = MagicMock()
        registry.get_latest.side_effect = Exception("none")
        orch = RetrainOrchestrator(config, registry=registry, audit=None, deployment_manager=dm)
        orch.set_pipeline_runner(_make_runner(version="3.0.0"))
        trigger = orch.evaluator.manual("test")
        result = orch.run("my_model", trigger)
        assert result["status"] == "promoted"
        assert "deployment" in result
        dm.start.assert_called_once_with(model_name="my_model", to_version="3.0.0")

    def test_no_deployment_without_manager(self) -> None:
        config = _make_config(deploy_on_promote=True)
        registry = MagicMock()
        registry.get_latest.side_effect = Exception("none")
        orch = RetrainOrchestrator(config, registry=registry, audit=None, deployment_manager=None)
        orch.set_pipeline_runner(_make_runner())
        trigger = orch.evaluator.manual("test")
        result = orch.run("my_model", trigger)
        assert result["status"] == "promoted"
        assert "deployment" not in result


class TestApproveTriggersDeployment:
    def test_approve_triggers_deploy(self) -> None:
        config = _make_config(deploy_on_promote=True)
        config = config.model_copy(update={"approval": ApprovalConfig(mode="human_in_loop")})
        dm = MagicMock(name="DeploymentManager")
        deploy_state = MagicMock()
        deploy_state.model_dump.return_value = {"phase": "running"}
        dm.start.return_value = deploy_state
        registry = MagicMock()
        registry.get_latest.side_effect = Exception("none")
        orch = RetrainOrchestrator(config, registry=registry, audit=None, deployment_manager=dm)
        orch.set_pipeline_runner(_make_runner(version="4.0.0"))
        trigger = orch.evaluator.manual("test")
        run_result = orch.run("my_model", trigger)
        assert run_result["status"] == "pending_approval"
        dm.start.assert_not_called()

        # Now approve
        request_id = run_result["request_id"]
        approve_result = orch.approve(request_id, by="alice")
        assert approve_result["status"] == "promoted"
        assert "deployment" in approve_result
        dm.start.assert_called_once_with(model_name="my_model", to_version="4.0.0")
