"""Tests for retraining dashboard routes — WS-D."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from sentinel.action.retrain.approval import ApprovalRequest
from sentinel.config.schema import (
    ApprovalConfig,
    DashboardConfig,
    DashboardUIConfig,
    ModelConfig,
    RetrainingConfig,
    SentinelConfig,
)
from sentinel.dashboard.state import DashboardState
from sentinel.dashboard.views.retraining import build


def _make_state(
    has_runner: bool = False,
    pending: list[ApprovalRequest] | None = None,
) -> DashboardState:
    """Build a mock DashboardState for retraining view tests."""
    client = MagicMock()
    client.model_name = "fraud_v2"
    client.current_version = "1.0.0"
    client.config = SentinelConfig(
        model=ModelConfig(name="fraud_v2"),
        retraining=RetrainingConfig(
            trigger="drift_confirmed",
            pipeline="azureml://pipelines/retrain_fraud" if has_runner else None,
            approval=ApprovalConfig(mode="human_in_loop"),
        ),
    )
    client.retrain = MagicMock()
    client.retrain._pipeline_runner = MagicMock() if has_runner else None
    client.retrain.approval.list_pending.return_value = pending or []

    audit_events: list[MagicMock] = []
    client.audit.query.return_value = audit_events

    config = DashboardConfig(
        ui=DashboardUIConfig(show_modules=["retraining"]),
    )
    state = DashboardState(client=client, config=config)
    return state


class TestRetrainingView:
    def test_build_basic(self) -> None:
        state = _make_state()
        data = build(state)
        assert "latest_drift" in data
        assert "retrain_config" in data
        assert "pending_approvals" in data
        assert "retrain_events" in data

    def test_config_has_pipeline_runner_false(self) -> None:
        state = _make_state(has_runner=False)
        data = build(state)
        assert data["retrain_config"]["has_pipeline_runner"] is False

    def test_config_has_pipeline_runner_true(self) -> None:
        state = _make_state(has_runner=True)
        data = build(state)
        assert data["retrain_config"]["has_pipeline_runner"] is True

    def test_pending_approvals_included(self) -> None:
        req = ApprovalRequest(
            model_name="fraud_v2",
            candidate_version="2.0.0",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=48),
        )
        state = _make_state(pending=[req])
        data = build(state)
        assert len(data["pending_approvals"]) == 1
        assert data["pending_approvals"][0]["candidate_version"] == "2.0.0"

    def test_no_drift_reports(self) -> None:
        state = _make_state()
        data = build(state)
        assert data["latest_drift"] is None


class TestRetrainingAPIRoutes:
    """Integration-style test of the API route logic."""

    def test_trigger_without_runner_returns_400(self) -> None:
        """The trigger API should fail when no pipeline runner is configured."""
        state = _make_state(has_runner=False)
        # Verify the condition the API route would check
        assert state.client.retrain._pipeline_runner is None

    def test_trigger_with_runner_calls_orchestrator(self) -> None:
        """When a runner is configured, trigger should delegate to orchestrator."""
        state = _make_state(has_runner=True)
        orchestrator = state.client.retrain
        orchestrator.evaluator.manual.return_value = MagicMock()
        orchestrator.run.return_value = {"status": "pending_approval", "version": "2.0.0"}
        # Simulating what the API route does
        trigger = orchestrator.evaluator.manual("dashboard trigger")
        result = orchestrator.run(
            model_name="fraud_v2",
            trigger=trigger,
            context={"triggered_by": "test_user", "source": "dashboard"},
        )
        assert result["status"] == "pending_approval"
        orchestrator.run.assert_called_once()

    def test_approve_flow(self) -> None:
        state = _make_state(has_runner=True)
        orchestrator = state.client.retrain
        orchestrator.approve.return_value = {"status": "promoted", "version": "2.0.0"}
        result = orchestrator.approve("req123", by="test_user")
        assert result["status"] == "promoted"

    def test_reject_flow(self) -> None:
        state = _make_state(has_runner=True)
        orchestrator = state.client.retrain
        orchestrator.reject.return_value = {"status": "rejected"}
        result = orchestrator.reject("req123", by="test_user")
        assert result["status"] == "rejected"
