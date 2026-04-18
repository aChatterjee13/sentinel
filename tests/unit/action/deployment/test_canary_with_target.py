"""Integration: CanaryStrategy + a recording BaseDeploymentTarget."""

from __future__ import annotations

from sentinel.action.deployment.strategies.base import DeploymentPhase
from sentinel.action.deployment.strategies.canary import CanaryStrategy
from sentinel.action.deployment.targets.base import BaseDeploymentTarget
from sentinel.core.exceptions import DeploymentError


class _RecordingTarget(BaseDeploymentTarget):
    """Records every call for assertion in tests."""

    name = "recording"

    def __init__(self) -> None:
        self.splits: list[tuple[str, dict[str, int]]] = []
        self.rollbacks: list[tuple[str, str]] = []
        self.raise_on_next_split = False

    def set_traffic_split(self, model_name: str, weights: dict[str, int]) -> None:
        if self.raise_on_next_split:
            self.raise_on_next_split = False
            raise DeploymentError("boom")
        self.splits.append((model_name, dict(weights)))

    def health_check(self, model_name: str, version: str) -> bool:
        return True

    def rollback_to(self, model_name: str, version: str) -> None:
        self.rollbacks.append((model_name, version))


class TestCanaryAdvanceCallsTarget:
    def test_advance_calls_set_traffic_split_with_ramp_weights(self) -> None:
        target = _RecordingTarget()
        strategy = CanaryStrategy(
            model_name="fraud",
            ramp_steps=[5, 25, 50, 100],
            target=target,
        )
        state = strategy.start(from_version="1.0.0", to_version="1.1.0")
        # state starts at 5% — advance to 25%
        state = strategy.advance(state, observed_metrics={})
        assert target.splits[-1] == ("fraud", {"1.1.0": 25, "1.0.0": 75})
        # advance to 50%
        state = strategy.advance(state, observed_metrics={})
        assert target.splits[-1] == ("fraud", {"1.1.0": 50, "1.0.0": 50})
        # advance to 100%
        state = strategy.advance(state, observed_metrics={})
        assert target.splits[-1] == ("fraud", {"1.1.0": 100, "1.0.0": 0})
        assert state.phase == DeploymentPhase.PROMOTED

    def test_target_failure_triggers_rollback(self) -> None:
        target = _RecordingTarget()
        target.raise_on_next_split = True
        strategy = CanaryStrategy(
            model_name="fraud",
            ramp_steps=[5, 25, 50, 100],
            target=target,
        )
        state = strategy.start(from_version="1.0.0", to_version="1.1.0")
        new_state = strategy.advance(state, observed_metrics={})
        assert new_state.phase == DeploymentPhase.ROLLED_BACK
        assert target.rollbacks == [("fraud", "1.0.0")]
        assert "target error" in (new_state.error or "")

    def test_metric_rollback_uses_target(self) -> None:
        target = _RecordingTarget()
        strategy = CanaryStrategy(
            model_name="fraud",
            ramp_steps=[5, 25, 50, 100],
            rollback_on={"error_rate_increase": 0.01},
            target=target,
        )
        state = strategy.start(from_version="1.0.0", to_version="1.1.0")
        new_state = strategy.advance(state, observed_metrics={"error_rate_increase": 0.05})
        assert new_state.phase == DeploymentPhase.ROLLED_BACK
        # Metric-triggered rollback should also call target.rollback_to
        assert target.rollbacks == [("fraud", "1.0.0")]

    def test_no_from_version_skips_rollback_target_call(self) -> None:
        target = _RecordingTarget()
        strategy = CanaryStrategy(
            model_name="fraud",
            ramp_steps=[5, 25, 50, 100],
            rollback_on={"error_rate_increase": 0.01},
            target=target,
        )
        state = strategy.start(from_version=None, to_version="1.0.0")
        new_state = strategy.advance(state, observed_metrics={"error_rate_increase": 0.05})
        assert new_state.phase == DeploymentPhase.ROLLED_BACK
        # No from_version → no rollback_to call
        assert target.rollbacks == []
