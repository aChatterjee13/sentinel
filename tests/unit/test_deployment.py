"""Unit tests for deployment strategies, promotion policy, and manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.action.deployment import (
    STRATEGY_REGISTRY,
    BlueGreenStrategy,
    CanaryStrategy,
    DeploymentManager,
    DirectStrategy,
    PromotionPolicy,
    ShadowStrategy,
    register_strategy,
)
from sentinel.action.deployment.strategies.base import (
    DeploymentPhase,
    DeploymentState,
)
from sentinel.config.schema import (
    BlueGreenConfig,
    CanaryConfig,
    DeploymentConfig,
)
from sentinel.core.exceptions import DeploymentError
from sentinel.foundation.registry.backends.local import LocalRegistryBackend
from sentinel.foundation.registry.model_registry import ModelRegistry


class TestCanaryStrategy:
    def test_start_at_initial_step(self) -> None:
        s = CanaryStrategy(model_name="m", ramp_steps=[5, 25, 50, 100])
        state = s.start(from_version="1.0.0", to_version="2.0.0")
        assert state.traffic_pct == 5
        assert state.phase == DeploymentPhase.RUNNING
        assert state.from_version == "1.0.0"
        assert state.to_version == "2.0.0"

    def test_advance_ramps_up(self) -> None:
        s = CanaryStrategy(model_name="m", ramp_steps=[5, 25, 50, 100])
        state = s.start(None, "2.0.0")
        state = s.advance(state, observed_metrics={})
        assert state.traffic_pct == 25
        state = s.advance(state, observed_metrics={})
        assert state.traffic_pct == 50

    def test_advance_to_100_promotes(self) -> None:
        s = CanaryStrategy(model_name="m", ramp_steps=[50, 100])
        state = s.start(None, "2.0.0")
        state = s.advance(state, observed_metrics={})
        assert state.traffic_pct == 100
        assert state.phase == DeploymentPhase.PROMOTED

    def test_rollback_on_metric_breach(self) -> None:
        s = CanaryStrategy(
            model_name="m",
            ramp_steps=[5, 25, 50, 100],
            rollback_on={"error_rate_increase": 0.02},
        )
        state = s.start(None, "2.0.0")
        state = s.advance(state, observed_metrics={"error_rate_increase": 0.05})
        assert state.phase == DeploymentPhase.ROLLED_BACK
        assert state.traffic_pct == 0
        assert "error_rate_increase" in (state.error or "")

    def test_metric_within_threshold_continues_rampup(self) -> None:
        s = CanaryStrategy(
            model_name="m",
            ramp_steps=[5, 25, 100],
            rollback_on={"error_rate_increase": 0.02},
        )
        state = s.start(None, "2.0.0")
        state = s.advance(state, observed_metrics={"error_rate_increase": 0.001})
        assert state.phase == DeploymentPhase.RUNNING
        assert state.traffic_pct == 25

    def test_metrics_accumulate(self) -> None:
        s = CanaryStrategy(model_name="m", ramp_steps=[5, 100])
        state = s.start(None, "2.0.0")
        state = s.advance(state, observed_metrics={"latency_p99": 120.0})
        assert state.metrics["latency_p99"] == 120.0


class TestShadowStrategy:
    def test_start_zero_traffic(self) -> None:
        s = ShadowStrategy(model_name="m", duration="24h")
        state = s.start(None, "1.0.0")
        assert state.traffic_pct == 0
        assert state.phase == DeploymentPhase.RUNNING
        assert state.metrics["shadow"] == 1.0

    def test_advance_records_metrics(self) -> None:
        s = ShadowStrategy(model_name="m")
        state = s.start(None, "1.0.0")
        state = s.advance(state, {"f1_diff": 0.02})
        assert state.metrics["f1_diff"] == 0.02
        # Shadow advance does not change phase or traffic
        assert state.phase == DeploymentPhase.RUNNING
        assert state.traffic_pct == 0

    def test_promote_goes_to_full_traffic(self) -> None:
        s = ShadowStrategy(model_name="m")
        state = s.start(None, "1.0.0")
        state = s.promote(state)
        assert state.traffic_pct == 100
        assert state.phase == DeploymentPhase.PROMOTED


class TestBlueGreenStrategy:
    def test_start_zero_traffic(self) -> None:
        s = BlueGreenStrategy(model_name="m", warmup_seconds=10)
        state = s.start("1.0.0", "2.0.0")
        assert state.traffic_pct == 0
        assert state.phase == DeploymentPhase.RUNNING

    def test_advance_health_check_passes_promotes(self) -> None:
        s = BlueGreenStrategy(model_name="m")
        state = s.start("1.0.0", "2.0.0")
        state = s.advance(state, {"health_check_passed": 1.0})
        assert state.phase == DeploymentPhase.PROMOTED
        assert state.traffic_pct == 100

    def test_advance_health_check_fails_rolls_back(self) -> None:
        s = BlueGreenStrategy(model_name="m")
        state = s.start("1.0.0", "2.0.0")
        state = s.advance(state, {"health_check_passed": 0.0})
        assert state.phase == DeploymentPhase.ROLLED_BACK
        assert "health check" in (state.error or "")


class TestDirectStrategy:
    def test_start_immediately_promoted(self) -> None:
        s = DirectStrategy(model_name="m")
        state = s.start("1.0.0", "2.0.0")
        assert state.phase == DeploymentPhase.PROMOTED
        assert state.traffic_pct == 100

    def test_rollback_marks_state(self) -> None:
        s = DirectStrategy(model_name="m")
        state = s.start("1.0.0", "2.0.0")
        state = s.rollback(state, "manual abort")
        assert state.phase == DeploymentPhase.ROLLED_BACK
        assert state.error == "manual abort"


class TestPromotionPolicy:
    def test_improvement_pct_pass(self) -> None:
        p = PromotionPolicy(metric="f1", improvement_pct=2.0)
        assert p.should_promote({"f1": 0.85}, {"f1": 0.88})

    def test_improvement_pct_fail(self) -> None:
        p = PromotionPolicy(metric="f1", improvement_pct=5.0)
        assert not p.should_promote({"f1": 0.85}, {"f1": 0.86})

    def test_min_metrics_floor(self) -> None:
        p = PromotionPolicy(metric="f1", min_metrics={"f1": 0.9})
        assert not p.should_promote({"f1": 0.5}, {"f1": 0.85})
        assert p.should_promote({"f1": 0.5}, {"f1": 0.92})

    def test_require_all_metrics_better(self) -> None:
        p = PromotionPolicy(require_all_metrics_better=True)
        # Challenger better on f1 but worse on auc
        assert not p.should_promote({"f1": 0.8, "auc": 0.9}, {"f1": 0.85, "auc": 0.85})
        assert p.should_promote({"f1": 0.8, "auc": 0.9}, {"f1": 0.85, "auc": 0.92})

    def test_explain_returns_diagnostic(self) -> None:
        p = PromotionPolicy(metric="f1", improvement_pct=2.0)
        diag = p.explain({"f1": 0.8}, {"f1": 0.84})
        assert diag["should_promote"] is True
        assert diag["improvements"]["f1"]["delta"] == pytest.approx(0.04)

    def test_zero_champion_promotes_if_challenger_positive(self) -> None:
        p = PromotionPolicy(metric="f1")
        assert p.should_promote({"f1": 0}, {"f1": 0.5})
        assert not p.should_promote({"f1": 0}, {"f1": 0})


class TestStrategyRegistry:
    def test_default_strategies_registered(self) -> None:
        for name in ["shadow", "canary", "blue_green", "direct"]:
            assert name in STRATEGY_REGISTRY

    def test_register_custom_strategy(self) -> None:
        class _Custom(DirectStrategy):
            name = "custom_strategy"

        register_strategy("custom_strategy", _Custom)
        try:
            assert STRATEGY_REGISTRY["custom_strategy"] is _Custom
        finally:
            STRATEGY_REGISTRY.pop("custom_strategy", None)


class TestDeploymentManager:
    def _registry(self, tmp_path: Path) -> ModelRegistry:
        return ModelRegistry(backend=LocalRegistryBackend(root=tmp_path / "reg"))

    def _manager(self, tmp_path: Path, strategy: str = "canary") -> DeploymentManager:
        cfg = DeploymentConfig(
            strategy=strategy,  # type: ignore[arg-type]
            canary=CanaryConfig(ramp_steps=[10, 100], rollback_on={"err": 0.05}),
            blue_green=BlueGreenConfig(warmup_seconds=1),
        )
        return DeploymentManager(cfg, registry=self._registry(tmp_path))

    def test_start_creates_state(self, tmp_path: Path) -> None:
        m = self._manager(tmp_path)
        m.registry.register("fraud", "2.0.0")  # type: ignore[union-attr]
        state = m.start("fraud", "2.0.0")
        assert state.model_name == "fraud"
        assert state.to_version == "2.0.0"
        assert state.strategy == "canary"
        assert state.traffic_pct == 10

    def test_start_uses_existing_production_as_from_version(self, tmp_path: Path) -> None:
        m = self._manager(tmp_path)
        m.registry.register("fraud", "1.0.0")  # type: ignore[union-attr]
        m.registry.promote("fraud", "1.0.0", status="production")  # type: ignore[union-attr]
        m.registry.register("fraud", "2.0.0")  # type: ignore[union-attr]
        state = m.start("fraud", "2.0.0")
        assert state.from_version == "1.0.0"

    def test_advance_promotes_in_registry(self, tmp_path: Path) -> None:
        m = self._manager(tmp_path)
        m.registry.register("fraud", "2.0.0")  # type: ignore[union-attr]
        state = m.start("fraud", "2.0.0")
        # Advance to 100% (single hop with ramp_steps=[10,100])
        state = m.advance(state, {})
        assert state.phase == DeploymentPhase.PROMOTED
        # Registry should reflect the promotion
        assert m.registry.get("fraud", "2.0.0").status == "production"  # type: ignore[union-attr]

    def test_advance_rollback_breach(self, tmp_path: Path) -> None:
        m = self._manager(tmp_path)
        m.registry.register("fraud", "2.0.0")  # type: ignore[union-attr]
        state = m.start("fraud", "2.0.0")
        state = m.advance(state, {"err": 0.99})
        assert state.phase == DeploymentPhase.ROLLED_BACK

    def test_get_unknown_raises(self, tmp_path: Path) -> None:
        m = self._manager(tmp_path)
        with pytest.raises(DeploymentError):
            m.get("not-a-real-id")

    def test_list_active_filters_running(self, tmp_path: Path) -> None:
        m = self._manager(tmp_path)
        m.registry.register("model_a", "1.0.0")  # type: ignore[union-attr]
        m.registry.register("model_b", "1.0.0")  # type: ignore[union-attr]
        s1 = m.start("model_a", "1.0.0")
        s2 = m.start("model_b", "1.0.0")
        # Roll back one
        m.rollback(s1, "test")
        active = m.list_active()
        active_ids = {s.deployment_id for s in active}
        assert s2.deployment_id in active_ids
        assert s1.deployment_id not in active_ids

    def test_unknown_strategy_raises(self, tmp_path: Path) -> None:
        # Use a known strategy at config level, then override with unknown name
        m = self._manager(tmp_path)
        m.registry.register("fraud", "2.0.0")  # type: ignore[union-attr]
        with pytest.raises(DeploymentError):
            m.start("fraud", "2.0.0", strategy_override="bogus_strategy")


class TestDeploymentState:
    def test_state_phase_default(self) -> None:
        s = DeploymentState(
            deployment_id="abc",
            model_name="m",
            from_version=None,
            to_version="1.0.0",
            strategy="canary",
        )
        assert s.phase == DeploymentPhase.PENDING
        assert s.traffic_pct == 0
