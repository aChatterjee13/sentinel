"""Tests for DeploymentManager hardening — gaps 1, 3, 4, 5, 6."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

import pytest

from sentinel.action.deployment.manager import DeploymentManager
from sentinel.action.deployment.strategies.base import DeploymentPhase, DeploymentState
from sentinel.action.retrain.approval import ApprovalGate, ApprovalStatus
from sentinel.config.schema import (
    DeploymentConfig,
    SageMakerEndpointTargetConfig,
    VertexAIEndpointTargetConfig,
)
from sentinel.core.exceptions import DeploymentError

# ── Helpers ────────────────────────────────────────────────────────


def _local_config(**overrides) -> DeploymentConfig:
    defaults = {"strategy": "canary", "target": "local"}
    defaults.update(overrides)
    return DeploymentConfig(**defaults)


def _make_manager(
    config: DeploymentConfig | None = None,
    registry: MagicMock | None = None,
) -> DeploymentManager:
    """Build a manager with a local target so no cloud SDKs are needed."""
    cfg = config or _local_config()
    return DeploymentManager(cfg, registry=registry)


# ── Gap 1: _build_target for SageMaker / Vertex AI ─────────────────


class TestBuildTargetSageMaker:
    def test_build_target_sagemaker(self) -> None:
        cfg = DeploymentConfig(
            strategy="canary",
            target="sagemaker_endpoint",
            sagemaker_endpoint=SageMakerEndpointTargetConfig(
                endpoint_name="my-ep",
                region_name="us-east-1",
            ),
        )
        with patch(
            "sentinel.action.deployment.targets.resolve_target"
        ) as mock_resolve:
            mock_resolve.return_value = MagicMock()
            target = DeploymentManager._build_target(cfg)
            mock_resolve.assert_called_once_with(
                "sagemaker_endpoint",
                endpoint_name="my-ep",
                region_name="us-east-1",
                variant_name_pattern="{model_name}-{version}",
            )
            assert target is mock_resolve.return_value


class TestBuildTargetVertexAI:
    def test_build_target_vertex_ai(self) -> None:
        cfg = DeploymentConfig(
            strategy="canary",
            target="vertex_ai_endpoint",
            vertex_ai_endpoint=VertexAIEndpointTargetConfig(
                endpoint_name="my-vertex-ep",
                project="gcp-proj",
                location="europe-west1",
            ),
        )
        with patch(
            "sentinel.action.deployment.targets.resolve_target"
        ) as mock_resolve:
            mock_resolve.return_value = MagicMock()
            target = DeploymentManager._build_target(cfg)
            mock_resolve.assert_called_once_with(
                "vertex_ai_endpoint",
                endpoint_name="my-vertex-ep",
                project="gcp-proj",
                location="europe-west1",
            )
            assert target is mock_resolve.return_value


# ── Gap 3: Thread safety ───────────────────────────────────────────


class TestThreadSafetyConcurrentStarts:
    def test_concurrent_starts_no_lost_deployments(self) -> None:
        dm = _make_manager()
        n_threads = 10

        def _start(i: int) -> DeploymentState:
            return dm.start(model_name="fraud", to_version=f"1.0.{i}")

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(_start, i) for i in range(n_threads)]
            states = [f.result() for f in as_completed(futures)]

        assert len(states) == n_threads
        # Every deployment should be stored
        ids = {s.deployment_id for s in states}
        assert len(ids) == n_threads
        for s in states:
            assert dm.get(s.deployment_id) == s


# ── Gap 4: advance passes baseline to canary ───────────────────────


class TestAdvancePassesBaseline:
    def test_advance_passes_baseline_to_canary(self) -> None:
        registry = MagicMock()
        baseline_entry = MagicMock()
        baseline_entry.metrics = {"f1": 0.90, "accuracy": 0.92}
        registry.get_latest.return_value = baseline_entry
        registry.list_by_status.return_value = []
        registry.backend = MagicMock()
        registry.backend.exists.return_value = True

        dm = _make_manager(registry=registry)
        state = dm.start(model_name="fraud", to_version="2.0.0")
        assert state.strategy == "canary"

        # Verify the canary strategy receives
        # baseline_metrics by spying on the actual call
        strategy = dm._build_strategy(state.model_name, override=state.strategy)
        original_advance = strategy.advance

        call_args: list[tuple] = []

        def spy_advance(*args, **kwargs):
            call_args.append((args, kwargs))
            return original_advance(*args, **kwargs)

        strategy.advance = spy_advance  # type: ignore[assignment]

        with patch.object(dm, "_build_strategy", return_value=strategy):
            dm.advance(state, observed_metrics={"error_rate_increase": 0.001})

        assert len(call_args) == 1
        _, kwargs = call_args[0]
        assert "baseline_metrics" in kwargs
        assert kwargs["baseline_metrics"] == {"f1": 0.90, "accuracy": 0.92}

    def test_advance_no_baseline_when_registry_none(self) -> None:
        dm = _make_manager(registry=None)
        state = dm.start(model_name="fraud", to_version="2.0.0")

        strategy = dm._build_strategy(state.model_name, override=state.strategy)
        original_advance = strategy.advance
        call_args: list[tuple] = []

        def spy_advance(*args, **kwargs):
            call_args.append((args, kwargs))
            return original_advance(*args, **kwargs)

        strategy.advance = spy_advance  # type: ignore[assignment]

        with patch.object(dm, "_build_strategy", return_value=strategy):
            dm.advance(state, observed_metrics={"error_rate_increase": 0.001})

        assert len(call_args) == 1
        _, kwargs = call_args[0]
        # No baseline_metrics kwarg when registry is None
        assert "baseline_metrics" not in kwargs


# ── Gap 6: start() validates version exists ────────────────────────


class TestStartValidatesVersion:
    def test_start_validates_version_exists(self) -> None:
        registry = MagicMock()
        registry.backend = MagicMock()
        registry.backend.exists.return_value = False
        registry.list_by_status.return_value = []

        dm = _make_manager(registry=registry)
        with pytest.raises(DeploymentError, match="not found in registry"):
            dm.start(model_name="fraud", to_version="9.9.9")

    def test_start_allows_valid_version(self) -> None:
        registry = MagicMock()
        registry.backend = MagicMock()
        registry.backend.exists.return_value = True
        registry.list_by_status.return_value = []

        dm = _make_manager(registry=registry)
        state = dm.start(model_name="fraud", to_version="1.0.0")
        assert state.to_version == "1.0.0"
        assert state.phase == DeploymentPhase.RUNNING

    def test_start_skips_validation_when_no_registry(self) -> None:
        dm = _make_manager(registry=None)
        state = dm.start(model_name="fraud", to_version="1.0.0")
        assert state.to_version == "1.0.0"


# ── Gap 5: PromotionPolicy integration in ApprovalGate ─────────────


class TestPromotionPolicyIntegration:
    def test_meets_auto_promote_delegates_to_promotion_policy(self) -> None:
        from sentinel.config.schema import ApprovalConfig

        cfg = ApprovalConfig(
            mode="hybrid",
            timeout="48h",
            auto_promote_if={"metric": "f1", "improvement_pct": 2.0},
            approvers=["admin@co.com"],
        )
        gate = ApprovalGate(cfg)
        try:
            # challenger f1 = 0.90 vs champion f1 = 0.85 → ~5.9% → above 2%
            assert gate._meets_auto_promote(
                champion={"f1": 0.85}, challenger={"f1": 0.90}
            )
            # challenger f1 = 0.855 vs champion f1 = 0.85 → ~0.6% → below 2%
            assert not gate._meets_auto_promote(
                champion={"f1": 0.85}, challenger={"f1": 0.855}
            )
        finally:
            gate.close()

    def test_hybrid_auto_approves_when_criteria_met(self) -> None:
        from sentinel.config.schema import ApprovalConfig

        cfg = ApprovalConfig(
            mode="hybrid",
            timeout="48h",
            auto_promote_if={"metric": "f1", "improvement_pct": 2.0},
            approvers=["admin@co.com"],
        )
        gate = ApprovalGate(cfg)
        try:
            req = gate.request(
                model_name="fraud",
                candidate_version="2.0.0",
                champion_metrics={"f1": 0.85},
                challenger_metrics={"f1": 0.90},
            )
            assert req.status == ApprovalStatus.AUTO_APPROVED
        finally:
            gate.close()
