"""Tests for ShadowStrategy target interactions.

Verifies that the shadow strategy correctly delegates to deployment
targets for traffic splitting, health checks, and rollback.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sentinel.action.deployment.strategies.base import DeploymentPhase
from sentinel.action.deployment.strategies.shadow import ShadowStrategy
from sentinel.action.deployment.targets.local import LocalDeploymentTarget
from sentinel.core.exceptions import DeploymentError


@pytest.fixture()
def mock_target() -> MagicMock:
    """Return a mock deployment target with sane defaults."""
    target = MagicMock()
    target.health_check.return_value = True
    return target


# ── start ──────────────────────────────────────────────────────────


def test_start_calls_target_with_zero_traffic(mock_target: MagicMock) -> None:
    strategy = ShadowStrategy(model_name="fraud", target=mock_target)
    state = strategy.start(from_version="1.0", to_version="2.0")

    mock_target.set_traffic_split.assert_called_once_with(
        "fraud", {"1.0": 100, "2.0": 0}
    )
    assert state.traffic_pct == 0
    assert state.phase == DeploymentPhase.RUNNING
    assert state.to_version == "2.0"


def test_start_without_from_version_skips_target(mock_target: MagicMock) -> None:
    strategy = ShadowStrategy(model_name="fraud", target=mock_target)
    state = strategy.start(from_version=None, to_version="2.0")

    mock_target.set_traffic_split.assert_not_called()
    assert state.traffic_pct == 0
    assert state.phase == DeploymentPhase.RUNNING


def test_target_error_doesnt_crash_start(mock_target: MagicMock) -> None:
    mock_target.set_traffic_split.side_effect = DeploymentError("boom")
    strategy = ShadowStrategy(model_name="fraud", target=mock_target)

    state = strategy.start(from_version="1.0", to_version="2.0")

    assert state.phase == DeploymentPhase.RUNNING
    assert state.traffic_pct == 0


# ── advance ────────────────────────────────────────────────────────


def test_advance_calls_health_check(mock_target: MagicMock) -> None:
    strategy = ShadowStrategy(model_name="fraud", target=mock_target)
    state = strategy.start(from_version="1.0", to_version="2.0")

    updated = strategy.advance(state, {"accuracy": 0.95})

    mock_target.health_check.assert_called_once_with("fraud", "2.0")
    assert updated.metrics["accuracy"] == 0.95


def test_advance_continues_on_unhealthy(mock_target: MagicMock) -> None:
    mock_target.health_check.return_value = False
    strategy = ShadowStrategy(model_name="fraud", target=mock_target)
    state = strategy.start(from_version="1.0", to_version="2.0")

    updated = strategy.advance(state, {"accuracy": 0.80})

    assert updated.metrics["accuracy"] == 0.80
    assert updated.phase == DeploymentPhase.RUNNING


def test_advance_continues_on_health_check_error(mock_target: MagicMock) -> None:
    mock_target.health_check.side_effect = DeploymentError("unreachable")
    strategy = ShadowStrategy(model_name="fraud", target=mock_target)
    state = strategy.start(from_version="1.0", to_version="2.0")

    updated = strategy.advance(state, {"f1": 0.88})

    assert updated.metrics["f1"] == 0.88


# ── rollback ───────────────────────────────────────────────────────


def test_rollback_calls_target_rollback(mock_target: MagicMock) -> None:
    strategy = ShadowStrategy(model_name="fraud", target=mock_target)
    state = strategy.start(from_version="1.0", to_version="2.0")

    rolled = strategy.rollback(state, "metrics degraded")

    mock_target.rollback_to.assert_called_once_with("fraud", "1.0")
    assert rolled.phase == DeploymentPhase.ROLLED_BACK
    assert rolled.error == "metrics degraded"


def test_rollback_without_from_version_skips_target(mock_target: MagicMock) -> None:
    strategy = ShadowStrategy(model_name="fraud", target=mock_target)
    state = strategy.start(from_version=None, to_version="2.0")

    rolled = strategy.rollback(state, "bad shadow")

    mock_target.rollback_to.assert_not_called()
    assert rolled.phase == DeploymentPhase.ROLLED_BACK


# ── promote ────────────────────────────────────────────────────────


def test_promote_calls_target_full_traffic(mock_target: MagicMock) -> None:
    strategy = ShadowStrategy(model_name="fraud", target=mock_target)
    state = strategy.start(from_version="1.0", to_version="2.0")

    promoted = strategy.promote(state)

    mock_target.set_traffic_split.assert_called_with("fraud", {"2.0": 100})
    assert promoted.phase == DeploymentPhase.PROMOTED
    assert promoted.traffic_pct == 100


def test_target_error_doesnt_crash_promote(mock_target: MagicMock) -> None:
    strategy = ShadowStrategy(model_name="fraud", target=mock_target)
    state = strategy.start(from_version="1.0", to_version="2.0")

    mock_target.set_traffic_split.side_effect = DeploymentError("promote failed")
    promoted = strategy.promote(state)

    assert promoted.phase == DeploymentPhase.PROMOTED
    assert promoted.traffic_pct == 100


# ── LocalDeploymentTarget integration ─────────────────────────────


def test_shadow_with_local_target_noop() -> None:
    """LocalDeploymentTarget is a no-op — shadow lifecycle runs cleanly."""
    target = LocalDeploymentTarget()
    strategy = ShadowStrategy(model_name="fraud", target=target)

    state = strategy.start(from_version="1.0", to_version="2.0")
    assert state.phase == DeploymentPhase.RUNNING

    state = strategy.advance(state, {"accuracy": 0.91})
    assert state.metrics["accuracy"] == 0.91

    promoted = strategy.promote(state)
    assert promoted.phase == DeploymentPhase.PROMOTED
