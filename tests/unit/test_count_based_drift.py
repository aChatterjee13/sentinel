"""Tests for Gap-B: count-based automatic drift checking."""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np

from sentinel.config.schema import (
    AuditConfig,
    DataDriftConfig,
    DriftAutoCheckConfig,
    DriftConfig,
    ModelConfig,
    SentinelConfig,
)
from sentinel.core.client import SentinelClient


def _make_config(*, auto_enabled: bool = False, every_n: int = 5) -> SentinelConfig:
    return SentinelConfig(
        model=ModelConfig(name="test_model"),
        drift=DriftConfig(
            data=DataDriftConfig(method="psi", threshold=0.2),
            auto_check=DriftAutoCheckConfig(enabled=auto_enabled, every_n_predictions=every_n),
        ),
        audit=AuditConfig(storage="local"),
    )


class TestDriftAutoCheckConfig:
    def test_defaults(self) -> None:
        cfg = DriftAutoCheckConfig()
        assert cfg.enabled is False
        assert cfg.every_n_predictions == 1000

    def test_custom(self) -> None:
        cfg = DriftAutoCheckConfig(enabled=True, every_n_predictions=500)
        assert cfg.enabled is True
        assert cfg.every_n_predictions == 500


class TestCountBasedDrift:
    def test_no_auto_check_when_disabled(self) -> None:
        config = _make_config(auto_enabled=False, every_n=3)
        client = SentinelClient(config)
        try:
            with patch.object(client, "_safe_check_drift") as mock_check:
                for i in range(10):
                    client.log_prediction(features={"x": float(i)}, prediction=0)
                mock_check.assert_not_called()
        finally:
            client.close()

    def test_auto_check_triggers_at_threshold(self) -> None:
        config = _make_config(auto_enabled=True, every_n=5)
        client = SentinelClient(config)
        try:
            # Fit a baseline so check_drift doesn't fail
            baseline = np.random.default_rng(42).normal(size=(100, 1))
            client.fit_baseline(baseline)

            call_count = 0

            def counting_check() -> None:
                nonlocal call_count
                call_count += 1

            with patch.object(client, "_safe_check_drift", side_effect=counting_check):
                # Log 4 predictions — should NOT trigger
                for i in range(4):
                    client.log_prediction(features={"x": float(i)}, prediction=0)
                time.sleep(0.05)
                assert call_count == 0

                # 5th prediction should trigger
                client.log_prediction(features={"x": 5.0}, prediction=0)
                time.sleep(0.1)
                assert call_count == 1

        finally:
            client.close()

    def test_counter_resets_on_manual_check(self) -> None:
        config = _make_config(auto_enabled=True, every_n=5)
        client = SentinelClient(config)
        try:
            baseline = np.random.default_rng(42).normal(size=(100, 1))
            client.fit_baseline(baseline)

            # Log 3 predictions
            for i in range(3):
                client.log_prediction(features={"x": float(i)}, prediction=0)
            assert client._predictions_since_check == 3

            # Manual check resets counter
            client.check_drift(baseline)
            assert client._predictions_since_check == 0
        finally:
            client.close()
