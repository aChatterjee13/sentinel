"""Unit tests for the SentinelClient — the SDK entry point."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel import SentinelClient
from sentinel.config.schema import SentinelConfig
from sentinel.core.exceptions import SentinelError


class TestSentinelClientConstruction:
    def test_builds_from_minimal_config(self, minimal_config: SentinelConfig) -> None:
        client = SentinelClient(minimal_config)
        assert client.model_name == "test_model"
        assert client.config.model.domain == "tabular"

    def test_resolves_tabular_domain_adapter(self, minimal_config: SentinelConfig) -> None:
        client = SentinelClient(minimal_config)
        assert client.domain_adapter.domain == "tabular"

    def test_status_returns_summary(self, minimal_config: SentinelConfig) -> None:
        client = SentinelClient(minimal_config)
        status = client.status()
        assert status["model"] == "test_model"
        assert status["domain"] == "tabular"
        assert status["buffer"] == 0
        assert status["llmops_enabled"] is False
        assert status["agentops_enabled"] is False


class TestPredictionLogging:
    def test_log_prediction_appends_to_buffer(self, minimal_config: SentinelConfig) -> None:
        client = SentinelClient(minimal_config)
        pid = client.log_prediction(features={"a": 1.0, "b": 2.0}, prediction=1)
        assert isinstance(pid, str)
        assert len(pid) == 16
        assert client.buffer_size() == 1

    def test_clear_buffer(self, minimal_config: SentinelConfig) -> None:
        client = SentinelClient(minimal_config)
        for _ in range(10):
            client.log_prediction(features={"x": 1.0}, prediction=0)
        assert client.buffer_size() == 10
        client.clear_buffer()
        assert client.buffer_size() == 0

    def test_log_prediction_with_metadata(self, minimal_config: SentinelConfig) -> None:
        client = SentinelClient(minimal_config)
        pid = client.log_prediction(
            features={"a": 1.0},
            prediction=0,
            confidence=0.95,
            user_id="abc",
        )
        assert isinstance(pid, str)


class TestDriftDetection:
    def test_check_drift_raises_without_data(self, minimal_config: SentinelConfig) -> None:
        client = SentinelClient(minimal_config)
        with pytest.raises(SentinelError):
            client.check_drift()

    def test_check_drift_uses_buffer(self, minimal_config: SentinelConfig) -> None:
        client = SentinelClient(minimal_config)
        rng = np.random.default_rng(0)
        ref = rng.normal(size=(200, 1))
        client.fit_baseline(ref)
        for v in rng.normal(size=200):
            client.log_prediction(features={"x": float(v)}, prediction=0)
        report = client.check_drift()
        assert report.method == "psi"

    def test_check_drift_explicit_data(self, minimal_config: SentinelConfig) -> None:
        client = SentinelClient(minimal_config)
        rng = np.random.default_rng(0)
        ref = rng.normal(size=(200, 2))
        cur = rng.normal(loc=3.0, size=(200, 2))
        client.fit_baseline(ref)
        report = client.check_drift(cur)
        assert report.is_drifted


class TestRegistry:
    def test_register_model(self, minimal_config: SentinelConfig) -> None:
        client = SentinelClient(minimal_config)
        mv = client.register_model(version="1.0.0", framework="sklearn")
        assert mv.version == "1.0.0"
        assert client.current_version == "1.0.0"


class TestLazyAccessors:
    def test_llmops_disabled_raises(self, minimal_config: SentinelConfig) -> None:
        client = SentinelClient(minimal_config)
        with pytest.raises(SentinelError):
            _ = client.llmops

    def test_agentops_disabled_raises(self, minimal_config: SentinelConfig) -> None:
        client = SentinelClient(minimal_config)
        with pytest.raises(SentinelError):
            _ = client.agentops
