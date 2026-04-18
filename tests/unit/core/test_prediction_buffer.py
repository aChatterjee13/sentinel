"""Tests for prediction buffer: IDs, log_actual, and flush_buffer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentinel import SentinelClient
from sentinel.config.schema import SentinelConfig


class TestPredictionId:
    """prediction_id is generated, unique, and returned by log_prediction."""

    def test_prediction_id_is_generated(self, minimal_config: SentinelConfig) -> None:
        client = SentinelClient(minimal_config)
        pid = client.log_prediction(features={"x": 1.0}, prediction=0)
        assert isinstance(pid, str)
        assert len(pid) == 16

    def test_prediction_ids_are_unique(self, minimal_config: SentinelConfig) -> None:
        client = SentinelClient(minimal_config)
        ids = {client.log_prediction(features={"x": float(i)}, prediction=0) for i in range(50)}
        assert len(ids) == 50


class TestLogActual:
    """log_actual attaches ground truth to a buffered prediction."""

    def test_attaches_actual_to_correct_prediction(
        self, minimal_config: SentinelConfig
    ) -> None:
        client = SentinelClient(minimal_config)
        pid = client.log_prediction(features={"x": 1.0}, prediction=0)
        client.log_actual(pid, actual=1)
        # Verify via _actuals mapping
        assert client._actuals[pid] == 1

    def test_unknown_id_logs_warning(
        self, minimal_config: SentinelConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = SentinelClient(minimal_config)
        with caplog.at_level("WARNING"):
            client.log_actual("nonexistent_id_00", actual=42)
        assert "nonexistent_id_00" not in client._actuals

    def test_actual_appears_in_flush(
        self, minimal_config: SentinelConfig, tmp_path: Path
    ) -> None:
        client = SentinelClient(minimal_config)
        pid = client.log_prediction(features={"x": 1.0}, prediction=0)
        client.log_actual(pid, actual=1)
        out = tmp_path / "buf.jsonl"
        client.flush_buffer(out)
        record = json.loads(out.read_text().strip())
        assert record["actual"] == 1
        assert record["prediction_id"] == pid


class TestFlushBuffer:
    """flush_buffer writes valid JSONL with all expected fields."""

    def test_writes_valid_jsonl(
        self, minimal_config: SentinelConfig, tmp_path: Path
    ) -> None:
        client = SentinelClient(minimal_config)
        for i in range(5):
            client.log_prediction(features={"x": float(i)}, prediction=i % 2)
        out = tmp_path / "buf.jsonl"
        count = client.flush_buffer(out)
        assert count == 5
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 5
        for line in lines:
            obj = json.loads(line)
            assert isinstance(obj, dict)

    def test_includes_all_fields(
        self, minimal_config: SentinelConfig, tmp_path: Path
    ) -> None:
        client = SentinelClient(minimal_config)
        client.log_prediction(
            features={"a": 1.0, "b": 2.0},
            prediction=1,
            confidence=0.95,
        )
        out = tmp_path / "buf.jsonl"
        client.flush_buffer(out)
        record = json.loads(out.read_text().strip())
        assert set(record.keys()) == {
            "prediction_id",
            "features",
            "prediction",
            "actual",
            "timestamp",
            "confidence",
        }
        assert record["features"] == {"a": 1.0, "b": 2.0}
        assert record["prediction"] == 1
        assert record["confidence"] == 0.95
        assert record["timestamp"] is not None

    def test_does_not_clear_buffer(
        self, minimal_config: SentinelConfig, tmp_path: Path
    ) -> None:
        client = SentinelClient(minimal_config)
        for i in range(3):
            client.log_prediction(features={"x": float(i)}, prediction=0)
        out = tmp_path / "buf.jsonl"
        client.flush_buffer(out)
        assert client.buffer_size() == 3

    def test_unsupported_format_raises(
        self, minimal_config: SentinelConfig, tmp_path: Path
    ) -> None:
        client = SentinelClient(minimal_config)
        with pytest.raises(ValueError, match="unsupported format"):
            client.flush_buffer(tmp_path / "out.csv", format="csv")

    def test_flush_empty_buffer(
        self, minimal_config: SentinelConfig, tmp_path: Path
    ) -> None:
        client = SentinelClient(minimal_config)
        out = tmp_path / "empty.jsonl"
        count = client.flush_buffer(out)
        assert count == 0
        assert out.read_text() == ""


class TestBufferOverflow:
    """Oldest predictions are evicted when the buffer exceeds maxlen."""

    def test_overflow_drops_oldest(self, minimal_config: SentinelConfig) -> None:
        client = SentinelClient(minimal_config)
        # The default maxlen is 10_000
        first_ids: list[str] = []
        for i in range(10_000):
            pid = client.log_prediction(features={"x": float(i)}, prediction=0)
            if i < 5:
                first_ids.append(pid)

        assert client.buffer_size() == 10_000

        # Push 5 more — the first 5 should be evicted
        for i in range(5):
            client.log_prediction(features={"x": float(10_000 + i)}, prediction=0)

        assert client.buffer_size() == 10_000
        remaining_ids = {r.prediction_id for r in client._prediction_buffer}
        for old_id in first_ids:
            assert old_id not in remaining_ids
