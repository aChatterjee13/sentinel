"""Unit tests for sentinel.agentops.tool_audit.replay — ToolReplayStore."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sentinel.agentops.tool_audit.monitor import ToolCallRecord
from sentinel.agentops.tool_audit.replay import ToolReplayStore, _hash_inputs


@pytest.fixture
def store(tmp_path) -> ToolReplayStore:
    return ToolReplayStore(root=tmp_path / "replay")


def _make_record(
    tool: str = "search",
    agent: str = "agent_a",
    inputs: dict | None = None,
    output: object = "result",
) -> ToolCallRecord:
    return ToolCallRecord(
        agent=agent,
        tool=tool,
        inputs=inputs or {"query": "test"},
        output=output,
        success=True,
        latency_ms=10.0,
        timestamp=datetime.now(timezone.utc),
    )


class TestSaveAndReplay:
    """Round-trip: save a record then replay by (tool, inputs)."""

    def test_save_then_replay_from_cache(self, store: ToolReplayStore) -> None:
        rec = _make_record(output={"data": [1, 2, 3]})
        store.save(rec)
        result = store.replay("search", {"query": "test"})
        assert result == {"data": [1, 2, 3]}

    def test_replay_from_disk_after_cache_miss(self, tmp_path) -> None:
        store1 = ToolReplayStore(root=tmp_path / "replay")
        rec = _make_record(output="disk_result")
        store1.save(rec)

        # New store instance — empty cache, must read from disk
        store2 = ToolReplayStore(root=tmp_path / "replay")
        result = store2.replay("search", {"query": "test"})
        assert result == "disk_result"

    def test_replay_returns_none_for_unknown_tool(self, store: ToolReplayStore) -> None:
        assert store.replay("nonexistent_tool", {"x": 1}) is None

    def test_replay_returns_none_for_unknown_inputs(self, store: ToolReplayStore) -> None:
        store.save(_make_record(inputs={"q": "a"}, output="found"))
        assert store.replay("search", {"q": "other"}) is None

    def test_save_multiple_same_tool(self, store: ToolReplayStore) -> None:
        store.save(_make_record(inputs={"q": "alpha"}, output="A"))
        store.save(_make_record(inputs={"q": "beta"}, output="B"))

        assert store.replay("search", {"q": "alpha"}) == "A"
        assert store.replay("search", {"q": "beta"}) == "B"


class TestListRecordings:
    """Listing stored recording files."""

    def test_list_recordings_empty(self, store: ToolReplayStore) -> None:
        # Tool directory doesn't exist yet
        assert store.list_recordings("search") == []

    def test_list_recordings_after_save(self, store: ToolReplayStore) -> None:
        store.save(_make_record())
        files = store.list_recordings("search")
        assert len(files) == 1
        assert files[0].suffix == ".jsonl"


class TestHashInputs:
    """Deterministic input hashing."""

    def test_same_inputs_same_hash(self) -> None:
        a = _hash_inputs({"x": 1, "y": "hello"})
        b = _hash_inputs({"y": "hello", "x": 1})
        assert a == b

    def test_different_inputs_different_hash(self) -> None:
        a = _hash_inputs({"x": 1})
        b = _hash_inputs({"x": 2})
        assert a != b

    def test_empty_inputs(self) -> None:
        h = _hash_inputs({})
        assert h == "{}"

    def test_nested_inputs(self) -> None:
        a = _hash_inputs({"a": {"b": 1}})
        b = _hash_inputs({"a": {"b": 1}})
        assert a == b


class TestEdgeCases:
    """None values, empty data, corrupt files."""

    def test_save_none_output(self, store: ToolReplayStore) -> None:
        rec = _make_record(output=None)
        store.save(rec)
        result = store.replay("search", {"query": "test"})
        assert result is None

    def test_replay_ignores_corrupt_lines(self, tmp_path) -> None:
        store = ToolReplayStore(root=tmp_path / "replay")
        store.save(_make_record(output="good"))

        # Append a corrupt line to the JSONL file
        tool_dir = tmp_path / "replay" / "search"
        jsonl_files = list(tool_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        with jsonl_files[0].open("a") as f:
            f.write("NOT VALID JSON\n")

        # New store reads from disk — corrupt line is skipped
        store2 = ToolReplayStore(root=tmp_path / "replay")
        result = store2.replay("search", {"query": "test"})
        assert result == "good"
