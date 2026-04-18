"""Tests for sentinel.config.source — config field source tracing."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.config.loader import ConfigLoader
from sentinel.config.source import (
    ConfigSource,
    SourceMap,
    annotate,
    harvest,
    merge_with_sources,
)
from sentinel.core.exceptions import ConfigValidationError


class TestSourceMap:
    def test_set_and_get(self) -> None:
        sm = SourceMap()
        src = ConfigSource(file=Path("/tmp/a.yaml"), merged_from=(Path("/tmp/a.yaml"),))
        sm.set("model.name", src)
        assert sm.get("model.name") == src
        assert sm.get("missing") is None
        assert "model.name" in sm
        assert len(sm) == 1

    def test_lookup_walks_back_to_parent_path(self) -> None:
        sm = SourceMap()
        src = ConfigSource(file=Path("/tmp/a.yaml"), merged_from=(Path("/tmp/a.yaml"),))
        sm.set("alerts.channels.0", src)
        # Pydantic will report loc=("alerts", "channels", 0, "webhook_url")
        # — we should fall back to the parent path's source.
        result = sm.lookup(("alerts", "channels", 0, "webhook_url"))
        assert result == src

    def test_lookup_returns_none_when_no_match(self) -> None:
        sm = SourceMap()
        assert sm.lookup(("missing", "field")) is None


class TestAnnotateAndHarvest:
    def test_round_trip_preserves_values(self) -> None:
        path = Path("/tmp/x.yaml")
        annotated = annotate({"model": {"name": "m"}}, path)
        cleaned, sm = harvest(annotated)
        assert cleaned == {"model": {"name": "m"}}
        assert sm.get("model.name") is not None
        assert sm.get("model.name").file == path  # type: ignore[union-attr]

    def test_lists_are_walked(self) -> None:
        path = Path("/tmp/x.yaml")
        annotated = annotate({"items": [{"name": "a"}, {"name": "b"}]}, path)
        cleaned, sm = harvest(annotated)
        assert cleaned == {"items": [{"name": "a"}, {"name": "b"}]}
        assert sm.get("items.0.name") is not None
        assert sm.get("items.1.name") is not None


class TestMergeWithSources:
    def test_child_overrides_parent_chain(self) -> None:
        parent_path = Path("/tmp/base.yaml")
        child_path = Path("/tmp/child.yaml")
        parent = annotate({"model": {"name": "base"}}, parent_path)
        child = annotate({"model": {"name": "child"}}, child_path)
        merged = merge_with_sources(parent, child, child_path)
        cleaned, sm = harvest(merged)
        assert cleaned == {"model": {"name": "child"}}
        src = sm.get("model.name")
        assert src is not None
        assert src.file == child_path
        assert parent_path in src.merged_from
        assert child_path in src.merged_from

    def test_parent_only_field_kept(self) -> None:
        parent_path = Path("/tmp/base.yaml")
        child_path = Path("/tmp/child.yaml")
        parent = annotate({"model": {"name": "p"}, "drift": {"method": "psi"}}, parent_path)
        child = annotate({"model": {"name": "c"}}, child_path)
        merged = merge_with_sources(parent, child, child_path)
        cleaned, sm = harvest(merged)
        assert cleaned["drift"]["method"] == "psi"
        drift_src = sm.get("drift.method")
        assert drift_src is not None
        assert drift_src.file == parent_path


class TestSourceTracingInLoaderErrors:
    def test_parent_field_error_mentions_parent_file(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text(
            """
version: "1.0"
model:
  name: bm
  type: not_a_type
"""
        )
        child = tmp_path / "child.yaml"
        child.write_text(
            """
extends: base.yaml
model:
  domain: tabular
"""
        )
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader(child).load()
        msg = str(exc_info.value)
        assert "base.yaml" in msg

    def test_child_override_error_mentions_child_file(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text(
            """
version: "1.0"
model:
  name: bm
  type: classification
"""
        )
        child = tmp_path / "child.yaml"
        child.write_text(
            """
extends: base.yaml
model:
  type: not_a_type
"""
        )
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader(child).load()
        msg = str(exc_info.value)
        assert "child.yaml" in msg
        # Both files appear in the merged_from chain.
        assert "base.yaml" in msg
