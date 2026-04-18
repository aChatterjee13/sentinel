"""Unit tests for foundation modules: AuditTrail and ModelRegistry."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sentinel.config.schema import AuditConfig
from sentinel.core.exceptions import ModelNotFoundError
from sentinel.core.types import AuditEvent
from sentinel.foundation.audit.trail import AuditTrail
from sentinel.foundation.registry.backends.local import LocalRegistryBackend
from sentinel.foundation.registry.model_registry import ModelRegistry, ModelVersion
from sentinel.foundation.registry.versioning import bump_version, parse_version


class TestAuditTrail:
    def _trail(self, tmp_path: Path) -> AuditTrail:
        return AuditTrail(AuditConfig(path=str(tmp_path / "audit"), retention_days=30))

    def test_log_appends_event(self, tmp_path: Path) -> None:
        trail = self._trail(tmp_path)
        event = trail.log("model_registered", model_name="m", model_version="1.0.0")
        assert event.event_type == "model_registered"
        # File should exist
        files = list((tmp_path / "audit").glob("audit-*.jsonl"))
        assert len(files) == 1

    def test_query_filters_by_event_type(self, tmp_path: Path) -> None:
        trail = self._trail(tmp_path)
        trail.log("drift_detected", model_name="m")
        trail.log("alert_sent", model_name="m")
        trail.log("drift_detected", model_name="m")
        events = list(trail.query(event_type="drift_detected"))
        assert len(events) == 2
        assert all(e.event_type == "drift_detected" for e in events)

    def test_query_filters_by_model_name(self, tmp_path: Path) -> None:
        trail = self._trail(tmp_path)
        trail.log("drift_detected", model_name="m1")
        trail.log("drift_detected", model_name="m2")
        events = list(trail.query(model_name="m1"))
        assert len(events) == 1
        assert events[0].model_name == "m1"

    def test_query_with_limit(self, tmp_path: Path) -> None:
        trail = self._trail(tmp_path)
        for i in range(10):
            trail.log("ping", model_name=f"m{i}")
        events = list(trail.query(limit=3))
        assert len(events) == 3

    def test_log_event_persists_prebuilt(self, tmp_path: Path) -> None:
        trail = self._trail(tmp_path)
        ev = AuditEvent(event_type="custom", model_name="m", payload={"k": "v"})
        trail.log_event(ev)
        events = list(trail.query(event_type="custom"))
        assert len(events) == 1
        assert events[0].payload["k"] == "v"

    def test_latest_returns_tail(self, tmp_path: Path) -> None:
        trail = self._trail(tmp_path)
        for i in range(5):
            trail.log("ping", model_name=f"m{i}")
        recent = trail.latest(n=2)
        assert len(recent) == 2
        assert recent[-1].model_name == "m4"

    def test_query_skips_blank_lines(self, tmp_path: Path) -> None:
        trail = self._trail(tmp_path)
        trail.log("ping", model_name="m")
        # Inject a blank line into the file
        files = list((tmp_path / "audit").glob("audit-*.jsonl"))
        with files[0].open("a", encoding="utf-8") as f:
            f.write("\n   \n")
        trail.log("ping", model_name="n")
        events = list(trail.query())
        assert len(events) == 2

    def test_query_skips_invalid_json(self, tmp_path: Path) -> None:
        trail = self._trail(tmp_path)
        trail.log("ping", model_name="m")
        files = list((tmp_path / "audit").glob("audit-*.jsonl"))
        with files[0].open("a", encoding="utf-8") as f:
            f.write("not valid json\n")
        events = list(trail.query())
        assert len(events) == 1

    def test_enforce_retention_removes_old_files(self, tmp_path: Path) -> None:
        trail = AuditTrail(AuditConfig(path=str(tmp_path / "audit"), retention_days=1))
        trail.log("ping", model_name="m")
        # Create a stale file from 5 days ago
        old_day = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
        stale = tmp_path / "audit" / f"audit-{old_day}.jsonl"
        stale.write_text('{"event_type": "old"}\n')
        removed = trail.enforce_retention()
        assert removed == 1
        assert not stale.exists()

    def test_enforce_retention_ignores_unparseable_filenames(self, tmp_path: Path) -> None:
        trail = self._trail(tmp_path)
        (tmp_path / "audit").mkdir(exist_ok=True)
        # Create a file that doesn't match the audit-YYYY-MM-DD pattern
        bad = tmp_path / "audit" / "audit-not-a-date.jsonl"
        bad.write_text("garbage")
        removed = trail.enforce_retention()
        assert removed == 0
        assert bad.exists()


class TestModelRegistry:
    def _registry(self, tmp_path: Path) -> ModelRegistry:
        return ModelRegistry(backend=LocalRegistryBackend(root=tmp_path / "registry"))

    def test_register_creates_version(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        mv = r.register("fraud", "1.0.0", framework="sklearn", metrics={"f1": 0.9})
        assert mv.name == "fraud"
        assert mv.version == "1.0.0"
        assert mv.framework == "sklearn"
        assert mv.metrics["f1"] == 0.9

    def test_get_returns_version(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        r.register("fraud", "1.0.0", framework="sklearn")
        loaded = r.get("fraud", "1.0.0")
        assert loaded.name == "fraud"
        assert loaded.version == "1.0.0"

    def test_get_unknown_raises(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        with pytest.raises(ModelNotFoundError):
            r.get("missing", "1.0.0")

    def test_get_latest_returns_highest_semver(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        r.register("fraud", "1.0.0")
        r.register("fraud", "1.10.0")
        r.register("fraud", "1.2.0")
        latest = r.get_latest("fraud")
        # Semver sort: 1.10.0 > 1.2.0 > 1.0.0
        assert latest.version == "1.10.0"

    def test_get_latest_no_versions_raises(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        with pytest.raises(ModelNotFoundError):
            r.get_latest("none")

    def test_list_versions_sorted_semver(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        r.register("m", "2.0.0")
        r.register("m", "1.0.0")
        r.register("m", "1.5.0")
        assert r.list_versions("m") == ["1.0.0", "1.5.0", "2.0.0"]

    def test_list_models(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        r.register("a", "1.0.0")
        r.register("b", "1.0.0")
        models = r.list_models()
        assert set(models) == {"a", "b"}

    def test_update_patches_fields(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        r.register("m", "1.0.0", framework="sklearn", metrics={"f1": 0.8})
        updated = r.update("m", "1.0.0", metrics={"f1": 0.9, "auc": 0.95})
        assert updated.metrics["f1"] == 0.9
        assert updated.metrics["auc"] == 0.95
        # Framework preserved
        assert updated.framework == "sklearn"

    def test_promote_changes_status(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        r.register("m", "1.0.0")
        promoted = r.promote("m", "1.0.0", status="production")
        assert promoted.status == "production"

    def test_list_by_status(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        r.register("m", "1.0.0")
        r.register("m", "1.1.0")
        r.promote("m", "1.0.0", status="production")
        prod = r.list_by_status("m", "production")
        assert len(prod) == 1
        assert prod[0].version == "1.0.0"

    def test_get_baseline_returns_metrics_and_schema(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        r.register(
            "m",
            "1.0.0",
            metrics={"f1": 0.9},
            feature_schema={"x": "float"},
            baseline_dataset="s3://bucket/ref.parquet",
        )
        baseline = r.get_baseline("m", "1.0.0")
        assert baseline["metrics"]["f1"] == 0.9
        assert baseline["feature_schema"] == {"x": "float"}
        assert baseline["baseline_dataset"] == "s3://bucket/ref.parquet"

    def test_get_baseline_uses_latest_when_version_omitted(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        r.register("m", "1.0.0", metrics={"f1": 0.8})
        r.register("m", "1.1.0", metrics={"f1": 0.9})
        baseline = r.get_baseline("m")
        assert baseline["metrics"]["f1"] == 0.9

    def test_compare_computes_metric_delta(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        r.register("m", "1.0.0", metrics={"f1": 0.8, "auc": 0.85})
        r.register("m", "1.1.0", metrics={"f1": 0.9, "auc": 0.90})
        diff = r.compare("m", "1.0.0", "1.1.0")
        assert abs(diff["metrics"]["f1"]["delta"] - 0.1) < 1e-9
        assert abs(diff["metrics"]["auc"]["delta"] - 0.05) < 1e-9

    def test_next_version_starts_at_default(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        assert r.next_version("brand_new") == "0.1.0"

    def test_next_version_bumps_patch(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        r.register("m", "1.2.3")
        assert r.next_version("m", level="patch") == "1.2.4"

    def test_next_version_bumps_minor(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        r.register("m", "1.2.3")
        assert r.next_version("m", level="minor") == "1.3.0"

    def test_next_version_bumps_major(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        r.register("m", "1.2.3")
        assert r.next_version("m", level="major") == "2.0.0"

    def test_register_if_new_skips_existing(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        r.register("m", "1.0.0", framework="sklearn")
        result = r.register_if_new("m", "1.0.0", framework="xgboost")
        # Should return existing, not overwrite
        assert result.framework == "sklearn"

    def test_register_if_new_creates_when_missing(self, tmp_path: Path) -> None:
        r = self._registry(tmp_path)
        result = r.register_if_new("m", "1.0.0", framework="sklearn")
        assert result.framework == "sklearn"
        assert r.backend.exists("m", "1.0.0")


class TestVersioning:
    def test_parse_version_basic(self) -> None:
        v = parse_version("1.2.3")
        assert (v.major, v.minor, v.patch) == (1, 2, 3)

    def test_parse_version_with_suffix(self) -> None:
        v = parse_version("2.0.0-rc1")
        assert (v.major, v.minor, v.patch) == (2, 0, 0)

    def test_parse_version_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_version("not-a-version")

    def test_bump_patch(self) -> None:
        assert bump_version("1.2.3", "patch") == "1.2.4"

    def test_bump_minor_resets_patch(self) -> None:
        assert bump_version("1.2.3", "minor") == "1.3.0"

    def test_bump_major_resets_minor_and_patch(self) -> None:
        assert bump_version("1.2.3", "major") == "2.0.0"

    def test_bump_unknown_level_raises(self) -> None:
        with pytest.raises(ValueError):
            bump_version("1.2.3", "spicy")

    def test_version_comparison(self) -> None:
        assert parse_version("1.0.0") < parse_version("1.0.1")
        assert parse_version("1.9.0") < parse_version("1.10.0")
        assert parse_version("2.0.0") > parse_version("1.99.99")


class TestModelVersionModel:
    def test_default_status(self) -> None:
        mv = ModelVersion(name="m", version="1.0.0")
        assert mv.status == "registered"

    def test_extra_fields_allowed(self) -> None:
        mv = ModelVersion(name="m", version="1.0.0", custom_field="hello")
        # extra="allow" → custom_field should be accessible via model_dump
        assert mv.model_dump().get("custom_field") == "hello"
