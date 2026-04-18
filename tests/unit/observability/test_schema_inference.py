"""Unit tests for schema inference, fit/merge, enum/pattern validation, and data profile."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sentinel.config.schema import DataQualityConfig, FreshnessConfig, OutlierConfig, SchemaConfig
from sentinel.core.types import AlertSeverity
from sentinel.observability.data_quality import DataQualityChecker


def _make_config(**overrides) -> DataQualityConfig:
    defaults = {
        "schema": SchemaConfig(enforce=True),
        "freshness": FreshnessConfig(max_age_hours=24),
        "outlier_detection": OutlierConfig(method="iqr", contamination=0.05),
        "null_threshold": 0.1,
        "duplicate_threshold": 0.05,
    }
    defaults.update(overrides)
    return DataQualityConfig(**defaults)


# ── infer_schema ───────────────────────────────────────────────────


class TestInferSchema:
    """Tests for DataQualityChecker.infer_schema()."""

    def test_mixed_types(self) -> None:
        """Infer correct JSON Schema types from mixed-type data."""
        rows = [
            {"age": 30, "score": 0.95, "name": "Alice", "active": True},
            {"age": 25, "score": 0.80, "name": "Bob", "active": False},
        ]
        config = _make_config(schema=SchemaConfig(enforce=True))
        checker = DataQualityChecker(config, model_name="test")
        schema = checker.infer_schema(rows)

        props = schema["properties"]
        assert props["age"]["type"] == "integer"
        assert props["score"]["type"] == "number"
        assert props["name"]["type"] == "string"
        assert props["active"]["type"] == "boolean"

    def test_enum_inference_low_cardinality(self) -> None:
        """String columns with ≤20 unique values produce an enum."""
        rows = [{"status": v} for v in ["open", "closed", "pending"] * 5]
        config = _make_config(schema=SchemaConfig(enforce=True))
        checker = DataQualityChecker(config, model_name="test")
        schema = checker.infer_schema(rows)

        assert "enum" in schema["properties"]["status"]
        assert set(schema["properties"]["status"]["enum"]) == {"open", "closed", "pending"}

    def test_no_enum_high_cardinality(self) -> None:
        """String columns with >20 unique values should not produce an enum."""
        rows = [{"id": f"user_{i}"} for i in range(25)]
        config = _make_config(schema=SchemaConfig(enforce=True))
        checker = DataQualityChecker(config, model_name="test")
        schema = checker.infer_schema(rows)

        assert "enum" not in schema["properties"]["id"]

    def test_required_field_inference(self) -> None:
        """Columns with zero nulls are marked required."""
        rows = [
            {"age": 30, "name": "Alice"},
            {"age": 25, "name": None},
        ]
        config = _make_config(schema=SchemaConfig(enforce=True))
        checker = DataQualityChecker(config, model_name="test")
        schema = checker.infer_schema(rows)

        assert "age" in schema["required"]
        assert "name" not in schema["required"]

    def test_min_max_range_inference(self) -> None:
        """Numeric columns should have min/max inferred."""
        rows = [{"val": v} for v in [10, 20, 30, 40, 50]]
        config = _make_config(schema=SchemaConfig(enforce=True))
        checker = DataQualityChecker(config, model_name="test")
        schema = checker.infer_schema(rows)

        assert schema["properties"]["val"]["minimum"] == 10
        assert schema["properties"]["val"]["maximum"] == 50

    def test_numpy_array_input(self) -> None:
        """infer_schema accepts a 2-D numpy array with feature_names."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        config = _make_config(schema=SchemaConfig(enforce=True))
        checker = DataQualityChecker(config, model_name="test")
        schema = checker.infer_schema(arr, feature_names=["x", "y"])

        assert "x" in schema["properties"]
        assert "y" in schema["properties"]
        assert schema["properties"]["x"]["type"] == "number"
        assert schema["properties"]["x"]["minimum"] == 1.0
        assert schema["properties"]["y"]["maximum"] == 6.0
        assert "x" in schema["required"]

    def test_numpy_array_without_names_autogenerates(self) -> None:
        """Passing a numpy array without feature_names auto-generates names."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        config = _make_config(schema=SchemaConfig(enforce=True))
        checker = DataQualityChecker(config, model_name="test")
        schema = checker.infer_schema(arr)
        assert "feature_0" in schema["properties"]
        assert "feature_1" in schema["properties"]

    def test_saves_to_disk(self, tmp_path) -> None:
        """Schema is saved to disk when schema_.path is configured."""
        out = tmp_path / "sub" / "schema.json"
        config = _make_config(schema=SchemaConfig(enforce=True, path=str(out)))
        checker = DataQualityChecker(config, model_name="test")
        rows = [{"a": 1, "b": "x"}]
        checker.infer_schema(rows)

        assert out.exists()
        saved = json.loads(out.read_text())
        assert "a" in saved["properties"]

    def test_stores_as_internal_schema(self) -> None:
        """infer_schema sets self._schema for future check() calls."""
        config = _make_config(schema=SchemaConfig(enforce=True))
        checker = DataQualityChecker(config, model_name="test")
        assert checker._schema is None
        checker.infer_schema([{"x": 1}])
        assert checker._schema is not None
        assert "x" in checker._schema["properties"]

    def test_empty_data(self) -> None:
        """infer_schema on empty data returns a valid but empty schema."""
        config = _make_config(schema=SchemaConfig(enforce=True))
        checker = DataQualityChecker(config, model_name="test")
        schema = checker.infer_schema([])
        assert schema["properties"] == {}
        assert schema["required"] == []


# ── fit ────────────────────────────────────────────────────────────


class TestFit:
    """Tests for DataQualityChecker.fit()."""

    def test_fit_no_existing_schema(self) -> None:
        """fit() with no on-disk schema creates one from data."""
        config = _make_config(schema=SchemaConfig(enforce=True))
        checker = DataQualityChecker(config, model_name="test")
        assert checker._schema is None

        schema = checker.fit([{"age": 30}, {"age": 25}])
        assert "age" in schema["properties"]
        assert checker._schema is not None

    def test_fit_merges_with_existing_schema(self, tmp_path) -> None:
        """fit() merges inferred with on-disk schema; manual overrides win."""
        manual_schema = {
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 200},
            },
            "required": ["age"],
        }
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(manual_schema))

        config = _make_config(schema=SchemaConfig(enforce=True, path=str(schema_path)))
        checker = DataQualityChecker(config, model_name="test")

        # Reference data has age and score
        rows = [{"age": 30, "score": 0.9}, {"age": 25, "score": 0.8}]
        schema = checker.fit(rows)

        # Manual override for age wins (min=0, max=200 instead of inferred)
        assert schema["properties"]["age"]["minimum"] == 0
        assert schema["properties"]["age"]["maximum"] == 200
        # Inferred score is also present
        assert "score" in schema["properties"]

    def test_fit_stores_reference_stats(self) -> None:
        """fit() stores per-feature statistics internally."""
        config = _make_config(schema=SchemaConfig(enforce=True))
        checker = DataQualityChecker(config, model_name="test")
        checker.fit([{"x": 10}, {"x": 20}, {"x": 30}])

        assert checker._reference_stats is not None
        assert "x" in checker._reference_stats
        assert checker._reference_stats["x"]["mean"] == pytest.approx(20.0)

    def test_fit_numpy(self) -> None:
        """fit() works with numpy array input."""
        arr = np.array([[1, 2], [3, 4]])
        config = _make_config(schema=SchemaConfig(enforce=True))
        checker = DataQualityChecker(config, model_name="test")
        schema = checker.fit(arr, feature_names=["a", "b"])
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]


# ── enum validation ────────────────────────────────────────────────


class TestEnumValidation:
    """Tests for schema.enum validation in _validate_schema."""

    def test_enum_violation(self, tmp_path) -> None:
        """Invalid enum value produces a WARNING issue."""
        schema = {
            "properties": {"color": {"type": "string", "enum": ["red", "blue"]}},
        }
        path = tmp_path / "s.json"
        path.write_text(json.dumps(schema))
        config = _make_config(schema=SchemaConfig(enforce=True, path=str(path)))
        checker = DataQualityChecker(config, model_name="test")

        report = checker.check({"color": "green"})
        enum_issues = [i for i in report.issues if i.rule == "schema.enum"]
        assert len(enum_issues) == 1
        assert enum_issues[0].severity == AlertSeverity.WARNING

    def test_enum_valid_value(self, tmp_path) -> None:
        """Valid enum value produces no issue."""
        schema = {
            "properties": {"color": {"type": "string", "enum": ["red", "blue"]}},
        }
        path = tmp_path / "s.json"
        path.write_text(json.dumps(schema))
        config = _make_config(schema=SchemaConfig(enforce=True, path=str(path)))
        checker = DataQualityChecker(config, model_name="test")

        report = checker.check({"color": "red"})
        enum_issues = [i for i in report.issues if i.rule == "schema.enum"]
        assert len(enum_issues) == 0


# ── pattern validation ─────────────────────────────────────────────


class TestPatternValidation:
    """Tests for schema.pattern validation in _validate_schema."""

    def test_pattern_violation(self, tmp_path) -> None:
        """Non-matching pattern produces a WARNING issue."""
        schema = {
            "properties": {"zip": {"type": "string", "pattern": r"^\d{5}$"}},
        }
        path = tmp_path / "s.json"
        path.write_text(json.dumps(schema))
        config = _make_config(schema=SchemaConfig(enforce=True, path=str(path)))
        checker = DataQualityChecker(config, model_name="test")

        report = checker.check({"zip": "ABCDE"})
        pat_issues = [i for i in report.issues if i.rule == "schema.pattern"]
        assert len(pat_issues) == 1
        assert pat_issues[0].severity == AlertSeverity.WARNING

    def test_pattern_valid_value(self, tmp_path) -> None:
        """Matching pattern produces no issue."""
        schema = {
            "properties": {"zip": {"type": "string", "pattern": r"^\d{5}$"}},
        }
        path = tmp_path / "s.json"
        path.write_text(json.dumps(schema))
        config = _make_config(schema=SchemaConfig(enforce=True, path=str(path)))
        checker = DataQualityChecker(config, model_name="test")

        report = checker.check({"zip": "12345"})
        pat_issues = [i for i in report.issues if i.rule == "schema.pattern"]
        assert len(pat_issues) == 0

    def test_pattern_none_value_skipped(self, tmp_path) -> None:
        """None values skip pattern check without error."""
        schema = {
            "properties": {"zip": {"type": "string", "pattern": r"^\d{5}$"}},
        }
        path = tmp_path / "s.json"
        path.write_text(json.dumps(schema))
        config = _make_config(schema=SchemaConfig(enforce=True, path=str(path)))
        checker = DataQualityChecker(config, model_name="test")

        report = checker.check({"zip": None})
        pat_issues = [i for i in report.issues if i.rule == "schema.pattern"]
        assert len(pat_issues) == 0


# ── data profile ───────────────────────────────────────────────────


class TestDataProfile:
    """Tests for profile population in QualityReport."""

    def test_profile_populated(self) -> None:
        """check() should populate the profile field."""
        config = _make_config(schema=SchemaConfig(enforce=False))
        checker = DataQualityChecker(config, model_name="test")
        rows = [{"x": 10, "y": "a"}, {"x": 20, "y": "b"}, {"x": 30, "y": "a"}]
        report = checker.check(rows)

        assert "x" in report.profile
        assert "y" in report.profile
        assert report.profile["x"]["type"] == "integer"
        assert report.profile["x"]["mean"] == pytest.approx(20.0)
        assert report.profile["x"]["min"] == pytest.approx(10.0)
        assert report.profile["x"]["max"] == pytest.approx(30.0)
        assert report.profile["x"]["null_rate"] == 0.0
        assert report.profile["y"]["type"] == "string"
        assert report.profile["y"]["mean"] is None
        assert report.profile["y"]["unique_count"] == 2

    def test_profile_with_nulls(self) -> None:
        """Profile correctly reports null_rate when nulls exist."""
        config = _make_config(schema=SchemaConfig(enforce=False), null_threshold=1.0)
        checker = DataQualityChecker(config, model_name="test")
        rows = [{"x": 1}, {"x": None}, {"x": 3}]
        report = checker.check(rows)

        assert report.profile["x"]["null_rate"] == pytest.approx(1.0 / 3.0)

    def test_profile_empty_rows(self) -> None:
        """Empty input produces empty profile."""
        config = _make_config(schema=SchemaConfig(enforce=False))
        checker = DataQualityChecker(config, model_name="test")
        report = checker.check([])
        assert report.profile == {}

    def test_check_still_works_without_fit(self) -> None:
        """check() works without calling fit() first — backward compat."""
        config = _make_config(schema=SchemaConfig(enforce=False))
        checker = DataQualityChecker(config, model_name="test")
        report = checker.check({"x": 42})
        assert report.is_valid
        assert report.rows_checked == 1
        assert "x" in report.profile
