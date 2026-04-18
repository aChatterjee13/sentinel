"""Unit tests for DataQualityChecker (expanded coverage)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from sentinel.config.schema import (
    DataQualityConfig,
    FreshnessConfig,
    OutlierConfig,
    SchemaConfig,
)
from sentinel.core.types import AlertSeverity
from sentinel.observability.data_quality import DataQualityChecker


def _make_config(**overrides) -> DataQualityConfig:
    defaults = {
        "schema": SchemaConfig(enforce=False),
        "freshness": FreshnessConfig(max_age_hours=24),
        "outlier_detection": OutlierConfig(method="iqr", contamination=0.05),
        "null_threshold": 0.1,
        "duplicate_threshold": 0.05,
    }
    defaults.update(overrides)
    return DataQualityConfig(**defaults)


class TestSchemaValidation:
    """Schema enforcement tests."""

    def test_valid_data_passes(self, tmp_path) -> None:
        schema = {"required": ["age"], "properties": {"age": {"type": "integer"}}}
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(schema))
        config = _make_config(schema=SchemaConfig(enforce=True, path=str(schema_path)))
        checker = DataQualityChecker(config, model_name="test")
        report = checker.check({"age": 25})
        assert report.is_valid
        assert report.rows_checked == 1

    def test_missing_required_field(self, tmp_path) -> None:
        schema = {"required": ["age", "name"]}
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(schema))
        config = _make_config(schema=SchemaConfig(enforce=True, path=str(schema_path)))
        checker = DataQualityChecker(config, model_name="test")
        report = checker.check({"age": 25})
        critical = [i for i in report.issues if i.severity == AlertSeverity.CRITICAL]
        assert len(critical) == 1
        assert "name" in critical[0].message

    def test_required_field_is_none(self, tmp_path) -> None:
        schema = {"required": ["age"]}
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(schema))
        config = _make_config(schema=SchemaConfig(enforce=True, path=str(schema_path)))
        checker = DataQualityChecker(config, model_name="test")
        report = checker.check({"age": None})
        assert any(i.rule == "schema.required" for i in report.issues)

    def test_type_mismatch(self, tmp_path) -> None:
        schema = {"properties": {"age": {"type": "integer"}}}
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(schema))
        config = _make_config(schema=SchemaConfig(enforce=True, path=str(schema_path)))
        checker = DataQualityChecker(config, model_name="test")
        report = checker.check({"age": "not_a_number"})
        assert any(i.rule == "schema.type" for i in report.issues)

    def test_minimum_violation(self, tmp_path) -> None:
        schema = {"properties": {"age": {"type": "integer", "minimum": 0}}}
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(schema))
        config = _make_config(schema=SchemaConfig(enforce=True, path=str(schema_path)))
        checker = DataQualityChecker(config, model_name="test")
        report = checker.check({"age": -5})
        assert any(i.rule == "schema.minimum" for i in report.issues)

    def test_maximum_violation(self, tmp_path) -> None:
        schema = {"properties": {"age": {"type": "integer", "maximum": 120}}}
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(schema))
        config = _make_config(schema=SchemaConfig(enforce=True, path=str(schema_path)))
        checker = DataQualityChecker(config, model_name="test")
        report = checker.check({"age": 200})
        assert any(i.rule == "schema.maximum" for i in report.issues)

    def test_schema_not_enforced_skips_checks(self) -> None:
        config = _make_config(schema=SchemaConfig(enforce=False))
        checker = DataQualityChecker(config, model_name="test")
        report = checker.check({"anything": "goes"})
        assert not any(i.rule.startswith("schema") for i in report.issues)

    def test_schema_file_missing_no_crash(self) -> None:
        config = _make_config(schema=SchemaConfig(enforce=True, path="/nonexistent/schema.json"))
        checker = DataQualityChecker(config, model_name="test")
        report = checker.check({"age": 25})
        assert report.is_valid


class TestFreshnessChecks:
    """Freshness monitoring tests."""

    def test_fresh_data_no_issue(self) -> None:
        config = _make_config(freshness=FreshnessConfig(max_age_hours=24))
        checker = DataQualityChecker(config, model_name="test")
        checker.mark_fresh()
        report = checker.check({"x": 1})
        freshness_issues = [i for i in report.issues if i.rule == "freshness"]
        assert len(freshness_issues) == 0

    def test_stale_data_raises_issue(self) -> None:
        config = _make_config(freshness=FreshnessConfig(max_age_hours=1))
        checker = DataQualityChecker(config, model_name="test")
        checker._last_seen = datetime.now(timezone.utc) - timedelta(hours=2)
        report = checker.check({"x": 1})
        freshness_issues = [i for i in report.issues if i.rule == "freshness"]
        assert len(freshness_issues) == 1
        assert freshness_issues[0].severity == AlertSeverity.HIGH

    def test_no_previous_data_no_freshness_issue(self) -> None:
        config = _make_config()
        checker = DataQualityChecker(config, model_name="test")
        report = checker.check({"x": 1})
        freshness_issues = [i for i in report.issues if i.rule == "freshness"]
        assert len(freshness_issues) == 0

    def test_mark_fresh_resets_timestamp(self) -> None:
        config = _make_config()
        checker = DataQualityChecker(config, model_name="test")
        assert checker._last_seen is None
        checker.mark_fresh()
        assert checker._last_seen is not None


class TestNullChecks:
    """Null rate detection tests."""

    def test_high_null_rate_flagged(self) -> None:
        config = _make_config(null_threshold=0.1)
        checker = DataQualityChecker(config, model_name="test")
        rows = [{"x": None} for _ in range(10)]
        report = checker.check(rows)
        null_issues = [i for i in report.issues if i.rule == "nulls"]
        assert len(null_issues) == 1

    def test_low_null_rate_passes(self) -> None:
        config = _make_config(null_threshold=0.5)
        checker = DataQualityChecker(config, model_name="test")
        rows = [{"x": 1}, {"x": 2}, {"x": None}]
        report = checker.check(rows)
        null_issues = [i for i in report.issues if i.rule == "nulls"]
        assert len(null_issues) == 0

    def test_empty_rows_no_null_issues(self) -> None:
        config = _make_config()
        checker = DataQualityChecker(config, model_name="test")
        report = checker.check([])
        assert report.rows_checked == 0


class TestDuplicateChecks:
    """Duplicate row detection tests."""

    def test_high_duplicate_rate_flagged(self) -> None:
        config = _make_config(duplicate_threshold=0.01)
        checker = DataQualityChecker(config, model_name="test")
        rows = [{"x": 1, "y": 2}] * 10
        report = checker.check(rows)
        dupe_issues = [i for i in report.issues if i.rule == "duplicates"]
        assert len(dupe_issues) == 1

    def test_no_duplicates_passes(self) -> None:
        config = _make_config()
        checker = DataQualityChecker(config, model_name="test")
        rows = [{"x": i} for i in range(10)]
        report = checker.check(rows)
        dupe_issues = [i for i in report.issues if i.rule == "duplicates"]
        assert len(dupe_issues) == 0

    def test_single_row_no_duplicate_check(self) -> None:
        config = _make_config()
        checker = DataQualityChecker(config, model_name="test")
        report = checker.check({"x": 1})
        dupe_issues = [i for i in report.issues if i.rule == "duplicates"]
        assert len(dupe_issues) == 0


class TestOutlierDetection:
    """Outlier detection tests."""

    def test_zscore_outlier_detection(self) -> None:
        config = _make_config(outlier_detection=OutlierConfig(method="zscore", contamination=0.01))
        checker = DataQualityChecker(config, model_name="test")
        rows = [{"x": float(i)} for i in range(20)] + [{"x": 1000.0}]
        report = checker.check(rows)
        outlier_issues = [i for i in report.issues if i.rule == "outliers"]
        assert len(outlier_issues) >= 1

    def test_iqr_outlier_detection(self) -> None:
        config = _make_config(outlier_detection=OutlierConfig(method="iqr", contamination=0.01))
        checker = DataQualityChecker(config, model_name="test")
        rows = [{"x": float(i)} for i in range(20)] + [{"x": 1000.0}]
        report = checker.check(rows)
        outlier_issues = [i for i in report.issues if i.rule == "outliers"]
        assert len(outlier_issues) >= 1

    def test_no_outliers_clean_data(self) -> None:
        config = _make_config(outlier_detection=OutlierConfig(method="zscore", contamination=0.05))
        checker = DataQualityChecker(config, model_name="test")
        rows = [{"x": float(i)} for i in range(20)]
        report = checker.check(rows)
        outlier_issues = [i for i in report.issues if i.rule == "outliers"]
        assert len(outlier_issues) == 0

    def test_isolation_forest_with_sklearn(self) -> None:
        config = _make_config(
            outlier_detection=OutlierConfig(method="isolation_forest", contamination=0.1)
        )
        checker = DataQualityChecker(config, model_name="test")
        rows = [{"x": float(i), "y": float(i * 2)} for i in range(15)] + [{"x": 999.0, "y": 999.0}]
        report = checker.check(rows)
        # Depending on sklearn availability, this should not crash
        assert report.rows_checked == 16

    def test_isolation_forest_too_few_rows(self) -> None:
        config = _make_config(
            outlier_detection=OutlierConfig(method="isolation_forest", contamination=0.1)
        )
        checker = DataQualityChecker(config, model_name="test")
        rows = [{"x": 1.0}] * 5
        report = checker.check(rows)
        outlier_issues = [i for i in report.issues if "outlier" in i.rule]
        assert len(outlier_issues) == 0

    def test_too_few_numeric_values_skipped(self) -> None:
        config = _make_config(outlier_detection=OutlierConfig(method="zscore", contamination=0.01))
        checker = DataQualityChecker(config, model_name="test")
        rows = [{"x": 1.0}, {"x": 2.0}]  # less than 5
        report = checker.check(rows)
        outlier_issues = [i for i in report.issues if i.rule == "outliers"]
        assert len(outlier_issues) == 0


class TestBatchAndMiscellaneous:
    """Batch processing and edge cases."""

    def test_batch_check(self) -> None:
        config = _make_config()
        checker = DataQualityChecker(config, model_name="test")
        rows = [{"x": i} for i in range(5)]
        report = checker.check(rows)
        assert report.rows_checked == 5

    def test_has_critical_issues(self, tmp_path) -> None:
        schema = {"required": ["id"]}
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(schema))
        config = _make_config(schema=SchemaConfig(enforce=True, path=str(schema_path)))
        checker = DataQualityChecker(config, model_name="test")
        report = checker.check({"no_id_here": True})
        assert report.has_critical_issues

    def test_report_is_valid_false_on_high_severity(self, tmp_path) -> None:
        schema = {"properties": {"age": {"type": "integer"}}}
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(schema))
        config = _make_config(schema=SchemaConfig(enforce=True, path=str(schema_path)))
        checker = DataQualityChecker(config, model_name="test")
        report = checker.check({"age": "invalid"})
        assert not report.is_valid

    def test_matches_type_various(self) -> None:
        assert DataQualityChecker._matches_type("hello", "string")
        assert DataQualityChecker._matches_type(42, "integer")
        assert DataQualityChecker._matches_type(3.14, "number")
        assert DataQualityChecker._matches_type(True, "boolean")
        assert DataQualityChecker._matches_type([1, 2], "array")
        assert DataQualityChecker._matches_type({"a": 1}, "object")
        assert not DataQualityChecker._matches_type(None, "string")
        assert not DataQualityChecker._matches_type("x", "integer")
