"""Tests for sentinel.config.references — file-reference validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.config.references import (
    ReferenceIssue,
    _is_remote_uri,
    validate_file_references,
)
from sentinel.config.schema import (
    AuditConfig,
    DataQualityConfig,
    ModelConfig,
    RetrainingConfig,
    SchemaConfig,
    SentinelConfig,
    ValidationConfig,
)


class TestRemoteUriDetection:
    @pytest.mark.parametrize(
        "uri",
        [
            "s3://bucket/data.parquet",
            "azure://container/file.csv",
            "azureml://datasets/baseline",
            "gs://bucket/data.csv",
            "https://example.com/file.csv",
            "http://example.com/file.csv",
            "abfss://container@account.dfs.core.windows.net/path",
            "warehouse://analytics.fraud_metrics",
        ],
    )
    def test_remote_uri_recognised(self, uri: str) -> None:
        assert _is_remote_uri(uri) is True

    @pytest.mark.parametrize(
        "path",
        [
            "data/baseline.parquet",
            "/abs/path/data.csv",
            "./relative.json",
            "schemas/claims.json",
        ],
    )
    def test_local_path_not_remote(self, path: str) -> None:
        assert _is_remote_uri(path) is False


class TestValidateFileReferences:
    def _build_cfg(
        self,
        *,
        baseline: str | None = None,
        schema_path: str | None = None,
        holdout: str | None = None,
        audit_path: str = "./audit/",
        pipeline: str | None = None,
    ) -> SentinelConfig:
        return SentinelConfig(
            model=ModelConfig(name="m", domain="tabular", baseline_dataset=baseline),
            data_quality=DataQualityConfig(schema=SchemaConfig(path=schema_path)),  # type: ignore[arg-type]
            retraining=RetrainingConfig(
                pipeline=pipeline,
                validation=ValidationConfig(holdout_dataset=holdout),
            ),
            audit=AuditConfig(storage="local", path=audit_path),
        )

    def test_no_references_no_issues(self, tmp_path: Path) -> None:
        cfg = self._build_cfg(audit_path=str(tmp_path))
        assert validate_file_references(cfg, tmp_path) == []

    def test_existing_baseline_passes(self, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline.parquet"
        baseline.touch()
        cfg = self._build_cfg(baseline="baseline.parquet", audit_path=str(tmp_path))
        issues = validate_file_references(cfg, tmp_path)
        assert issues == []

    def test_missing_baseline_reported(self, tmp_path: Path) -> None:
        cfg = self._build_cfg(baseline="missing.parquet", audit_path=str(tmp_path))
        issues = validate_file_references(cfg, tmp_path)
        errors = [i for i in issues if i.field == "model.baseline_dataset"]
        assert len(errors) == 1
        assert errors[0].severity == "error"
        assert "missing.parquet" in errors[0].path

    def test_remote_baseline_skipped(self, tmp_path: Path) -> None:
        cfg = self._build_cfg(
            baseline="s3://bucket/baseline.parquet",
            audit_path=str(tmp_path),
        )
        issues = validate_file_references(cfg, tmp_path)
        assert all(i.field != "model.baseline_dataset" for i in issues)

    def test_missing_schema_reported(self, tmp_path: Path) -> None:
        cfg = self._build_cfg(schema_path="schemas/missing.json", audit_path=str(tmp_path))
        issues = validate_file_references(cfg, tmp_path)
        assert any(i.field == "data_quality.schema.path" for i in issues)

    def test_existing_schema_passes(self, tmp_path: Path) -> None:
        sd = tmp_path / "schemas"
        sd.mkdir()
        (sd / "claims.json").touch()
        cfg = self._build_cfg(schema_path="schemas/claims.json", audit_path=str(tmp_path))
        issues = validate_file_references(cfg, tmp_path)
        assert all(i.field != "data_quality.schema.path" for i in issues)

    def test_missing_holdout_reported(self, tmp_path: Path) -> None:
        cfg = self._build_cfg(holdout="data/holdout.csv", audit_path=str(tmp_path))
        issues = validate_file_references(cfg, tmp_path)
        assert any(i.field == "retraining.validation.holdout_dataset" for i in issues)

    def test_pipeline_missing_is_warning(self, tmp_path: Path) -> None:
        cfg = self._build_cfg(pipeline="pipelines/retrain.py", audit_path=str(tmp_path))
        issues = validate_file_references(cfg, tmp_path)
        relevant = [i for i in issues if i.field == "retraining.pipeline"]
        assert len(relevant) == 1
        assert relevant[0].severity == "warning"

    def test_pipeline_remote_uri_ignored(self, tmp_path: Path) -> None:
        cfg = self._build_cfg(pipeline="azureml://pipelines/retrain", audit_path=str(tmp_path))
        issues = validate_file_references(cfg, tmp_path)
        assert all(i.field != "retraining.pipeline" for i in issues)

    def test_audit_path_writable(self, tmp_path: Path) -> None:
        cfg = self._build_cfg(audit_path=str(tmp_path / "audit"))
        issues = validate_file_references(cfg, tmp_path)
        # tmp_path exists and is writable; no issues expected
        assert all(i.field != "audit.path" for i in issues)

    def test_audit_path_unwritable_parent(self, tmp_path: Path) -> None:
        cfg = self._build_cfg(audit_path="/totally/missing/parent/audit")
        issues = validate_file_references(cfg, tmp_path)
        assert any(i.field == "audit.path" for i in issues)

    def test_absolute_path_resolution(self, tmp_path: Path) -> None:
        target = tmp_path / "abs_baseline.parquet"
        target.touch()
        cfg = self._build_cfg(baseline=str(target.resolve()), audit_path=str(tmp_path))
        # base_dir doesn't matter for absolute paths
        issues = validate_file_references(cfg, tmp_path / "elsewhere")
        assert all(i.field != "model.baseline_dataset" for i in issues)


class TestReferenceIssueFormat:
    def test_format_includes_severity_and_path(self) -> None:
        issue = ReferenceIssue(
            field="model.baseline_dataset",
            path="missing.parquet",
            message="not found",
            severity="error",
        )
        formatted = issue.format()
        assert "[error]" in formatted
        assert "model.baseline_dataset" in formatted
        assert "missing.parquet" in formatted
