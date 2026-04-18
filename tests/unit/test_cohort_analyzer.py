"""Tests for CohortAnalyzer."""

from __future__ import annotations

import pytest

from sentinel.config.schema import CohortAnalysisConfig
from sentinel.core.types import CohortComparativeReport, CohortMetrics, CohortPerformanceReport
from sentinel.observability.cohort_analyzer import CohortAnalyzer

# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture()
def config() -> CohortAnalysisConfig:
    return CohortAnalysisConfig(
        enabled=True,
        cohort_column="cohort",
        max_cohorts=5,
        min_samples_per_cohort=2,
        disparity_threshold=0.10,
        buffer_size=100,
    )


@pytest.fixture()
def analyzer(config: CohortAnalysisConfig) -> CohortAnalyzer:
    return CohortAnalyzer(config, model_name="test_model")


def _seed(analyzer: CohortAnalyzer, cohort: str, n: int, pred: float, actual: float) -> None:
    for _ in range(n):
        analyzer.add_prediction({"x": 1.0}, pred, actual, cohort_id=cohort)


# ── Ingestion ──────────────────────────────────────────────────────


class TestIngestion:
    def test_add_prediction_basic(self, analyzer: CohortAnalyzer) -> None:
        analyzer.add_prediction({"x": 1.0}, 0.8, 1.0, cohort_id="a")
        assert "a" in analyzer.cohort_ids
        assert analyzer.cohort_count("a") == 1

    def test_add_multiple_cohorts(self, analyzer: CohortAnalyzer) -> None:
        analyzer.add_prediction({"x": 1.0}, 0.8, cohort_id="a")
        analyzer.add_prediction({"x": 2.0}, 0.3, cohort_id="b")
        assert set(analyzer.cohort_ids) == {"a", "b"}

    def test_max_cohorts_enforced(self, analyzer: CohortAnalyzer) -> None:
        for i in range(6):
            analyzer.add_prediction({"x": float(i)}, 0.5, cohort_id=f"c{i}")
        # max_cohorts=5, so the 6th should be dropped
        assert len(analyzer.cohort_ids) == 5

    def test_no_cohort_id_dropped(self, analyzer: CohortAnalyzer) -> None:
        analyzer.add_prediction({"x": 1.0}, 0.8)
        assert len(analyzer.cohort_ids) == 0

    def test_cohort_column_auto_derive(self, config: CohortAnalysisConfig) -> None:
        config_with_col = CohortAnalysisConfig(
            enabled=True,
            cohort_column="region",
            max_cohorts=10,
            min_samples_per_cohort=1,
        )
        az = CohortAnalyzer(config_with_col, model_name="m")
        az.add_prediction({"region": "north", "x": 1.0}, 0.5)
        assert "north" in az.cohort_ids

    def test_buffer_size_respected(self, config: CohortAnalysisConfig) -> None:
        small_config = CohortAnalysisConfig(
            enabled=True, cohort_column="x", buffer_size=3, min_samples_per_cohort=1
        )
        az = CohortAnalyzer(small_config, model_name="m")
        for i in range(10):
            az.add_prediction({"x": float(i)}, 0.5, cohort_id="a")
        assert az.cohort_count("a") == 3


# ── Single cohort report ──────────────────────────────────────────


class TestCohortReport:
    def test_returns_report(self, analyzer: CohortAnalyzer) -> None:
        _seed(analyzer, "a", 5, pred=0.9, actual=1.0)
        report = analyzer.get_cohort_report("a")
        assert isinstance(report, CohortPerformanceReport)
        assert report.cohort_id == "a"
        assert report.metrics.count == 5

    def test_unknown_cohort_returns_none(self, analyzer: CohortAnalyzer) -> None:
        assert analyzer.get_cohort_report("missing") is None

    def test_mean_prediction(self, analyzer: CohortAnalyzer) -> None:
        analyzer.add_prediction({"x": 1.0}, 0.6, cohort_id="a")
        analyzer.add_prediction({"x": 1.0}, 0.8, cohort_id="a")
        report = analyzer.get_cohort_report("a")
        assert report is not None
        assert abs(report.metrics.mean_prediction - 0.7) < 1e-6  # type: ignore[union-attr]

    def test_accuracy_binary(self, analyzer: CohortAnalyzer) -> None:
        analyzer.add_prediction({"x": 1.0}, 0.9, 1.0, cohort_id="a")
        analyzer.add_prediction({"x": 1.0}, 0.8, 1.0, cohort_id="a")
        analyzer.add_prediction({"x": 1.0}, 0.3, 0.0, cohort_id="a")  # correct
        analyzer.add_prediction({"x": 1.0}, 0.7, 0.0, cohort_id="a")  # wrong
        report = analyzer.get_cohort_report("a")
        assert report is not None
        assert report.metrics.accuracy == 0.75


# ── Comparative report ────────────────────────────────────────────


class TestCompareCohorts:
    def test_basic_comparison(self, analyzer: CohortAnalyzer) -> None:
        _seed(analyzer, "good", 5, pred=0.9, actual=1.0)
        _seed(analyzer, "bad", 5, pred=0.3, actual=1.0)
        report = analyzer.compare_cohorts()
        assert isinstance(report, CohortComparativeReport)
        assert len(report.cohorts) == 2

    def test_disparity_flagged(self, analyzer: CohortAnalyzer) -> None:
        # "good" cohort: all correct (accuracy=1.0)
        for _ in range(5):
            analyzer.add_prediction({"x": 1.0}, 0.9, 1.0, cohort_id="good")
        # "bad" cohort: all wrong (accuracy=0.0)
        for _ in range(5):
            analyzer.add_prediction({"x": 1.0}, 0.3, 1.0, cohort_id="bad")
        report = analyzer.compare_cohorts()
        assert "bad" in report.disparity_flags

    def test_no_disparity_when_similar(self, analyzer: CohortAnalyzer) -> None:
        _seed(analyzer, "a", 5, pred=0.9, actual=1.0)
        _seed(analyzer, "b", 5, pred=0.8, actual=1.0)
        report = analyzer.compare_cohorts()
        assert report.disparity_flags == []

    def test_summary_property(self, analyzer: CohortAnalyzer) -> None:
        _seed(analyzer, "x", 5, pred=0.9, actual=1.0)
        report = analyzer.compare_cohorts()
        assert "no disparity" in report.summary or "cohort" in report.summary

    def test_skips_small_cohorts(self, analyzer: CohortAnalyzer) -> None:
        # min_samples_per_cohort=2, only 1 sample → skipped
        analyzer.add_prediction({"x": 1.0}, 0.9, 1.0, cohort_id="tiny")
        _seed(analyzer, "ok", 5, pred=0.9, actual=1.0)
        report = analyzer.compare_cohorts()
        assert len(report.cohorts) == 1  # "tiny" excluded

    def test_global_accuracy(self, analyzer: CohortAnalyzer) -> None:
        _seed(analyzer, "a", 5, pred=0.9, actual=1.0)
        report = analyzer.compare_cohorts()
        assert report.global_accuracy is not None


# ── Reset ──────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_buffers(self, analyzer: CohortAnalyzer) -> None:
        _seed(analyzer, "a", 5, pred=0.9, actual=1.0)
        analyzer.reset()
        assert len(analyzer.cohort_ids) == 0


# ── Type frozen checks ────────────────────────────────────────────


class TestTypes:
    def test_cohort_metrics_frozen(self) -> None:
        m = CohortMetrics(cohort_id="a", count=10)
        with pytest.raises(Exception):
            m.count = 20  # type: ignore[misc]

    def test_comparative_report_frozen(self) -> None:
        r = CohortComparativeReport(model_name="m")
        with pytest.raises(Exception):
            r.model_name = "other"  # type: ignore[misc]
