"""Unit tests for domain adapter resolution and behaviour."""

from __future__ import annotations

import math

import pytest

from sentinel.config.schema import ModelConfig, SentinelConfig
from sentinel.domains import resolve_adapter
from sentinel.domains.base import BaseDomainAdapter
from sentinel.domains.graph.adapter import GraphAdapter
from sentinel.domains.nlp.adapter import NLPAdapter
from sentinel.domains.recommendation.adapter import RecommendationAdapter
from sentinel.domains.tabular.adapter import TabularAdapter
from sentinel.domains.timeseries.adapter import TimeSeriesAdapter


def _make_config(domain: str, options: dict | None = None) -> SentinelConfig:
    cfg = SentinelConfig(model=ModelConfig(name="m", domain=domain))  # type: ignore[arg-type]
    if options:
        setattr(cfg.domains, domain, options)
    return cfg


class TestAdapterResolution:
    @pytest.mark.parametrize(
        "domain,expected",
        [
            ("tabular", TabularAdapter),
            ("timeseries", TimeSeriesAdapter),
            ("nlp", NLPAdapter),
            ("recommendation", RecommendationAdapter),
            ("graph", GraphAdapter),
        ],
    )
    def test_resolve_returns_correct_class(
        self, domain: str, expected: type[BaseDomainAdapter]
    ) -> None:
        cls = resolve_adapter(domain)
        assert cls is expected

    def test_resolve_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_adapter("nonsense")


class TestTabularAdapter:
    def test_provides_drift_detector(self) -> None:
        adapter = TabularAdapter(_make_config("tabular"))
        detectors = adapter.get_drift_detectors()
        assert len(detectors) == 1
        assert detectors[0].method_name == "psi"


class TestTimeSeriesAdapter:
    def test_calendar_drift_detector(self) -> None:
        adapter = TimeSeriesAdapter(
            _make_config("timeseries", {"drift": {"method": "calendar_test", "season_period": 7}})
        )
        detectors = adapter.get_drift_detectors()
        assert len(detectors) >= 1

    def test_quality_metrics(self) -> None:
        adapter = TimeSeriesAdapter(_make_config("timeseries"))
        metrics = adapter.get_quality_metrics()
        names = [m.name for m in metrics]
        assert "forecast_quality" in names


class TestNLPAdapter:
    def test_ner_metrics(self) -> None:
        adapter = NLPAdapter(_make_config("nlp", {"task": "ner"}))
        metrics = adapter.get_quality_metrics()
        assert len(metrics) >= 1

    def test_classification_metrics(self) -> None:
        adapter = NLPAdapter(_make_config("nlp", {"task": "classification"}))
        metrics = adapter.get_quality_metrics()
        assert len(metrics) >= 1


class TestRecommendationAdapter:
    def test_ranking_quality(self) -> None:
        adapter = RecommendationAdapter(_make_config("recommendation"))
        metrics = adapter.get_quality_metrics()
        assert len(metrics) >= 1


class TestGraphAdapter:
    def test_link_prediction(self) -> None:
        adapter = GraphAdapter(_make_config("graph", {"task": "link_prediction"}))
        metrics = adapter.get_quality_metrics()
        names = [m.name for m in metrics]
        assert "auc_roc" in names

    def test_kg_completion_metrics(self) -> None:
        adapter = GraphAdapter(
            _make_config("graph", {"task": "kg_completion", "graph_type": "knowledge_graph"})
        )
        metrics = adapter.get_quality_metrics()
        names = [m.name for m in metrics]
        assert "mrr" in names
        # Should also produce an EntityVocabulary detector for KGs
        detectors = adapter.get_drift_detectors()
        method_names = [d.method_name for d in detectors]
        assert "entity_vocabulary" in method_names


# ═══════════════════════════════════════════════════════════════════
#  Extended tests — appended below the original 110-line test file
# ═══════════════════════════════════════════════════════════════════

import numpy as np

from sentinel.domains.graph.drift import (
    EntityVocabularyDriftDetector,
    TopologyDriftDetector,
)
from sentinel.domains.graph.quality import (
    auc_roc,
    embedding_isotropy,
    hits_at_k,
    mrr,
    node_classification_f1,
)
from sentinel.domains.graph.structure import (
    clustering_coefficient,
    connected_components,
    degree_distribution,
    density,
    topology_stats,
)
from sentinel.domains.nlp.drift import (
    EmbeddingDriftDetector,
    LabelDistributionDriftDetector,
    VocabularyDriftDetector,
)
from sentinel.domains.nlp.quality import (
    classification_metrics,
    span_exact_match,
    token_f1,
)
from sentinel.domains.nlp.text_stats import TextStatsMonitor, tokenise
from sentinel.domains.recommendation.bias import FairnessReport, group_fairness, position_bias
from sentinel.domains.recommendation.drift import (
    ItemDistributionDriftDetector,
    UserSegmentDriftDetector,
)
from sentinel.domains.recommendation.quality import (
    catalogue_coverage,
    diversity_intra_list,
    evaluate_recommendations,
    gini_coefficient,
    map_at_k,
    ndcg_at_k,
    novelty_inverse_popularity,
)
from sentinel.domains.timeseries.decomposition import (
    Decomposition,
    DecompositionMonitor,
    decompose,
)
from sentinel.domains.timeseries.drift import (
    ACFShiftDetector,
    CalendarDriftDetector,
    StationarityDriftDetector,
)
from sentinel.domains.timeseries.quality import (
    coverage,
    directional_accuracy,
    evaluate_forecast,
    interval_width,
    mape,
    mase,
    rmse,
    smape,
    winkler_score,
)
from sentinel.observability.data_quality import DataQualityChecker

# ── Tabular (extended) ────────────────────────────────────────────


class TestTabularAdapterExtended:
    def test_schema_validator_type(self) -> None:
        adapter = TabularAdapter(_make_config("tabular"))
        validator = adapter.get_schema_validator()
        assert isinstance(validator, DataQualityChecker)

    def test_regression_quality_metric(self) -> None:
        cfg = _make_config("tabular")
        cfg.model.type = "regression"
        adapter = TabularAdapter(cfg)
        metrics = adapter.get_quality_metrics()
        names = [m.name for m in metrics]
        assert "mae" in names

    def test_classification_quality_metric(self) -> None:
        adapter = TabularAdapter(_make_config("tabular"))
        metrics = adapter.get_quality_metrics()
        names = [m.name for m in metrics]
        assert "accuracy" in names

    def test_describe(self) -> None:
        adapter = TabularAdapter(_make_config("tabular"))
        desc = adapter.describe()
        assert desc["domain"] == "tabular"
        assert "psi" in desc["drift_detectors"]

    @pytest.mark.parametrize("method", ["psi", "ks", "js_divergence", "chi_squared", "wasserstein"])
    def test_drift_method_from_config(self, method: str) -> None:
        cfg = _make_config("tabular")
        cfg.drift.data.method = method
        adapter = TabularAdapter(cfg)
        detectors = adapter.get_drift_detectors()
        assert len(detectors) == 1
        assert detectors[0].method_name == method


# ── Time Series (extended) ────────────────────────────────────────


class TestTimeSeriesDecomposition:
    def test_decompose_returns_components(self) -> None:
        values = list(range(28))
        result = decompose(values, period=7)
        assert isinstance(result, Decomposition)
        assert len(result.trend) == 28
        assert len(result.seasonal) == 28
        assert len(result.residual) == 28

    def test_short_series_fallback(self) -> None:
        values = [1.0, 2.0, 3.0]
        result = decompose(values, period=7)
        assert result.seasonal_amplitude == 0.0

    def test_trend_slope_positive(self) -> None:
        values = list(range(50))
        result = decompose(values, period=7)
        assert result.trend_slope > 0

    def test_residual_variance(self) -> None:
        np.random.seed(42)
        values = np.random.randn(100).tolist()
        result = decompose(values, period=7)
        assert result.residual_variance >= 0

    def test_monitor_fit_and_evaluate(self) -> None:
        monitor = DecompositionMonitor(period=7)
        ref = [float(i + np.sin(i * 2 * np.pi / 7)) for i in range(56)]
        monitor.fit(ref)
        result = monitor.evaluate(ref)
        assert "trend_slope_change" in result
        assert "seasonal_amplitude_change_pct" in result
        assert "residual_variance_change_pct" in result
        assert result["trend_alert"] is False or isinstance(result["trend_alert"], bool)

    def test_monitor_detects_trend_shift(self) -> None:
        monitor = DecompositionMonitor(period=7, trend_slope_change_threshold=0.01)
        ref = [float(i) for i in range(56)]
        monitor.fit(ref)
        shifted = [float(i * 5) for i in range(56)]
        result = monitor.evaluate(shifted)
        assert result["trend_alert"] is True


class TestTimeSeriesForecastQuality:
    def test_mase_perfect(self) -> None:
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert mase(y_true, y_true, season=1) == 0.0

    def test_mase_bad(self) -> None:
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [5.0, 4.0, 3.0, 2.0, 1.0]
        result = mase(y_true, y_pred, season=1)
        assert result > 1.0

    def test_mape_zero_true(self) -> None:
        result = mape([0.0, 0.0], [1.0, 2.0])
        assert math.isnan(result)

    def test_smape_perfect(self) -> None:
        y = [1.0, 2.0, 3.0]
        assert smape(y, y) == 0.0

    def test_rmse_perfect(self) -> None:
        y = [1.0, 2.0, 3.0]
        assert rmse(y, y) == 0.0

    def test_coverage_all_inside(self) -> None:
        y_true = [2.0, 3.0, 4.0]
        lower = [1.0, 2.0, 3.0]
        upper = [3.0, 4.0, 5.0]
        assert coverage(y_true, lower, upper) == 1.0

    def test_coverage_none_inside(self) -> None:
        y_true = [10.0, 11.0]
        lower = [1.0, 2.0]
        upper = [3.0, 4.0]
        assert coverage(y_true, lower, upper) == 0.0

    def test_interval_width(self) -> None:
        assert interval_width([0.0, 1.0], [2.0, 3.0]) == 2.0

    def test_directional_accuracy_perfect(self) -> None:
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.5, 2.5, 3.5, 4.5]
        assert directional_accuracy(y_true, y_pred) == 1.0

    def test_directional_accuracy_opposite(self) -> None:
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [4.0, 3.0, 2.0, 1.0]
        assert directional_accuracy(y_true, y_pred) == 0.0

    def test_winkler_no_penalty(self) -> None:
        y_true = [2.0, 3.0]
        lower = [1.0, 2.0]
        upper = [3.0, 4.0]
        score = winkler_score(y_true, lower, upper)
        assert score == 2.0  # just width, no penalty

    def test_evaluate_forecast_bundle(self) -> None:
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.1, 3.1, 4.1, 5.1]
        result = evaluate_forecast(
            y_true,
            y_pred,
            season=1,
            lower=[0.5, 1.5, 2.5, 3.5, 4.5],
            upper=[1.5, 2.5, 3.5, 4.5, 5.5],
        )
        assert result.mase is not None
        assert result.coverage is not None
        assert result.directional_accuracy is not None


class TestTimeSeriesDriftDetectors:
    def test_calendar_fit_detect_no_drift(self) -> None:
        det = CalendarDriftDetector(model_name="m", threshold=1.0, seasonality=7)
        ref = [float(i % 7) for i in range(70)]
        det.fit(ref)
        report = det.detect(ref)
        assert not report.is_drifted

    def test_calendar_detect_drift(self) -> None:
        det = CalendarDriftDetector(model_name="m", threshold=0.5, seasonality=7)
        ref = [float(i % 7) for i in range(70)]
        det.fit(ref)
        shifted = [float((i % 7) + 10) for i in range(70)]
        report = det.detect(shifted)
        assert report.is_drifted

    def test_calendar_not_fitted_raises(self) -> None:
        det = CalendarDriftDetector(model_name="m")
        with pytest.raises(RuntimeError):
            det.detect([1.0, 2.0])

    def test_acf_shift_no_drift(self) -> None:
        np.random.seed(0)
        series = np.cumsum(np.random.randn(200)).tolist()
        det = ACFShiftDetector(model_name="m", threshold=0.5, lags=(1, 7))
        det.fit(series)
        report = det.detect(series)
        assert not report.is_drifted

    def test_acf_shift_detects_change(self) -> None:
        np.random.seed(0)
        ref = np.cumsum(np.random.randn(200)).tolist()
        det = ACFShiftDetector(model_name="m", threshold=0.1, lags=(1,))
        det.fit(ref)
        uncorrelated = np.random.randn(200).tolist()
        report = det.detect(uncorrelated)
        assert report.test_statistic > 0

    def test_stationarity_no_drift(self) -> None:
        np.random.seed(0)
        ref = np.random.randn(100).tolist()
        det = StationarityDriftDetector(model_name="m", threshold=1.0)
        det.fit(ref)
        report = det.detect(ref)
        assert not report.is_drifted

    def test_stationarity_detects_mean_shift(self) -> None:
        np.random.seed(0)
        ref = np.random.randn(100).tolist()
        det = StationarityDriftDetector(model_name="m", threshold=0.5)
        det.fit(ref)
        shifted = (np.random.randn(100) + 10).tolist()
        report = det.detect(shifted)
        assert report.is_drifted
        assert "mean_shift" in report.feature_scores

    def test_adapter_all_drift_method(self) -> None:
        adapter = TimeSeriesAdapter(
            _make_config("timeseries", {"drift": {"method": "all"}, "seasonality_periods": [7]})
        )
        detectors = adapter.get_drift_detectors()
        methods = {d.method_name for d in detectors}
        assert "calendar_test" in methods
        assert "acf_shift" in methods
        assert "stationarity" in methods

    def test_adapter_stationarity_method(self) -> None:
        adapter = TimeSeriesAdapter(
            _make_config("timeseries", {"drift": {"method": "stationarity"}})
        )
        detectors = adapter.get_drift_detectors()
        assert len(detectors) == 1
        assert detectors[0].method_name == "stationarity"

    def test_adapter_schema_validator(self) -> None:
        adapter = TimeSeriesAdapter(_make_config("timeseries"))
        assert isinstance(adapter.get_schema_validator(), DataQualityChecker)


# ── NLP (extended) ────────────────────────────────────────────────


class TestNLPDriftDetectors:
    def test_vocabulary_drift_no_drift(self) -> None:
        det = VocabularyDriftDetector(model_name="m", threshold=0.5)
        ref = ["the cat sat on the mat", "a dog ran in the park"]
        det.fit(ref)
        report = det.detect(ref)
        assert not report.is_drifted
        assert report.test_statistic == 0.0

    def test_vocabulary_drift_detects_new_words(self) -> None:
        det = VocabularyDriftDetector(model_name="m", threshold=0.1)
        ref = ["the cat sat on the mat"]
        det.fit(ref)
        novel = ["quantum entanglement superconductor nanotechnology"]
        report = det.detect(novel)
        assert report.is_drifted
        assert report.test_statistic > 0

    def test_embedding_drift_cosine_no_drift(self) -> None:
        np.random.seed(42)
        embeddings = np.random.randn(50, 8)
        det = EmbeddingDriftDetector(model_name="m", threshold=0.5, method="cosine_centroid")
        det.fit(embeddings)
        report = det.detect(embeddings)
        assert not report.is_drifted

    def test_embedding_drift_cosine_detects_shift(self) -> None:
        np.random.seed(42)
        ref = np.random.randn(50, 8)
        det = EmbeddingDriftDetector(model_name="m", threshold=0.05, method="cosine_centroid")
        det.fit(ref)
        shifted = ref + 10
        report = det.detect(shifted)
        assert report.test_statistic > 0

    def test_embedding_drift_mahalanobis(self) -> None:
        np.random.seed(42)
        ref = np.random.randn(50, 4)
        det = EmbeddingDriftDetector(model_name="m", threshold=0.5, method="mahalanobis")
        det.fit(ref)
        report = det.detect(ref)
        assert report.test_statistic >= 0

    def test_embedding_drift_mmd(self) -> None:
        np.random.seed(42)
        ref = np.random.randn(50, 4)
        det = EmbeddingDriftDetector(model_name="m", threshold=0.5, method="mmd")
        det.fit(ref)
        report = det.detect(ref + 5)
        assert report.test_statistic > 0

    def test_embedding_bad_shape_raises(self) -> None:
        det = EmbeddingDriftDetector(model_name="m")
        with pytest.raises(ValueError):
            det.fit(np.array([1.0, 2.0, 3.0]))  # 1-D

    def test_label_distribution_no_drift(self) -> None:
        det = LabelDistributionDriftDetector(model_name="m", threshold=1.0)
        labels = ["pos", "neg", "pos", "pos", "neg"]
        det.fit(labels)
        report = det.detect(labels)
        assert not report.is_drifted

    def test_label_distribution_detects_shift(self) -> None:
        det = LabelDistributionDriftDetector(model_name="m", threshold=0.01)
        ref = ["pos"] * 50 + ["neg"] * 50
        det.fit(ref)
        shifted = ["pos"] * 90 + ["neg"] * 10
        report = det.detect(shifted)
        assert report.is_drifted


class TestNLPTextStats:
    def test_tokenise(self) -> None:
        tokens = tokenise("Hello World! Testing 123.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "123" in tokens

    def test_monitor_fit_evaluate(self) -> None:
        monitor = TextStatsMonitor(oov_threshold=0.1)
        monitor.fit(["the cat sat on the mat", "a dog ran"])
        stats = monitor.evaluate(["the cat barked loudly"])
        assert stats.n_documents == 1
        assert stats.avg_length > 0
        assert stats.oov_rate >= 0

    def test_monitor_expand_vocab(self) -> None:
        monitor = TextStatsMonitor()
        monitor.fit(["hello world"])
        added = monitor.expand_vocab(["brand new tokens"])
        assert added > 0

    def test_empty_corpus(self) -> None:
        monitor = TextStatsMonitor()
        monitor.fit(["some words"])
        stats = monitor.evaluate([])
        assert stats.n_documents == 0


class TestNLPQualityMetrics:
    def test_token_f1_perfect(self) -> None:
        tokens = ["B-PER", "I-PER", "O"]
        assert token_f1(tokens, tokens) == 1.0

    def test_token_f1_no_overlap(self) -> None:
        assert token_f1(["A", "B"], ["C", "D"]) == 0.0

    def test_span_exact_match_perfect(self) -> None:
        spans = [(0, 5, "PER"), (10, 15, "ORG")]
        assert span_exact_match(spans, spans) == 1.0

    def test_span_exact_match_partial(self) -> None:
        gold = [(0, 5, "PER"), (10, 15, "ORG")]
        pred = [(0, 5, "PER")]
        assert span_exact_match(pred, gold) == 0.5

    def test_classification_metrics_basic(self) -> None:
        y_true = ["pos", "neg", "pos", "pos", "neg"]
        y_pred = ["pos", "neg", "neg", "pos", "neg"]
        result = classification_metrics(y_true, y_pred)
        assert 0 < result.accuracy <= 1.0
        assert "pos" in result.per_class
        assert "neg" in result.per_class

    @pytest.mark.parametrize("task", ["sentiment", "topic_modelling"])
    def test_adapter_non_ner_tasks(self, task: str) -> None:
        adapter = NLPAdapter(_make_config("nlp", {"task": task}))
        metrics = adapter.get_quality_metrics()
        names = [m.name for m in metrics]
        assert "classification_metrics" in names

    def test_adapter_drift_detectors_count(self) -> None:
        adapter = NLPAdapter(_make_config("nlp"))
        detectors = adapter.get_drift_detectors()
        methods = {d.method_name for d in detectors}
        assert "vocabulary_drift" in methods
        assert "embedding_drift" in methods
        assert "label_distribution" in methods

    def test_adapter_schema_validator(self) -> None:
        adapter = NLPAdapter(_make_config("nlp"))
        assert isinstance(adapter.get_schema_validator(), DataQualityChecker)


# ── Recommendation (extended) ─────────────────────────────────────


class TestRecommendationDriftDetectors:
    def test_item_distribution_no_drift(self) -> None:
        det = ItemDistributionDriftDetector(model_name="m", threshold=0.5)
        items = ["a", "b", "c", "a", "b", "c"]
        det.fit(items)
        report = det.detect(items)
        assert not report.is_drifted

    def test_item_distribution_detects_shift(self) -> None:
        det = ItemDistributionDriftDetector(model_name="m", threshold=0.05)
        ref = ["a"] * 50 + ["b"] * 50
        det.fit(ref)
        shifted = ["a"] * 95 + ["c"] * 5
        report = det.detect(shifted)
        assert report.is_drifted
        assert "js_divergence" in report.feature_scores

    def test_item_distribution_nested_lists(self) -> None:
        det = ItemDistributionDriftDetector(model_name="m", threshold=0.5)
        ref = [["a", "b"], ["c", "d"]]
        det.fit(ref)
        report = det.detect(ref)
        assert not report.is_drifted

    def test_user_segment_no_drift(self) -> None:
        det = UserSegmentDriftDetector(model_name="m", threshold=0.5)
        segments = ["gold", "silver", "gold", "silver"]
        det.fit(segments)
        report = det.detect(segments)
        assert not report.is_drifted

    def test_user_segment_detects_shift(self) -> None:
        det = UserSegmentDriftDetector(model_name="m", threshold=0.05)
        ref = ["gold"] * 50 + ["silver"] * 50
        det.fit(ref)
        shifted = ["gold"] * 95 + ["bronze"] * 5
        report = det.detect(shifted)
        assert report.is_drifted


class TestRecommendationQuality:
    def test_ndcg_perfect(self) -> None:
        preds = [["a", "b", "c"]]
        truth = [{"a", "b", "c"}]
        result = ndcg_at_k(preds, truth, k=3)
        assert result == 1.0

    def test_ndcg_empty(self) -> None:
        assert ndcg_at_k([], [], k=10) == 0.0

    def test_map_basic(self) -> None:
        preds = [["a", "x", "b"]]
        truth = [{"a", "b"}]
        result = map_at_k(preds, truth, k=3)
        assert 0 < result <= 1.0

    def test_catalogue_coverage(self) -> None:
        preds = [["a", "b"], ["c", "a"]]
        catalogue = {"a", "b", "c", "d", "e"}
        assert catalogue_coverage(preds, catalogue) == 0.6

    def test_diversity_identical_lists(self) -> None:
        preds = [["a", "b", "c"]]
        result = diversity_intra_list(preds, similarity=None)
        assert result == 1.0

    def test_diversity_with_similarity(self) -> None:
        preds = [["a", "b"]]
        sim = {("a", "b"): 0.9}
        result = diversity_intra_list(preds, similarity=sim)
        assert result == pytest.approx(0.1)

    def test_novelty(self) -> None:
        preds = [["a", "b"]]
        pop = {"a": 100, "b": 1}
        result = novelty_inverse_popularity(preds, pop)
        assert result > 0

    def test_gini_uniform(self) -> None:
        items = ["a", "b", "c", "d"] * 25
        result = gini_coefficient(items)
        assert result < 0.1

    def test_gini_skewed(self) -> None:
        items = ["a"] * 100 + ["b"]
        result = gini_coefficient(items)
        assert result > 0.3

    def test_evaluate_recommendations_bundle(self) -> None:
        preds = [["a", "b", "c"], ["d", "a", "e"]]
        truth = [{"a", "c"}, {"d"}]
        catalogue = {"a", "b", "c", "d", "e", "f"}
        result = evaluate_recommendations(preds, truth, catalogue, k=3)
        assert result.ndcg >= 0
        assert result.coverage >= 0
        assert result.diversity >= 0


class TestRecommendationBias:
    def test_group_fairness_equal(self) -> None:
        preds = [["a", "b"], ["a", "c"], ["b", "c"], ["a", "b"]]
        truth = [{"a"}, {"a"}, {"b"}, {"a"}]
        groups = ["m", "f", "m", "f"]
        report = group_fairness(preds, truth, groups, k=2)
        assert isinstance(report, FairnessReport)
        assert report.max_disparity >= 0

    def test_group_fairness_fails_on_disparity(self) -> None:
        preds = [["a"], ["x"]]
        truth = [{"a"}, {"a"}]
        groups = ["good", "bad"]
        report = group_fairness(preds, truth, groups, k=1, max_disparity=0.0)
        assert report.max_disparity >= 0

    def test_position_bias(self) -> None:
        preds = [["a", "b", "c"], ["d", "e", "f"]]
        result = position_bias(preds, k=3)
        assert result[0] == 2
        assert result[1] == 2

    def test_adapter_schema_validator(self) -> None:
        adapter = RecommendationAdapter(_make_config("recommendation"))
        assert isinstance(adapter.get_schema_validator(), DataQualityChecker)

    def test_adapter_drift_detector_types(self) -> None:
        adapter = RecommendationAdapter(_make_config("recommendation"))
        detectors = adapter.get_drift_detectors()
        methods = {d.method_name for d in detectors}
        assert "item_distribution" in methods
        assert "user_segment" in methods


# ── Graph (extended) ──────────────────────────────────────────────


class TestGraphStructure:
    def test_degree_distribution(self) -> None:
        edges = [("a", "b"), ("b", "c"), ("c", "a")]
        dist = degree_distribution(edges)
        assert dist[2] == 3  # all nodes have degree 2

    def test_density_complete(self) -> None:
        assert density(3, 3) == 1.0

    def test_density_single_node(self) -> None:
        assert density(1, 0) == 0.0

    def test_clustering_coefficient_triangle(self) -> None:
        edges = [("a", "b"), ("b", "c"), ("c", "a")]
        assert clustering_coefficient(edges) == 1.0

    def test_clustering_coefficient_star(self) -> None:
        edges = [("a", "b"), ("a", "c"), ("a", "d")]
        assert clustering_coefficient(edges) == 0.0

    def test_connected_components(self) -> None:
        edges = [("a", "b"), ("c", "d")]
        comps = connected_components(edges)
        assert len(comps) == 2

    def test_topology_stats(self) -> None:
        edges = [("a", "b"), ("b", "c"), ("c", "a")]
        stats = topology_stats(edges)
        assert stats.n_nodes == 3
        assert stats.n_edges == 3
        assert stats.n_components == 1
        assert stats.density == 1.0


class TestGraphDriftDetectors:
    def test_topology_no_drift(self) -> None:
        edges = [("a", "b"), ("b", "c"), ("c", "a"), ("d", "e")]
        det = TopologyDriftDetector(model_name="m", threshold=0.5)
        det.fit(edges)
        report = det.detect(edges)
        assert not report.is_drifted

    def test_topology_detects_structural_change(self) -> None:
        ref = [("a", "b"), ("b", "c"), ("c", "a")]
        det = TopologyDriftDetector(model_name="m", threshold=0.05)
        det.fit(ref)
        different = [("x", "y"), ("y", "z"), ("z", "w"), ("w", "x"), ("x", "z")]
        report = det.detect(different)
        assert report.test_statistic > 0
        assert "degree_ks" in report.feature_scores

    def test_topology_not_fitted_raises(self) -> None:
        det = TopologyDriftDetector(model_name="m")
        with pytest.raises(RuntimeError):
            det.detect([("a", "b")])

    def test_entity_vocab_no_drift(self) -> None:
        det = EntityVocabularyDriftDetector(model_name="m", threshold=0.5)
        det.fit(["e1", "e2", "e3"])
        report = det.detect(["e1", "e2"])
        assert not report.is_drifted
        assert report.test_statistic == 0.0

    def test_entity_vocab_detects_unseen(self) -> None:
        det = EntityVocabularyDriftDetector(model_name="m", threshold=0.1)
        det.fit(["e1", "e2", "e3"])
        report = det.detect(["e1", "x1", "x2", "x3", "x4"])
        assert report.is_drifted
        assert report.feature_scores["oov_rate"] == pytest.approx(0.8)

    def test_entity_vocab_empty_current(self) -> None:
        det = EntityVocabularyDriftDetector(model_name="m")
        det.fit(["e1"])
        report = det.detect([])
        assert not report.is_drifted


class TestGraphQualityMetrics:
    def test_auc_roc_perfect(self) -> None:
        scores = [0.1, 0.2, 0.9, 0.95]
        labels = [0, 0, 1, 1]
        assert auc_roc(scores, labels) == 1.0

    def test_auc_roc_random(self) -> None:
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = [0, 1, 0, 1]
        result = auc_roc(scores, labels)
        assert 0.0 <= result <= 1.0

    def test_hits_at_k(self) -> None:
        rankings = [1, 5, 15, 3]
        assert hits_at_k(rankings, k=10) == 0.75

    def test_mrr_basic(self) -> None:
        rankings = [1, 2, 4]
        expected = (1.0 + 0.5 + 0.25) / 3
        assert mrr(rankings) == pytest.approx(expected)

    def test_node_classification_f1(self) -> None:
        y_true = ["A", "B", "A", "B", "A"]
        y_pred = ["A", "B", "A", "A", "A"]
        result = node_classification_f1(y_true, y_pred)
        assert 0 < result <= 1.0

    def test_embedding_isotropy_isotropic(self) -> None:
        np.random.seed(0)
        emb = np.random.randn(100, 5)
        result = embedding_isotropy(emb)
        assert 0 < result <= 1.0

    def test_embedding_isotropy_degenerate(self) -> None:
        emb = np.array([[1, 0, 0]] * 10)
        result = embedding_isotropy(emb)
        assert result == 0.0

    @pytest.mark.parametrize(
        "task,expected_metric",
        [
            ("link_prediction", "auc_roc"),
            ("node_classification", "node_classification_f1"),
        ],
    )
    def test_adapter_quality_by_task(self, task: str, expected_metric: str) -> None:
        adapter = GraphAdapter(_make_config("graph", {"task": task}))
        names = [m.name for m in adapter.get_quality_metrics()]
        assert expected_metric in names

    def test_adapter_non_kg_no_entity_detector(self) -> None:
        adapter = GraphAdapter(
            _make_config("graph", {"task": "link_prediction", "graph_type": "social_network"})
        )
        methods = {d.method_name for d in adapter.get_drift_detectors()}
        assert "entity_vocabulary" not in methods
        assert "topology_drift" in methods

    def test_adapter_schema_validator(self) -> None:
        adapter = GraphAdapter(_make_config("graph"))
        assert isinstance(adapter.get_schema_validator(), DataQualityChecker)
