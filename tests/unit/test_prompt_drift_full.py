"""Unit tests for PromptDriftDetector (expanded coverage)."""

from __future__ import annotations

from sentinel.config.schema import PromptDriftConfig
from sentinel.core.types import AlertSeverity
from sentinel.llmops.prompt_drift import PromptDriftDetector


class TestPromptDriftDetector:
    """Tests for the PromptDriftDetector."""

    def _make_detector(self, **kwargs) -> PromptDriftDetector:
        config = PromptDriftConfig(**kwargs)
        return PromptDriftDetector(config=config)

    # ── insufficient data ─────────────────────────────────────────

    def test_insufficient_data_returns_stable(self) -> None:
        det = self._make_detector()
        for _i in range(10):
            det.observe("p", "1.0", quality_score=0.8)
        report = det.detect("p", "1.0")
        assert not report.is_drifted
        assert report.metadata.get("reason") == "insufficient_data"

    def test_no_observations_returns_stable(self) -> None:
        det = self._make_detector()
        report = det.detect("p", "1.0")
        assert not report.is_drifted
        assert report.severity == AlertSeverity.INFO

    # ── stable data (no drift) ────────────────────────────────────

    def test_stable_quality_no_drift(self) -> None:
        det = self._make_detector()
        # All scores are the same → no decline
        for _ in range(40):
            det.observe("p", "1.0", quality_score=0.8, guardrail_violations=0, total_tokens=100)
        report = det.detect("p", "1.0")
        assert not report.is_drifted
        assert report.severity == AlertSeverity.INFO

    def test_improving_quality_no_drift(self) -> None:
        det = self._make_detector()
        # Scores improving over time
        for i in range(40):
            det.observe("p", "1.0", quality_score=0.5 + i * 0.01)
        report = det.detect("p", "1.0")
        assert not report.is_drifted

    # ── quality decline drift ─────────────────────────────────────

    def test_quality_decline_drift(self) -> None:
        det = self._make_detector(signals={"quality_score_decline": 0.1})
        # First half: high scores, second half: low scores
        for _ in range(20):
            det.observe("p", "1.0", quality_score=0.9)
        for _ in range(20):
            det.observe("p", "1.0", quality_score=0.5)
        report = det.detect("p", "1.0")
        assert report.is_drifted
        assert "quality_decline" in report.drifted_features
        assert report.feature_scores["quality_decline"] > 0.1

    # ── guardrail violation increase ──────────────────────────────

    def test_guardrail_violation_increase(self) -> None:
        det = self._make_detector(signals={"guardrail_violation_rate_increase": 0.05})
        for _ in range(20):
            det.observe("p", "1.0", quality_score=0.8, guardrail_violations=0)
        for _ in range(20):
            det.observe("p", "1.0", quality_score=0.8, guardrail_violations=1)
        report = det.detect("p", "1.0")
        assert "guardrail_violations" in report.drifted_features

    # ── token usage increase ──────────────────────────────────────

    def test_token_usage_increase(self) -> None:
        det = self._make_detector(signals={"token_usage_increase_pct": 25.0})
        for _ in range(20):
            det.observe("p", "1.0", quality_score=0.8, total_tokens=100)
        for _ in range(20):
            det.observe("p", "1.0", quality_score=0.8, total_tokens=200)
        report = det.detect("p", "1.0")
        assert "token_usage" in report.drifted_features

    # ── semantic drift ────────────────────────────────────────────

    def test_semantic_drift_signal(self) -> None:
        det = self._make_detector()
        for _ in range(20):
            det.observe("p", "1.0", quality_score=0.8, semantic_distance=0.05)
        for _ in range(20):
            det.observe("p", "1.0", quality_score=0.8, semantic_distance=0.4)
        report = det.detect("p", "1.0")
        # avg semantic distance = (20*0.05 + 20*0.4)/40 = 0.225 > 0.2
        assert "semantic_drift" in report.drifted_features

    # ── multiple signals → HIGH severity ──────────────────────────

    def test_multiple_signals_high_severity(self) -> None:
        det = self._make_detector(
            signals={
                "quality_score_decline": 0.05,
                "guardrail_violation_rate_increase": 0.01,
            }
        )
        for _ in range(20):
            det.observe("p", "1.0", quality_score=0.9, guardrail_violations=0)
        for _ in range(20):
            det.observe("p", "1.0", quality_score=0.5, guardrail_violations=2)
        report = det.detect("p", "1.0")
        assert report.is_drifted
        assert report.severity == AlertSeverity.HIGH
        assert len(report.drifted_features) >= 2

    def test_single_signal_warning_severity(self) -> None:
        det = self._make_detector(signals={"quality_score_decline": 0.05})
        for _ in range(20):
            det.observe("p", "1.0", quality_score=0.9)
        for _ in range(20):
            det.observe("p", "1.0", quality_score=0.5)
        report = det.detect("p", "1.0")
        assert report.severity == AlertSeverity.WARNING

    # ── report metadata ───────────────────────────────────────────

    def test_report_metadata(self) -> None:
        det = self._make_detector()
        for _ in range(30):
            det.observe("my_prompt", "2.0", quality_score=0.8)
        report = det.detect("my_prompt", "2.0")
        assert report.model_name == "my_prompt"
        assert report.method == "prompt_drift"
        assert report.metadata.get("prompt_version") == "2.0"

    def test_report_test_statistic_is_signal_count(self) -> None:
        det = self._make_detector(signals={"quality_score_decline": 0.05})
        for _ in range(20):
            det.observe("p", "1.0", quality_score=0.9)
        for _ in range(20):
            det.observe("p", "1.0", quality_score=0.5)
        report = det.detect("p", "1.0")
        assert report.test_statistic >= 1.0

    # ── default config ────────────────────────────────────────────

    def test_default_config_signals(self) -> None:
        det = PromptDriftDetector()
        assert "quality_score_decline" in det.signals
        assert "guardrail_violation_rate_increase" in det.signals
        assert "token_usage_increase_pct" in det.signals

    # ── key format ────────────────────────────────────────────────

    def test_separate_versions_tracked_independently(self) -> None:
        det = self._make_detector()
        for _ in range(25):
            det.observe("p", "1.0", quality_score=0.9)
            det.observe("p", "2.0", quality_score=0.3)
        report_v1 = det.detect("p", "1.0")
        report_v2 = det.detect("p", "2.0")
        # v1 is stable, v2 is stable too (both halves same)
        assert not report_v1.is_drifted
        assert not report_v2.is_drifted
