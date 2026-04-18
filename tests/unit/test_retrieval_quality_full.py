"""Unit tests for RetrievalQualityMonitor (expanded coverage)."""

from __future__ import annotations

from sentinel.config.schema import RetrievalQualityConfig
from sentinel.llmops.quality.retrieval_quality import (
    RetrievalQualityMonitor,
    RetrievalQualityResult,
)


class TestRetrievalQualityResult:
    """Tests for the RetrievalQualityResult dataclass."""

    def test_is_acceptable_both_above(self) -> None:
        result = RetrievalQualityResult(relevance=0.8, faithfulness=0.9)
        assert result.is_acceptable

    def test_is_acceptable_low_relevance(self) -> None:
        result = RetrievalQualityResult(relevance=0.3, faithfulness=0.9)
        assert not result.is_acceptable

    def test_is_acceptable_low_faithfulness(self) -> None:
        result = RetrievalQualityResult(relevance=0.8, faithfulness=0.5)
        assert not result.is_acceptable

    def test_default_values(self) -> None:
        result = RetrievalQualityResult()
        assert result.relevance == 0.0
        assert result.chunk_utilisation == 0.0
        assert result.faithfulness == 0.0
        assert result.chunks_retrieved == 0


class TestRetrievalQualityMonitor:
    """Tests for RetrievalQualityMonitor."""

    def _make_monitor(self, **kwargs) -> RetrievalQualityMonitor:
        config = RetrievalQualityConfig(**kwargs)
        return RetrievalQualityMonitor(config=config)

    # ── relevance scoring ─────────────────────────────────────────

    def test_high_relevance(self) -> None:
        mon = self._make_monitor()
        result = mon.evaluate(
            query="What is the policy coverage limit?",
            response="The policy coverage limit is 100000.",
            chunks=["The policy coverage limit for this plan is 100000 dollars."],
        )
        assert result.relevance > 0.3

    def test_low_relevance(self) -> None:
        mon = self._make_monitor()
        result = mon.evaluate(
            query="What is the policy coverage limit?",
            response="The weather is sunny.",
            chunks=["Today's lunch menu includes pasta and salad."],
        )
        assert result.relevance < 0.5

    # ── chunk utilisation ─────────────────────────────────────────

    def test_all_chunks_used(self) -> None:
        mon = self._make_monitor()
        result = mon.evaluate(
            query="Tell me about the claim",
            response="The claim was filed on Monday and approved Tuesday.",
            chunks=[
                "The claim was filed on Monday.",
                "The claim was approved Tuesday.",
            ],
        )
        assert result.chunk_utilisation == 1.0
        assert result.chunks_used == 2

    def test_some_chunks_unused(self) -> None:
        mon = self._make_monitor()
        result = mon.evaluate(
            query="Tell me about the claim",
            response="The claim was filed on Monday.",
            chunks=[
                "The claim was filed on Monday.",
                "Completely irrelevant marketing text xyz.",
            ],
        )
        assert result.chunk_utilisation < 1.0

    # ── faithfulness ──────────────────────────────────────────────

    def test_high_faithfulness(self) -> None:
        mon = self._make_monitor()
        result = mon.evaluate(
            query="What is the deductible?",
            response="The deductible is 500 dollars per year.",
            chunks=["The annual deductible is 500 dollars per year for this plan."],
        )
        assert result.faithfulness > 0.5

    def test_low_faithfulness_hallucination(self) -> None:
        mon = self._make_monitor()
        result = mon.evaluate(
            query="What is the deductible?",
            response="The deductible is 1000 euros and covers dental implants.",
            chunks=["The plan covers vision care only."],
        )
        # Response tokens mostly not in chunks
        assert result.faithfulness < 0.8

    # ── answer coverage ───────────────────────────────────────────

    def test_high_answer_coverage(self) -> None:
        mon = self._make_monitor()
        result = mon.evaluate(
            query="What is the coverage and deductible?",
            response="The coverage is full and the deductible is 500.",
            chunks=["Full coverage plan with 500 dollar deductible."],
        )
        assert result.answer_coverage > 0.3

    def test_low_answer_coverage(self) -> None:
        mon = self._make_monitor()
        result = mon.evaluate(
            query="What are the coverage limits and exclusions?",
            response="Hello",
            chunks=["Coverage limits are 100k."],
        )
        assert result.answer_coverage < 0.3

    # ── empty inputs ──────────────────────────────────────────────

    def test_no_chunks_returns_defaults(self) -> None:
        mon = self._make_monitor()
        result = mon.evaluate(
            query="anything",
            response="some response",
            chunks=[],
        )
        assert result.chunks_retrieved == 0
        assert result.relevance == 0.0
        assert result.faithfulness == 0.0

    # ── dict chunks ───────────────────────────────────────────────

    def test_dict_chunks_supported(self) -> None:
        mon = self._make_monitor()
        result = mon.evaluate(
            query="What is the policy?",
            response="The policy covers fire damage.",
            chunks=[{"text": "The policy covers fire damage and flooding."}],
        )
        assert result.chunks_retrieved == 1
        assert result.relevance > 0

    def test_dict_chunks_without_text_key(self) -> None:
        mon = self._make_monitor()
        result = mon.evaluate(
            query="query",
            response="response content",
            chunks=[{"content": "some content"}],
        )
        assert result.chunks_retrieved == 1

    # ── config thresholds ─────────────────────────────────────────

    def test_low_relevance_logs_warning(self) -> None:
        mon = self._make_monitor(min_relevance=0.9)
        result = mon.evaluate(
            query="unique query",
            response="irrelevant response",
            chunks=["unrelated chunk"],
        )
        # Should not raise, just log
        assert result.relevance < 0.9

    def test_low_faithfulness_logs_warning(self) -> None:
        mon = self._make_monitor(min_faithfulness=0.99)
        result = mon.evaluate(
            query="test",
            response="completely made up hallucination text",
            chunks=["actual chunk content"],
        )
        assert result.faithfulness < 0.99

    # ── multiple chunks aggregation ───────────────────────────────

    def test_multiple_chunks_relevance_averaged(self) -> None:
        mon = self._make_monitor()
        result = mon.evaluate(
            query="insurance policy claim",
            response="policy details here",
            chunks=[
                "insurance policy details about coverage",
                "weather forecast for tomorrow",
                "claim processing information",
            ],
        )
        assert result.chunks_retrieved == 3
        assert 0 < result.relevance < 1.0
