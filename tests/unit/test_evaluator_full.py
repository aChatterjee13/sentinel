"""Unit tests for ResponseEvaluator (expanded coverage)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sentinel.config.schema import QualityEvaluatorConfig
from sentinel.llmops.quality.evaluator import ResponseEvaluator


class TestHeuristicScoring:
    """Test the heuristic scoring method."""

    def test_basic_response(self) -> None:
        evaluator = ResponseEvaluator()
        score = evaluator.evaluate(
            response="The insurance claim was processed successfully.",
            query="What happened to my claim?",
        )
        assert 0 < score.overall <= 1.0
        assert score.method == "heuristic"
        assert "relevance" in score.rubric_scores
        assert "completeness" in score.rubric_scores
        assert "clarity" in score.rubric_scores
        assert "safety" in score.rubric_scores

    def test_empty_response_low_completeness(self) -> None:
        evaluator = ResponseEvaluator()
        score = evaluator.evaluate(response="Ok.", query="Explain the policy.")
        assert score.rubric_scores["completeness"] < 0.5

    def test_long_response_better_completeness(self) -> None:
        evaluator = ResponseEvaluator()
        text = "This is a detailed explanation. " * 20
        score = evaluator.evaluate(response=text, query="Explain the policy.")
        assert score.rubric_scores["completeness"] >= 0.5

    def test_no_query_default_relevance(self) -> None:
        evaluator = ResponseEvaluator()
        score = evaluator.evaluate(response="Some text.")
        assert score.rubric_scores["relevance"] == 0.5

    def test_high_overlap_with_query(self) -> None:
        evaluator = ResponseEvaluator()
        query = "What is the fraud detection accuracy"
        response = "The fraud detection accuracy is 95 percent."
        score = evaluator.evaluate(response=response, query=query)
        assert score.rubric_scores["relevance"] > 0.3

    def test_all_caps_reduces_safety(self) -> None:
        evaluator = ResponseEvaluator()
        score = evaluator.evaluate(response="THIS IS ALL CAPS SHOUTING AT YOU!")
        assert score.rubric_scores["safety"] < 0.5

    def test_metadata_includes_word_and_sentence_count(self) -> None:
        evaluator = ResponseEvaluator()
        score = evaluator.evaluate(response="Hello world. This is a test.")
        assert "word_count" in score.metadata
        assert "sentence_count" in score.metadata

    def test_no_sentences_zero_clarity(self) -> None:
        evaluator = ResponseEvaluator()
        # A single long string with no sentence-ending punctuation
        score = evaluator.evaluate(response="")
        assert score.rubric_scores["clarity"] == 0.0


class TestLLMJudge:
    """Test the LLM-as-judge method."""

    def test_llm_judge_calls_judge_fn(self) -> None:
        mock_judge = MagicMock(
            return_value={
                "relevance": 0.9,
                "completeness": 0.8,
                "clarity": 0.7,
                "safety": 1.0,
            }
        )
        config = QualityEvaluatorConfig(method="llm_judge")
        evaluator = ResponseEvaluator(config=config, judge_fn=mock_judge)
        score = evaluator.evaluate(response="answer", query="question")
        assert score.method == "llm_judge"
        assert score.overall > 0
        mock_judge.assert_called_once()

    def test_llm_judge_fallback_on_error(self) -> None:
        mock_judge = MagicMock(side_effect=RuntimeError("LLM unavailable"))
        config = QualityEvaluatorConfig(method="llm_judge")
        evaluator = ResponseEvaluator(config=config, judge_fn=mock_judge)
        score = evaluator.evaluate(response="answer", query="question")
        # Should fall back to heuristic
        assert score.method == "heuristic"

    def test_llm_judge_without_fn_falls_back(self) -> None:
        config = QualityEvaluatorConfig(method="llm_judge")
        evaluator = ResponseEvaluator(config=config, judge_fn=None)
        score = evaluator.evaluate(response="answer", query="question")
        assert score.method == "heuristic"


class TestReferenceBased:
    """Test the reference-based method."""

    def test_reference_based_exact_match(self) -> None:
        config = QualityEvaluatorConfig(method="reference_based")
        evaluator = ResponseEvaluator(config=config)
        score = evaluator.evaluate(
            response="The claim was approved for 5000 dollars.",
            reference="The claim was approved for 5000 dollars.",
        )
        assert score.method == "reference_based"
        assert score.overall > 0.9

    def test_reference_based_no_overlap(self) -> None:
        config = QualityEvaluatorConfig(method="reference_based")
        evaluator = ResponseEvaluator(config=config)
        score = evaluator.evaluate(
            response="alpha beta gamma",
            reference="delta epsilon zeta",
        )
        assert score.overall == 0.0

    def test_reference_based_partial_overlap(self) -> None:
        config = QualityEvaluatorConfig(method="reference_based")
        evaluator = ResponseEvaluator(config=config)
        score = evaluator.evaluate(
            response="The weather is sunny today.",
            reference="The weather is rainy today.",
        )
        assert 0 < score.overall < 1.0
        assert "precision" in score.rubric_scores
        assert "recall" in score.rubric_scores
        assert "f1" in score.rubric_scores

    def test_reference_based_empty_reference(self) -> None:
        config = QualityEvaluatorConfig(method="reference_based")
        evaluator = ResponseEvaluator(config=config)
        # Empty string is falsy, so evaluate() falls through to heuristic
        score = evaluator.evaluate(response="something", reference="")
        assert score.method == "heuristic"

    def test_reference_based_without_reference_falls_back(self) -> None:
        config = QualityEvaluatorConfig(method="reference_based")
        evaluator = ResponseEvaluator(config=config)
        score = evaluator.evaluate(response="something", reference=None)
        # Falls through to heuristic because no reference
        assert score.method == "heuristic"


class TestHybrid:
    """Test the hybrid method."""

    def test_hybrid_combines_methods(self) -> None:
        config = QualityEvaluatorConfig(method="hybrid")
        evaluator = ResponseEvaluator(config=config)
        score = evaluator.evaluate(
            response="The claim is valid.",
            query="Is the claim valid?",
            reference="The claim is valid and approved.",
        )
        assert score.method == "hybrid"
        assert any("heuristic" in k for k in score.rubric_scores)
        assert any("reference_based" in k for k in score.rubric_scores)

    def test_hybrid_with_judge(self) -> None:
        mock_judge = MagicMock(return_value={"relevance": 0.8, "clarity": 0.9})
        config = QualityEvaluatorConfig(method="hybrid")
        evaluator = ResponseEvaluator(config=config, judge_fn=mock_judge)
        score = evaluator.evaluate(
            response="answer",
            query="question",
            reference="ref",
        )
        assert score.method == "hybrid"
        # Should include llm_judge scores too
        assert any("llm_judge" in k for k in score.rubric_scores)

    def test_hybrid_without_reference_or_judge(self) -> None:
        config = QualityEvaluatorConfig(method="hybrid")
        evaluator = ResponseEvaluator(config=config)
        score = evaluator.evaluate(response="answer", query="question")
        assert score.method == "hybrid"


class TestRubricConfiguration:
    """Test configurable rubrics and weights."""

    def test_custom_rubrics(self) -> None:
        config = QualityEvaluatorConfig(
            method="heuristic",
            rubrics={
                "relevance": {"weight": 1.0, "scale": 5},
                "completeness": {"weight": 0.0, "scale": 5},
                "clarity": {"weight": 0.0, "scale": 5},
                "safety": {"weight": 0.0, "scale": 5},
            },
        )
        evaluator = ResponseEvaluator(config=config)
        score = evaluator.evaluate(
            response="relevant words from the query here.",
            query="relevant words",
        )
        # Overall should be driven entirely by relevance
        assert score.overall == pytest.approx(score.rubric_scores["relevance"], abs=0.01)

    def test_default_rubrics_assigned(self) -> None:
        evaluator = ResponseEvaluator()
        assert "relevance" in evaluator.rubrics
        assert "completeness" in evaluator.rubrics
        assert evaluator.rubrics["relevance"]["weight"] == 0.3


class TestSampling:
    """Test the sampling mechanism."""

    def test_should_evaluate_respects_sample_rate(self) -> None:
        config = QualityEvaluatorConfig(sample_rate=1.0)
        evaluator = ResponseEvaluator(config=config)
        assert evaluator.should_evaluate()

    def test_zero_sample_rate_never_evaluates(self) -> None:
        config = QualityEvaluatorConfig(sample_rate=0.0)
        evaluator = ResponseEvaluator(config=config)
        # With rate=0, should never evaluate
        results = [evaluator.should_evaluate() for _ in range(100)]
        assert not any(results)


class TestWeightedAverage:
    """Test the _weighted_average helper."""

    def test_weighted_average(self) -> None:
        evaluator = ResponseEvaluator()
        scores = {"relevance": 1.0, "completeness": 0.5}
        result = evaluator._weighted_average(scores)
        assert 0 < result < 1.0

    def test_weighted_average_empty(self) -> None:
        evaluator = ResponseEvaluator()
        assert evaluator._weighted_average({}) == 0.0

    def test_weighted_average_unknown_key(self) -> None:
        evaluator = ResponseEvaluator()
        # Unknown keys get weight 1.0
        result = evaluator._weighted_average({"unknown": 0.5})
        assert result == 0.5
