"""NLP domain adapter."""

from __future__ import annotations

from typing import Any

from sentinel.config.schema import SentinelConfig
from sentinel.domains.base import BaseDomainAdapter
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
from sentinel.domains.nlp.text_stats import TextStatsMonitor
from sentinel.observability.drift.base import BaseDriftDetector


class _NLPQualityMetric:
    def __init__(self, name: str, fn):
        self.name = name
        self.fn = fn

    def __call__(self, *args: Any, **kwargs: Any):
        return self.fn(*args, **kwargs)


class NLPAdapter(BaseDomainAdapter):
    """Adapter for traditional NLP models — NER, classification, sentiment."""

    domain = "nlp"

    VALID_TASKS: frozenset[str] = frozenset(
        {"ner", "classification", "sentiment", "topic_modelling"}
    )

    def __init__(self, config: SentinelConfig):
        super().__init__(config)
        self.task = self.options.get("task", "classification")
        if self.task not in self.VALID_TASKS:
            raise ValueError(
                f"invalid NLP task '{self.task}', must be one of {sorted(self.VALID_TASKS)}"
            )
        drift_cfg = self.options.get("drift", {})
        vocab_cfg = drift_cfg.get("vocabulary", {})
        self.oov_threshold = float(vocab_cfg.get("oov_rate_threshold", 0.05))
        if self.oov_threshold < 0:
            raise ValueError(
                f"threshold must be non-negative, got {self.oov_threshold}"
            )
        self.top_new_tokens_k = int(vocab_cfg.get("top_new_tokens_k", 50))
        embed_cfg = drift_cfg.get("embedding", {})
        self.embed_method = embed_cfg.get("method", "cosine_centroid")
        self.embed_threshold = float(embed_cfg.get("threshold", 0.1))
        if self.embed_threshold < 0:
            raise ValueError(
                f"threshold must be non-negative, got {self.embed_threshold}"
            )
        label_cfg = drift_cfg.get("label_distribution", {})
        self.label_threshold = float(label_cfg.get("threshold", 0.05))
        if self.label_threshold < 0:
            raise ValueError(
                f"threshold must be non-negative, got {self.label_threshold}"
            )
        self.text_stats = TextStatsMonitor(
            oov_threshold=self.oov_threshold,
            top_new_tokens_k=self.top_new_tokens_k,
        )

    def get_drift_detectors(self) -> list[BaseDriftDetector]:
        return [
            VocabularyDriftDetector(
                model_name=self.model_name,
                threshold=self.oov_threshold,
                top_new_tokens_k=self.top_new_tokens_k,
            ),
            EmbeddingDriftDetector(
                model_name=self.model_name,
                threshold=self.embed_threshold,
                method=self.embed_method,
            ),
            LabelDistributionDriftDetector(
                model_name=self.model_name,
                threshold=self.label_threshold,
            ),
        ]

    def get_quality_metrics(self) -> list[_NLPQualityMetric]:
        if self.task == "ner":
            return [
                _NLPQualityMetric("token_f1", token_f1),
                _NLPQualityMetric("span_exact_match", span_exact_match),
            ]
        return [
            _NLPQualityMetric("classification_metrics", classification_metrics),
        ]

    def get_schema_validator(self) -> Any:
        from sentinel.observability.data_quality import DataQualityChecker

        return DataQualityChecker(self.config.data_quality, model_name=self.model_name)
