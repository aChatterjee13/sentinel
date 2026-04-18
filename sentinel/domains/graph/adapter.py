"""Graph ML / knowledge graph domain adapter."""

from __future__ import annotations

from typing import Any

from sentinel.config.schema import SentinelConfig
from sentinel.domains.base import BaseDomainAdapter
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
from sentinel.observability.drift.base import BaseDriftDetector


class _GraphMetric:
    def __init__(self, name: str, fn):
        self.name = name
        self.fn = fn

    def __call__(self, *args: Any, **kwargs: Any):
        return self.fn(*args, **kwargs)


class GraphAdapter(BaseDomainAdapter):
    """Adapter for graph ML / knowledge graph models."""

    domain = "graph"

    VALID_TASKS: frozenset[str] = frozenset({
        "link_prediction",
        "node_classification",
        "graph_classification",
        "kg_completion",
        "knowledge_graph_completion",
    })
    VALID_GRAPH_TYPES: frozenset[str] = frozenset({
        "knowledge_graph",
        "social_network",
        "transaction_graph",
        "molecular",
    })

    def __init__(self, config: SentinelConfig):
        super().__init__(config)
        self.task = self.options.get("task", "link_prediction")
        if self.task not in self.VALID_TASKS:
            raise ValueError(
                f"invalid graph task '{self.task}', must be one of {sorted(self.VALID_TASKS)}"
            )
        self.graph_type = self.options.get("graph_type", "knowledge_graph")
        if self.graph_type not in self.VALID_GRAPH_TYPES:
            raise ValueError(
                f"invalid graph_type '{self.graph_type}', "
                f"must be one of {sorted(self.VALID_GRAPH_TYPES)}"
            )
        drift_cfg = self.options.get("drift", {})
        topo_cfg = drift_cfg.get("topology", {})
        self.topology_threshold = float(
            topo_cfg.get("threshold", topo_cfg.get("degree_distribution", {}).get("threshold", 0.1))
        )
        if self.topology_threshold < 0:
            raise ValueError(
                f"threshold must be non-negative, got {self.topology_threshold}"
            )
        kg_cfg = self.options.get("knowledge_graph", {})
        self.entity_oov_threshold = float(
            kg_cfg.get("entity_vocabulary", {}).get("oov_entity_threshold", 0.1)
        )
        if self.entity_oov_threshold < 0:
            raise ValueError(
                f"threshold must be non-negative, got {self.entity_oov_threshold}"
            )

    def get_drift_detectors(self) -> list[BaseDriftDetector]:
        detectors: list[BaseDriftDetector] = [
            TopologyDriftDetector(model_name=self.model_name, threshold=self.topology_threshold),
        ]
        if self.graph_type == "knowledge_graph":
            detectors.append(
                EntityVocabularyDriftDetector(
                    model_name=self.model_name,
                    threshold=self.entity_oov_threshold,
                )
            )
        return detectors

    def get_quality_metrics(self) -> list[_GraphMetric]:
        if self.task == "link_prediction":
            return [_GraphMetric("auc_roc", auc_roc), _GraphMetric("isotropy", embedding_isotropy)]
        if self.task in {"kg_completion", "knowledge_graph_completion"}:
            return [
                _GraphMetric("hits_at_10", lambda r: hits_at_k(r, k=10)),
                _GraphMetric("mrr", mrr),
                _GraphMetric("isotropy", embedding_isotropy),
            ]
        return [_GraphMetric("node_classification_f1", node_classification_f1)]

    def get_schema_validator(self) -> Any:
        from sentinel.observability.data_quality import DataQualityChecker

        return DataQualityChecker(self.config.data_quality, model_name=self.model_name)
