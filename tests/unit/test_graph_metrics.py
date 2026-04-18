"""Unit tests for graph ML quality and topology drift."""

from __future__ import annotations

import math

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


class TestStructure:
    def test_density_triangle(self) -> None:
        # Triangle: 3 nodes, 3 edges → density = 1.0
        assert math.isclose(density(3, 3), 1.0)

    def test_density_empty_graph(self) -> None:
        assert density(1, 0) == 0.0

    def test_degree_distribution(self) -> None:
        edges = [("a", "b"), ("a", "c"), ("b", "c")]  # triangle
        dist = degree_distribution(edges)
        assert dist[2] == 3  # all 3 nodes have degree 2

    def test_clustering_coefficient_complete_graph(self) -> None:
        edges = [("a", "b"), ("a", "c"), ("b", "c")]
        score = clustering_coefficient(edges)
        assert math.isclose(score, 1.0)

    def test_connected_components(self) -> None:
        edges = [("a", "b"), ("c", "d"), ("e", "f")]
        comps = connected_components(edges)
        assert len(comps) == 3

    def test_topology_stats_full_bundle(self) -> None:
        edges = [("a", "b"), ("a", "c"), ("b", "c"), ("d", "e")]
        stats = topology_stats(edges)
        assert stats.n_nodes == 5
        assert stats.n_edges == 4
        assert stats.n_components == 2


class TestQualityMetrics:
    def test_auc_roc_perfect(self) -> None:
        scores = [0.1, 0.4, 0.6, 0.9]
        labels = [0, 0, 1, 1]
        assert math.isclose(auc_roc(scores, labels), 1.0)

    def test_auc_roc_random(self) -> None:
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = [0, 1, 0, 1]
        # Tied predictions → AUC ≈ 0.5
        score = auc_roc(scores, labels)
        assert 0.0 <= score <= 1.0

    def test_hits_at_k(self) -> None:
        rankings = [1, 5, 11, 3]
        assert math.isclose(hits_at_k(rankings, k=5), 0.75)

    def test_mrr(self) -> None:
        rankings = [1, 2, 4]
        # (1/1 + 1/2 + 1/4) / 3 ≈ 0.583
        score = mrr(rankings)
        assert math.isclose(score, (1 + 0.5 + 0.25) / 3)

    def test_node_classification_f1(self) -> None:
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "a", "b", "b"]
        score = node_classification_f1(y_true, y_pred)
        assert math.isclose(score, 1.0)

    def test_embedding_isotropy_uniform(self) -> None:
        import numpy as np

        # Identity-like embedding → high isotropy ratio
        emb = np.eye(4)
        score = embedding_isotropy(emb)
        assert 0.0 <= score <= 1.0


class TestTopologyDrift:
    def test_no_drift_on_identical_graph(self) -> None:
        edges = [("a", "b"), ("b", "c"), ("c", "a")]
        d = TopologyDriftDetector(model_name="m", threshold=0.2)
        d.fit(edges)
        report = d.detect(edges)
        assert not report.is_drifted

    def test_detects_density_change(self) -> None:
        ref = [("a", "b"), ("b", "c")]
        cur = [
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("d", "a"),
            ("a", "c"),
            ("b", "d"),
        ]
        d = TopologyDriftDetector(model_name="m", threshold=0.1)
        d.fit(ref)
        report = d.detect(cur)
        assert report.is_drifted


class TestEntityVocabularyDrift:
    def test_no_drift_when_subset(self) -> None:
        d = EntityVocabularyDriftDetector(model_name="m", threshold=0.1)
        d.fit({"a", "b", "c"})
        report = d.detect(["a", "b"])
        assert not report.is_drifted

    def test_detects_oov_entities(self) -> None:
        d = EntityVocabularyDriftDetector(model_name="m", threshold=0.1)
        d.fit({"a", "b"})
        report = d.detect(["a", "x", "y", "z"])
        assert report.is_drifted
