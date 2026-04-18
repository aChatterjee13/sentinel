"""Unit tests for LineageTracker (expanded coverage)."""

from __future__ import annotations

from sentinel.foundation.audit.lineage import LineageEdge, LineageNode, LineageTracker


class TestLineageNode:
    """Tests for the LineageNode dataclass."""

    def test_create_node(self) -> None:
        node = LineageNode(node_id="ds1", kind="dataset", metadata={"path": "/data"})
        assert node.node_id == "ds1"
        assert node.kind == "dataset"
        assert node.metadata["path"] == "/data"

    def test_default_metadata(self) -> None:
        node = LineageNode(node_id="n1", kind="model_version")
        assert node.metadata == {}


class TestLineageEdge:
    """Tests for the LineageEdge dataclass."""

    def test_create_edge(self) -> None:
        edge = LineageEdge(src="ds1", dst="model1", relation="trained_on")
        assert edge.src == "ds1"
        assert edge.dst == "model1"
        assert edge.relation == "trained_on"


class TestLineageTracker:
    """Tests for LineageTracker."""

    def _make_tracker(self) -> LineageTracker:
        return LineageTracker()

    # ── add_node ──────────────────────────────────────────────────

    def test_add_node(self) -> None:
        lt = self._make_tracker()
        node = lt.add_node("ds1", "dataset", path="/data/train.csv")
        assert node.node_id == "ds1"
        assert node.kind == "dataset"
        assert node.metadata["path"] == "/data/train.csv"

    def test_add_multiple_nodes(self) -> None:
        lt = self._make_tracker()
        lt.add_node("ds1", "dataset")
        lt.add_node("exp1", "experiment")
        lt.add_node("model1", "model_version")
        assert len(lt._nodes) == 3

    def test_overwrite_node(self) -> None:
        lt = self._make_tracker()
        lt.add_node("n1", "dataset", version="v1")
        lt.add_node("n1", "dataset", version="v2")
        assert lt._nodes["n1"].metadata["version"] == "v2"

    # ── add_edge ──────────────────────────────────────────────────

    def test_add_edge(self) -> None:
        lt = self._make_tracker()
        lt.add_node("ds1", "dataset")
        lt.add_node("model1", "model_version")
        lt.add_edge("ds1", "model1", "trained_on")
        assert len(lt._edges_out["ds1"]) == 1
        assert len(lt._edges_in["model1"]) == 1

    def test_add_multiple_edges(self) -> None:
        lt = self._make_tracker()
        lt.add_node("ds1", "dataset")
        lt.add_node("ds2", "dataset")
        lt.add_node("model1", "model_version")
        lt.add_edge("ds1", "model1", "trained_on")
        lt.add_edge("ds2", "model1", "validated_on")
        assert len(lt._edges_in["model1"]) == 2

    # ── get_ancestors ─────────────────────────────────────────────

    def test_get_ancestors_direct(self) -> None:
        lt = self._make_tracker()
        lt.add_node("ds1", "dataset")
        lt.add_node("model1", "model_version")
        lt.add_edge("ds1", "model1", "trained_on")
        ancestors = lt.get_ancestors("model1")
        assert len(ancestors) == 1
        assert ancestors[0].node_id == "ds1"

    def test_get_ancestors_transitive(self) -> None:
        lt = self._make_tracker()
        lt.add_node("raw", "dataset")
        lt.add_node("processed", "dataset")
        lt.add_node("model", "model_version")
        lt.add_edge("raw", "processed", "derived_from")
        lt.add_edge("processed", "model", "trained_on")
        ancestors = lt.get_ancestors("model")
        ids = {a.node_id for a in ancestors}
        assert ids == {"raw", "processed"}

    def test_get_ancestors_no_ancestors(self) -> None:
        lt = self._make_tracker()
        lt.add_node("root", "dataset")
        ancestors = lt.get_ancestors("root")
        assert ancestors == []

    def test_get_ancestors_handles_cycle(self) -> None:
        lt = self._make_tracker()
        lt.add_node("a", "experiment")
        lt.add_node("b", "experiment")
        lt.add_edge("a", "b", "produced_by")
        lt.add_edge("b", "a", "produced_by")
        # Should not infinite loop
        ancestors = lt.get_ancestors("a")
        assert len(ancestors) <= 2

    def test_get_ancestors_deep_chain(self) -> None:
        lt = self._make_tracker()
        for i in range(10):
            lt.add_node(f"n{i}", "step")
        for i in range(9):
            lt.add_edge(f"n{i}", f"n{i + 1}", "feeds")
        ancestors = lt.get_ancestors("n9")
        assert len(ancestors) == 9

    # ── get_descendants ───────────────────────────────────────────

    def test_get_descendants_direct(self) -> None:
        lt = self._make_tracker()
        lt.add_node("model", "model_version")
        lt.add_node("pred", "prediction")
        lt.add_edge("model", "pred", "served_by")
        descendants = lt.get_descendants("model")
        assert len(descendants) == 1
        assert descendants[0].node_id == "pred"

    def test_get_descendants_transitive(self) -> None:
        lt = self._make_tracker()
        lt.add_node("a", "step")
        lt.add_node("b", "step")
        lt.add_node("c", "step")
        lt.add_edge("a", "b", "feeds")
        lt.add_edge("b", "c", "feeds")
        descendants = lt.get_descendants("a")
        ids = {d.node_id for d in descendants}
        assert ids == {"b", "c"}

    def test_get_descendants_no_descendants(self) -> None:
        lt = self._make_tracker()
        lt.add_node("leaf", "prediction")
        descendants = lt.get_descendants("leaf")
        assert descendants == []

    def test_get_descendants_handles_cycle(self) -> None:
        lt = self._make_tracker()
        lt.add_node("x", "step")
        lt.add_node("y", "step")
        lt.add_edge("x", "y", "feeds")
        lt.add_edge("y", "x", "feeds")
        descendants = lt.get_descendants("x")
        assert len(descendants) <= 2

    # ── to_dict serialization ─────────────────────────────────────

    def test_to_dict_empty(self) -> None:
        lt = self._make_tracker()
        d = lt.to_dict()
        assert d == {"nodes": [], "edges": []}

    def test_to_dict_with_data(self) -> None:
        lt = self._make_tracker()
        lt.add_node("ds1", "dataset", rows=1000)
        lt.add_node("model1", "model_version")
        lt.add_edge("ds1", "model1", "trained_on")
        d = lt.to_dict()
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1
        assert d["edges"][0]["src"] == "ds1"
        assert d["edges"][0]["dst"] == "model1"
        assert d["edges"][0]["relation"] == "trained_on"

    def test_to_dict_node_metadata(self) -> None:
        lt = self._make_tracker()
        lt.add_node("n1", "dataset", path="/data", version="v2")
        d = lt.to_dict()
        node = d["nodes"][0]
        assert node["id"] == "n1"
        assert node["kind"] == "dataset"
        assert node["metadata"]["path"] == "/data"

    def test_to_dict_multiple_edges(self) -> None:
        lt = self._make_tracker()
        lt.add_node("a", "step")
        lt.add_node("b", "step")
        lt.add_node("c", "step")
        lt.add_edge("a", "b", "feeds")
        lt.add_edge("a", "c", "feeds")
        d = lt.to_dict()
        assert len(d["edges"]) == 2

    # ── edge between nonexistent nodes ────────────────────────────

    def test_edge_to_unknown_node_ancestors(self) -> None:
        lt = self._make_tracker()
        lt.add_node("b", "step")
        lt.add_edge("unknown", "b", "feeds")
        # ancestors of b includes "unknown" which has no node entry
        ancestors = lt.get_ancestors("b")
        assert len(ancestors) == 0  # unknown is not in _nodes

    def test_get_descendants_from_nonexistent(self) -> None:
        lt = self._make_tracker()
        descendants = lt.get_descendants("nonexistent")
        assert descendants == []
