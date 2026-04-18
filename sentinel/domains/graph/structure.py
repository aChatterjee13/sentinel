"""Pure-Python graph topology metrics — no networkx required."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

Edge = tuple[str, str]


@dataclass
class TopologyStats:
    """Aggregated topology statistics for a graph snapshot."""

    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    clustering_coefficient: float
    n_components: int
    largest_component_size: int


def _adjacency(edges: Iterable[Edge]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def degree_distribution(edges: Sequence[Edge]) -> Counter:
    """Counter mapping degree → number of nodes with that degree."""
    adj = _adjacency(edges)
    return Counter(len(neigh) for neigh in adj.values())


def density(n_nodes: int, n_edges: int) -> float:
    """Edge density: actual / possible undirected edges."""
    if n_nodes < 2:
        return 0.0
    return (2 * n_edges) / (n_nodes * (n_nodes - 1))


def clustering_coefficient(edges: Sequence[Edge]) -> float:
    """Average local clustering coefficient over all nodes with degree ≥ 2."""
    adj = _adjacency(edges)
    coeffs = []
    for _node, neigh in adj.items():
        if len(neigh) < 2:
            continue
        possible = len(neigh) * (len(neigh) - 1) / 2
        actual = 0
        neigh_list = list(neigh)
        for i, a in enumerate(neigh_list):
            for b in neigh_list[i + 1 :]:
                if b in adj[a]:
                    actual += 1
        coeffs.append(actual / possible if possible else 0.0)
    return float(sum(coeffs) / len(coeffs)) if coeffs else 0.0


def connected_components(
    edges: Sequence[Edge], all_nodes: Iterable[str] | None = None
) -> list[set[str]]:
    """Return the list of connected components in the undirected graph."""
    adj = _adjacency(edges)
    nodes = set(adj.keys())
    if all_nodes is not None:
        nodes |= set(all_nodes)
    visited: set[str] = set()
    components: list[set[str]] = []
    for node in nodes:
        if node in visited:
            continue
        comp: set[str] = set()
        queue = deque([node])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            comp.add(current)
            queue.extend(adj.get(current, ()))
        components.append(comp)
    return components


def topology_stats(edges: Sequence[Edge], all_nodes: Iterable[str] | None = None) -> TopologyStats:
    adj = _adjacency(edges)
    nodes = set(adj.keys())
    if all_nodes is not None:
        nodes |= set(all_nodes)
    n_nodes = len(nodes)
    n_edges = len(edges)
    deg_total = sum(len(n) for n in adj.values())
    components = connected_components(edges, all_nodes=nodes)
    largest = max((len(c) for c in components), default=0)
    return TopologyStats(
        n_nodes=n_nodes,
        n_edges=n_edges,
        density=density(n_nodes, n_edges),
        avg_degree=deg_total / n_nodes if n_nodes else 0.0,
        clustering_coefficient=clustering_coefficient(edges),
        n_components=len(components),
        largest_component_size=largest,
    )
