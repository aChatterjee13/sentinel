"""Multi-agent orchestration monitor — chains, consensus, bottlenecks."""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import structlog

from sentinel.agentops.multi_agent.consensus import ConsensusEvaluator, ConsensusResult
from sentinel.agentops.multi_agent.delegation import DelegationLink, DelegationTracker
from sentinel.config.schema import MultiAgentConfig
from sentinel.core.types import AgentTrace

log = structlog.get_logger(__name__)


@dataclass
class _AgentLatency:
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=200))


class MultiAgentMonitor:
    """Track delegation chains, consensus, and bottlenecks across agents.

    The monitor is plugged into the agent runtime via three hooks:

    - :meth:`on_delegation` — recorded when one agent calls another
    - :meth:`on_agent_complete` — recorded when an agent finishes a step
    - :meth:`evaluate_consensus` — computes vote agreement across agents
    """

    def __init__(self, config: MultiAgentConfig | None = None):
        self.config = config or MultiAgentConfig()
        self.delegations = DelegationTracker()
        self.consensus = ConsensusEvaluator(self.config)
        self._latency: dict[str, _AgentLatency] = defaultdict(_AgentLatency)
        self._consensus_history: deque[ConsensusResult] = deque(maxlen=500)
        self._lock = threading.RLock()
        bottleneck_cfg = self.config.bottleneck_detection or {}
        self._bottleneck_percentile = bottleneck_cfg.get("latency_percentile", "p95")
        self._bottleneck_threshold_ms = float(bottleneck_cfg.get("threshold_ms", 5000.0))

    # ── Hooks ─────────────────────────────────────────────────────

    def on_delegation(
        self,
        run_id: str,
        source: str,
        target: str,
        task: str,
        **metadata: Any,
    ) -> DelegationLink:
        if not self.config.delegation_tracking:
            return DelegationLink(
                run_id=run_id,
                source=source,
                target=target,
                task=task,
                timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            )
        return self.delegations.record(run_id, source, target, task, **metadata)

    def on_agent_complete(self, agent: str, latency_ms: float) -> None:
        with self._lock:
            self._latency[agent].samples.append(float(latency_ms))

    def end_run(self, run_id: str) -> list[DelegationLink]:
        return self.delegations.end_run(run_id)

    # ── Consensus ─────────────────────────────────────────────────

    def evaluate_consensus(
        self,
        votes: dict[str, Any],
        weights: dict[str, float] | None = None,
    ) -> ConsensusResult:
        result = self.consensus.evaluate(votes, weights=weights)
        with self._lock:
            self._consensus_history.append(result)
        if result.conflict:
            log.warning(
                "multi_agent.conflict",
                voters=result.n_voters,
                agreement=result.agreement,
            )
        return result

    def consensus_disagreement_rate(self) -> float:
        return self.consensus.disagreement_rate(self._consensus_history)

    # ── Bottleneck detection ──────────────────────────────────────

    def bottlenecks(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with self._lock:
            for agent, stats in self._latency.items():
                if not stats.samples:
                    continue
                sorted_samples = sorted(stats.samples)
                p = self._percentile(sorted_samples, self._bottleneck_percentile)
                if p > self._bottleneck_threshold_ms:
                    out.append(
                        {
                            "agent": agent,
                            "percentile": self._bottleneck_percentile,
                            "value_ms": p,
                            "threshold_ms": self._bottleneck_threshold_ms,
                            "n_samples": len(stats.samples),
                        }
                    )
        return out

    @staticmethod
    def _percentile(sorted_samples: list[float], label: str) -> float:
        if not sorted_samples:
            return 0.0
        pct_map = {"p50": 0.5, "p90": 0.9, "p95": 0.95, "p99": 0.99}
        q = pct_map.get(label, 0.95)
        idx = round(q * (len(sorted_samples) - 1))
        return sorted_samples[idx]

    # ── Trace ingestion ───────────────────────────────────────────

    def ingest_trace(self, trace: AgentTrace) -> None:
        """Update latency stats from a finished trace."""
        if trace.duration_ms is None:
            return
        with self._lock:
            self._latency[trace.agent_name].samples.append(float(trace.duration_ms))

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_runs": len(self.delegations._by_run),
                "tracked_agents": list(self._latency.keys()),
                "consensus_disagreement_rate": self.consensus_disagreement_rate(),
                "bottlenecks": self.bottlenecks(),
            }
