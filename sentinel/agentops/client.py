"""AgentOpsClient — single-entry-point wrapper around the AgentOps modules."""

from __future__ import annotations

from typing import Any

import structlog

from sentinel.agentops.agent_registry import AgentRegistry, AgentSpec
from sentinel.agentops.eval.golden_datasets import GoldenSuiteRunner
from sentinel.agentops.eval.task_completion import TaskCompletionTracker
from sentinel.agentops.eval.trajectory import TrajectoryEvaluator
from sentinel.agentops.multi_agent.orchestration import MultiAgentMonitor
from sentinel.agentops.safety.budget_guard import BudgetGuard
from sentinel.agentops.safety.escalation import EscalationManager
from sentinel.agentops.safety.loop_detector import LoopDetector
from sentinel.agentops.safety.sandbox import ActionSandbox
from sentinel.agentops.tool_audit.monitor import ToolAuditMonitor
from sentinel.agentops.tool_audit.permissions import PermissionMatrix
from sentinel.agentops.trace.tracer import AgentTracer
from sentinel.config.schema import AgentOpsConfig
from sentinel.core.exceptions import EscalationRequired
from sentinel.core.types import AgentTrace
from sentinel.foundation.audit.trail import AuditTrail

log = structlog.get_logger(__name__)


class AgentOpsClient:
    """Composite client that wires every AgentOps subsystem together.

    The client owns one tracer, one tool-audit monitor, one safety
    layer (loop detector + budget guard + escalation manager + sandbox),
    one agent registry, one multi-agent monitor, and one evaluation
    suite. It exposes ``begin_run``/``end_run`` to give callers a single
    place to start and finalise an agent execution.
    """

    def __init__(
        self,
        config: AgentOpsConfig,
        audit: AuditTrail | None = None,
    ):
        self.config = config
        self.audit = audit
        self.tracer = AgentTracer(config.tracing)
        self.permissions = PermissionMatrix(config.tool_audit)
        self.tool_monitor = ToolAuditMonitor(config.tool_audit, permissions=self.permissions)
        self.loop_detector = LoopDetector(config.safety.loop_detection)
        self.budget_guard = BudgetGuard(config.safety.budget)
        self.escalation = EscalationManager(config.safety.escalation)
        self.sandbox = ActionSandbox(config.safety.sandbox)
        self.registry = AgentRegistry(config.agent_registry)
        self.multi_agent = MultiAgentMonitor(config.multi_agent)
        self.task_completion = TaskCompletionTracker(config.evaluation)
        self.trajectory = TrajectoryEvaluator(config.evaluation)
        self.golden_runner = GoldenSuiteRunner(config.evaluation, trajectory=self.trajectory)

    # ── Run lifecycle ─────────────────────────────────────────────

    def begin_run(self, run_id: str, agent_name: str, **metadata: Any) -> None:
        self.loop_detector.begin_run(run_id)
        self.budget_guard.begin_run(run_id)
        if self.audit is not None:
            self.audit.log(
                event_type="agent.run.start",
                model_name=agent_name,
                run_id=run_id,
                **metadata,
            )
        log.info("agent.run.begin", run_id=run_id, agent=agent_name)

    def end_run(
        self,
        run_id: str,
        agent_name: str,
        *,
        success: bool = True,
        task_type: str = "default",
        score: float | None = None,
        trace: AgentTrace | None = None,
        **metadata: Any,
    ) -> dict[str, Any]:
        budget_state = self.budget_guard.end_run(run_id)
        self.loop_detector.end_run(run_id)
        delegations = self.multi_agent.end_run(run_id)
        if trace is not None:
            self.multi_agent.ingest_trace(trace)
        duration_ms = trace.duration_ms if trace else None
        completion = self.task_completion.record(
            agent=agent_name,
            task_type=task_type,
            success=success,
            score=score,
            duration_ms=duration_ms,
            **metadata,
        )
        if self.audit is not None:
            self.audit.log(
                event_type="agent.run.end",
                model_name=agent_name,
                run_id=run_id,
                success=success,
                tokens_used=getattr(budget_state, "tokens_used", 0),
                cost_used=getattr(budget_state, "cost_used", 0.0),
                tool_calls=getattr(budget_state, "tool_calls", 0),
                delegations=len(delegations),
                duration_ms=duration_ms,
            )
        log.info(
            "agent.run.end",
            run_id=run_id,
            agent=agent_name,
            success=success,
            tokens=getattr(budget_state, "tokens_used", 0),
            cost=getattr(budget_state, "cost_used", 0.0),
        )
        return {
            "tokens": getattr(budget_state, "tokens_used", 0),
            "cost": getattr(budget_state, "cost_used", 0.0),
            "tool_calls": getattr(budget_state, "tool_calls", 0),
            "delegations": [d.target for d in delegations],
            "completion": completion,
        }

    # ── Per-step helpers ──────────────────────────────────────────

    def step(self, run_id: str) -> None:
        self.loop_detector.step(run_id)
        self.budget_guard.check_time(run_id)

    def call_tool(
        self,
        run_id: str,
        agent_name: str,
        tool_name: str,
        inputs: dict[str, Any],
        *,
        output: Any = None,
        latency_ms: float = 0.0,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        self.tool_monitor.authorise(agent_name, tool_name, inputs=inputs)
        self.loop_detector.record_tool_call(run_id, tool_name, inputs)
        self.budget_guard.record_tool_call(run_id)
        self.tool_monitor.record(
            agent=agent_name,
            tool=tool_name,
            inputs=inputs,
            output=output,
            success=success,
            latency_ms=latency_ms,
            error=error,
        )
        decision = self.escalation.check(run_id, success=success)
        if decision.triggered:
            log.warning("agent.escalation", run_id=run_id, reason=decision.reason)

    def add_tokens(self, run_id: str, tokens: int) -> None:
        self.budget_guard.add_tokens(run_id, tokens)

    def add_cost(self, run_id: str, cost: float) -> None:
        self.budget_guard.add_cost(run_id, cost)

    def delegate(self, run_id: str, source: str, target: str, task: str, **metadata: Any) -> None:
        self.loop_detector.record_delegation(run_id, target)
        self.multi_agent.on_delegation(run_id, source, target, task, **metadata)

    def check_confidence(
        self, run_id: str, confidence: float, action_text: str | None = None
    ) -> None:
        decision = self.escalation.check(run_id, confidence=confidence, action_text=action_text)
        if decision.triggered:
            raise EscalationRequired(decision.reason or "escalation required")

    # ── Registry helpers ──────────────────────────────────────────

    def register_agent(self, spec: AgentSpec) -> AgentSpec:
        spec = self.registry.register(spec)
        if self.audit is not None:
            self.audit.log(
                event_type="agent.registered",
                model_name=spec.name,
                model_version=spec.version,
                capabilities=spec.capabilities,
            )
        return spec

    def discover(self, capability: str) -> list[AgentSpec]:
        return self.registry.find_by_capability(capability)

    # ── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "agents": self.registry.list_agents(),
            "tools": self.tool_monitor.all_stats(),
            "task_completion": self.task_completion.stats(),
            "multi_agent": self.multi_agent.stats(),
            "recent_traces": [
                {
                    "trace_id": t.trace_id,
                    "agent": t.agent_name,
                    "duration_ms": t.duration_ms,
                    "tokens": t.total_tokens,
                    "tool_calls": t.tool_call_count,
                }
                for t in self.tracer.get_recent(10)
            ],
        }
