"""Thread safety tests for the AgentOps subsystem.

Each test hammers a shared object from 10 threads x 50 operations and
verifies no exceptions, no data corruption, and consistent final state.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sentinel.agentops.agent_registry import AgentRegistry, AgentSpec
from sentinel.agentops.eval.task_completion import TaskCompletionTracker
from sentinel.agentops.safety.budget_guard import BudgetGuard
from sentinel.agentops.safety.escalation import EscalationManager
from sentinel.agentops.safety.loop_detector import LoopDetector
from sentinel.agentops.tool_audit.monitor import ToolAuditMonitor
from sentinel.agentops.tool_audit.replay import ToolReplayStore
from sentinel.config.schema import (
    AgentRegistryConfig,
    BudgetConfig,
    EscalationConfig,
    EscalationTrigger,
    LoopDetectionConfig,
    ToolAuditConfig,
)

THREADS = 10
OPS = 50


# ── Fix 1: LoopDetector ─────────────────────────────────────────────


class TestLoopDetectorThreadSafety:
    """Concurrent step() + record_tool_call() on same run_id."""

    def test_concurrent_operations(self) -> None:
        detector = LoopDetector(LoopDetectionConfig(max_iterations=10_000))
        run_id = "run-1"
        detector.begin_run(run_id)
        errors: list[Exception] = []

        def work(tid: int) -> None:
            for i in range(OPS):
                try:
                    detector.step(run_id)
                    detector.record_tool_call(
                        run_id, f"tool_{tid}", {"idx": i, "tid": tid}
                    )
                except Exception as exc:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            futures = [pool.submit(work, t) for t in range(THREADS)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"unexpected errors: {errors}"
        state = detector._runs[run_id]
        assert state.iterations == THREADS * OPS


# ── Fix 2: BudgetGuard ──────────────────────────────────────────────


class TestBudgetGuardThreadSafety:
    """Concurrent add_tokens() — total must equal expected sum."""

    def test_concurrent_add_tokens(self) -> None:
        guard = BudgetGuard(BudgetConfig(max_tokens_per_run=10_000_000))
        run_id = "run-budget"
        guard.begin_run(run_id)

        def work(_tid: int) -> None:
            for _ in range(OPS):
                guard.add_tokens(run_id, 1)

        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            futures = [pool.submit(work, t) for t in range(THREADS)]
            for f in as_completed(futures):
                f.result()

        state = guard._runs[run_id]
        assert state.tokens_used == THREADS * OPS


# ── Fix 3: EscalationManager ────────────────────────────────────────


class TestEscalationManagerThreadSafety:
    """Concurrent check() calls — no exceptions."""

    def test_concurrent_checks(self) -> None:
        config = EscalationConfig(
            triggers=[
                EscalationTrigger(
                    condition="consecutive_tool_failures",
                    threshold=9999,
                    action="human_handoff",
                ),
            ]
        )
        mgr = EscalationManager(config)
        errors: list[Exception] = []

        def work(tid: int) -> None:
            run_id = f"run-{tid}"
            for _ in range(OPS):
                try:
                    mgr.check(run_id, success=random.choice([True, False]))
                except Exception as exc:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            futures = [pool.submit(work, t) for t in range(THREADS)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"unexpected errors: {errors}"


# ── Fix 4: ToolAuditMonitor ─────────────────────────────────────────


class TestToolAuditMonitorThreadSafety:
    """Concurrent record() + stats() — no corruption."""

    def test_concurrent_record_and_stats(self) -> None:
        monitor = ToolAuditMonitor(ToolAuditConfig())
        errors: list[Exception] = []

        def work(tid: int) -> None:
            for i in range(OPS):
                try:
                    monitor.record(
                        agent=f"agent-{tid}",
                        tool="my_tool",
                        inputs={"i": i},
                        latency_ms=float(i),
                    )
                    monitor.stats("my_tool")
                except Exception as exc:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            futures = [pool.submit(work, t) for t in range(THREADS)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"unexpected errors: {errors}"
        s = monitor.stats("my_tool")
        assert s["calls"] == THREADS * OPS


# ── Fix 5: AgentRegistry ────────────────────────────────────────────


class TestAgentRegistryThreadSafety:
    """Concurrent register() + find_by_capability() — no lost registrations."""

    def test_concurrent_register_and_find(self, tmp_path: Path) -> None:
        registry = AgentRegistry(
            AgentRegistryConfig(capability_manifest=False),
            root=tmp_path / "agents",
        )
        errors: list[Exception] = []

        def work(tid: int) -> None:
            for i in range(OPS):
                try:
                    spec = AgentSpec(
                        name=f"agent-{tid}",
                        version=f"0.{i}.0",
                        capabilities=["search"],
                    )
                    registry.register(spec)
                    registry.find_by_capability("search")
                except Exception as exc:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            futures = [pool.submit(work, t) for t in range(THREADS)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"unexpected errors: {errors}"
        agents = registry.list_agents()
        assert len(agents) == THREADS


# ── Fix 6: TaskCompletionTracker ─────────────────────────────────────


class TestTaskCompletionTrackerThreadSafety:
    """Concurrent record() + success_rate() — rate must be in [0, 1]."""

    def test_concurrent_record_and_rate(self) -> None:
        tracker = TaskCompletionTracker()
        errors: list[Exception] = []

        def work(tid: int) -> None:
            for _ in range(OPS):
                try:
                    tracker.record(
                        agent="agent-a",
                        task_type="classify",
                        success=random.choice([True, False]),
                    )
                    rate = tracker.success_rate(agent="agent-a", task_type="classify")
                    if rate is not None:
                        assert 0.0 <= rate <= 1.0, f"invalid rate: {rate}"
                except Exception as exc:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            futures = [pool.submit(work, t) for t in range(THREADS)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"unexpected errors: {errors}"


# ── Fix 7: ReplayStore ──────────────────────────────────────────────


class TestReplayStoreThreadSafety:
    """Concurrent save() + replay() — no corruption."""

    def test_concurrent_save_and_replay(self, tmp_path: Path) -> None:
        from sentinel.agentops.tool_audit.monitor import ToolCallRecord

        store = ToolReplayStore(root=tmp_path / "replay")
        errors: list[Exception] = []

        def work(tid: int) -> None:
            for i in range(OPS):
                try:
                    rec = ToolCallRecord(
                        agent=f"agent-{tid}",
                        tool="my_tool",
                        inputs={"tid": tid, "i": i},
                        output={"result": tid * 1000 + i},
                    )
                    store.save(rec)
                    store.replay("my_tool", {"tid": tid, "i": i})
                except Exception as exc:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            futures = [pool.submit(work, t) for t in range(THREADS)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"unexpected errors: {errors}"
        # Verify at least some cached entries are retrievable
        result = store.replay("my_tool", {"tid": 0, "i": 0})
        assert result is not None
