"""Golden dataset management and CI/CD friendly suite runner."""

from __future__ import annotations

import concurrent.futures
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from sentinel.agentops.eval.trajectory import TrajectoryEvaluator, TrajectoryScore
from sentinel.config.schema import AgentEvaluationConfig
from sentinel.core.exceptions import BudgetExceededError

log = structlog.get_logger(__name__)

AgentRunner = Callable[[dict[str, Any]], Any]
AsyncAgentRunner = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class GoldenExample:
    """A single input/expected-output pair with optional optimal trajectory."""

    example_id: str
    input: dict[str, Any]
    expected_output: Any
    expected_steps: list[str] = field(default_factory=list)
    success_criteria: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenExample:
        return cls(
            example_id=str(data.get("id", data.get("example_id", ""))),
            input=data.get("input", {}),
            expected_output=data.get("expected_output"),
            expected_steps=list(data.get("expected_steps", [])),
            success_criteria=data.get("success_criteria", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GoldenDataset:
    """A versioned set of :class:`GoldenExample` instances."""

    name: str
    version: str
    examples: list[GoldenExample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    @classmethod
    def from_file(cls, path: str | Path) -> GoldenDataset:
        p = Path(path)
        data = json.loads(p.read_text())
        return cls(
            name=data.get("name", p.stem),
            version=data.get("version", "1.0"),
            examples=[GoldenExample.from_dict(e) for e in data.get("examples", [])],
        )

    @classmethod
    def from_directory(cls, directory: str | Path) -> list[GoldenDataset]:
        return [cls.from_file(f) for f in sorted(Path(directory).glob("*.json"))]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "examples": [
                {
                    "example_id": e.example_id,
                    "input": e.input,
                    "expected_output": e.expected_output,
                    "expected_steps": e.expected_steps,
                    "success_criteria": e.success_criteria,
                    "metadata": e.metadata,
                }
                for e in self.examples
            ],
        }


@dataclass
class GoldenSuiteResult:
    """Aggregate result of running a golden dataset against an agent."""

    dataset: str
    version: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    examples: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class GoldenSuiteRunner:
    """Run golden datasets against an agent and report pass/fail metrics.

    The runner accepts a synchronous callable that takes the example
    input dict and returns ``{"output": ..., "steps": [...]}``. The
    expected output is matched via deep equality (or a custom matcher in
    ``success_criteria.matcher``) and trajectories are compared via the
    :class:`TrajectoryEvaluator`.
    """

    def __init__(
        self,
        config: AgentEvaluationConfig | None = None,
        trajectory: TrajectoryEvaluator | None = None,
    ):
        self.config = config or AgentEvaluationConfig()
        self.trajectory = trajectory or TrajectoryEvaluator(self.config)

    def run(
        self,
        dataset: GoldenDataset,
        runner: AgentRunner,
        *,
        matcher: Callable[[Any, Any], bool] | None = None,
        timeout_seconds: float | None = None,
    ) -> GoldenSuiteResult:
        """Run a golden dataset against an agent runner.

        Args:
            dataset: The golden dataset to evaluate.
            runner: Synchronous callable taking the example input dict.
            matcher: Optional custom matcher for output comparison.
            timeout_seconds: Per-example timeout in seconds. ``None``
                disables the timeout.

        Returns:
            Aggregate suite result with per-example details.
        """
        match_fn = matcher or _default_matcher
        records: list[dict[str, Any]] = []
        passed = 0
        for example in dataset.examples:
            try:
                if timeout_seconds is not None:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(runner, example.input)
                        output = future.result(timeout=timeout_seconds)
                else:
                    output = runner(example.input)
            except (TimeoutError, concurrent.futures.TimeoutError):
                log.warning(
                    "golden.example_timeout",
                    example=example.example_id,
                    timeout=timeout_seconds,
                )
                records.append(
                    {
                        "example_id": example.example_id,
                        "passed": False,
                        "failure_reason": "timeout",
                        "error": f"Timed out after {timeout_seconds}s",
                        "trajectory": None,
                    }
                )
                continue
            except BudgetExceededError as exc:
                log.warning(
                    "golden.example_budget_exceeded",
                    example=example.example_id,
                    error=str(exc),
                )
                records.append(
                    {
                        "example_id": example.example_id,
                        "passed": False,
                        "failure_reason": f"budget_exceeded: {exc}",
                        "error": str(exc),
                        "trajectory": None,
                    }
                )
                continue
            except Exception as exc:
                log.warning("golden.example_error", example=example.example_id, error=str(exc))
                records.append(
                    {
                        "example_id": example.example_id,
                        "passed": False,
                        "failure_reason": f"error: {type(exc).__name__}: {exc}",
                        "error": str(exc),
                        "trajectory": None,
                    }
                )
                continue

            actual_output, actual_steps = _unwrap_output(output)
            output_match = match_fn(actual_output, example.expected_output)

            traj_score: TrajectoryScore | None = None
            if example.expected_steps:
                traj_score = self.trajectory.score(actual_steps, example.expected_steps)

            example_passed = bool(output_match) and (traj_score is None or traj_score.passed)
            if example_passed:
                passed += 1
            records.append(
                {
                    "example_id": example.example_id,
                    "passed": example_passed,
                    "failure_reason": None,
                    "output_match": output_match,
                    "trajectory": (
                        {
                            "score": traj_score.score,
                            "extra_steps": traj_score.extra_steps,
                            "missing_steps": traj_score.missing_steps,
                        }
                        if traj_score
                        else None
                    ),
                }
            )

        total = len(dataset.examples)
        return GoldenSuiteResult(
            dataset=dataset.name,
            version=dataset.version,
            total=total,
            passed=passed,
            failed=total - passed,
            pass_rate=passed / total if total else 1.0,
            examples=records,
        )

    async def run_async(
        self,
        dataset: GoldenDataset,
        runner: AsyncAgentRunner,
        *,
        matcher: Callable[[Any, Any], bool] | None = None,
    ) -> GoldenSuiteResult:
        match_fn = matcher or _default_matcher
        records: list[dict[str, Any]] = []
        passed = 0
        for example in dataset.examples:
            try:
                output = await runner(example.input)
            except Exception as exc:
                records.append(
                    {
                        "example_id": example.example_id,
                        "passed": False,
                        "error": str(exc),
                        "trajectory": None,
                    }
                )
                continue
            actual_output, actual_steps = _unwrap_output(output)
            output_match = match_fn(actual_output, example.expected_output)
            traj_score: TrajectoryScore | None = None
            if example.expected_steps:
                traj_score = self.trajectory.score(actual_steps, example.expected_steps)
            example_passed = bool(output_match) and (traj_score is None or traj_score.passed)
            if example_passed:
                passed += 1
            records.append(
                {
                    "example_id": example.example_id,
                    "passed": example_passed,
                    "output_match": output_match,
                    "trajectory": (
                        {
                            "score": traj_score.score,
                            "extra_steps": traj_score.extra_steps,
                            "missing_steps": traj_score.missing_steps,
                        }
                        if traj_score
                        else None
                    ),
                }
            )
        total = len(dataset.examples)
        return GoldenSuiteResult(
            dataset=dataset.name,
            version=dataset.version,
            total=total,
            passed=passed,
            failed=total - passed,
            pass_rate=passed / total if total else 1.0,
            examples=records,
        )


def _unwrap_output(value: Any) -> tuple[Any, Sequence[str]]:
    if isinstance(value, dict) and ("output" in value or "steps" in value):
        return value.get("output"), value.get("steps", [])
    return value, []


def _default_matcher(actual: Any, expected: Any) -> bool:
    if expected is None:
        return True
    if isinstance(expected, str) and isinstance(actual, str):
        return actual.strip().lower() == expected.strip().lower()
    return actual == expected
