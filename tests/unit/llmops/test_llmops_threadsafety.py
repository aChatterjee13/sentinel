"""Thread-safety and correctness tests for LLMOps subsystem (fixes 5-8)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from sentinel.config.schema import PromptDriftConfig, SemanticDriftConfig
from sentinel.core.exceptions import LLMOpsError
from sentinel.core.types import AlertSeverity
from sentinel.llmops.guardrails.base import BaseGuardrail, GuardrailResult
from sentinel.llmops.guardrails.engine import GuardrailPipeline
from sentinel.llmops.prompt_drift import PromptDriftDetector
from sentinel.llmops.prompt_manager import PromptManager
from sentinel.llmops.quality.semantic_drift import SemanticDriftMonitor

# ── Helpers ──────────────────────────────────────────────────────────

def _dummy_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic 4-d embeddings based on text length."""
    return [[len(t) * 0.1, len(t) * 0.2, 0.5, 0.5] for t in texts]


class _PassGuardrail(BaseGuardrail):
    """Guardrail that always passes."""

    name = "pass_guard"

    def __init__(self, **kwargs):
        super().__init__(action="warn", **kwargs)

    def check(self, content: str, context: dict | None = None) -> GuardrailResult:
        return GuardrailResult(
            name=self.name, passed=True, blocked=False, severity=AlertSeverity.INFO,
        )


# ── Fix 5: SemanticDriftMonitor thread safety ────────────────────────


class TestSemanticDriftThreadSafety:
    """Concurrent observe()/detect() must not corrupt the rolling window."""

    def test_concurrent_observe_no_corruption(self) -> None:
        monitor = SemanticDriftMonitor(
            config=SemanticDriftConfig(window_size=2000),
            embed_fn=_dummy_embed,
        )
        monitor.fit(["reference text number one", "reference text number two"])

        errors: list[Exception] = []

        def _observe(i: int) -> None:
            try:
                monitor.observe(f"text {i}")
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_observe, i) for i in range(1000)]
            for f in as_completed(futures):
                f.result()

        assert not errors
        assert len(monitor._window) == 1000

    def test_concurrent_observe_and_detect(self) -> None:
        monitor = SemanticDriftMonitor(
            embed_fn=_dummy_embed, window_size=500,
        )
        monitor.fit(["baseline a", "baseline b"])

        errors: list[Exception] = []

        def _work(i: int) -> None:
            try:
                if i % 3 == 0:
                    monitor.detect()
                else:
                    monitor.observe(f"response {i}")
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_work, i) for i in range(500)]
            for f in as_completed(futures):
                f.result()

        assert not errors

    def test_window_size_from_config(self) -> None:
        cfg = SemanticDriftConfig(window_size=42)
        monitor = SemanticDriftMonitor(config=cfg, embed_fn=_dummy_embed)
        assert monitor._window.maxlen == 42

    def test_window_size_falls_back_to_constructor_default(self) -> None:
        """When config has default window_size, constructor arg is ignored."""
        monitor = SemanticDriftMonitor(embed_fn=_dummy_embed, window_size=123)
        # Config default (500) takes precedence over constructor arg
        assert monitor._window.maxlen == 500

    def test_lock_exists(self) -> None:
        monitor = SemanticDriftMonitor(embed_fn=_dummy_embed)
        assert isinstance(monitor._lock, type(threading.Lock()))


# ── Fix 6: PromptDriftDetector thread safety ─────────────────────────


class TestPromptDriftThreadSafety:
    """Concurrent observe() must not corrupt _stats."""

    def test_concurrent_observe_no_corruption(self) -> None:
        detector = PromptDriftDetector()
        errors: list[Exception] = []

        def _observe(i: int) -> None:
            try:
                detector.observe(
                    prompt_name="p",
                    prompt_version="1",
                    quality_score=0.8 - i * 0.001,
                    guardrail_violations=0,
                    total_tokens=100 + i,
                )
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_observe, i) for i in range(1000)]
            for f in as_completed(futures):
                f.result()

        assert not errors
        stats = detector._stats["p@1"]
        # deque maxlen is 200, so at most 200 entries
        assert len(stats.quality_scores) == min(1000, stats.quality_scores.maxlen)

    def test_concurrent_observe_and_detect(self) -> None:
        detector = PromptDriftDetector()
        # Seed enough data to cross min_samples
        for _ in range(30):
            detector.observe("p", "1", quality_score=0.9, total_tokens=100)

        errors: list[Exception] = []

        def _work(i: int) -> None:
            try:
                if i % 5 == 0:
                    detector.detect("p", "1")
                else:
                    detector.observe("p", "1", quality_score=0.7, total_tokens=200)
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_work, i) for i in range(500)]
            for f in as_completed(futures):
                f.result()

        assert not errors

    def test_min_samples_from_config(self) -> None:
        cfg = PromptDriftConfig(min_samples=50)
        detector = PromptDriftDetector(config=cfg)
        # Only 30 samples — should be insufficient
        for _ in range(30):
            detector.observe("p", "1", quality_score=0.9)
        report = detector.detect("p", "1")
        assert not report.is_drifted
        assert report.metadata.get("reason") == "insufficient_data"

    def test_lock_exists(self) -> None:
        detector = PromptDriftDetector()
        assert isinstance(detector._lock, type(threading.Lock()))


# ── Fix 7: GuardrailPipeline thread safety ───────────────────────────


class TestGuardrailPipelineThreadSafety:
    """Concurrent check_input() calls must complete without errors."""

    def test_concurrent_check_input(self) -> None:
        pipeline = GuardrailPipeline(
            input_guardrails=[_PassGuardrail()],
            output_guardrails=[_PassGuardrail()],
        )
        errors: list[Exception] = []

        def _check(i: int) -> None:
            try:
                result = pipeline.check_input(f"message {i}")
                assert not result.blocked
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_check, i) for i in range(100)]
            for f in as_completed(futures):
                f.result()

        assert not errors

    def test_concurrent_check_input_and_output(self) -> None:
        pipeline = GuardrailPipeline(
            input_guardrails=[_PassGuardrail()],
            output_guardrails=[_PassGuardrail()],
        )
        errors: list[Exception] = []

        def _work(i: int) -> None:
            try:
                if i % 2 == 0:
                    pipeline.check_input(f"input {i}")
                else:
                    pipeline.check_output(f"output {i}")
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_work, i) for i in range(200)]
            for f in as_completed(futures):
                f.result()

        assert not errors

    def test_lock_is_rlock(self) -> None:
        pipeline = GuardrailPipeline()
        assert isinstance(pipeline._lock, type(threading.RLock()))


# ── Fix 8: PromptManager idempotent re-registration ─────────────────


class TestPromptManagerReRegister:
    """Re-registering same content with different mutable fields updates them."""

    @pytest.fixture()
    def manager(self, tmp_path: object) -> PromptManager:
        from sentinel.config.schema import LLMOpsConfig

        return PromptManager(config=LLMOpsConfig(), root=str(tmp_path))

    def test_reregister_same_content_updates_traffic(self, manager: PromptManager) -> None:
        v1 = manager.register(
            name="test_prompt",
            version="1.0",
            system_prompt="sys",
            template="hello {{name}}",
            traffic_pct=10,
        )
        assert v1.traffic_pct == 10

        v2 = manager.register(
            name="test_prompt",
            version="1.0",
            system_prompt="sys",
            template="hello {{name}}",
            traffic_pct=50,
        )
        assert v2.traffic_pct == 50
        assert v2 is v1  # same object, mutated in place

    def test_reregister_same_content_updates_metadata(self, manager: PromptManager) -> None:
        v1 = manager.register(
            name="test_prompt",
            version="1.0",
            system_prompt="sys",
            template="hello {{name}}",
            metadata={"author": "alice"},
        )
        assert v1.metadata == {"author": "alice"}

        v2 = manager.register(
            name="test_prompt",
            version="1.0",
            system_prompt="sys",
            template="hello {{name}}",
            metadata={"author": "bob"},
        )
        assert v2.metadata == {"author": "bob"}

    def test_reregister_different_content_raises(self, manager: PromptManager) -> None:
        manager.register(
            name="test_prompt",
            version="1.0",
            system_prompt="sys v1",
            template="hello {{name}}",
        )
        with pytest.raises(LLMOpsError, match="already exists with different content"):
            manager.register(
                name="test_prompt",
                version="1.0",
                system_prompt="sys v2",
                template="hello {{name}}",
            )

    def test_reregister_identical_is_noop(self, manager: PromptManager) -> None:
        v1 = manager.register(
            name="test_prompt",
            version="1.0",
            system_prompt="sys",
            template="hello {{name}}",
            traffic_pct=10,
            metadata={"a": 1},
        )
        v2 = manager.register(
            name="test_prompt",
            version="1.0",
            system_prompt="sys",
            template="hello {{name}}",
            traffic_pct=10,
            metadata={"a": 1},
        )
        assert v2 is v1
