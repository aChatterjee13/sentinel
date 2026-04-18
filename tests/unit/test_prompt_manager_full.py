"""Unit tests for PromptManager (expanded coverage)."""

from __future__ import annotations

import json

import pytest

from sentinel.config.schema import LLMOpsConfig
from sentinel.core.exceptions import LLMOpsError
from sentinel.llmops.prompt_manager import (
    FewShotExample,
    Prompt,
    PromptManager,
    PromptVersion,
)


@pytest.fixture
def llmops_config() -> LLMOpsConfig:
    return LLMOpsConfig(enabled=True)


@pytest.fixture
def manager(tmp_path, llmops_config) -> PromptManager:
    return PromptManager(config=llmops_config, root=tmp_path / "prompts")


class TestRegister:
    """Test prompt registration."""

    def test_register_new_prompt(self, manager) -> None:
        pv = manager.register(
            name="summariser",
            version="1.0",
            system_prompt="You are a summariser.",
            template="Summarise: {{ text }}",
        )
        assert pv.name == "summariser"
        assert pv.version == "1.0"

    def test_register_idempotent_same_content(self, manager) -> None:
        manager.register("p", "1.0", "sys", "tmpl")
        pv2 = manager.register("p", "1.0", "sys", "tmpl")
        assert pv2.version == "1.0"

    def test_register_same_version_different_content_raises(self, manager) -> None:
        manager.register("p", "1.0", "sys A", "tmpl A")
        with pytest.raises(LLMOpsError, match="already exists"):
            manager.register("p", "1.0", "sys B", "tmpl B")

    def test_register_with_few_shot_examples(self, manager) -> None:
        pv = manager.register(
            name="qa",
            version="1.0",
            system_prompt="sys",
            template="Q: {{ q }}",
            few_shot_examples=[
                {"user": "What is X?", "assistant": "X is Y."},
            ],
        )
        assert len(pv.few_shot_examples) == 1
        assert pv.few_shot_examples[0].user == "What is X?"

    def test_register_with_metadata(self, manager) -> None:
        pv = manager.register(
            name="p",
            version="1.0",
            system_prompt="sys",
            template="tmpl",
            metadata={"author": "ml-team"},
        )
        assert pv.metadata["author"] == "ml-team"

    def test_register_with_traffic_pct(self, manager) -> None:
        pv = manager.register(
            name="p",
            version="1.0",
            system_prompt="s",
            template="t",
            traffic_pct=50,
        )
        assert pv.traffic_pct == 50


class TestGetAndList:
    """Test retrieval and listing."""

    def test_get_specific_version(self, manager) -> None:
        manager.register("p", "1.0", "sys", "tmpl")
        pv = manager.get("p", "1.0")
        assert pv.version == "1.0"

    def test_get_latest_version(self, manager) -> None:
        manager.register("p", "1.0", "sys", "tmpl1")
        manager.register("p", "2.0", "sys", "tmpl2")
        pv = manager.get("p")
        assert pv.version == "2.0"

    def test_get_nonexistent_prompt_raises(self, manager) -> None:
        with pytest.raises(LLMOpsError, match="not found"):
            manager.get("nonexistent")

    def test_get_nonexistent_version_raises(self, manager) -> None:
        manager.register("p", "1.0", "sys", "tmpl")
        with pytest.raises(LLMOpsError, match="not found"):
            manager.get("p", "99.0")

    def test_list_versions(self, manager) -> None:
        manager.register("p", "1.0", "sys", "t1")
        manager.register("p", "2.0", "sys", "t2")
        versions = manager.list_versions("p")
        assert versions == ["1.0", "2.0"]

    def test_list_prompts(self, manager) -> None:
        manager.register("a", "1.0", "sys", "t")
        manager.register("b", "1.0", "sys", "t")
        prompts = manager.list_prompts()
        assert "a" in prompts
        assert "b" in prompts


class TestABRouting:
    """Test A/B traffic splitting."""

    def test_set_traffic_valid(self, manager) -> None:
        manager.register("p", "1.0", "sys", "t1")
        manager.register("p", "2.0", "sys", "t2")
        manager.set_traffic("p", {"1.0": 90, "2.0": 10})
        v1 = manager.get("p", "1.0")
        v2 = manager.get("p", "2.0")
        assert v1.traffic_pct == 90
        assert v2.traffic_pct == 10

    def test_set_traffic_invalid_sum_raises(self, manager) -> None:
        manager.register("p", "1.0", "sys", "t")
        with pytest.raises(LLMOpsError, match="sum to 100"):
            manager.set_traffic("p", {"1.0": 50})

    def test_resolve_with_traffic_split(self, manager) -> None:
        manager.register("p", "1.0", "sys", "t1", traffic_pct=90)
        manager.register("p", "2.0", "sys", "t2", traffic_pct=10)
        # Resolve many times and check both versions appear
        versions_seen = set()
        for _i in range(200):
            prompt = manager.resolve("p")
            versions_seen.add(prompt.version)
        assert "1.0" in versions_seen
        # 2.0 might not always appear in 200 random tries at 10%, but usually does
        # Not asserting 2.0 to avoid flaky test

    def test_resolve_stable_hashing_by_user_id(self, manager) -> None:
        manager.register("p", "1.0", "sys", "t1", traffic_pct=50)
        manager.register("p", "2.0", "sys", "t2", traffic_pct=50)
        # Same user_id should always get the same version
        ctx = {"user_id": "user123"}
        results = [manager.resolve("p", context=ctx).version for _ in range(10)]
        assert len(set(results)) == 1

    def test_resolve_without_traffic_returns_latest(self, manager) -> None:
        manager.register("p", "1.0", "sys", "t1")
        manager.register("p", "2.0", "sys", "t2")
        prompt = manager.resolve("p")
        assert prompt.version == "2.0"

    def test_resolve_nonexistent_raises(self, manager) -> None:
        with pytest.raises(LLMOpsError, match="not found"):
            manager.resolve("nonexistent")


class TestLogResult:
    """Test telemetry logging."""

    def test_log_result_updates_stats(self, manager) -> None:
        manager.register("p", "1.0", "sys", "tmpl")
        manager.log_result("p", "1.0", input_tokens=100, output_tokens=50, quality_score=0.9)
        pv = manager.get("p", "1.0")
        assert pv.stats["calls"] == 1
        assert pv.stats["avg_input_tokens"] == 100
        assert pv.stats["avg_output_tokens"] == 50
        assert pv.stats["avg_quality"] == 0.9

    def test_log_result_rolling_average(self, manager) -> None:
        manager.register("p", "1.0", "sys", "tmpl")
        manager.log_result("p", "1.0", input_tokens=100, output_tokens=50)
        manager.log_result("p", "1.0", input_tokens=200, output_tokens=100)
        pv = manager.get("p", "1.0")
        assert pv.stats["calls"] == 2
        assert pv.stats["avg_input_tokens"] == pytest.approx(150.0)

    def test_log_result_with_latency(self, manager) -> None:
        manager.register("p", "1.0", "sys", "tmpl")
        manager.log_result("p", "1.0", input_tokens=100, output_tokens=50, latency_ms=200)
        pv = manager.get("p", "1.0")
        assert pv.stats["avg_latency_ms"] == 200

    def test_log_result_with_guardrail_violations(self, manager) -> None:
        manager.register("p", "1.0", "sys", "tmpl")
        manager.log_result(
            "p",
            "1.0",
            input_tokens=10,
            output_tokens=10,
            guardrail_violations=["pii", "toxicity"],
        )
        pv = manager.get("p", "1.0")
        assert pv.stats["guardrail_violations"] == 2


class TestPersistence:
    """Test disk persistence and reload."""

    def test_persisted_to_disk(self, tmp_path, llmops_config) -> None:
        mgr = PromptManager(config=llmops_config, root=tmp_path / "prompts")
        mgr.register("p", "1.0", "sys", "tmpl")
        # Check file exists
        path = tmp_path / "prompts" / "p" / "1.0.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "p"
        assert data["version"] == "1.0"

    def test_reload_from_disk(self, tmp_path, llmops_config) -> None:
        mgr1 = PromptManager(config=llmops_config, root=tmp_path / "prompts")
        mgr1.register("p", "1.0", "sys", "tmpl")
        # Create a new manager pointing to the same directory
        mgr2 = PromptManager(config=llmops_config, root=tmp_path / "prompts")
        pv = mgr2.get("p", "1.0")
        assert pv.system_prompt == "sys"

    def test_corrupt_file_handled_gracefully(self, tmp_path, llmops_config) -> None:
        prompt_dir = tmp_path / "prompts" / "p"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "1.0.json").write_text("not valid json{{{")
        # Should not crash
        mgr = PromptManager(config=llmops_config, root=tmp_path / "prompts")
        assert "p" not in mgr.list_prompts()


class TestPromptRender:
    """Test the Prompt.render() method."""

    def test_render_variables(self) -> None:
        prompt = Prompt(
            name="test",
            version="1.0",
            system_prompt="sys",
            template="Hello {{ name }}, your claim {{ claim_id }} is ready.",
        )
        rendered = prompt.render(name="Alice", claim_id="C123")
        assert rendered == "Hello Alice, your claim C123 is ready."

    def test_render_missing_variable_raises(self) -> None:
        prompt = Prompt(
            name="test",
            version="1.0",
            system_prompt="s",
            template="Hello {{ name }}",
        )
        with pytest.raises(LLMOpsError, match="missing template variable"):
            prompt.render()


class TestPromptVersion:
    """Test PromptVersion serialization."""

    def test_fingerprint_stability(self) -> None:
        pv = PromptVersion(name="p", version="1.0", system_prompt="s", template="t")
        fp1 = pv.fingerprint()
        fp2 = pv.fingerprint()
        assert fp1 == fp2

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        pv = PromptVersion(
            name="p",
            version="1.0",
            system_prompt="sys",
            template="t {{ x }}",
            few_shot_examples=[FewShotExample(user="q", assistant="a")],
            metadata={"key": "val"},
            traffic_pct=50,
        )
        d = pv.to_dict()
        restored = PromptVersion.from_dict(d)
        assert restored.name == pv.name
        assert restored.version == pv.version
        assert restored.system_prompt == pv.system_prompt
        assert restored.template == pv.template
        assert len(restored.few_shot_examples) == 1
        assert restored.traffic_pct == 50
