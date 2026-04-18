"""Unit tests for sentinel.agentops.agent_registry — AgentRegistry & AgentSpec."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sentinel.agentops.agent_registry import AgentRegistry, AgentSpec
from sentinel.config.schema import AgentRegistryConfig
from sentinel.core.exceptions import AgentError


@pytest.fixture
def registry(tmp_path) -> AgentRegistry:
    return AgentRegistry(
        config=AgentRegistryConfig(capability_manifest=True),
        root=tmp_path / "agents",
    )


def _make_spec(
    name: str = "claims_processor",
    version: str = "1.0",
    capabilities: list[str] | None = None,
    tools: list[str] | None = None,
) -> AgentSpec:
    return AgentSpec(
        name=name,
        version=version,
        description="A test agent",
        capabilities=capabilities or ["claims_processing"],
        tools=tools or ["policy_search", "llm_extraction"],
    )


class TestRegister:
    """Registering agents in the registry."""

    def test_register_and_get(self, registry: AgentRegistry) -> None:
        spec = _make_spec()
        registry.register(spec)
        result = registry.get("claims_processor", "1.0")
        assert result.name == "claims_processor"
        assert result.version == "1.0"

    def test_register_without_capabilities_raises(self, registry: AgentRegistry) -> None:
        spec = AgentSpec(name="no_caps", version="1.0", capabilities=[])
        with pytest.raises(AgentError, match="must declare capabilities"):
            registry.register(spec)

    def test_register_without_capabilities_ok_when_manifest_disabled(self, tmp_path) -> None:
        reg = AgentRegistry(
            config=AgentRegistryConfig(capability_manifest=False),
            root=tmp_path / "agents",
        )
        spec = AgentSpec(name="relaxed", version="1.0", capabilities=[])
        result = reg.register(spec)
        assert result.name == "relaxed"

    def test_register_multiple_versions(self, registry: AgentRegistry) -> None:
        registry.register(_make_spec(version="1.0"))
        registry.register(_make_spec(version="2.0"))
        versions = registry.list_versions("claims_processor")
        assert "1.0" in versions
        assert "2.0" in versions


class TestGet:
    """Looking up agents by name and version."""

    def test_get_latest_version(self, registry: AgentRegistry) -> None:
        registry.register(_make_spec(version="1.0"))
        registry.register(_make_spec(version="2.0"))
        latest = registry.get("claims_processor")
        assert latest.version == "2.0"

    def test_get_specific_version(self, registry: AgentRegistry) -> None:
        registry.register(_make_spec(version="1.0"))
        registry.register(_make_spec(version="2.0"))
        v1 = registry.get("claims_processor", "1.0")
        assert v1.version == "1.0"

    def test_get_unknown_agent_raises(self, registry: AgentRegistry) -> None:
        with pytest.raises(AgentError, match="agent not found"):
            registry.get("nonexistent")

    def test_get_unknown_version_raises(self, registry: AgentRegistry) -> None:
        registry.register(_make_spec(version="1.0"))
        with pytest.raises(AgentError, match="not found"):
            registry.get("claims_processor", "99.0")


class TestListAndDiscovery:
    """list_agents, list_versions, find_by_capability."""

    def test_list_agents_empty(self, registry: AgentRegistry) -> None:
        assert registry.list_agents() == []

    def test_list_agents(self, registry: AgentRegistry) -> None:
        registry.register(_make_spec(name="a", capabilities=["cap"]))
        registry.register(_make_spec(name="b", capabilities=["cap"]))
        assert registry.list_agents() == ["a", "b"]

    def test_find_by_capability(self, registry: AgentRegistry) -> None:
        registry.register(_make_spec(name="agent_a", capabilities=["claims", "search"]))
        registry.register(_make_spec(name="agent_b", capabilities=["search"]))
        registry.register(_make_spec(name="agent_c", capabilities=["payments"]))

        found = registry.find_by_capability("search")
        names = {s.name for s in found}
        assert names == {"agent_a", "agent_b"}

    def test_find_by_capability_no_match(self, registry: AgentRegistry) -> None:
        registry.register(_make_spec(capabilities=["claims"]))
        assert registry.find_by_capability("nonexistent") == []


class TestUpdateHealthAndBaseline:
    """Mutating agent status and baselines."""

    def test_update_health(self, registry: AgentRegistry) -> None:
        registry.register(_make_spec())
        registry.update_health("claims_processor", "1.0", "healthy")
        spec = registry.get("claims_processor", "1.0")
        assert spec.health_status == "healthy"

    def test_update_baseline(self, registry: AgentRegistry) -> None:
        registry.register(_make_spec())
        registry.update_baseline("claims_processor", "1.0", accuracy=0.95, latency_ms=120.0)
        spec = registry.get("claims_processor", "1.0")
        assert spec.baselines["accuracy"] == pytest.approx(0.95)
        assert spec.baselines["latency_ms"] == pytest.approx(120.0)


class TestPersistence:
    """Disk persistence and reload."""

    def test_persisted_to_disk(self, tmp_path) -> None:
        root = tmp_path / "agents"
        reg = AgentRegistry(config=AgentRegistryConfig(capability_manifest=True), root=root)
        reg.register(_make_spec(version="1.0"))

        json_file = root / "claims_processor" / "1.0.json"
        assert json_file.exists()

    def test_reload_from_disk(self, tmp_path) -> None:
        root = tmp_path / "agents"
        reg1 = AgentRegistry(config=AgentRegistryConfig(capability_manifest=True), root=root)
        reg1.register(_make_spec(version="1.0"))

        # New registry instance loading from the same directory
        reg2 = AgentRegistry(config=AgentRegistryConfig(capability_manifest=True), root=root)
        spec = reg2.get("claims_processor", "1.0")
        assert spec.name == "claims_processor"


class TestAgentSpec:
    """AgentSpec serialisation round-trip."""

    def test_to_dict_and_from_dict(self) -> None:
        spec = _make_spec()
        d = spec.to_dict()
        restored = AgentSpec.from_dict(d)
        assert restored.name == spec.name
        assert restored.version == spec.version
        assert restored.capabilities == spec.capabilities

    def test_from_dict_with_missing_optional_fields(self) -> None:
        minimal = {"name": "minimal", "version": "0.1"}
        spec = AgentSpec.from_dict(minimal)
        assert spec.name == "minimal"
        assert spec.capabilities == []
        assert spec.health_status == "unknown"

    def test_from_dict_with_iso_timestamp(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat()
        spec = AgentSpec.from_dict({"name": "a", "version": "1", "registered_at": ts})
        assert spec.registered_at.year == 2025
