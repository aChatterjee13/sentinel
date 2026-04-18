"""Agent registry — versioning, capabilities, A2A discovery."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from sentinel.config.schema import AgentRegistryConfig
from sentinel.core.exceptions import AgentError

log = structlog.get_logger(__name__)


@dataclass
class AgentSpec:
    """A versioned agent definition."""

    name: str
    version: str
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    llm_config: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    safety_policies: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    health_status: str = "unknown"
    baselines: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["registered_at"] = self.registered_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSpec:
        ts = data.get("registered_at")
        return cls(
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            capabilities=data.get("capabilities", []),
            tools=data.get("tools", []),
            llm_config=data.get("llm_config", {}),
            budget=data.get("budget", {}),
            safety_policies=data.get("safety_policies", {}),
            dependencies=data.get("dependencies", []),
            metadata=data.get("metadata", {}),
            registered_at=datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc),
            health_status=data.get("health_status", "unknown"),
            baselines=data.get("baselines", {}),
        )


class AgentRegistry:
    """Local agent registry with capability-based discovery (A2A)."""

    def __init__(self, config: AgentRegistryConfig | None = None, root: str | Path = "./agents"):
        self.config = config or AgentRegistryConfig()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._agents: dict[str, dict[str, AgentSpec]] = {}
        self._capability_index: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        self._load_from_disk()
        self._rebuild_capability_index()

    # ── CRUD ──────────────────────────────────────────────────────

    def register(self, spec: AgentSpec) -> AgentSpec:
        with self._lock:
            if self.config.capability_manifest and not spec.capabilities:
                raise AgentError(f"agent {spec.name} must declare capabilities")
            self._agents.setdefault(spec.name, {})[spec.version] = spec
            for cap in spec.capabilities:
                self._capability_index[cap].add(spec.name)
            self._persist(spec)
            log.info("agent.registered", name=spec.name, version=spec.version)
            return spec

    def get(self, name: str, version: str | None = None) -> AgentSpec:
        with self._lock:
            versions = self._agents.get(name)
            if not versions:
                raise AgentError(f"agent not found: {name}")
            if version is None:
                return max(versions.values(), key=lambda s: s.registered_at)
            if version not in versions:
                raise AgentError(f"agent {name}@{version} not found")
            return versions[version]

    def list_agents(self) -> list[str]:
        with self._lock:
            return sorted(self._agents.keys())

    def list_versions(self, name: str) -> list[str]:
        with self._lock:
            return sorted(self._agents.get(name, {}).keys())

    # ── A2A discovery ─────────────────────────────────────────────

    def find_by_capability(self, capability: str) -> list[AgentSpec]:
        """Find agents with a specific capability using the reverse index.

        Args:
            capability: The capability name to search for.

        Returns:
            List of latest :class:`AgentSpec` instances that declare
            the requested capability.
        """
        with self._lock:
            agent_names = self._capability_index.get(capability, set())
            result: list[AgentSpec] = []
            for name in agent_names:
                versions = self._agents.get(name)
                if versions:
                    latest = max(versions.values(), key=lambda s: s.registered_at)
                    if capability in latest.capabilities:
                        result.append(latest)
            return result

    def update_health(self, name: str, version: str, status: str) -> None:
        with self._lock:
            spec = self.get(name, version)
            spec.health_status = status
            self._persist(spec)

    def update_baseline(self, name: str, version: str, **metrics: float) -> None:
        with self._lock:
            spec = self.get(name, version)
            spec.baselines.update(metrics)
            self._persist(spec)

    # ── Persistence ───────────────────────────────────────────────

    def _persist(self, spec: AgentSpec) -> None:
        agent_dir = self.root / spec.name
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / f"{spec.version}.json").write_text(
            json.dumps(spec.to_dict(), indent=2, default=str)
        )

    def _load_from_disk(self) -> None:
        for agent_dir in self.root.iterdir():
            if not agent_dir.is_dir():
                continue
            for version_file in agent_dir.glob("*.json"):
                try:
                    spec = AgentSpec.from_dict(json.loads(version_file.read_text()))
                    self._agents.setdefault(spec.name, {})[spec.version] = spec
                except Exception as e:
                    log.warning("agent.load_failed", file=str(version_file), error=str(e))

    def _rebuild_capability_index(self) -> None:
        """Rebuild the capability → agent-name reverse index from scratch."""
        self._capability_index.clear()
        for name, versions in self._agents.items():
            latest = max(versions.values(), key=lambda s: s.registered_at)
            for cap in latest.capabilities:
                self._capability_index[cap].add(name)
