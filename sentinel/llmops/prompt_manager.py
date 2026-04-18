"""Prompt registry, versioning, and A/B routing.

Prompts are the source code of LLM applications. This module gives them
the same governance traditional models get: a registry, versioned
history, A/B testing, and per-version performance tracking.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from sentinel.config.schema import LLMOpsConfig
from sentinel.core.exceptions import LLMOpsError

log = structlog.get_logger(__name__)


_TEMPLATE_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


@dataclass
class FewShotExample:
    """A single in-context example for a prompt."""

    user: str
    assistant: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prompt:
    """A resolved prompt — system message, rendered user template, and few-shots.

    Returned by :meth:`PromptManager.resolve`. Contains the version that
    was selected so the caller can log the same version against the
    response for performance attribution.
    """

    name: str
    version: str
    system_prompt: str
    template: str
    few_shot_examples: list[FewShotExample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, **variables: Any) -> str:
        """Render the user template with `{{ variable }}` substitution."""

        def _sub(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in variables:
                raise LLMOpsError(f"missing template variable: {key}")
            return str(variables[key])

        return _TEMPLATE_VAR.sub(_sub, self.template)


@dataclass
class PromptVersion:
    """A single version of a registered prompt."""

    name: str
    version: str
    system_prompt: str
    template: str
    few_shot_examples: list[FewShotExample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    traffic_pct: int = 0  # 0-100, used for A/B routing
    stats: dict[str, float] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Stable hash of the prompt content (system + template + few-shots)."""
        body = {
            "system": self.system_prompt,
            "template": self.template,
            "few_shots": [(e.user, e.assistant) for e in self.few_shot_examples],
        }
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "system_prompt": self.system_prompt,
            "template": self.template,
            "few_shot_examples": [
                {"user": e.user, "assistant": e.assistant, "metadata": e.metadata}
                for e in self.few_shot_examples
            ],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "traffic_pct": self.traffic_pct,
            "stats": self.stats,
            "fingerprint": self.fingerprint(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptVersion:
        examples = [
            FewShotExample(user=e["user"], assistant=e["assistant"], metadata=e.get("metadata", {}))
            for e in data.get("few_shot_examples", [])
        ]
        created = data.get("created_at")
        return cls(
            name=data["name"],
            version=data["version"],
            system_prompt=data.get("system_prompt", ""),
            template=data.get("template", ""),
            few_shot_examples=examples,
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(created) if created else datetime.now(timezone.utc),
            traffic_pct=data.get("traffic_pct", 0),
            stats=data.get("stats", {}),
        )


class PromptManager:
    """Versioned prompt registry with A/B traffic splitting.

    Backends are filesystem-based by default. ``register_backend()`` can
    plug in cloud-backed storage (Azure Blob, S3) using the same
    interface as the model registry backends.
    """

    def __init__(self, config: LLMOpsConfig, root: str | Path = "./prompts"):
        self.config = config
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._versions: dict[str, dict[str, PromptVersion]] = defaultdict(dict)
        self._lock = threading.RLock()
        self._reload_from_disk()

    @classmethod
    def from_config(cls, config: LLMOpsConfig | str | Path) -> PromptManager:
        """Build from an :class:`LLMOpsConfig` or a YAML file path."""
        if isinstance(config, (str, Path)):
            from sentinel.config.loader import load_config

            cfg = load_config(config)
            return cls(cfg.llmops)
        return cls(config)

    # ── CRUD ──────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        version: str,
        system_prompt: str,
        template: str,
        few_shot_examples: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        traffic_pct: int = 0,
    ) -> PromptVersion:
        """Register a new prompt version. Idempotent on identical content."""
        with self._lock:
            if version in self._versions[name]:
                existing = self._versions[name][version]
                new_fp = PromptVersion(
                    name=name,
                    version=version,
                    system_prompt=system_prompt,
                    template=template,
                    few_shot_examples=[FewShotExample(**fs) for fs in (few_shot_examples or [])],
                ).fingerprint()
                if existing.fingerprint() != new_fp:
                    raise LLMOpsError(
                        f"prompt {name}@{version} already exists with different content"
                    )
                # Update mutable fields if they differ
                changed = False
                if traffic_pct is not None and existing.traffic_pct != traffic_pct:
                    existing.traffic_pct = traffic_pct
                    changed = True
                if metadata and existing.metadata != metadata:
                    existing.metadata = metadata
                    changed = True
                if changed:
                    self._persist(existing)
                return existing
            pv = PromptVersion(
                name=name,
                version=version,
                system_prompt=system_prompt,
                template=template,
                few_shot_examples=[FewShotExample(**fs) for fs in (few_shot_examples or [])],
                metadata=metadata or {},
                traffic_pct=traffic_pct,
            )
            self._versions[name][version] = pv
            self._persist(pv)
            log.info("prompt.registered", name=name, version=version, fp=pv.fingerprint())
            return pv

    def get(self, name: str, version: str | None = None) -> PromptVersion:
        """Look up a specific version, or the latest if `version` is None."""
        with self._lock:
            versions = self._versions.get(name)
            if not versions:
                raise LLMOpsError(f"prompt not found: {name}")
            if version is None:
                return max(versions.values(), key=lambda v: v.created_at)
            if version not in versions:
                raise LLMOpsError(f"prompt {name}@{version} not found")
            return versions[version]

    def list_versions(self, name: str) -> list[str]:
        with self._lock:
            return sorted(self._versions.get(name, {}).keys())

    def list_prompts(self) -> list[str]:
        with self._lock:
            return sorted(self._versions.keys())

    # ── A/B routing ───────────────────────────────────────────────

    def set_traffic(self, name: str, allocations: dict[str, int]) -> None:
        """Assign traffic % across versions. Must sum to 100."""
        with self._lock:
            if sum(allocations.values()) != 100:
                raise LLMOpsError(
                    f"traffic allocations must sum to 100, got {sum(allocations.values())}"
                )
            for version, pct in allocations.items():
                pv = self.get(name, version)
                pv.traffic_pct = pct
                self._persist(pv)

    def resolve(self, name: str, context: dict[str, Any] | None = None) -> Prompt:
        """Pick a prompt version based on A/B traffic split.

        If only one version exists or no traffic split is configured, the
        latest version is returned.
        """
        with self._lock:
            versions = list(self._versions.get(name, {}).values())
            if not versions:
                raise LLMOpsError(f"prompt not found: {name}")

            with_traffic = [v for v in versions if v.traffic_pct > 0]
            if not with_traffic:
                chosen = max(versions, key=lambda v: v.created_at)
            else:
                # Stable hashing if context contains a user_id, otherwise random.
                user_id = (context or {}).get("user_id")
                if user_id:
                    bucket = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 100
                else:
                    bucket = random.randint(0, 99)
                cumulative = 0
                chosen = with_traffic[0]
                for v in with_traffic:
                    cumulative += v.traffic_pct
                    if bucket < cumulative:
                        chosen = v
                        break

            return Prompt(
                name=chosen.name,
                version=chosen.version,
                system_prompt=chosen.system_prompt,
                template=chosen.template,
                few_shot_examples=list(chosen.few_shot_examples),
                metadata=dict(chosen.metadata),
            )

    # ── Telemetry ─────────────────────────────────────────────────

    def log_result(
        self,
        prompt_name: str,
        version: str,
        input_tokens: int,
        output_tokens: int,
        quality_score: float | None = None,
        guardrail_violations: list[str] | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Update rolling stats on a prompt version."""
        with self._lock:
            pv = self.get(prompt_name, version)
            stats = pv.stats
            n = stats.get("calls", 0) + 1
            stats["calls"] = n
            stats["avg_input_tokens"] = self._rolling_avg(
                stats.get("avg_input_tokens", 0.0), input_tokens, n
            )
            stats["avg_output_tokens"] = self._rolling_avg(
                stats.get("avg_output_tokens", 0.0), output_tokens, n
            )
            if quality_score is not None:
                stats["avg_quality"] = self._rolling_avg(
                    stats.get("avg_quality", 0.0), quality_score, n
                )
            if latency_ms is not None:
                stats["avg_latency_ms"] = self._rolling_avg(
                    stats.get("avg_latency_ms", 0.0), latency_ms, n
                )
            if guardrail_violations:
                stats["guardrail_violations"] = stats.get("guardrail_violations", 0) + len(
                    guardrail_violations
                )
            pv.stats = stats
            self._persist(pv)

    def get_stats(self, name: str, version: str) -> dict[str, float]:
        """Retrieve rolling stats for a prompt version.

        Args:
            name: Prompt name.
            version: Prompt version string.

        Returns:
            Dict with keys: calls, avg_input_tokens, avg_output_tokens,
            avg_quality, avg_latency_ms, guardrail_violations.

        Raises:
            LLMOpsError: If prompt/version not found.
        """
        with self._lock:
            pv = self.get(name, version)
            return dict(pv.stats)

    def get_all_stats(self) -> dict[str, dict[str, dict[str, float]]]:
        """Retrieve stats for all prompt versions.

        Returns:
            Nested dict: {prompt_name: {version: {stat_key: value}}}.
        """
        with self._lock:
            result: dict[str, dict[str, dict[str, float]]] = {}
            for name, versions in self._versions.items():
                result[name] = {}
                for ver, pv in versions.items():
                    result[name][ver] = dict(pv.stats)
            return result

    @staticmethod
    def _rolling_avg(prev: float, new: float, n: int) -> float:
        return prev + (new - prev) / n

    # ── Persistence ───────────────────────────────────────────────

    def _persist(self, pv: PromptVersion) -> None:
        prompt_dir = self.root / pv.name
        prompt_dir.mkdir(parents=True, exist_ok=True)
        path = prompt_dir / f"{pv.version}.json"
        content = json.dumps(pv.to_dict(), indent=2, default=str)
        import os
        import tempfile

        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            os.write(fd, content.encode())
            os.close(fd)
            os.replace(tmp, str(path))
        except Exception:
            with contextlib.suppress(OSError):
                os.close(fd)
            Path(tmp).unlink(missing_ok=True)
            raise

    def _reload_from_disk(self) -> None:
        with self._lock:
            for prompt_dir in self.root.iterdir():
                if not prompt_dir.is_dir():
                    continue
                for version_file in prompt_dir.glob("*.json"):
                    try:
                        data = json.loads(version_file.read_text())
                        pv = PromptVersion.from_dict(data)
                        self._versions[pv.name][pv.version] = pv
                    except Exception as e:
                        log.warning("prompt.load_failed", file=str(version_file), error=str(e))
