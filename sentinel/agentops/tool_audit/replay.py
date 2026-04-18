"""Tool call replay store — record and replay tool interactions."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentinel.agentops.tool_audit.monitor import ToolCallRecord


class ToolReplayStore:
    """Persist tool calls so failed agent runs can be replayed deterministically.

    Each call is stored as a JSON line in ``<root>/<tool>/<YYYY-MM-DD>.jsonl``.
    Replay returns the recorded output for the same (tool, inputs) tuple.
    """

    def __init__(self, root: str | Path = "./tool_replay"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], Any] = {}
        self._cache_maxsize = 10_000

    def save(self, record: ToolCallRecord) -> None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        tool_dir = self.root / record.tool
        tool_dir.mkdir(parents=True, exist_ok=True)
        path = tool_dir / f"{day}.jsonl"
        line = json.dumps(
            {
                "tool": record.tool,
                "agent": record.agent,
                "inputs": record.inputs,
                "output": record.output,
                "success": record.success,
                "error": record.error,
                "latency_ms": record.latency_ms,
                "timestamp": record.timestamp.isoformat(),
            },
            default=str,
        )
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            if len(self._cache) >= self._cache_maxsize:
                self._cache.clear()
            self._cache[(record.tool, _hash_inputs(record.inputs))] = record.output

    def replay(self, tool: str, inputs: dict[str, Any]) -> Any:
        key = (tool, _hash_inputs(inputs))
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        # Lazy disk scan (no lock held — file reads are append-safe)
        tool_dir = self.root / tool
        if not tool_dir.exists():
            return None
        for path in sorted(tool_dir.glob("*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict) or "inputs" not in rec or "output" not in rec:
                    continue
                if _hash_inputs(rec.get("inputs", {})) == key[1]:
                    with self._lock:
                        if len(self._cache) >= self._cache_maxsize:
                            self._cache.clear()
                        self._cache[key] = rec.get("output")
                    return rec.get("output")
        return None

    def list_recordings(self, tool: str) -> list[Path]:
        return sorted((self.root / tool).glob("*.jsonl"))


def _hash_inputs(inputs: dict[str, Any]) -> str:
    return json.dumps(inputs, sort_keys=True, default=str)
