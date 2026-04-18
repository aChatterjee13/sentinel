"""Append finished traces to a daily JSON-Lines file."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from sentinel.agentops.trace.exporters.base import BaseExporter
from sentinel.core.types import AgentTrace


class JSONLExporter(BaseExporter):
    """Default exporter — writes one trace per line into ``traces-YYYY-MM-DD.jsonl``."""

    name = "jsonl"

    def __init__(self, root: str | Path = "./traces"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, trace: AgentTrace) -> None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.root / f"traces-{day}.jsonl"
        line = trace.model_dump_json()
        with self._lock, path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
