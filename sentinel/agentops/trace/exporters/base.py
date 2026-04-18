"""Abstract base class for trace exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sentinel.core.types import AgentTrace


class BaseExporter(ABC):
    """Sink for finished agent traces.

    Exporters never modify traces. They serialise and forward them to
    storage, dashboards, or APM systems.
    """

    name: str = "base"

    @abstractmethod
    def export(self, trace: AgentTrace) -> None:
        """Persist or forward a single trace."""
