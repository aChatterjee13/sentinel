"""Per-agent tool permission matrix."""

from __future__ import annotations

from sentinel.config.schema import ToolAuditConfig
from sentinel.core.exceptions import ToolPermissionError


class PermissionMatrix:
    """Resolve which tools an agent is allowed to call.

    The config layout mirrors the YAML schema:

    ::

        tool_audit:
          permissions:
            agent_a:
              allowed: [tool_x, tool_y]
              blocked: [tool_z]
    """

    def __init__(self, config: ToolAuditConfig | None = None):
        self.config = config or ToolAuditConfig()
        self._matrix = config.permissions if config else {}

    def is_allowed(self, agent: str, tool: str) -> bool:
        rules = self._matrix.get(agent, {})
        blocked = set(rules.get("blocked", []))
        allowed = set(rules.get("allowed", []))
        if tool in blocked:
            return False
        if not allowed:
            return True  # no allowlist = permissive
        return tool in allowed

    def enforce(self, agent: str, tool: str) -> None:
        if not self.is_allowed(agent, tool):
            raise ToolPermissionError(f"agent '{agent}' is not allowed to call tool '{tool}'")

    def list_allowed(self, agent: str) -> list[str]:
        return sorted(self._matrix.get(agent, {}).get("allowed", []))

    def list_blocked(self, agent: str) -> list[str]:
        return sorted(self._matrix.get(agent, {}).get("blocked", []))
