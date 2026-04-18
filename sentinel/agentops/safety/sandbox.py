"""Action sandbox — gate destructive operations."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import structlog

from sentinel.config.schema import SandboxConfig
from sentinel.core.exceptions import AgentError

log = structlog.get_logger(__name__)


@dataclass
class SandboxDecision:
    """Result of sandbox evaluation."""

    allowed: bool
    requires_approval: bool = False
    dry_run: bool = False
    reason: str | None = None


class ActionSandbox:
    """Intercept destructive tool calls and apply the configured policy.

    Modes:
        - ``approve_first``: caller must check :attr:`SandboxDecision.requires_approval`
          and route to a human before executing.
        - ``dry_run``: action is allowed only as a no-op simulation.
        - ``sandbox_then_apply``: caller should run the action in a
          sandbox first, then apply the resulting diff in production.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        audit_trail: Any = None,
    ):
        self.config = config or SandboxConfig()
        self.destructive_ops = {op.lower() for op in self.config.destructive_ops}
        self._audit = audit_trail

    def _is_destructive(self, text: str) -> bool:
        """Check if text contains any destructive operation as a whole word."""
        return any(re.search(rf"\b{re.escape(op)}\b", text) for op in self.destructive_ops)

    def evaluate(self, action: str, tool: str | None = None) -> SandboxDecision:
        action_l = action.lower()
        is_destructive = self._is_destructive(action_l) or (
            tool is not None and self._is_destructive(tool.lower())
        )
        if not is_destructive:
            return SandboxDecision(allowed=True)

        mode: Literal["approve_first", "dry_run", "sandbox_then_apply"] = self.config.mode
        log.info("sandbox.intercepted", action=action, tool=tool, mode=mode)
        if mode == "approve_first":
            return SandboxDecision(
                allowed=False,
                requires_approval=True,
                reason="destructive op requires human approval",
            )
        if mode == "dry_run":
            return SandboxDecision(allowed=True, dry_run=True, reason="dry-run only")
        return SandboxDecision(allowed=True, reason="sandbox-then-apply")

    def execute(
        self,
        action: str,
        operation: Callable[[], Any],
        approver: Callable[[str], bool] | None = None,
        tool: str | None = None,
    ) -> Any:
        decision = self.evaluate(action, tool=tool)

        if self._audit is not None:
            self._audit.log(
                event_type="sandbox.evaluated",
                action=action,
                tool=tool,
                allowed=decision.allowed,
                requires_approval=decision.requires_approval,
                dry_run=decision.dry_run,
                mode=self.config.mode,
            )

        if not decision.allowed:
            if decision.requires_approval and approver is None:
                log.warning("sandbox.no_approver_configured", action=action, tool=tool)
                raise AgentError(
                    f"destructive action requires approval but no approver configured: {action}"
                )
            if approver is None:
                raise AgentError(f"sandbox blocked action without approver: {action}")
            approved = approver(action)
            if self._audit is not None:
                self._audit.log(
                    event_type="sandbox.approval_decision",
                    action=action,
                    approved=approved,
                )
            if not approved:
                raise AgentError(f"action denied by approver: {action}")
        if decision.dry_run:
            log.info("sandbox.dry_run", action=action)
            return None
        return operation()
