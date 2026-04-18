"""Plugin / hook system for extending Sentinel.

Hooks let users register callbacks that fire on lifecycle events such as
``before_drift_check``, ``after_alert``, ``before_deployment``, etc. They are
the primary extension point for behaviour that doesn't warrant a full module.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from typing import Any

import structlog

log = structlog.get_logger(__name__)

HookCallback = Callable[..., Any]


class HookType(str, Enum):
    """Lifecycle events that hooks can subscribe to."""

    BEFORE_PREDICTION = "before_prediction"
    AFTER_PREDICTION = "after_prediction"
    BEFORE_DRIFT_CHECK = "before_drift_check"
    AFTER_DRIFT_CHECK = "after_drift_check"
    BEFORE_ALERT = "before_alert"
    AFTER_ALERT = "after_alert"
    BEFORE_DEPLOYMENT = "before_deployment"
    AFTER_DEPLOYMENT = "after_deployment"
    BEFORE_RETRAIN = "before_retrain"
    AFTER_RETRAIN = "after_retrain"
    BEFORE_GUARDRAIL = "before_guardrail"
    AFTER_GUARDRAIL = "after_guardrail"
    BEFORE_AGENT_RUN = "before_agent_run"
    AFTER_AGENT_RUN = "after_agent_run"
    ON_ESCALATION = "on_escalation"


class HookManager:
    """Registry and dispatcher for lifecycle hooks.

    Example:
        >>> manager = HookManager()
        >>> @manager.on(HookType.AFTER_DRIFT_CHECK)
        ... def log_drift(report):
        ...     print(f"Drift: {report.summary}")
        >>> manager.dispatch(HookType.AFTER_DRIFT_CHECK, report)
    """

    def __init__(self) -> None:
        self._hooks: dict[HookType, list[HookCallback]] = defaultdict(list)

    def register(self, hook_type: HookType, callback: HookCallback) -> None:
        """Register a callback for a hook type."""
        self._hooks[hook_type].append(callback)
        log.debug("hook.registered", hook=hook_type.value, callback=callback.__name__)

    def on(self, hook_type: HookType) -> Callable[[HookCallback], HookCallback]:
        """Decorator form of `register`."""

        def decorator(func: HookCallback) -> HookCallback:
            self.register(hook_type, func)
            return func

        return decorator

    def dispatch(self, hook_type: HookType, *args: Any, **kwargs: Any) -> list[Any]:
        """Invoke all callbacks for a hook type, returning their results.

        Hook callbacks must not raise — exceptions are caught and logged so
        that one buggy hook does not break the SDK.
        """
        results: list[Any] = []
        for callback in self._hooks.get(hook_type, []):
            try:
                results.append(callback(*args, **kwargs))
            except Exception as e:
                log.error(
                    "hook.failed",
                    hook=hook_type.value,
                    callback=callback.__name__,
                    error=str(e),
                )
        return results

    def clear(self, hook_type: HookType | None = None) -> None:
        """Remove all callbacks (optionally for a single hook type)."""
        if hook_type is None:
            self._hooks.clear()
        else:
            self._hooks.pop(hook_type, None)

    def count(self, hook_type: HookType) -> int:
        """Return the number of callbacks registered for a hook type."""
        return len(self._hooks.get(hook_type, []))
