"""Token usage and cost tracking for LLM applications."""

from __future__ import annotations

import contextlib
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import structlog

from sentinel.config.defaults import DEFAULT_PRICING
from sentinel.config.schema import LLMOpsConfig, TokenEconomicsConfig
from sentinel.core.types import LLMUsage

log = structlog.get_logger(__name__)


def provider_from_model(model: str) -> str:
    """Classify a model name into a provider bucket.

    Returns one of ``"azure"``, ``"openai"``, ``"anthropic"``, or
    ``"unknown"``. Azure OpenAI deployments are expected to be
    prefixed with ``azure/`` — e.g. ``azure/gpt-4o`` — so cost and
    usage dashboards can tell Azure-hosted calls apart from direct
    OpenAI ones. Unknown prefixes return ``"unknown"`` rather than
    raising so the caller can still log the call.

    Example:
        >>> provider_from_model("azure/gpt-4o")
        'azure'
        >>> provider_from_model("gpt-4o-mini")
        'openai'
        >>> provider_from_model("claude-sonnet-4-6")
        'anthropic'
    """
    if model.startswith("azure/"):
        return "azure"
    if model.startswith(("gpt-", "text-embedding-", "o1-", "o3-")):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    return "unknown"


class TokenTracker:
    """Track token usage, cost, and budget compliance for LLM calls.

    Aggregates by model and by configurable dimensions (user segment,
    prompt version, use case). Emits alerts via the provided audit
    trail when daily budgets, per-query budgets, or trend thresholds
    are crossed.
    """

    def __init__(
        self,
        config: TokenEconomicsConfig | None = None,
        audit_trail: Any = None,
        pricing: dict[str, dict[str, float]] | None = None,
    ):
        self.config = config or TokenEconomicsConfig()
        self.audit = audit_trail
        self.pricing = {**DEFAULT_PRICING, **(pricing or {}), **self.config.pricing}
        self._totals: dict[str, dict[str, float]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": 0}
        )
        self._daily: dict[str, float] = defaultdict(float)
        self._recent_costs: deque[float] = deque(maxlen=200)
        self._budget_warned: set[str] = set()
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: LLMOpsConfig | str | Any, audit_trail: Any = None) -> TokenTracker:
        if isinstance(config, str):
            from sentinel.config.loader import load_config

            cfg = load_config(config)
            return cls(cfg.llmops.token_economics, audit_trail=audit_trail)
        if isinstance(config, LLMOpsConfig):
            return cls(config.token_economics, audit_trail=audit_trail)
        return cls(config.llmops.token_economics, audit_trail=audit_trail)

    # ── Cost calculation ──────────────────────────────────────────

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Compute USD cost using the configured pricing table."""
        prices = self.pricing.get(model) or self.pricing.get(
            "default", {"input": 0.0, "output": 0.0}
        )
        input_price = prices.get("input", 0.0)
        output_price = prices.get("output", 0.0)
        return (input_tokens * input_price + output_tokens * output_price) / 1000.0

    # ── Recording ─────────────────────────────────────────────────

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float = 0.0,
        cached: bool = False,
        **dimensions: Any,
    ) -> LLMUsage:
        cost = 0.0 if cached else self.estimate_cost(model, input_tokens, output_tokens)
        usage = LLMUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            cached=cached,
        )

        with self._lock:
            # Aggregate by model + provider + each tracked dimension.
            # The provider bucket (azure/openai/anthropic/unknown) is
            # always tracked so cost dashboards can group by provider
            # without any extra config.
            provider = provider_from_model(model)
            keys = [f"model:{model}", f"provider:{provider}"]
            for dim in self.config.track_by:
                value = dimensions.get(dim)
                if value is not None:
                    keys.append(f"{dim}:{value}")
            for key in keys:
                stats = self._totals[key]
                stats["input_tokens"] += input_tokens
                stats["output_tokens"] += output_tokens
                stats["cost"] += cost
                stats["calls"] += 1

            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self._daily[day] += cost
            self._recent_costs.append(cost)
            self._check_budgets(day, cost, model)
        return usage

    # ── Budget enforcement ────────────────────────────────────────

    def _check_budgets(self, day: str, cost: float, model: str) -> None:
        budgets = self.config.budgets or {}
        alerts = self.config.alerts or {}

        per_query_max_cost = budgets.get("per_query_max_cost")
        if per_query_max_cost and cost > per_query_max_cost:
            self._emit_alert(
                "per_query_budget_exceeded", model=model, cost=cost, limit=per_query_max_cost
            )

        daily_max = budgets.get("daily_max_cost")
        if daily_max and self._daily[day] > daily_max and day not in self._budget_warned:
            self._emit_alert(
                "daily_budget_exceeded", day=day, total=self._daily[day], limit=daily_max
            )
            self._budget_warned.add(day)

        threshold = alerts.get("daily_cost_threshold")
        if threshold and self._daily[day] > threshold and f"warn-{day}" not in self._budget_warned:
            self._emit_alert(
                "daily_cost_threshold", day=day, total=self._daily[day], limit=threshold
            )
            self._budget_warned.add(f"warn-{day}")

    def _emit_alert(self, event: str, **payload: Any) -> None:
        log.warning(f"token_economics.{event}", **payload)
        if self.audit is not None:
            with contextlib.suppress(Exception):
                self.audit.log(event_type=f"llmops.{event}", **payload)

    # ── Reporting ─────────────────────────────────────────────────

    def totals(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {k: dict(v) for k, v in self._totals.items()}

    def daily_total(self, day: str | None = None) -> float:
        if day is None:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            return self._daily.get(day, 0.0)

    def trend(self, window: int = 50) -> float:
        """Return the average cost across the last `window` calls."""
        with self._lock:
            if not self._recent_costs:
                return 0.0
            items = list(self._recent_costs)[-window:]
        return sum(items) / len(items)
