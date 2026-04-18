"""Token budget guardrail — block oversized inputs."""

from __future__ import annotations

from typing import Any, Literal

from sentinel.core.types import GuardrailResult
from sentinel.llmops.guardrails.base import BaseGuardrail


class TokenBudgetGuardrail(BaseGuardrail):
    """Reject inputs that would exceed cost or context budgets.

    Uses ``tiktoken`` for accurate counts when available; otherwise
    falls back to a simple whitespace + punctuation heuristic.
    """

    name = "token_budget"
    direction = "input"

    def __init__(
        self,
        action: Literal["block", "warn", "redact"] = "block",
        max_input_tokens: int = 4000,
        encoding: str = "cl100k_base",
        **kwargs: Any,
    ):
        super().__init__(action=action, **kwargs)
        self.max_input_tokens = max_input_tokens
        self.encoding = encoding
        self._encoder = self._try_load_encoder(encoding)

    @staticmethod
    def _try_load_encoder(encoding: str) -> Any:
        try:
            import tiktoken  # type: ignore[import-not-found]

            return tiktoken.get_encoding(encoding)
        except Exception:
            return None

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        n_tokens = self._count(content)
        if n_tokens > self.max_input_tokens:
            return self._result(
                passed=False,
                score=n_tokens / self.max_input_tokens,
                reason=f"input has {n_tokens} tokens, max {self.max_input_tokens}",
                metadata={"tokens": n_tokens, "max": self.max_input_tokens},
            )
        return self._result(passed=True, score=n_tokens / self.max_input_tokens)

    def _count(self, content: str) -> int:
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(content))
            except Exception:
                pass
        # Heuristic fallback: ~4 chars per token for English text
        return max(1, len(content) // 4)
