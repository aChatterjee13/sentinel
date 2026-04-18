"""Custom guardrail DSL — user-defined rules evaluated at runtime.

Supports 11 rule types (regex, keyword, length, JSON schema, sentiment,
language detection, word count, not-empty) that can be combined with
``all`` (lenient — block only when every rule fails) or ``any`` (strict —
block on the first failure).
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from typing import Any, Literal

import structlog

from sentinel.core.types import GuardrailResult
from sentinel.llmops.guardrails.base import BaseGuardrail

log = structlog.get_logger(__name__)

# ── Regex safety helpers ──────────────────────────────────────────

_REGEX_TIMEOUT_SECONDS = 2.0


def _safe_regex_search(pattern: str, content: str, flags: int = 0) -> re.Match[str] | None:
    """Execute regex with timeout protection against catastrophic backtracking."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(re.search, pattern, content, flags)
        try:
            return future.result(timeout=_REGEX_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            log.warning(
                "custom_guardrail.regex_timeout",
                pattern=pattern,
                timeout_s=_REGEX_TIMEOUT_SECONDS,
            )
            future.cancel()
            return None


def _validate_regex_pattern(pattern: str) -> None:
    """Compile a regex pattern upfront to catch syntax errors early.

    Raises:
        re.error: If the pattern is not a valid regular expression.
    """
    re.compile(pattern)

# ── Sentiment word lists ───────────────────────────────────────────

_POSITIVE_WORDS = frozenset(
    {
        "good",
        "great",
        "excellent",
        "happy",
        "love",
        "wonderful",
        "best",
        "nice",
        "thank",
        "please",
    }
)

_NEGATIVE_WORDS = frozenset(
    {
        "bad",
        "terrible",
        "horrible",
        "worst",
        "hate",
        "awful",
        "poor",
        "wrong",
        "stupid",
        "ugly",
    }
)

# ── Language detection word lists ──────────────────────────────────

_LANGUAGE_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset({"the", "is", "and", "of", "to"}),
    "es": frozenset({"el", "la", "de", "en", "y"}),
    "fr": frozenset({"le", "la", "de", "et", "les"}),
    "de": frozenset({"der", "die", "und", "den", "ist"}),
}


# ── Rule evaluators ───────────────────────────────────────────────
# Each returns ``(passed: bool, message: str)``.  The message is only
# meaningful when ``passed`` is False.


def _eval_regex_match(content: str, rule: dict[str, Any]) -> tuple[bool, str]:
    """Content must match the given regex pattern."""
    pattern = rule.get("pattern", "")
    flags = re.IGNORECASE if rule.get("case_insensitive", False) else 0
    if _safe_regex_search(pattern, content, flags):
        return True, ""
    return False, f"regex_match: pattern '{pattern}' not found"


def _eval_regex_absent(content: str, rule: dict[str, Any]) -> tuple[bool, str]:
    """Content must NOT match the given regex pattern."""
    pattern = rule.get("pattern", "")
    flags = re.IGNORECASE if rule.get("case_insensitive", False) else 0
    if _safe_regex_search(pattern, content, flags):
        return False, f"regex_absent: pattern '{pattern}' found"
    return True, ""


def _eval_keyword_present(content: str, rule: dict[str, Any]) -> tuple[bool, str]:
    """At least one keyword must appear in the content."""
    keywords: list[str] = rule.get("keywords", [])
    lower = content.lower()
    if any(kw.lower() in lower for kw in keywords):
        return True, ""
    return False, f"keyword_present: none of {keywords} found"


def _eval_keyword_absent(content: str, rule: dict[str, Any]) -> tuple[bool, str]:
    """None of the keywords may appear in the content."""
    keywords: list[str] = rule.get("keywords", [])
    lower = content.lower()
    found = [kw for kw in keywords if kw.lower() in lower]
    if found:
        return False, f"keyword_absent: found {found}"
    return True, ""


def _eval_min_length(content: str, rule: dict[str, Any]) -> tuple[bool, str]:
    """Content length must be at least ``min_chars``."""
    min_chars: int = rule.get("min_chars", 0)
    if len(content) >= min_chars:
        return True, ""
    return False, f"min_length: {len(content)} < {min_chars}"


def _eval_max_length(content: str, rule: dict[str, Any]) -> tuple[bool, str]:
    """Content length must be at most ``max_chars``."""
    max_chars: int = rule.get("max_chars", 0)
    if len(content) <= max_chars:
        return True, ""
    return False, f"max_length: {len(content)} > {max_chars}"


def _eval_json_schema(content: str, rule: dict[str, Any]) -> tuple[bool, str]:
    """Content must parse as JSON and validate against the given schema."""
    schema: dict[str, Any] = rule.get("schema", {})
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        return False, f"json_schema: invalid JSON — {exc}"

    try:
        import jsonschema  # type: ignore[import-untyped]

        jsonschema.validate(data, schema)
        return True, ""
    except ImportError:
        # Fallback: check that all required top-level keys are present.
        required = schema.get("required", [])
        if not isinstance(data, dict):
            return False, "json_schema: content is not a JSON object"
        missing = [k for k in required if k not in data]
        if missing:
            return False, f"json_schema: missing keys {missing}"
        return True, ""
    except jsonschema.ValidationError as exc:  # type: ignore[union-attr]
        return False, f"json_schema: {exc.message}"


def _eval_sentiment(content: str, rule: dict[str, Any]) -> tuple[bool, str]:
    """Heuristic sentiment score must fall within [min_score, max_score].

    Score = (positive - negative) / total_words, clamped to [-1, 1].
    """
    min_score: float = rule.get("min_score", -1.0)
    max_score: float = rule.get("max_score", 1.0)
    words = content.lower().split()
    if not words:
        score = 0.0
    else:
        pos = sum(1 for w in words if w.strip(".,!?;:'\"") in _POSITIVE_WORDS)
        neg = sum(1 for w in words if w.strip(".,!?;:'\"") in _NEGATIVE_WORDS)
        score = max(-1.0, min(1.0, (pos - neg) / len(words)))
    if min_score <= score <= max_score:
        return True, ""
    return False, f"sentiment: score {score:.2f} not in [{min_score}, {max_score}]"


def _eval_language(content: str, rule: dict[str, Any]) -> tuple[bool, str]:
    """Detect language via common function-word frequency heuristic."""
    allowed: list[str] = rule.get("allowed", [])
    words = set(content.lower().split())
    best_lang = "unknown"
    best_count = 0
    for lang, common in _LANGUAGE_WORDS.items():
        count = len(words & common)
        if count > best_count:
            best_count = count
            best_lang = lang
    if best_lang in allowed:
        return True, ""
    return False, f"language: detected '{best_lang}', allowed {allowed}"


def _eval_word_count(content: str, rule: dict[str, Any]) -> tuple[bool, str]:
    """Word count must fall within [min_words, max_words]."""
    min_words: int = rule.get("min_words", 0)
    max_words: int = rule.get("max_words", 2**31)
    wc = len(content.split())
    if min_words <= wc <= max_words:
        return True, ""
    return False, f"word_count: {wc} not in [{min_words}, {max_words}]"


def _eval_not_empty(content: str, rule: dict[str, Any]) -> tuple[bool, str]:
    """Content must not be blank or whitespace-only."""
    if content.strip():
        return True, ""
    return False, "not_empty: content is blank"


_RULE_EVALUATORS: dict[str, Any] = {
    "regex_match": _eval_regex_match,
    "regex_absent": _eval_regex_absent,
    "keyword_present": _eval_keyword_present,
    "keyword_absent": _eval_keyword_absent,
    "min_length": _eval_min_length,
    "max_length": _eval_max_length,
    "json_schema": _eval_json_schema,
    "sentiment": _eval_sentiment,
    "language": _eval_language,
    "word_count": _eval_word_count,
    "not_empty": _eval_not_empty,
}


class CustomGuardrail(BaseGuardrail):
    """A guardrail composed of user-defined DSL rules.

    Rules are specified in the YAML config as a list of dicts, each
    containing a ``rule`` key naming the evaluator plus any params that
    evaluator expects.  The ``combine`` strategy controls how individual
    results are aggregated:

    * ``"any"`` (strict) — block if **any** rule fails.
    * ``"all"`` (lenient) — block only if **every** rule fails.

    Args:
        name: Human-readable guardrail name (from YAML config).
        action: What to do when the guardrail triggers (``block`` / ``warn``).
        rules: List of rule dicts, each with at least a ``rule`` key.
        combine: Combination strategy — ``"all"`` or ``"any"``.

    Example:
        >>> g = CustomGuardrail(
        ...     name="profanity_check",
        ...     action="block",
        ...     rules=[{"rule": "keyword_absent", "keywords": ["spam"]}],
        ...     combine="any",
        ... )
        >>> result = g.check("This is spam")
        >>> result.passed
        False
    """

    name: str = "custom"
    direction: Literal["input", "output", "both"] = "both"

    def __init__(
        self,
        *,
        name: str,
        action: Literal["block", "warn", "redact"] = "warn",
        rules: list[dict[str, Any]],
        combine: Literal["all", "any"] = "all",
        **kwargs: Any,
    ) -> None:
        super().__init__(action=action, **kwargs)
        self.custom_name = name
        self._rules = rules
        self._combine = combine
        # Validate regex patterns upfront to fail fast on invalid patterns.
        for rule_cfg in rules:
            rule_type = rule_cfg.get("rule", "")
            if rule_type in ("regex_match", "regex_absent"):
                pattern = rule_cfg.get("pattern", "")
                try:
                    _validate_regex_pattern(pattern)
                except re.error as exc:
                    raise ValueError(
                        f"Invalid regex pattern in rule '{rule_type}': {exc}"
                    ) from exc

    def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        """Evaluate all configured rules against *content*.

        Args:
            content: The text being checked.
            context: Optional context dict (unused by built-in rule types
                but forwarded for future extensibility).

        Returns:
            A :class:`GuardrailResult` reflecting the combined evaluation.
        """
        if not self._rules:
            return self._result(passed=True, score=0.0)

        failures: list[str] = []
        for rule_cfg in self._rules:
            rule_type = rule_cfg.get("rule", "")
            evaluator = _RULE_EVALUATORS.get(rule_type)
            if evaluator is None:
                msg = f"unknown rule type: {rule_type}"
                log.warning("custom_guardrail.unknown_rule", rule=rule_type, name=self.custom_name)
                failures.append(msg)
                continue
            passed, msg = evaluator(content, rule_cfg)
            if not passed:
                failures.append(msg)

        if self._combine == "any":
            # Strict: any single failure triggers
            triggered = len(failures) > 0
        else:
            # Lenient ("all"): trigger only if every rule fails
            triggered = len(failures) == len(self._rules)

        if triggered:
            score = len(failures) / len(self._rules)
            reason = "; ".join(failures[:3])
            return self._result(
                passed=False,
                score=score,
                reason=reason,
                metadata={"guardrail": self.custom_name, "failures": failures},
            )
        return self._result(passed=True, score=0.0)
