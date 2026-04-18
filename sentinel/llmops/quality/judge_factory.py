"""Factory for creating LLM judge callables from config.

Auto-detects the available LLM provider package and returns a callable
that scores responses on the configured rubrics.  When no provider is
installed, returns ``None`` so the evaluator degrades to heuristic mode.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import structlog

from sentinel.config.schema import QualityEvaluatorConfig

log = structlog.get_logger(__name__)

JudgeFn = Callable[[str, str, str | None], dict[str, float]]


def create_judge_fn(config: QualityEvaluatorConfig) -> JudgeFn | None:
    """Build an LLM-judge callable from the evaluator config.

    Args:
        config: Quality evaluator configuration containing
            ``judge_model`` and ``rubrics``.

    Returns:
        A callable ``(response, query, context) -> {rubric: score}``
        or ``None`` if no supported LLM package is available.

    Example:
        >>> fn = create_judge_fn(cfg)
        >>> if fn is not None:
        ...     scores = fn("The claim is valid.", "Is the claim valid?", None)
        ...     print(scores)  # {"relevance": 4.2, "completeness": 3.8, ...}
    """
    if config.method != "llm_judge" and config.method != "hybrid":
        return None

    model = config.judge_model
    if not model:
        log.warning("judge_factory.no_judge_model", hint="set llmops.quality.evaluator.judge_model")
        return None

    rubrics = config.rubrics or _default_rubrics()

    if model.startswith("claude-"):
        fn = _try_anthropic(model, rubrics)
        if fn is not None:
            return fn
        log.warning(
            "judge_factory.anthropic_unavailable",
            model=model,
            hint="pip install anthropic",
        )
        return None

    fn = _try_openai(model, rubrics)
    if fn is not None:
        return fn

    log.warning(
        "judge_factory.no_llm_package",
        model=model,
        hint="pip install openai  (or anthropic for claude models)",
    )
    return None


# ── Internals ────────────────────────────────────────────────────────


def _default_rubrics() -> dict[str, dict[str, Any]]:
    return {
        "relevance": {"weight": 0.3, "scale": 5},
        "completeness": {"weight": 0.3, "scale": 5},
        "clarity": {"weight": 0.2, "scale": 5},
        "safety": {"weight": 0.2, "scale": 5},
    }


def _build_system_prompt(rubrics: dict[str, dict[str, Any]]) -> str:
    """Build the system prompt instructing the judge model."""
    rubric_lines = []
    for name, cfg in rubrics.items():
        scale = cfg.get("scale", 5)
        rubric_lines.append(f"- {name}: score from 0 to {scale}")
    rubric_block = "\n".join(rubric_lines)

    return (
        "You are a strict response quality evaluator. "
        "Score the assistant response on each rubric below.\n\n"
        f"Rubrics:\n{rubric_block}\n\n"
        'Return ONLY a JSON object mapping each rubric name to its numeric score. '
        'Example: {"relevance": 4.2, "completeness": 3.8}\n'
        "No explanation. JSON only."
    )


def _build_user_prompt(response: str, query: str, context: str | None) -> str:
    parts = [f"Query: {query}", f"Response: {response}"]
    if context:
        parts.insert(1, f"Context: {context}")
    return "\n\n".join(parts)


def _parse_scores(text: str, rubrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Extract rubric scores from the judge model's reply."""
    match = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"judge model did not return valid JSON: {text[:200]}")
    raw = json.loads(match.group())
    scores: dict[str, float] = {}
    for name, cfg in rubrics.items():
        val = raw.get(name)
        if val is not None:
            scale = cfg.get("scale", 5)
            scores[name] = max(0.0, min(float(val), float(scale)))
    return scores


def _try_openai(
    model: str, rubrics: dict[str, dict[str, Any]]
) -> JudgeFn | None:
    """Attempt to build a judge using the ``openai`` package."""
    try:
        import openai
    except ImportError:
        return None

    client = openai.OpenAI()
    system_prompt = _build_system_prompt(rubrics)

    def _judge(response: str, query: str, context: str | None) -> dict[str, float]:
        user_prompt = _build_user_prompt(response, query, context)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        text = completion.choices[0].message.content or ""
        return _parse_scores(text, rubrics)

    log.info("judge_factory.openai_ready", model=model)
    return _judge


def _try_anthropic(
    model: str, rubrics: dict[str, dict[str, Any]]
) -> JudgeFn | None:
    """Attempt to build a judge using the ``anthropic`` package."""
    try:
        import anthropic
    except ImportError:
        return None

    client = anthropic.Anthropic()
    system_prompt = _build_system_prompt(rubrics)

    def _judge(response: str, query: str, context: str | None) -> dict[str, float]:
        user_prompt = _build_user_prompt(response, query, context)
        message = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        text = message.content[0].text if message.content else ""
        return _parse_scores(text, rubrics)

    log.info("judge_factory.anthropic_ready", model=model)
    return _judge
