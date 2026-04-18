"""LLMOps — prompt management, guardrails, quality, token economics.

Provides the LLM monitoring and governance layer of Project Sentinel.
The :class:`LLMOpsClient` is the entry point exposed via
``SentinelClient.llmops`` when ``llmops.enabled`` is true in the config.
"""

from sentinel.llmops.client import LLMOpsClient
from sentinel.llmops.guardrails.engine import GuardrailPipeline
from sentinel.llmops.prompt_drift import PromptDriftDetector
from sentinel.llmops.prompt_manager import Prompt, PromptManager, PromptVersion
from sentinel.llmops.quality.evaluator import ResponseEvaluator
from sentinel.llmops.quality.semantic_drift import SemanticDriftMonitor
from sentinel.llmops.token_economics import TokenTracker

__all__ = [
    "GuardrailPipeline",
    "LLMOpsClient",
    "Prompt",
    "PromptDriftDetector",
    "PromptManager",
    "PromptVersion",
    "ResponseEvaluator",
    "SemanticDriftMonitor",
    "TokenTracker",
]
