"""Sentinel exception hierarchy.

All Sentinel errors derive from `SentinelError`. Module-specific errors
extend module-specific bases so callers can catch broad or narrow categories.

Example:
    >>> try:
    ...     client.check_drift()
    ... except DriftDetectionError as e:
    ...     log.error("drift detection failed", error=str(e))
    ... except SentinelError as e:
    ...     log.error("sentinel failed", error=str(e))
"""

from __future__ import annotations


class SentinelError(Exception):
    """Base class for all Sentinel exceptions."""


# ── Configuration ──────────────────────────────────────────────────


class ConfigError(SentinelError):
    """Raised when a config file is invalid or cannot be loaded."""


class ConfigValidationError(ConfigError):
    """Raised when a config fails Pydantic validation."""


class ConfigNotFoundError(ConfigError):
    """Raised when the requested config file does not exist."""


class ConfigMissingEnvVarError(ConfigValidationError):
    """Raised when ``${VAR}`` substitution finds an unset env var in strict mode."""


class ConfigCircularInheritanceError(ConfigError):
    """Raised when ``extends:`` chains form a cycle."""


class ConfigFileReferenceError(ConfigValidationError):
    """Raised when a path referenced by the config does not exist (strict mode)."""


class ConfigSignatureError(ConfigError):
    """Raised when a config signature is missing, malformed, or does not match."""


class ConfigSecretError(ConfigError):
    """Base class for secret-resolution failures during config load."""


class ConfigKeyVaultError(ConfigSecretError):
    """Raised when an ``${azkv:vault/secret}`` token cannot be resolved."""


class ConfigAWSSecretsError(ConfigSecretError):
    """Raised when an ``${awssm:secret-name}`` token cannot be resolved."""


class ConfigGCPSecretsError(ConfigSecretError):
    """Raised when a ``${gcpsm:project/secret-name}`` token cannot be resolved."""


# ── Observability ──────────────────────────────────────────────────


class ObservabilityError(SentinelError):
    """Base for observability-layer errors."""


class DriftDetectionError(ObservabilityError):
    """Raised when a drift detector cannot compute a result."""


class DataQualityError(ObservabilityError):
    """Raised when data quality validation fails catastrophically."""


class FeatureHealthError(ObservabilityError):
    """Raised when feature health calculation fails."""


# ── Deployment / Action ────────────────────────────────────────────


class ActionError(SentinelError):
    """Base for action-layer errors."""


class DeploymentError(ActionError):
    """Raised when a deployment operation fails."""


class RollbackError(DeploymentError):
    """Raised when an automatic rollback fails."""


class NotificationError(ActionError):
    """Raised when a notification channel fails to deliver."""


class RetrainError(ActionError):
    """Raised when a retrain pipeline fails."""


class ApprovalTimeoutError(RetrainError):
    """Raised when a human approval gate times out."""


# ── Foundation ─────────────────────────────────────────────────────


class FoundationError(SentinelError):
    """Base for foundation-layer errors."""


class RegistryError(FoundationError):
    """Raised when model registry operations fail."""


class ModelNotFoundError(RegistryError):
    """Raised when a requested model version is not in the registry."""


class AuditError(FoundationError):
    """Raised when audit trail operations fail."""


class AuditTamperedError(AuditError):
    """Raised when audit trail integrity verification detects tampering."""


class AuditKeystoreError(AuditError):
    """Raised when an audit signing key cannot be loaded."""


# ── LLMOps ─────────────────────────────────────────────────────────


class LLMOpsError(SentinelError):
    """Base for LLMOps errors."""


class GuardrailError(LLMOpsError):
    """Raised when a guardrail fails to evaluate."""


class GuardrailBlockedError(GuardrailError):
    """Raised when a guardrail blocks a request (use as control flow signal)."""


class PromptError(LLMOpsError):
    """Raised when prompt resolution or rendering fails."""


class TokenBudgetExceededError(LLMOpsError):
    """Raised when an LLM call exceeds its token budget."""


# Backwards-compatible alias
TokenBudgetExceeded = TokenBudgetExceededError


# ── AgentOps ───────────────────────────────────────────────────────


class AgentError(SentinelError):
    """Base for AgentOps errors."""


class TraceError(AgentError):
    """Raised when tracing infrastructure fails."""


class BudgetExceededError(AgentError):
    """Raised when an agent run exceeds its token, cost, or time budget."""


class LoopDetectedError(AgentError):
    """Raised when an infinite loop or thrashing pattern is detected."""


class ToolPermissionError(AgentError):
    """Raised when an agent attempts a forbidden tool call."""


class EscalationRequiredError(AgentError):
    """Raised to signal that human handoff is required."""


# Backwards-compatible alias
EscalationRequired = EscalationRequiredError


# ── Domain adapters ────────────────────────────────────────────────


class DomainAdapterError(SentinelError):
    """Raised when a domain adapter cannot be resolved or used."""


# ── Dashboard ──────────────────────────────────────────────────────


class DashboardError(SentinelError):
    """Raised when a dashboard route or view cannot be served."""


class DashboardNotInstalledError(DashboardError):
    """Raised when the dashboard extras (`sentinel[dashboard]`) are missing."""


class AuthorizationError(DashboardError):
    """Raised when a principal lacks the permission required for a route."""


class CSRFError(DashboardError):
    """Raised when a mutating request fails CSRF token verification."""


class RateLimitExceededError(DashboardError):
    """Raised when a request exceeds its configured rate-limit budget."""


class BearerTokenError(DashboardError):
    """Raised when a Bearer JWT fails validation."""
