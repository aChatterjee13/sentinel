"""Config-as-code layer.

Every Sentinel behaviour is driven by a YAML/JSON config file that is
version-controlled alongside the model code.
"""

from sentinel.config.defaults import default_config
from sentinel.config.loader import ConfigLoader, load_config
from sentinel.config.references import (
    ReferenceIssue,
    ReferenceSeverity,
    validate_file_references,
)
from sentinel.config.schema import (
    AgentOpsConfig,
    AlertsConfig,
    AuditConfig,
    DataQualityConfig,
    DeploymentConfig,
    DomainConfig,
    DriftConfig,
    LLMOpsConfig,
    ModelConfig,
    SentinelConfig,
)
from sentinel.config.secrets import REDACTED, Secret, masked_dump, unwrap
from sentinel.config.source import ConfigSource, SourceMap
from sentinel.core.exceptions import (
    ConfigCircularInheritanceError,
    ConfigFileReferenceError,
    ConfigMissingEnvVarError,
    ConfigNotFoundError,
    ConfigValidationError,
)

__all__ = [  # noqa: RUF022 — grouped by topic, not alphabetically
    "ConfigLoader",
    "load_config",
    "default_config",
    "SentinelConfig",
    "ModelConfig",
    "DriftConfig",
    "DataQualityConfig",
    "AlertsConfig",
    "DeploymentConfig",
    "AuditConfig",
    "LLMOpsConfig",
    "AgentOpsConfig",
    "DomainConfig",
    # Secrets
    "Secret",
    "unwrap",
    "masked_dump",
    "REDACTED",
    # File references
    "ReferenceIssue",
    "ReferenceSeverity",
    "validate_file_references",
    # Source tracing
    "ConfigSource",
    "SourceMap",
    # Errors
    "ConfigValidationError",
    "ConfigNotFoundError",
    "ConfigMissingEnvVarError",
    "ConfigCircularInheritanceError",
    "ConfigFileReferenceError",
]
