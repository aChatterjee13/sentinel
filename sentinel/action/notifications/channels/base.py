"""Abstract notification channel."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import structlog

from sentinel.core.types import Alert, DeliveryResult

log = structlog.get_logger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds: 2, 4, 8

DEFAULT_TEMPLATE = """\
*[{{ severity }}]* {{ title }}
*Model:* {{ model_name }}
*Source:* {{ source }}

{{ body }}
{% if payload %}
*Details:*
{% for k, v in payload.items() %}
- {{ k }}: {{ v }}
{% endfor %}
{% endif %}
"""


class BaseChannel(ABC):
    """A notification channel — Slack, Teams, PagerDuty, etc.

    Concrete channels override `send` and (optionally) `format_message`.
    The engine should call `send_with_retry` to get automatic exponential
    backoff on transient failures.
    """

    name: str = "base"

    def __init__(self, **config: Any):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.max_retries = int(config.get("max_retries", MAX_RETRIES))
        self._template = self._compile_template(config.get("template"))

    @abstractmethod
    def send(self, alert: Alert) -> DeliveryResult:
        """Deliver an alert. Must not raise — return DeliveryResult instead."""

    def send_with_retry(self, alert: Alert) -> DeliveryResult:
        """Send with exponential backoff retry on transient failures.

        Retries up to ``max_retries`` times with exponential backoff
        (2s, 4s, 8s). Only retries when ``DeliveryResult.delivered``
        is False and an error string is present (indicating a transient
        failure rather than a permanent config issue like "channel disabled").

        Args:
            alert: The alert to deliver.

        Returns:
            The final DeliveryResult after all attempts.
        """
        last_result = DeliveryResult(channel=self.name, delivered=False, error="no attempt made")
        for attempt in range(self.max_retries + 1):
            result = self.send(alert)
            if result.delivered:
                if attempt > 0:
                    log.info(
                        "notification.retry_succeeded",
                        channel=self.name,
                        attempt=attempt + 1,
                    )
                return result
            last_result = result
            # Don't retry permanent failures (disabled channel, missing config)
            if result.error and "disabled" in result.error.lower():
                return result
            if attempt < self.max_retries:
                backoff = RETRY_BACKOFF_BASE ** (attempt + 1)
                log.warning(
                    "notification.retry",
                    channel=self.name,
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    backoff_s=backoff,
                    error=result.error,
                )
                time.sleep(backoff)
        log.error(
            "notification.all_retries_failed",
            channel=self.name,
            max_retries=self.max_retries,
            last_error=last_result.error,
        )
        return last_result

    def _compile_template(self, template_str: str | None) -> Any:
        """Compile a Jinja2 template, or return ``None`` if unavailable."""
        if template_str is None:
            return None
        try:
            from jinja2 import Template

            return Template(template_str)
        except ImportError:
            log.warning("notification.jinja2_not_installed", channel=self.name)
            return None

    def format_message(self, alert: Alert) -> str:
        """Render the alert using a Jinja2 template (if set), else hardcoded."""
        if self._template is not None:
            try:
                return self._template.render(
                    alert_id=alert.alert_id,
                    model_name=alert.model_name,
                    title=alert.title,
                    body=alert.body,
                    severity=alert.severity.value.upper(),
                    source=alert.source,
                    timestamp=alert.timestamp.isoformat(),
                    payload=alert.payload,
                    fingerprint=alert.fingerprint or "",
                )
            except Exception:
                log.warning(
                    "notification.template_render_failed",
                    channel=self.name,
                    exc_info=True,
                )
        return self._format_default(alert)

    def _format_default(self, alert: Alert) -> str:
        """Original hardcoded Markdown-style format (fallback)."""
        body_lines = [
            f"*[{alert.severity.value.upper()}]* {alert.title}",
            f"*Model:* {alert.model_name}",
            f"*Source:* {alert.source}",
            "",
            alert.body,
        ]
        if alert.payload:
            body_lines.append("")
            body_lines.append("*Details:*")
            for k, v in list(alert.payload.items())[:10]:
                body_lines.append(f"- {k}: {v}")
        return "\n".join(body_lines)
