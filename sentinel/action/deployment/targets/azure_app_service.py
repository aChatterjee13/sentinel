"""Azure App Service deployment target — slot swap for blue/green.

App Service slot traffic routing is brittle for % splits (it relies
on cookie-based routing with a 0-100 slider that behaves differently
per tier), so this target supports **blue/green only**. A canary
strategy paired with this target is rejected at config validation
time — see :class:`~sentinel.config.schema.DeploymentConfig`.

Blue/green is the slot swap model: a staging slot holds the new
version, health is checked via HTTP probe, and ``set_traffic_split``
triggers ``begin_swap_slot``.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from sentinel.action.deployment.targets.base import BaseDeploymentTarget
from sentinel.core.exceptions import DeploymentError

log = structlog.get_logger(__name__)


class AzureAppServiceTarget(BaseDeploymentTarget):
    """Azure App Service slot-swap deployment target.

    Requires the ``azure`` extra with ``azure-mgmt-web`` installed.
    """

    name = "azure_app_service"

    def __init__(
        self,
        *,
        subscription_id: str,
        resource_group: str,
        site_name: str,
        production_slot: str = "production",
        staging_slot: str = "staging",
        health_check_path: str = "/healthz",
        credential: Any = None,
        timeout_seconds: int = 300,
        health_check_timeout: int = 30,
    ) -> None:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.web import WebSiteManagementClient  # type: ignore[import-untyped]
        except ImportError as e:
            raise DeploymentError(
                "azure extra not installed — "
                "`pip install sentinel-mlops[azure]` (needs azure-mgmt-web)"
            ) from e

        self._resource_group = resource_group
        self._site_name = site_name
        self._production_slot = production_slot
        self._staging_slot = staging_slot
        self._health_check_path = health_check_path
        self._timeout = timeout_seconds
        self._health_timeout = health_check_timeout
        self._client = WebSiteManagementClient(
            credential=credential or DefaultAzureCredential(),
            subscription_id=subscription_id,
        )

    def set_traffic_split(self, model_name: str, weights: dict[str, int]) -> None:
        """Swap slots for blue/green. Only 0/100 splits are supported.

        A weights dict of ``{new_version: 100, old_version: 0}`` triggers
        a slot swap from staging to production. Anything else raises a
        :class:`DeploymentError`.
        """
        total = sum(weights.values())
        if total != 100:
            raise DeploymentError(f"traffic weights must sum to 100, got {total}")
        ramping = {v: w for v, w in weights.items() if 0 < w < 100}
        if ramping:
            raise DeploymentError(
                "azure_app_service only supports atomic 0/100 swaps — "
                f"received partial weights {weights}. Use a blue_green "
                "strategy, not canary."
            )
        try:
            poller = self._client.web_apps.begin_swap_slot(
                resource_group_name=self._resource_group,
                name=self._site_name,
                slot=self._staging_slot,
                target_slot=self._production_slot,
            )
            poller.result(timeout=self._timeout)
        except Exception as e:
            raise DeploymentError(
                f"azure_app_service swap_slot failed for {self._site_name}: {e}"
            ) from e
        log.info(
            "deployment.azure_app_service.swapped",
            site=self._site_name,
            from_slot=self._staging_slot,
            to_slot=self._production_slot,
        )

    def health_check(self, model_name: str, version: str) -> bool:
        """Probe the production slot's health endpoint over HTTPS."""
        hostname = f"{self._site_name}.azurewebsites.net"
        url = f"https://{hostname}{self._health_check_path}"
        try:
            resp = httpx.get(url, timeout=float(self._health_timeout))
        except Exception as e:
            log.warning(
                "deployment.azure_app_service.health_check_failed",
                url=url,
                error=str(e),
            )
            return False
        return 200 <= resp.status_code < 300

    def rollback_to(self, model_name: str, version: str) -> None:
        """Swap slots back — production becomes staging and vice versa.

        App Service slot swaps are symmetric, so the rollback is
        literally the same swap call in reverse.
        """
        try:
            poller = self._client.web_apps.begin_swap_slot(
                resource_group_name=self._resource_group,
                name=self._site_name,
                slot=self._production_slot,
                target_slot=self._staging_slot,
            )
            poller.result(timeout=self._timeout)
        except Exception as e:
            raise DeploymentError(
                f"azure_app_service rollback_to failed for {self._site_name}: {e}"
            ) from e
        log.info(
            "deployment.azure_app_service.rolled_back",
            site=self._site_name,
            version=version,
        )

    def describe(self, model_name: str) -> dict[str, Any]:
        return {
            "target": self.name,
            "site": self._site_name,
            "production_slot": self._production_slot,
            "staging_slot": self._staging_slot,
        }
