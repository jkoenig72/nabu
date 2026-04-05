"""Home Assistant REST API client for service calls and state queries."""

import logging

import httpx

log = logging.getLogger(__name__)


class HAClient:

    def __init__(self, config):
        ha_cfg = config.get("homeassistant", {})
        self._url = ha_cfg.get("url", "").rstrip("/")
        self._token = ha_cfg.get("token", "")
        self._timeout = ha_cfg.get("timeout", 5.0)
        self._enabled = bool(self._url and self._token)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def call_service(self, domain: str, service: str, entity_id: str) -> bool:
        """Call a HA service. Returns True on success."""
        if not self._enabled:
            log.warning("HA not configured")
            return False

        url = f"{self._url}/api/services/{domain}/{service}"
        log.debug("HA call: %s/%s → %s", domain, service, entity_id)

        try:
            resp = httpx.post(
                url,
                json={"entity_id": entity_id},
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            log.debug("HA response: %d", resp.status_code)
            return True
        except Exception as e:
            log.error("HA service call failed: %s", e)
            return False

    def get_state(self, entity_id: str) -> str | None:
        """Get current state of an entity."""
        if not self._enabled:
            return None

        url = f"{self._url}/api/states/{entity_id}"
        try:
            resp = httpx.get(
                url,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("state")
        except Exception as e:
            log.error("HA get_state failed: %s", e)
            return None
