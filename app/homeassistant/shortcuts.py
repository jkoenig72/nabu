"""Voice shortcuts that execute HA commands directly from wake phrase."""

import logging
import re

log = logging.getLogger(__name__)


class ShortcutHandler:

    def __init__(self, config, ha_client):
        self._ha = ha_client
        self._shortcuts = []

        for shortcut in config.get("homeassistant", {}).get("shortcuts", []):
            patterns = [re.compile(p, re.IGNORECASE) for p in shortcut["patterns"]]
            self._shortcuts.append({
                "name": shortcut["name"],
                "patterns": patterns,
                "entity_id": shortcut["entity_id"],
                "domain": shortcut["domain"],
                "service_on": shortcut.get("service_on", "turn_on"),
                "service_off": shortcut.get("service_off", "turn_off"),
                "response_on": shortcut.get("response_on", f"{shortcut['name']} eingeschaltet."),
                "response_off": shortcut.get("response_off", f"{shortcut['name']} ausgeschaltet."),
            })

        log.debug("Loaded %d HA shortcuts", len(self._shortcuts))

    def check(self, transcript: str) -> dict | None:
        """Return {"response": str} if shortcut matched and executed, else None."""
        if not self._ha or not self._ha.enabled:
            return None

        normalized = transcript.lower()
        normalized = re.sub(r'[.,!?;:\-\'"]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        for shortcut in self._shortcuts:
            for pattern in shortcut["patterns"]:
                match = pattern.search(normalized)
                if match:
                    return self._execute(shortcut, normalized)

        return None

    def _execute(self, shortcut: dict, text: str) -> dict:
        is_off = any(w in text for w in ["aus", "abschalt", "deaktiv", "stopp"])
        service = shortcut["service_off"] if is_off else shortcut["service_on"]
        action = "off" if is_off else "on"

        log.info("Shortcut: %s → %s/%s (%s)",
                 shortcut["name"], shortcut["domain"], service, shortcut["entity_id"])

        success = self._ha.call_service(shortcut["domain"], service, shortcut["entity_id"])

        if success:
            response = shortcut["response_off"] if is_off else shortcut["response_on"]
        else:
            response = f"Fehler beim Schalten von {shortcut['name']}."

        return {"response": response}
