"""Speaker identification via alias matching in transcripts."""

import logging
import re

log = logging.getLogger(__name__)


class SpeakerParser:
    def __init__(self, config):
        wake_cfg = config["wake"]
        self._alias_map = {}  # alias -> user_id
        self._display_names = {}  # user_id -> display_name
        for user_id, info in wake_cfg.get("speakers", {}).items():
            self._display_names[user_id] = info["display_name"]
            for alias in info["aliases"]:
                self._alias_map[alias.lower()] = user_id
        log.debug("Speaker aliases: %s", dict(self._alias_map))

    def parse(self, transcript: str) -> str | None:
        """Extract user_id from transcript, or None."""
        normalized = self._normalize(transcript)
        for alias in sorted(self._alias_map, key=len, reverse=True):
            if alias in normalized:
                log.debug("Speaker matched: '%s' → %s (%s)",
                          alias, self._alias_map[alias], self._display_names.get(self._alias_map[alias]))
                return self._alias_map[alias]
        log.debug("No speaker found in: '%s'", normalized)
        return None

    def display_name(self, user_id: str) -> str | None:
        """Get display name for a user_id."""
        return self._display_names.get(user_id)

    def speaker_names_list(self) -> str:
        """Return 'Name1 oder Name2' string for prompts."""
        names = list(self._display_names.values())
        if len(names) == 2:
            return f"{names[0]} oder {names[1]}"
        return ", ".join(names[:-1]) + f" oder {names[-1]}" if names else ""

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[.,!?;:\-\'"]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
