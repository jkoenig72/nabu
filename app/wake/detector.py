"""Wake word detection via fuzzy transcript matching."""

import logging
import re

log = logging.getLogger(__name__)

_NABU_SOUNDS = [
    "nabu", "nabo", "na bu", "na boo", "na boom", "naboo", "na boh",
    "nach oben", "na gut", "na pu", "na buh", "na wu", "na woo",
    "na mu", "na do", "na bo", "nah bu",
]

_OK_PATTERNS = ["ok", "okay", "o k"]


class WakeWordDetector:
    def __init__(self, config):
        wake_cfg = config["wake"]
        self.phrases = [p.lower() for p in wake_cfg.get("phrases", [])]

        self._wake_patterns = []
        for ok in _OK_PATTERNS:
            for nabu in _NABU_SOUNDS:
                self._wake_patterns.append(f"{ok} {nabu}")
        for phrase in self.phrases:
            if phrase not in self._wake_patterns:
                self._wake_patterns.append(phrase)

        log.debug("Wake patterns: %d total (%d configured + generated)",
                  len(self._wake_patterns), len(self.phrases))

    def check(self, transcript: str) -> bool:
        """Return True if transcript contains the wake phrase."""
        normalized = self._normalize(transcript)
        for pattern in self._wake_patterns:
            if pattern in normalized:
                log.debug("Wake MATCH: '%s' found in '%s'", pattern, normalized)
                return True
        log.debug("Wake check: no match in '%s'", normalized)
        return False

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[.,!?;:\-\'"]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
