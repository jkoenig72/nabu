"""Regex-based intent classification for German voice commands."""

import re
import logging

log = logging.getLogger(__name__)

# Default keyword patterns per intent (German)
_DEFAULT_KEYWORDS = {
    "time_date": [
        r"wie sp[aä]e?t",
        r"uhrzeit",
        r"welcher tag",
        r"welches datum",
        r"wochentag",
        r"welcher monat",
        r"wie viel uhr",
    ],
    "home_control": [
        r"licht",
        r"lampe",
        r"heizung",
        r"thermostat",
        r"temperatur i[mn]",
        r"rollladen",
        r"rolladen",
        r"jalousie",
        r"schalte",
        r"mach.{0,20}an\b",
        r"mach.{0,20}aus\b",
        r"kaffeemaschine",
        r"steckdose",
        r"ventilator",
    ],
    "web_search": [
        r"wetter morgen",
        r"wetter heute",
        r"wie wird das wetter",
        r"wer hat.{0,20}gewonnen",
        r"aktuelle.{0,5}nachrichten",
        r"nachrichten",
        r"aktuell.{0,10}news",
        r"was kostet",
        r"aktienkurs",
        r"such.{0,5}(im internet|im web|online)",
    ],
    "system": [
        r"wer bin ich",
        r"was kannst du",
        r"was bist du",
        r"wie hei[sß]{1,2}t du",
        r"welche funktionen",
        r"\bhilfe\b",
    ],
    "memory_store": [
        r"merk dir",
        r"merke dir",
        r"vergiss nicht",
        r"erinnere dich",
        r"speicher",
    ],
    "memory_query": [
        r"was wei[sß]{1,2}t du [uü]ber",
        r"erinnerst du dich",
        r"was hast du .{0,20}gemerkt",
        r"kennst du .{0,10}(termin|zeitplan|vorlieb)",
    ],
    "delete_conversations": [
        r"konversationen? l[oö]schen",
        r"gespr[aä]che? l[oö]schen",
        r"alles l[oö]schen",
        r"unterhaltungen? l[oö]schen",
        r"verlauf l[oö]schen",
        r"historie l[oö]schen",
        r"zur[uü]cksetzen",
        r"reset",
    ],
    "volume_control": [
        r"lauter",
        r"leiser",
        r"lautst[aä]rke",
        r"sprich lauter",
        r"sprich leiser",
        r"kannst du lauter",
        r"kannst du leiser",
        r"etwas lauter",
        r"etwas leiser",
        r"bisschen lauter",
        r"bisschen leiser",
        r"zu laut",
        r"zu leise",
        r"volume",
    ],
    "end_conversation": [
        r"gespr[aä]ch beenden",
        r"tsch[uü][sß]",
        r"\bbye\b",
        r"\bciao\b",
        r"das war.{0,5}s\b",
        r"danke.{0,10}das reicht",
        r"auf wiedersehen",
        r"bis sp[aä]ter",
        r"ich bin fertig",
    ],
}

# Most specific first; general_chat is the implicit fallback
_INTENT_ORDER = ["end_conversation", "delete_conversations", "volume_control", "time_date", "home_control", "web_search", "system", "memory_store", "memory_query"]


class IntentRouter:
    def __init__(self, config=None):
        intent_cfg = {}
        if config:
            intent_cfg = config.get("intents", {}).get("keywords", {})

        self._patterns = {}
        for intent in _INTENT_ORDER:
            keywords = intent_cfg.get(intent, _DEFAULT_KEYWORDS.get(intent, []))
            # Compile all keywords for this intent into a single alternation regex
            if keywords:
                combined = "|".join(f"(?:{k})" for k in keywords)
                self._patterns[intent] = re.compile(combined, re.IGNORECASE)

    def classify(self, transcript: str) -> str:
        """Classify transcript into an intent string. Default: 'general_chat'."""
        if not transcript or not transcript.strip():
            return "general_chat"

        normalized = self._normalize(transcript)

        for intent in _INTENT_ORDER:
            pattern = self._patterns.get(intent)
            if pattern:
                match = pattern.search(normalized)
                if match:
                    log.debug("Intent '%s' matched pattern '%s' in: '%s'",
                              intent, match.group(), normalized[:80])
                    return intent

        log.debug("Intent fallback 'general_chat' for: '%s'", normalized[:80])
        return "general_chat"

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[.,!?;:\-\'"]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
