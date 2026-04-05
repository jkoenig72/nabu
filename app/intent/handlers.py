"""Response handlers for each intent type."""

import logging
import re
from datetime import datetime

log = logging.getLogger(__name__)

_WOCHENTAGE = [
    "Montag", "Dienstag", "Mittwoch", "Donnerstag",
    "Freitag", "Samstag", "Sonntag",
]
_MONATE = [
    "Januar", "Februar", "März", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",
]

_CAPABILITIES_PROMPT = """\
Du bist Nabu, ein lokaler Sprachassistent. Erkläre kurz deine Fähigkeiten:
- Allgemeine Fragen beantworten und Gespräche führen
- Uhrzeit und Datum nennen
- Smart-Home-Steuerung (kommt bald)
- Websuche (kommt bald)
Antworte in 2-3 Sätzen auf Deutsch."""


def handle_time_date(command_text: str, **kwargs) -> str:
    """Answer time/date questions from system clock."""
    log.debug("handle_time_date: '%s'", command_text[:80])
    now = datetime.now()
    text_lower = command_text.lower()

    wants_time = any(w in text_lower for w in ["spät", "uhr", "uhrzeit"])
    wants_date = any(w in text_lower for w in ["datum", "welcher tag", "wochentag", "monat"])

    # Ambiguous: give both
    if not wants_time and not wants_date:
        wants_time = True
        wants_date = True

    parts = []
    if wants_time:
        minute = now.minute
        if minute == 0:
            parts.append(f"Es ist {now.hour} Uhr")
        else:
            parts.append(f"Es ist {now.hour} Uhr {minute}")

    if wants_date:
        wochentag = _WOCHENTAGE[now.weekday()]
        monat = _MONATE[now.month - 1]
        parts.append(f"Heute ist {wochentag}, der {now.day}. {monat} {now.year}")

    return ". ".join(parts) + "."


def handle_home_control(command_text: str, **kwargs) -> str:
    """Stub for in-conversation HA control. Shortcuts work from wake phrase."""
    return "Home Assistant ist noch nicht verbunden. Diese Funktion kommt bald."


def handle_web_search(command_text: str, tavily=None, llm=None, **kwargs) -> str:
    """Search via Tavily, summarize with LLM."""
    log.debug("handle_web_search: '%s'", command_text[:80])
    if not tavily or not tavily.enabled:
        return "Die Websuche ist nicht konfiguriert."

    from app.search.tavily import TavilyError
    from app.search.llm_search import build_search_prompt

    query = command_text
    log.info("Web search: '%s'", query)

    try:
        results = tavily.search(query)
    except TavilyError as e:
        log.error("Tavily error: %s", e)
        return "Die Websuche hat leider nicht funktioniert. Versuche es später nochmal."

    if not results or results == "Keine Ergebnisse gefunden.":
        return "Ich konnte leider keine Ergebnisse finden."

    if llm:
        try:
            from app.llm.client import LLMError
            search_prompt = build_search_prompt(
                results, base_system_prompt=kwargs.get("base_system_prompt", ""),
            )
            response = llm.complete_sync(
                system_prompt=search_prompt,
                messages=[{"role": "user", "content": command_text}],
                max_tokens=200,
                temperature=0.3,
            )
            return response
        except Exception as e:
            log.error("LLM summary failed: %s", e)

    for line in results.split("\n"):
        if line.startswith("Zusammenfassung:"):
            return line.replace("Zusammenfassung: ", "")
    return results[:300]


def handle_system(command_text: str, display_name: str = None, llm=None, **kwargs) -> str:
    """Answer meta-questions (who am I, what can you do)."""
    log.debug("handle_system: '%s' (user=%s)", command_text[:80], display_name)
    text_lower = command_text.lower()

    if "wer bin ich" in text_lower:
        if display_name:
            return f"Du bist {display_name}."
        return "Das konnte ich leider nicht feststellen."

    if re.search(r"wie hei[sß]t du", text_lower):
        return "Ich bin Nabu, dein lokaler Sprachassistent."

    if "was bist du" in text_lower:
        return "Ich bin Nabu, ein privater Sprachassistent der komplett lokal läuft. Keine Cloud, keine Daten die das Haus verlassen."

    if llm:
        try:
            from app.llm.client import LLMError
            response = llm.complete_sync(
                system_prompt=_CAPABILITIES_PROMPT,
                messages=[{"role": "user", "content": command_text}],
                max_tokens=100,
                temperature=0.5,
            )
            return response
        except Exception:
            pass

    return "Ich bin Nabu. Ich kann Fragen beantworten, die Uhrzeit sagen, und bald auch dein Smart Home steuern."


def handle_volume_control(command_text: str, playback=None, **kwargs) -> str:
    """Adjust speaker volume."""
    log.debug("handle_volume_control: '%s'", command_text[:80])
    if not playback:
        return "Die Lautstärkeregelung ist nicht verfügbar."

    text_lower = command_text.lower()

    if any(w in text_lower for w in ["leiser", "zu laut"]):
        new_vol = playback.volume_down()
        pct = int(new_vol * 100)
        if new_vol <= playback.VOLUME_MIN:
            return f"Ich bin jetzt auf dem Minimum, {pct} Prozent."
        return f"Alles klar, ich spreche jetzt leiser. Lautstärke {pct} Prozent."

    if any(w in text_lower for w in ["lauter", "zu leise"]):
        new_vol = playback.volume_up()
        pct = int(new_vol * 100)
        if new_vol >= playback.VOLUME_MAX:
            return f"Ich bin schon auf Maximum, {pct} Prozent."
        return f"Alles klar, ich spreche jetzt lauter. Lautstärke {pct} Prozent."

    pct = int(playback.volume * 100)
    return f"Meine Lautstärke ist gerade bei {pct} Prozent. Sag lauter oder leiser um sie zu ändern."


def handle_memory_store(command_text: str, extractor=None, llm=None,
                        user_id: str = None, display_name: str = None, **kwargs) -> str:
    """Extract and store a fact the user asks Nabu to remember."""
    log.debug("handle_memory_store: '%s' (user=%s)", command_text[:80], user_id)
    if not extractor or not extractor.enabled:
        return "Das Gedächtnis ist nicht aktiviert."

    if not llm:
        return "Der Sprachserver ist nicht erreichbar."

    try:
        stored = extractor.extract_and_store(
            user_id=user_id or "unknown",
            user_message=command_text,
            assistant_response="",
            llm=llm,
            display_name=display_name,
        )
    except Exception as e:
        log.error("Memory store failed: %s", e)
        return "Das konnte ich leider nicht speichern."

    if stored:
        facts = ", ".join(f"{s['subject']} {s['fact']}" for s in stored)
        return f"Alles klar, ich merke mir: {facts}"
    return "Ich konnte keine konkreten Fakten erkennen. Sag es bitte nochmal genauer."


def handle_memory_query(command_text: str, extractor=None,
                        user_id: str = None, **kwargs) -> str:
    """Retrieve and format relevant memories."""
    log.debug("handle_memory_query: '%s' (user=%s)", command_text[:80], user_id)
    if not extractor or not extractor.enabled:
        return "Das Gedächtnis ist nicht aktiviert."

    try:
        context = extractor.retrieve_relevant(command_text, user_id=user_id)
    except Exception as e:
        log.error("Memory query failed: %s", e)
        return "Beim Suchen im Gedächtnis ist ein Fehler aufgetreten."

    if context:
        # Format nicely for speech
        lines = context.split("\n")
        facts = [l.lstrip("- ") for l in lines[1:] if l.startswith("- ")]
        if facts:
            return "Hier ist was ich weiß: " + ". ".join(facts) + "."
    return "Dazu habe ich leider nichts gespeichert."
