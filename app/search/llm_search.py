"""Search-augmented and no-search prompt templates for LLM."""

import logging
import re

log = logging.getLogger(__name__)

_SEARCH_PATTERN = re.compile(r'\[SEARCH:\s*(.+?)\]', re.IGNORECASE)

SEARCH_INSTRUCTIONS = """\
Dir stehen aktuelle Suchergebnisse zur Verfügung. Nutze diese als Grundlage
für deine Antwort. Wenn die Suchergebnisse relevant sind, basiere deine
Antwort darauf. Wenn nicht, antworte aus deinem Wissen, aber sage ehrlich
wenn du unsicher bist.

Suchergebnisse:
{results}"""

NOSEARCH_INSTRUCTIONS = """\
WICHTIG: Wenn du dir bei einer Antwort nicht sicher bist oder die
Information aktuell sein muss (Wetter, Nachrichten, Preise, Ergebnisse),
sage ehrlich: 'Das weiß ich leider nicht sicher' statt zu raten."""


def extract_search_query(llm_response: str) -> str | None:
    """Extract query from [SEARCH: ...] tag in LLM response, or None."""
    match = _SEARCH_PATTERN.search(llm_response)
    if match:
        return match.group(1).strip()
    return None


def build_search_prompt(search_results: str, display_name: str = None,
                        memory_context: str = "",
                        base_system_prompt: str = "") -> str:
    """Build system prompt with search results for grounded response."""
    parts = []
    if base_system_prompt:
        parts.append(base_system_prompt.rstrip())
    parts.append(SEARCH_INSTRUCTIONS.format(results=search_results))
    if memory_context:
        parts.append(memory_context)
    if display_name:
        parts.append(f"Der aktuelle Benutzer ist {display_name}.")
    prompt = "\n\n".join(parts)
    log.debug("Built search-augmented prompt: %d chars, memory=%s, user=%s",
              len(prompt), bool(memory_context), display_name)
    return prompt


def build_nosearch_prompt(display_name: str = None,
                          memory_context: str = "",
                          base_system_prompt: str = "") -> str:
    """Build system prompt for when no search results are available."""
    parts = []
    if base_system_prompt:
        parts.append(base_system_prompt.rstrip())
    parts.append(NOSEARCH_INSTRUCTIONS)
    if memory_context:
        parts.append(memory_context)
    if display_name:
        parts.append(f"Der aktuelle Benutzer ist {display_name}.")
    prompt = "\n\n".join(parts)
    log.debug("Built no-search prompt: %d chars, memory=%s, user=%s",
              len(prompt), bool(memory_context), display_name)
    return prompt
