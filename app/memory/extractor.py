"""LLM-based fact extraction and retrieval from conversations."""

import json
import logging

log = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
Extrahiere persönliche Fakten aus dieser Nachricht.
Nur konkrete, merkbare Informationen (Namen, Termine, Vorlieben, Adressen, Geburtstage, Gewohnheiten).
Antworte NUR mit einem JSON-Array oder dem Wort KEINE wenn nichts zu merken ist.
Format: [{"subject": "Name/Thema", "fact": "konkreter Fakt", "category": "schedule|preference|family|health|general"}]

Beispiele:
- "Isa hat montags Yoga" → [{"subject": "Isa", "fact": "hat montags Yoga", "category": "schedule"}]
- "Ich trinke gerne Cappuccino" → [{"subject": "{user}", "fact": "trinkt gerne Cappuccino", "category": "preference"}]
- "Wie wird das Wetter?" → KEINE
"""

MEMORY_CONTEXT_HEADER = "Bekannte Informationen über den Haushalt:"


class MemoryExtractor:

    def __init__(self, sqlite_store, vector_store):
        self._sqlite = sqlite_store
        self._vector = vector_store

    @property
    def enabled(self) -> bool:
        return self._sqlite is not None

    def delete_all_for_user(self, user_id: str):
        """Delete all memories for a user from both stores."""
        if self._sqlite:
            self._sqlite.delete_all_for_user(user_id)
        if self._vector:
            self._vector.delete_all_for_user(user_id)

    def extract_and_store(self, user_id: str, user_message: str,
                          assistant_response: str, llm,
                          display_name: str = None) -> list[dict]:
        """Extract facts from a user message and store them."""
        if not self.enabled or not llm:
            return []

        prompt = EXTRACTION_PROMPT.replace("{user}", display_name or user_id)

        log.debug("Memory extraction for user=%s: '%s'", user_id, user_message[:100])

        try:
            response = llm.complete_sync(
                system_prompt=prompt,
                messages=[{"role": "user", "content": user_message}],
                max_tokens=200,
                temperature=0.1,
            )
        except Exception as e:
            log.warning("Memory extraction LLM call failed: %s", e)
            return []

        log.debug("Memory extraction LLM response: '%s'", response[:150])
        return self._parse_and_store(user_id, response, user_message)

    def _parse_and_store(self, user_id: str, llm_response: str,
                         source_message: str) -> list[dict]:
        text = llm_response.strip()

        if text.upper() in ("KEINE", "KEINE.", "[]", ""):
            return []

        if "```" in text:
            parts = text.split("```")
            for part in parts[1:]:
                lines = part.strip().split("\n")
                if lines[0].strip().lower() in ("json", ""):
                    lines = lines[1:]
                candidate = "\n".join(lines).strip()
                if candidate.startswith("["):
                    text = candidate
                    break

        try:
            facts = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    facts = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    log.debug("Could not parse extraction response: %s", text[:100])
                    return []
            else:
                log.debug("No JSON array in extraction response: %s", text[:100])
                return []

        if not isinstance(facts, list):
            return []

        stored = []
        for fact_data in facts:
            if not isinstance(fact_data, dict):
                continue
            subject = fact_data.get("subject", "").strip()
            fact = fact_data.get("fact", "").strip()
            category = fact_data.get("category", "general").strip()

            if not subject or not fact:
                continue

            valid_categories = ("schedule", "preference", "family", "health", "general")
            if category not in valid_categories:
                category = "general"

            memory_id = self._sqlite.add(
                user_id=user_id,
                subject=subject,
                fact=fact,
                category=category,
                source_message=source_message,
            )

            if memory_id is not None:
                full_text = f"{subject}: {fact}"
                try:
                    self._vector.add(memory_id, user_id, full_text)
                except Exception as e:
                    log.warning("Vector store failed for memory #%d: %s", memory_id, e)

                stored.append({"subject": subject, "fact": fact, "category": category})

        return stored

    def retrieve_relevant(self, query: str, user_id: str = None,
                          limit: int = 5) -> str:
        """Search memories and return formatted context, or empty string."""
        if not self.enabled:
            return ""

        try:
            results = self._vector.search(query, user_id=user_id, limit=limit)
        except Exception as e:
            log.warning("Vector search failed: %s", e)
            results = []

        if not results:
            try:
                sql_results = self._sqlite.search_by_text(query)
                if sql_results:
                    results = [{"text": f"{r['subject']}: {r['fact']}"} for r in sql_results[:limit]]
            except Exception:
                pass

        if not results:
            log.debug("Memory search: no results for '%s'", query[:80])
            return ""

        log.debug("Memory search: %d results for '%s'", len(results), query[:80])
        lines = [MEMORY_CONTEXT_HEADER]
        for r in results:
            text = r.get("text", "")
            if text:
                lines.append(f"- {text}")
                log.debug("  Memory hit: '%s' (score=%.3f)", text[:60], r.get("score", 0))

        return "\n".join(lines)
