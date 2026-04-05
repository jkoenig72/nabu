"""Tavily web search client. Only external network dependency."""

import logging
import httpx

log = logging.getLogger(__name__)


class TavilyError(Exception):
    pass


class TavilyClient:
    SEARCH_URL = "https://api.tavily.com/search"

    def __init__(self, config):
        search_cfg = config.get("web_search", {})
        self.api_key = search_cfg.get("api_key")
        self.max_results = search_cfg.get("max_results", 3)
        self.search_depth = search_cfg.get("search_depth", "basic")
        self.timeout = search_cfg.get("timeout", 8.0)
        self._enabled = bool(self.api_key)

        if not self._enabled:
            log.warning("Tavily API key not configured — web search disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def search(self, query: str) -> str:
        """Return formatted search results string for LLM context."""
        import time
        if not self._enabled:
            return ""

        log.debug("Tavily request: query='%s', depth=%s, max=%d",
                  query, self.search_depth, self.max_results)
        t0 = time.monotonic()

        try:
            response = httpx.post(
                self.SEARCH_URL,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": self.search_depth,
                    "max_results": self.max_results,
                    "include_answer": True,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as e:
            log.error("Tavily timeout after %.1fs", time.monotonic() - t0)
            raise TavilyError(f"Search timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise TavilyError(f"Search HTTP error {e.response.status_code}") from e
        except httpx.ConnectError as e:
            raise TavilyError(f"Cannot connect to Tavily: {e}") from e
        except Exception as e:
            raise TavilyError(f"Search failed: {e}") from e

        elapsed = time.monotonic() - t0
        n_results = len(data.get("results", []))
        answer = data.get("answer", "")
        log.debug("Tavily response in %.2fs: %d results, answer=%d chars",
                  elapsed, n_results, len(answer))
        log.debug("Tavily answer: '%s'", answer[:150] if answer else "(none)")

        return self._format_results(data)

    @staticmethod
    def _format_results(data: dict) -> str:
        parts = []

        answer = data.get("answer")
        if answer:
            parts.append(f"Zusammenfassung: {answer}")

        results = data.get("results", [])
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            if content:
                parts.append(f"[{i}] {title}: {content}")

        if not parts:
            return "Keine Ergebnisse gefunden."

        return "\n".join(parts)
