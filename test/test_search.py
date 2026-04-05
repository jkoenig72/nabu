"""Tests for web search: Tavily client, LLM search extraction, handlers."""

from unittest.mock import patch, MagicMock

import pytest

from app.search.tavily import TavilyClient, TavilyError
from app.search.llm_search import extract_search_query, build_search_prompt, build_nosearch_prompt
from app.intent.handlers import handle_web_search


# --- Search query extraction ---

class TestExtractSearchQuery:
    def test_basic_search_tag(self):
        assert extract_search_query("[SEARCH: Wetter München morgen]") == "Wetter München morgen"

    def test_case_insensitive(self):
        assert extract_search_query("[search: test query]") == "test query"

    def test_with_surrounding_text(self):
        result = extract_search_query("Ich bin unsicher. [SEARCH: Bayern München Ergebnis] Mal schauen.")
        assert result == "Bayern München Ergebnis"

    def test_no_search_tag(self):
        assert extract_search_query("Das ist eine normale Antwort.") is None

    def test_empty_string(self):
        assert extract_search_query("") is None

    def test_whitespace_in_tag(self):
        assert extract_search_query("[SEARCH:   Wahlergebnisse BW  ]") == "Wahlergebnisse BW"

    def test_nosearch_prompt_warns_about_uncertainty(self):
        prompt = build_nosearch_prompt("Jörg")
        assert "nicht sicher" in prompt
        assert "Jörg" in prompt


class TestBuildSearchPrompt:
    def test_contains_results(self):
        prompt = build_search_prompt("Morgen 22 Grad und sonnig.")
        assert "Morgen 22 Grad" in prompt
        assert "Suchergebnisse" in prompt


# --- Tavily client ---

class TestTavilyClient:
    def test_disabled_without_key(self):
        client = TavilyClient({"web_search": {}})
        assert client.enabled is False

    def test_enabled_with_key(self):
        client = TavilyClient({"web_search": {"api_key": "test-key"}})
        assert client.enabled is True

    def test_search_disabled_returns_empty(self):
        client = TavilyClient({"web_search": {}})
        assert client.search("test") == ""

    @patch("app.search.tavily.httpx.post")
    def test_search_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "answer": "Es wird sonnig bei 22 Grad.",
            "results": [
                {"title": "Wetter.de", "content": "Sonnig, 22°C", "url": "https://wetter.de"}
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = TavilyClient({"web_search": {"api_key": "test-key"}})
        result = client.search("Wetter München")
        assert "sonnig" in result.lower() or "22" in result

    @patch("app.search.tavily.httpx.post")
    def test_search_formats_results(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "answer": "Summary answer",
            "results": [
                {"title": "Source 1", "content": "Content 1", "url": "https://a.com"},
                {"title": "Source 2", "content": "Content 2", "url": "https://b.com"},
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = TavilyClient({"web_search": {"api_key": "test-key"}})
        result = client.search("test")
        assert "Zusammenfassung: Summary answer" in result
        assert "[1] Source 1" in result
        assert "[2] Source 2" in result

    @patch("app.search.tavily.httpx.post")
    def test_search_empty_results(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = TavilyClient({"web_search": {"api_key": "test-key"}})
        result = client.search("nonexistent query")
        assert "Keine Ergebnisse" in result


# --- Web search handler ---

class TestHandleWebSearch:
    def test_no_tavily_returns_not_configured(self):
        result = handle_web_search("Wetter morgen")
        assert "nicht konfiguriert" in result

    def test_disabled_tavily(self):
        mock_tavily = MagicMock()
        mock_tavily.enabled = False
        result = handle_web_search("Wetter", tavily=mock_tavily)
        assert "nicht konfiguriert" in result

    def test_tavily_error_handled(self):
        mock_tavily = MagicMock()
        mock_tavily.enabled = True
        mock_tavily.search.side_effect = TavilyError("timeout")
        result = handle_web_search("Wetter", tavily=mock_tavily)
        assert "nicht funktioniert" in result

    def test_success_with_llm_summary(self):
        mock_tavily = MagicMock()
        mock_tavily.enabled = True
        mock_tavily.search.return_value = "Zusammenfassung: Morgen sonnig.\n[1] Wetter.de: 22 Grad"

        mock_llm = MagicMock()
        mock_llm.complete_sync.return_value = "Morgen wird es sonnig bei 22 Grad."

        result = handle_web_search("Wie wird das Wetter morgen?", tavily=mock_tavily, llm=mock_llm)
        assert result == "Morgen wird es sonnig bei 22 Grad."

    def test_fallback_to_tavily_answer_if_llm_fails(self):
        mock_tavily = MagicMock()
        mock_tavily.enabled = True
        mock_tavily.search.return_value = "Zusammenfassung: Morgen sonnig."

        mock_llm = MagicMock()
        mock_llm.complete_sync.side_effect = Exception("LLM down")

        result = handle_web_search("Wetter morgen", tavily=mock_tavily, llm=mock_llm)
        assert "Morgen sonnig" in result


# --- Live Tavily test (requires network + API key) ---

class TestTavilyLive:
    @pytest.mark.network
    def test_live_search(self):
        """Live test against real Tavily API."""
        config = {
            "web_search": {
                "api_key": "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",
                "max_results": 2,
                "search_depth": "basic",
                "timeout": 10.0,
            }
        }
        client = TavilyClient(config)
        result = client.search("Hauptstadt von Deutschland")
        assert "Berlin" in result
