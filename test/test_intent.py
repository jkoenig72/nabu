"""Tests for intent routing and handlers."""

from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from app.intent.router import IntentRouter
from app.intent.handlers import (
    handle_time_date,
    handle_home_control,
    handle_web_search,
    handle_system,
)


@pytest.fixture
def router():
    return IntentRouter()


# --- IntentRouter tests ---

class TestTimeDateIntent:
    def test_wie_spaet(self, router):
        assert router.classify("Wie spät ist es?") == "time_date"

    def test_wie_spaet_no_umlaut(self, router):
        assert router.classify("Wie spaet ist es?") == "time_date"

    def test_uhrzeit(self, router):
        assert router.classify("Sag mir die Uhrzeit") == "time_date"

    def test_welcher_tag(self, router):
        assert router.classify("Welcher Tag ist heute?") == "time_date"

    def test_welches_datum(self, router):
        assert router.classify("Welches Datum haben wir?") == "time_date"

    def test_wochentag(self, router):
        assert router.classify("Welcher Wochentag ist heute?") == "time_date"

    def test_wie_viel_uhr(self, router):
        assert router.classify("Wie viel Uhr ist es?") == "time_date"


class TestHomeControlIntent:
    def test_licht_an(self, router):
        assert router.classify("Mach das Licht an") == "home_control"

    def test_licht_aus(self, router):
        assert router.classify("Mach das Licht aus") == "home_control"

    def test_lampe(self, router):
        assert router.classify("Schalte die Lampe im Wohnzimmer ein") == "home_control"

    def test_heizung(self, router):
        assert router.classify("Stell die Heizung auf 22 Grad") == "home_control"

    def test_thermostat(self, router):
        assert router.classify("Wie steht der Thermostat?") == "home_control"

    def test_temperatur_im(self, router):
        assert router.classify("Wie ist die Temperatur im Schlafzimmer?") == "home_control"

    def test_rollladen(self, router):
        assert router.classify("Mach die Rollladen runter") == "home_control"

    def test_jalousie(self, router):
        assert router.classify("Öffne die Jalousie") == "home_control"

    def test_schalte(self, router):
        assert router.classify("Schalte den Fernseher ein") == "home_control"

    def test_kaffeemaschine(self, router):
        assert router.classify("Mach die Kaffeemaschine an") == "home_control"

    def test_steckdose(self, router):
        assert router.classify("Schalte die Steckdose aus") == "home_control"


class TestWebSearchIntent:
    def test_wetter_morgen(self, router):
        assert router.classify("Wie wird das Wetter morgen?") == "web_search"

    def test_wetter_heute(self, router):
        assert router.classify("Wie ist das Wetter heute?") == "web_search"

    def test_nachrichten(self, router):
        assert router.classify("Was sind die aktuellen Nachrichten?") == "web_search"

    def test_aktienkurs(self, router):
        assert router.classify("Wie ist der Aktienkurs von Apple?") == "web_search"

    def test_was_kostet(self, router):
        assert router.classify("Was kostet ein Tesla Model 3?") == "web_search"


class TestSystemIntent:
    def test_wer_bin_ich(self, router):
        assert router.classify("Wer bin ich?") == "system"

    def test_was_kannst_du(self, router):
        assert router.classify("Was kannst du alles?") == "system"

    def test_was_bist_du(self, router):
        assert router.classify("Was bist du?") == "system"

    def test_wie_heisst_du(self, router):
        assert router.classify("Wie heißt du?") == "system"

    def test_wie_heisst_du_ss(self, router):
        assert router.classify("Wie heisst du?") == "system"

    def test_hilfe(self, router):
        assert router.classify("Hilfe!") == "system"


class TestEndConversationIntent:
    @pytest.fixture
    def router(self):
        return IntentRouter()

    def test_gespraech_beenden(self, router):
        assert router.classify("Ich möchte dieses Gespräch beenden") == "end_conversation"

    def test_tschuess(self, router):
        assert router.classify("Tschüss Nabu") == "end_conversation"

    def test_bye(self, router):
        assert router.classify("Bye bye") == "end_conversation"

    def test_ciao(self, router):
        assert router.classify("Ciao Nabu") == "end_conversation"

    def test_das_wars(self, router):
        assert router.classify("Das wars, danke") == "end_conversation"

    def test_auf_wiedersehen(self, router):
        assert router.classify("Auf Wiedersehen") == "end_conversation"

    def test_bis_spaeter(self, router):
        assert router.classify("Bis später!") == "end_conversation"

    def test_ich_bin_fertig(self, router):
        assert router.classify("Ich bin fertig") == "end_conversation"


class TestGeneralChatFallback:
    def test_joke(self, router):
        assert router.classify("Erzähl mir einen Witz") == "general_chat"

    def test_knowledge(self, router):
        assert router.classify("Was ist die Hauptstadt von Frankreich?") == "general_chat"

    def test_empty_string(self, router):
        assert router.classify("") == "general_chat"

    def test_random_text(self, router):
        assert router.classify("Banane Apfel Kirsche") == "general_chat"

    def test_abstract_temperature(self, router):
        """Bare 'Temperatur' without 'im/in' should fall through to general."""
        assert router.classify("Bei welcher Temperatur schmilzt Eis?") == "general_chat"


class TestRouterEdgeCases:
    def test_case_insensitive(self, router):
        assert router.classify("WIE SPÄT IST ES?") == "time_date"

    def test_punctuation_stripped(self, router):
        assert router.classify("Wie spät ist es!!!") == "time_date"

    def test_keyword_in_long_sentence(self, router):
        assert router.classify("Kannst du mir sagen wie spät es jetzt gerade ist?") == "time_date"

    def test_config_override(self):
        config = {
            "intents": {
                "keywords": {
                    "time_date": ["test_keyword"],
                }
            }
        }
        r = IntentRouter(config)
        assert r.classify("test_keyword") == "time_date"
        # Original keywords should not work with config override
        assert r.classify("Wie spät ist es?") == "general_chat"


# --- Handler tests ---

class TestTimeDateHandler:
    @patch("app.intent.handlers.datetime")
    def test_time_response(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 4, 1, 14, 35)
        result = handle_time_date("Wie spät ist es?")
        assert "14 Uhr 35" in result

    @patch("app.intent.handlers.datetime")
    def test_time_on_the_hour(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 4, 1, 9, 0)
        result = handle_time_date("Wie spät ist es?")
        assert "9 Uhr" in result
        assert "0" not in result.split("Uhr")[1].split(".")[0]  # no minute number

    @patch("app.intent.handlers.datetime")
    def test_date_response(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 4, 1, 14, 35)
        result = handle_time_date("Welcher Tag ist heute?")
        assert "Mittwoch" in result
        assert "1. April 2026" in result

    @patch("app.intent.handlers.datetime")
    def test_wochentag(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 4, 3, 10, 0)  # Friday
        result = handle_time_date("Welcher Wochentag ist heute?")
        assert "Freitag" in result

    @patch("app.intent.handlers.datetime")
    def test_ambiguous_gives_both(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 4, 1, 14, 35)
        result = handle_time_date("Sag mir alles")
        assert "Uhr" in result
        assert "April" in result


class TestStubHandlers:
    def test_home_control_stub(self):
        result = handle_home_control("Mach das Licht an")
        assert "Home Assistant" in result
        assert "nicht verbunden" in result

    def test_web_search_no_tavily(self):
        result = handle_web_search("Wie wird das Wetter?")
        assert "nicht konfiguriert" in result


class TestSystemHandler:
    def test_wer_bin_ich(self):
        result = handle_system("Wer bin ich?", display_name="Jörg")
        assert "Jörg" in result

    def test_wer_bin_ich_unknown(self):
        result = handle_system("Wer bin ich?")
        assert "nicht" in result

    def test_wie_heisst_du(self):
        result = handle_system("Wie heißt du?")
        assert "Nabu" in result

    def test_was_bist_du(self):
        result = handle_system("Was bist du?")
        assert "Nabu" in result
        assert "lokal" in result

    def test_was_kannst_du_with_llm(self):
        mock_llm = MagicMock()
        mock_llm.complete_sync.return_value = "Ich kann vieles!"
        result = handle_system("Was kannst du?", display_name="Jörg", llm=mock_llm)
        assert result == "Ich kann vieles!"
        mock_llm.complete_sync.assert_called_once()

    def test_was_kannst_du_llm_fallback(self):
        """When LLM fails, return static fallback."""
        mock_llm = MagicMock()
        mock_llm.complete_sync.side_effect = Exception("timeout")
        result = handle_system("Was kannst du?", llm=mock_llm)
        assert "Nabu" in result

    def test_was_kannst_du_no_llm(self):
        """When no LLM provided, return static fallback."""
        result = handle_system("Was kannst du?")
        assert "Nabu" in result
