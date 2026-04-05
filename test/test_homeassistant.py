"""Tests for Home Assistant client and shortcuts."""

from unittest.mock import patch, MagicMock

import pytest

from app.homeassistant.client import HAClient
from app.homeassistant.shortcuts import ShortcutHandler


# --- HA Client ---

class TestHAClient:
    def test_disabled_without_config(self):
        client = HAClient({})
        assert client.enabled is False

    def test_disabled_without_token(self):
        client = HAClient({"homeassistant": {"url": "http://localhost:8123"}})
        assert client.enabled is False

    def test_enabled_with_config(self):
        client = HAClient({"homeassistant": {"url": "http://localhost:8123", "token": "abc"}})
        assert client.enabled is True

    def test_call_service_success(self):
        client = HAClient({"homeassistant": {"url": "http://localhost:8123", "token": "abc"}})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("app.homeassistant.client.httpx.post", return_value=mock_resp) as mock_post:
            result = client.call_service("switch", "turn_on", "switch.evening_lights")
            assert result is True
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "switch/turn_on" in call_args[0][0]
            assert call_args[1]["json"]["entity_id"] == "switch.evening_lights"

    def test_call_service_disabled(self):
        client = HAClient({})
        assert client.call_service("switch", "turn_on", "switch.test") is False

    def test_call_service_failure(self):
        client = HAClient({"homeassistant": {"url": "http://localhost:8123", "token": "abc"}})
        with patch("app.homeassistant.client.httpx.post", side_effect=Exception("timeout")):
            result = client.call_service("switch", "turn_on", "switch.test")
            assert result is False

    def test_get_state_success(self):
        client = HAClient({"homeassistant": {"url": "http://localhost:8123", "token": "abc"}})
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"state": "on"}
        with patch("app.homeassistant.client.httpx.get", return_value=mock_resp):
            assert client.get_state("switch.test") == "on"

    def test_get_state_disabled(self):
        client = HAClient({})
        assert client.get_state("switch.test") is None


# --- Shortcut Handler ---

_SHORTCUT_CONFIG = {
    "homeassistant": {
        "url": "http://localhost:8123",
        "token": "abc",
        "shortcuts": [
            {
                "name": "Abendbeleuchtung",
                "entity_id": "switch.evening_lights",
                "domain": "switch",
                "patterns": [
                    r"abendbeleuchtung.{0,10}ein",
                    r"abendbeleuchtung.{0,10}an",
                    r"abendbeleuchtung.{0,10}aus",
                ],
                "response_on": "Abendbeleuchtung ist eingeschaltet.",
                "response_off": "Abendbeleuchtung ist ausgeschaltet.",
            }
        ],
    }
}


class TestShortcutHandler:
    def _make_handler(self):
        mock_ha = MagicMock()
        mock_ha.enabled = True
        mock_ha.call_service.return_value = True
        handler = ShortcutHandler(_SHORTCUT_CONFIG, mock_ha)
        return handler, mock_ha

    def test_einschalten(self):
        handler, mock_ha = self._make_handler()
        result = handler.check("Okay Nabu, Abendbeleuchtung einschalten")
        assert result is not None
        assert "eingeschaltet" in result["response"]
        mock_ha.call_service.assert_called_once_with("switch", "turn_on", "switch.evening_lights")

    def test_ausschalten(self):
        handler, mock_ha = self._make_handler()
        result = handler.check("OK Nabu Abendbeleuchtung ausschalten")
        assert result is not None
        assert "ausgeschaltet" in result["response"]
        mock_ha.call_service.assert_called_once_with("switch", "turn_off", "switch.evening_lights")

    def test_anschalten(self):
        handler, mock_ha = self._make_handler()
        result = handler.check("okay nabu abendbeleuchtung anschalten")
        assert result is not None
        assert "eingeschaltet" in result["response"]

    def test_no_match(self):
        handler, _ = self._make_handler()
        result = handler.check("Okay Nabu wie wird das Wetter")
        assert result is None

    def test_ha_disabled(self):
        mock_ha = MagicMock()
        mock_ha.enabled = False
        handler = ShortcutHandler(_SHORTCUT_CONFIG, mock_ha)
        result = handler.check("Okay Nabu Abendbeleuchtung einschalten")
        assert result is None

    def test_service_failure(self):
        handler, mock_ha = self._make_handler()
        mock_ha.call_service.return_value = False
        result = handler.check("Okay Nabu Abendbeleuchtung einschalten")
        assert result is not None
        assert "Fehler" in result["response"]

    def test_punctuation_handled(self):
        handler, mock_ha = self._make_handler()
        result = handler.check("Okay, Nabu! Abendbeleuchtung einschalten!")
        assert result is not None

    def test_no_shortcuts_configured(self):
        mock_ha = MagicMock()
        mock_ha.enabled = True
        handler = ShortcutHandler({}, mock_ha)
        result = handler.check("Abendbeleuchtung einschalten")
        assert result is None
