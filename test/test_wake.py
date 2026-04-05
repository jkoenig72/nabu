"""Tests for wake word detection and speaker identification."""

import pytest

from app.wake.detector import WakeWordDetector
from app.wake.speaker import SpeakerParser


@pytest.fixture
def wake_config():
    return {
        "wake": {
            "phrases": ["ok nabu", "okay nabu", "ok nabo", "okay nabo", "o k nabu"],
            "speakers": {
                "joerg": {
                    "display_name": "Jörg",
                    "aliases": ["joerg", "jörg", "jorg", "georg"],
                },
                "isa": {
                    "display_name": "Isa",
                    "aliases": ["isa", "isabel", "isabelle"],
                },
            },
        }
    }


class TestWakeWordDetector:
    def test_ok_nabu(self, wake_config):
        d = WakeWordDetector(wake_config)
        assert d.check("OK Nabu") is True

    def test_okay_nabu(self, wake_config):
        d = WakeWordDetector(wake_config)
        assert d.check("Okay Nabu") is True

    def test_with_punctuation(self, wake_config):
        d = WakeWordDetector(wake_config)
        assert d.check("OK, Nabu!") is True

    def test_with_speaker(self, wake_config):
        d = WakeWordDetector(wake_config)
        assert d.check("OK Nabu Joerg hier") is True

    def test_ok_nabo_variant(self, wake_config):
        d = WakeWordDetector(wake_config)
        assert d.check("Okay Nabo") is True

    def test_o_k_nabu(self, wake_config):
        d = WakeWordDetector(wake_config)
        assert d.check("O.K. Nabu, Isa hier!") is True

    def test_case_insensitive(self, wake_config):
        d = WakeWordDetector(wake_config)
        assert d.check("OK NABU") is True

    def test_no_match(self, wake_config):
        d = WakeWordDetector(wake_config)
        assert d.check("Hallo Welt") is False

    def test_empty_string(self, wake_config):
        d = WakeWordDetector(wake_config)
        assert d.check("") is False

    def test_partial_no_match(self, wake_config):
        d = WakeWordDetector(wake_config)
        assert d.check("Nabu") is False

    def test_wrong_order(self, wake_config):
        d = WakeWordDetector(wake_config)
        assert d.check("Nabu ok") is False


class TestSpeakerParser:
    def test_joerg_hier(self, wake_config):
        p = SpeakerParser(wake_config)
        assert p.parse("OK Nabu Joerg hier") == "joerg"

    def test_joerg_umlaut(self, wake_config):
        p = SpeakerParser(wake_config)
        assert p.parse("OK Nabu Jörg hier") == "joerg"

    def test_georg_mishearing(self, wake_config):
        p = SpeakerParser(wake_config)
        assert p.parse("OK Nabu Georg hier") == "joerg"

    def test_isabel_maps_to_isa(self, wake_config):
        p = SpeakerParser(wake_config)
        assert p.parse("OK Nabu Isabel hier") == "isa"

    def test_isa_direct(self, wake_config):
        p = SpeakerParser(wake_config)
        assert p.parse("OK Nabu Isa hier") == "isa"

    def test_isabelle_maps_to_isa(self, wake_config):
        p = SpeakerParser(wake_config)
        assert p.parse("Okay Nabu, Isabelle hier!") == "isa"

    def test_no_speaker(self, wake_config):
        p = SpeakerParser(wake_config)
        assert p.parse("OK Nabu") is None

    def test_empty_string(self, wake_config):
        p = SpeakerParser(wake_config)
        assert p.parse("") is None

    def test_case_insensitive(self, wake_config):
        p = SpeakerParser(wake_config)
        assert p.parse("OK NABU JOERG HIER") == "joerg"

    def test_just_name_response(self, wake_config):
        """When asked 'Wer spricht?', user says just a name."""
        p = SpeakerParser(wake_config)
        assert p.parse("Joerg") == "joerg"

    def test_just_isa_response(self, wake_config):
        p = SpeakerParser(wake_config)
        assert p.parse("Isa") == "isa"

    def test_just_isabel_response(self, wake_config):
        p = SpeakerParser(wake_config)
        assert p.parse("Isabel") == "isa"

    def test_display_name(self, wake_config):
        p = SpeakerParser(wake_config)
        assert p.display_name("joerg") == "Jörg"
        assert p.display_name("isa") == "Isa"
        assert p.display_name("unknown") is None

    def test_speaker_names_list(self, wake_config):
        p = SpeakerParser(wake_config)
        names = p.speaker_names_list()
        assert "Jörg" in names
        assert "Isa" in names
        assert "oder" in names
