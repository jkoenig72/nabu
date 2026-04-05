"""Tests for LLM sentence splitter."""

from app.llm.sentence_splitter import is_sentence_end, split_sentences


class TestIsSentenceEnd:
    def test_period(self):
        assert is_sentence_end("Das ist ein Test.") is True

    def test_exclamation(self):
        assert is_sentence_end("Halt!") is True

    def test_question(self):
        assert is_sentence_end("Wie geht es dir?") is True

    def test_no_punctuation(self):
        assert is_sentence_end("Das ist ein Test") is False

    def test_comma(self):
        assert is_sentence_end("Hallo, wie geht es") is False

    def test_empty(self):
        assert is_sentence_end("") is False

    def test_whitespace_only(self):
        assert is_sentence_end("   ") is False

    def test_abbreviation_dr(self):
        assert is_sentence_end("Guten Tag Dr.") is False

    def test_abbreviation_prof(self):
        assert is_sentence_end("Das sagte Prof.") is False

    def test_abbreviation_zb(self):
        assert is_sentence_end("z.B.") is False

    def test_abbreviation_dh(self):
        assert is_sentence_end("d.h.") is False

    def test_decimal_number(self):
        assert is_sentence_end("Die Temperatur betraegt 22.") is False

    def test_decimal_mid_sentence(self):
        assert is_sentence_end("Es sind 3.") is False

    def test_sentence_after_abbreviation(self):
        assert is_sentence_end("Dr. Mueller ist da.") is True

    def test_trailing_whitespace(self):
        assert is_sentence_end("Das ist ein Test.  ") is True

    def test_colon_not_sentence_end(self):
        assert is_sentence_end("Folgendes:") is False

    def test_ellipsis(self):
        assert is_sentence_end("Hmm...") is True


class TestSplitSentences:
    def _tokens(self, text):
        """Simulate token stream by yielding individual words."""
        for word in text.split(" "):
            yield word + " "

    def test_single_sentence(self):
        tokens = self._tokens("Das ist ein Test.")
        result = list(split_sentences(tokens))
        assert result == ["Das ist ein Test."]

    def test_two_sentences(self):
        tokens = self._tokens("Satz eins. Satz zwei.")
        result = list(split_sentences(tokens))
        assert result == ["Satz eins.", "Satz zwei."]

    def test_three_sentences_mixed_punctuation(self):
        tokens = self._tokens("Hallo! Wie geht es? Gut.")
        result = list(split_sentences(tokens))
        assert result == ["Hallo!", "Wie geht es?", "Gut."]

    def test_remainder_without_punctuation(self):
        tokens = self._tokens("Satz eins. Kein Ende")
        result = list(split_sentences(tokens))
        assert result == ["Satz eins.", "Kein Ende"]

    def test_empty_stream(self):
        result = list(split_sentences(iter([])))
        assert result == []

    def test_abbreviation_not_split(self):
        tokens = self._tokens("Dr. Mueller ist da.")
        result = list(split_sentences(tokens))
        assert result == ["Dr. Mueller ist da."]

    def test_decimal_not_split(self):
        tokens = self._tokens("Es sind 22. Grad warm.")
        result = list(split_sentences(tokens))
        # "22." looks like decimal, so not split there
        assert len(result) == 1
        assert "22." in result[0]

    def test_character_by_character(self):
        """Simulate LLM yielding one character at a time."""
        text = "Hallo. Welt."
        tokens = (c for c in text)
        result = list(split_sentences(tokens))
        assert result == ["Hallo.", "Welt."]

    def test_whitespace_only_not_yielded(self):
        tokens = iter(["  ", "  "])
        result = list(split_sentences(tokens))
        assert result == []
