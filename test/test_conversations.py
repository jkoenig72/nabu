"""Tests for ConversationManager."""

import json
import os

import pytest

from app.wake.conversations import ConversationManager


@pytest.fixture
def mgr():
    return ConversationManager(max_conversations=10)


class TestBasicOperations:
    def test_no_conversations_initially(self, mgr):
        assert mgr.has_conversations("joerg") is False

    def test_get_active_creates_empty(self, mgr):
        history = mgr.get_active_history("joerg")
        assert history == []

    def test_has_conversations_after_message(self, mgr):
        history = mgr.get_active_history("joerg")
        history.append({"role": "user", "content": "Hallo"})
        assert mgr.has_conversations("joerg") is True

    def test_start_new_archives_current(self, mgr):
        history = mgr.get_active_history("joerg")
        history.append({"role": "user", "content": "Wetter morgen"})
        mgr.set_topic("joerg", "Wetter morgen")
        mgr.start_new("joerg")

        new_history = mgr.get_active_history("joerg")
        assert new_history == []
        assert mgr.list_topics("joerg") == ["Wetter morgen"]

    def test_start_new_doesnt_archive_empty(self, mgr):
        mgr.get_active_history("joerg")
        mgr.start_new("joerg")
        # Should still have just one empty conversation
        assert mgr.list_topics("joerg") == []


class TestTopics:
    def test_set_topic(self, mgr):
        mgr.get_active_history("joerg")
        mgr.set_topic("joerg", "Wie wird das Wetter morgen?")
        assert mgr.list_topics("joerg") == []  # no history yet, not listed

    def test_topic_listed_after_message(self, mgr):
        history = mgr.get_active_history("joerg")
        mgr.set_topic("joerg", "Wetter morgen")
        history.append({"role": "user", "content": "Wetter morgen"})
        assert mgr.list_topics("joerg") == ["Wetter morgen"]

    def test_topic_truncated(self, mgr):
        history = mgr.get_active_history("joerg")
        long_msg = "A" * 100
        mgr.set_topic("joerg", long_msg)
        history.append({"role": "user", "content": long_msg})
        topics = mgr.list_topics("joerg")
        assert len(topics[0]) == 63  # 60 + "..."
        assert topics[0].endswith("...")

    def test_multiple_topics(self, mgr):
        for topic in ["Wetter", "Rezepte", "Musik"]:
            history = mgr.get_active_history("joerg")
            mgr.set_topic("joerg", topic)
            history.append({"role": "user", "content": topic})
            mgr.start_new("joerg")

        topics = mgr.list_topics("joerg")
        assert topics == ["Wetter", "Rezepte", "Musik"]

    def test_format_topic_list(self, mgr):
        for topic in ["Wetter", "Rezepte"]:
            history = mgr.get_active_history("joerg")
            mgr.set_topic("joerg", topic)
            history.append({"role": "user", "content": topic})
            mgr.start_new("joerg")

        formatted = mgr.format_topic_list("joerg")
        assert "1. Wetter" in formatted
        assert "2. Rezepte" in formatted

    def test_no_topic_placeholder(self, mgr):
        history = mgr.get_active_history("joerg")
        history.append({"role": "user", "content": "test"})
        topics = mgr.list_topics("joerg")
        assert topics == ["(kein Thema)"]


class TestSelectConversation:
    def _setup_conversations(self, mgr):
        """Create 3 conversations for joerg."""
        for topic in ["Wetter morgen", "Kuchen backen", "Python lernen"]:
            history = mgr.get_active_history("joerg")
            mgr.set_topic("joerg", topic)
            history.append({"role": "user", "content": topic})
            history.append({"role": "assistant", "content": f"Antwort zu {topic}"})
            mgr.start_new("joerg")

    def test_select_by_digit(self, mgr):
        self._setup_conversations(mgr)
        assert mgr.select_conversation("joerg", "2") is True
        history = mgr.get_active_history("joerg")
        assert any("Kuchen" in m["content"] for m in history)

    def test_select_by_german_word(self, mgr):
        self._setup_conversations(mgr)
        assert mgr.select_conversation("joerg", "eins") is True
        history = mgr.get_active_history("joerg")
        assert any("Wetter" in m["content"] for m in history)

    def test_select_by_word_drei(self, mgr):
        self._setup_conversations(mgr)
        assert mgr.select_conversation("joerg", "drei") is True
        history = mgr.get_active_history("joerg")
        assert any("Python" in m["content"] for m in history)

    def test_select_invalid_number(self, mgr):
        self._setup_conversations(mgr)
        assert mgr.select_conversation("joerg", "99") is False

    def test_select_no_number(self, mgr):
        self._setup_conversations(mgr)
        assert mgr.select_conversation("joerg", "keine Ahnung") is False

    def test_select_zero(self, mgr):
        self._setup_conversations(mgr)
        assert mgr.select_conversation("joerg", "0") is False

    def test_select_with_punctuation(self, mgr):
        self._setup_conversations(mgr)
        assert mgr.select_conversation("joerg", "Die 2, bitte!") is True


class TestTrimAndLimits:
    def test_trim_active_history(self, mgr):
        history = mgr.get_active_history("joerg")
        for i in range(10):
            history.append({"role": "user", "content": f"msg {i}"})
        mgr.trim_active_history("joerg", max_turns=4)
        history = mgr.get_active_history("joerg")
        assert len(history) == 4

    def test_max_conversations_limit(self):
        mgr = ConversationManager(max_conversations=3)
        for i in range(5):
            history = mgr.get_active_history("joerg")
            mgr.set_topic("joerg", f"Topic {i}")
            history.append({"role": "user", "content": f"msg {i}"})
            mgr.start_new("joerg")

        # max=3 slots: 2 past with history + 1 empty active
        topics = mgr.list_topics("joerg")
        assert len(topics) == 2
        assert topics[0] == "Topic 3"
        assert topics[1] == "Topic 4"

    def test_per_user_isolation(self, mgr):
        h1 = mgr.get_active_history("joerg")
        h1.append({"role": "user", "content": "Joerg's message"})
        mgr.set_topic("joerg", "Joerg topic")

        h2 = mgr.get_active_history("isa")
        h2.append({"role": "user", "content": "Isa's message"})
        mgr.set_topic("isa", "Isa topic")

        assert mgr.list_topics("joerg") == ["Joerg topic"]
        assert mgr.list_topics("isa") == ["Isa topic"]


class TestParseNumber:
    @pytest.mark.parametrize("text,expected", [
        ("1", 1),
        ("2", 2),
        ("10", 10),
        ("eins", 1),
        ("zwei", 2),
        ("drei", 3),
        ("vier", 4),
        ("fünf", 5),
        ("sechs", 6),
        ("sieben", 7),
        ("acht", 8),
        ("neun", 9),
        ("zehn", 10),
        ("die zweite", 2),
        ("Nummer 3", 3),
        ("", None),
        ("keine Ahnung", None),
    ])
    def test_parse_number(self, text, expected):
        assert ConversationManager._parse_number(text) == expected


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        """Conversations survive manager restart."""
        conv_dir = str(tmp_path / "convos")
        mgr1 = ConversationManager(data_dir=conv_dir)
        history = mgr1.get_active_history("joerg")
        mgr1.set_topic("joerg", "Wetter")
        history.append({"role": "user", "content": "Wie wird das Wetter?"})
        history.append({"role": "assistant", "content": "Sonnig."})
        mgr1.save("joerg")

        # New manager, same directory
        mgr2 = ConversationManager(data_dir=conv_dir)
        assert mgr2.has_conversations("joerg")
        topics = mgr2.list_topics("joerg")
        assert topics == ["Wetter"]
        loaded = mgr2.get_active_history("joerg")
        assert len(loaded) == 2
        assert loaded[0]["content"] == "Wie wird das Wetter?"

    def test_missing_dir_created(self, tmp_path):
        conv_dir = str(tmp_path / "deep" / "nested" / "convos")
        mgr = ConversationManager(data_dir=conv_dir)
        assert os.path.isdir(conv_dir)
        # Should work without errors
        mgr.get_active_history("test")

    def test_corrupt_json_handled(self, tmp_path):
        conv_dir = str(tmp_path / "convos")
        os.makedirs(conv_dir)
        with open(os.path.join(conv_dir, "joerg.json"), "w") as f:
            f.write("{corrupt json!!")

        mgr = ConversationManager(data_dir=conv_dir)
        # Should not crash, starts with empty
        assert mgr.has_conversations("joerg") is False

    def test_no_persistence_when_none(self, tmp_path):
        mgr = ConversationManager(data_dir=None)
        history = mgr.get_active_history("joerg")
        history.append({"role": "user", "content": "test"})
        mgr.save("joerg")
        # No files created anywhere
        assert not list(tmp_path.iterdir())

    def test_atomic_write_valid_json(self, tmp_path):
        conv_dir = str(tmp_path / "convos")
        mgr = ConversationManager(data_dir=conv_dir)
        history = mgr.get_active_history("joerg")
        history.append({"role": "user", "content": "test"})
        mgr.save("joerg")

        filepath = os.path.join(conv_dir, "joerg.json")
        assert os.path.exists(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert data[0]["history"][0]["content"] == "test"

    def test_multiple_users_separate_files(self, tmp_path):
        conv_dir = str(tmp_path / "convos")
        mgr = ConversationManager(data_dir=conv_dir)

        for user in ["joerg", "isa"]:
            h = mgr.get_active_history(user)
            h.append({"role": "user", "content": f"Hello from {user}"})
            mgr.save(user)

        files = os.listdir(conv_dir)
        assert "joerg.json" in files
        assert "isa.json" in files

    def test_start_new_auto_saves(self, tmp_path):
        conv_dir = str(tmp_path / "convos")
        mgr = ConversationManager(data_dir=conv_dir)
        h = mgr.get_active_history("joerg")
        h.append({"role": "user", "content": "first"})
        mgr.save("joerg")
        mgr.start_new("joerg")

        # Reload and verify archived conversation
        mgr2 = ConversationManager(data_dir=conv_dir)
        topics = mgr2.list_topics("joerg")
        assert len(topics) == 1


class TestTokenTrimming:
    def test_short_history_untrimmed(self, mgr):
        history = mgr.get_active_history("joerg")
        history.append({"role": "user", "content": "Hallo"})
        history.append({"role": "assistant", "content": "Hi!"})
        history.append({"role": "user", "content": "Wie geht es?"})

        result = mgr.get_history_for_llm("joerg")
        assert len(result) == 3

    def test_long_history_trimmed(self):
        mgr = ConversationManager(max_context_tokens=100)
        history = mgr.get_active_history("joerg")
        # Each message ~100 chars = ~29 tokens + 4 overhead = ~33 tokens
        for i in range(20):
            history.append({"role": "user", "content": "A" * 100})
            history.append({"role": "assistant", "content": "B" * 100})

        result = mgr.get_history_for_llm("joerg")
        assert len(result) < 40  # trimmed
        assert len(result) >= 1  # at least last message
        # Last message should always be included
        assert result[-1] == history[-1]

    def test_always_includes_last_message(self):
        mgr = ConversationManager(max_context_tokens=1)  # tiny budget
        history = mgr.get_active_history("joerg")
        history.append({"role": "user", "content": "A" * 1000})

        result = mgr.get_history_for_llm("joerg")
        assert len(result) == 1
        assert result[0]["content"] == "A" * 1000

    def test_empty_history(self, mgr):
        mgr.get_active_history("joerg")
        result = mgr.get_history_for_llm("joerg")
        assert result == []

    def test_estimate_tokens(self):
        messages = [{"role": "user", "content": "A" * 350}]  # 350 chars / 3.5 = 100 + 4 overhead
        tokens = ConversationManager._estimate_tokens(messages)
        assert tokens == 104


class TestTopicSummary:
    def test_needs_summary_false_when_empty(self, mgr):
        mgr.get_active_history("joerg")
        assert mgr.needs_summary("joerg") is False

    def test_needs_summary_false_with_one_message(self, mgr):
        h = mgr.get_active_history("joerg")
        h.append({"role": "user", "content": "test"})
        assert mgr.needs_summary("joerg") is False

    def test_needs_summary_true_after_exchange(self, mgr):
        h = mgr.get_active_history("joerg")
        h.append({"role": "user", "content": "Wie wird das Wetter?"})
        h.append({"role": "assistant", "content": "Sonnig."})
        assert mgr.needs_summary("joerg") is True

    def test_needs_summary_false_after_update(self, mgr):
        h = mgr.get_active_history("joerg")
        h.append({"role": "user", "content": "Wetter"})
        h.append({"role": "assistant", "content": "Sonnig"})
        mgr.update_topic("joerg", "Wettervorhersage")
        assert mgr.needs_summary("joerg") is False

    def test_update_topic_replaces(self, mgr):
        h = mgr.get_active_history("joerg")
        mgr.set_topic("joerg", "Original topic")
        h.append({"role": "user", "content": "test"})
        mgr.update_topic("joerg", "LLM Summary")
        topics = mgr.list_topics("joerg")
        assert topics == ["LLM Summary"]

    def test_update_topic_saves_to_disk(self, tmp_path):
        conv_dir = str(tmp_path / "convos")
        mgr = ConversationManager(data_dir=conv_dir)
        h = mgr.get_active_history("joerg")
        h.append({"role": "user", "content": "test"})
        mgr.update_topic("joerg", "Summary")

        # Verify on disk
        with open(os.path.join(conv_dir, "joerg.json"), "r") as f:
            data = json.load(f)
        assert data[0]["topic"] == "Summary"
        assert data[0]["summarized"] is True
