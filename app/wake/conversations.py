"""Per-user conversation storage with JSON persistence and token trimming."""

import json
import logging
import os
import re

log = logging.getLogger(__name__)

_WORD_TO_NUM = {
    "eins": 1, "ein": 1, "erste": 1, "ersten": 1, "erstes": 1,
    "zwei": 2, "zweite": 2, "zweiten": 2, "zweites": 2,
    "drei": 3, "dritte": 3, "dritten": 3, "drittes": 3,
    "vier": 4, "vierte": 4,
    "fünf": 5, "fünfte": 5, "funf": 5,
    "sechs": 6, "sechste": 6,
    "sieben": 7, "siebte": 7,
    "acht": 8, "achte": 8,
    "neun": 9, "neunte": 9,
    "zehn": 10, "zehnte": 10,
}

MAX_CONVERSATIONS = 10
TOPIC_MAX_LENGTH = 60
CHARS_PER_TOKEN = 3.5


class ConversationManager:
    def __init__(self, max_conversations=MAX_CONVERSATIONS, data_dir=None,
                 max_context_tokens=28000):
        self.max_conversations = max_conversations
        self.max_context_tokens = max_context_tokens
        self._data_dir = data_dir
        self._store = {}  # user_id -> list of {"topic": str, "summarized": bool, "history": list}

        if self._data_dir:
            os.makedirs(self._data_dir, exist_ok=True)
            self._load_all()

    def _load_all(self):
        if not self._data_dir:
            return
        for filename in os.listdir(self._data_dir):
            if filename.endswith(".json"):
                user_id = filename[:-5]  # strip .json
                self._load(user_id)

    def _load(self, user_id: str):
        if not self._data_dir:
            return
        filepath = os.path.join(self._data_dir, f"{user_id}.json")
        if not os.path.exists(filepath):
            log.debug("No conversation file for %s", user_id)
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for conv in data:
                    conv.setdefault("summarized", False)
                self._store[user_id] = data
                total_turns = sum(len(c.get("history", [])) for c in data)
                log.info("Loaded %d conversations (%d turns) for %s", len(data), total_turns, user_id)
                log.debug("Conversation file: %s (%d bytes)", filepath, os.path.getsize(filepath))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to load conversations for %s: %s", user_id, e)
            self._store[user_id] = []

    def _save(self, user_id: str):
        """Atomic save via tmp file + rename."""
        if not self._data_dir:
            return
        filepath = os.path.join(self._data_dir, f"{user_id}.json")
        tmp_path = filepath + ".tmp"
        try:
            data = self._store.get(user_id, [])
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, filepath)
            log.debug("Saved %d conversations to %s", len(data), filepath)
        except OSError as e:
            log.error("Failed to save conversations for %s: %s", user_id, e)

    def save(self, user_id: str):
        self._save(user_id)

    def has_conversations(self, user_id: str) -> bool:
        convos = self._store.get(user_id, [])
        return any(c["history"] for c in convos)

    def get_active_history(self, user_id: str) -> list:
        """Get or create the active conversation history list."""
        convos = self._store.setdefault(user_id, [])
        if not convos:
            convos.append({"topic": None, "summarized": False, "history": []})
        return convos[-1]["history"]

    def start_new(self, user_id: str):
        """Archive current conversation (if non-empty) and start a new one."""
        convos = self._store.setdefault(user_id, [])
        if convos and not convos[-1]["history"]:
            return
        convos.append({"topic": None, "summarized": False, "history": []})
        if len(convos) > self.max_conversations:
            self._store[user_id] = convos[-self.max_conversations:]
        self._save(user_id)

    def set_topic(self, user_id: str, first_message: str):
        """Set topic from the first user message."""
        convos = self._store.get(user_id, [])
        if convos and convos[-1]["topic"] is None:
            topic = first_message.strip()
            if len(topic) > TOPIC_MAX_LENGTH:
                topic = topic[:TOPIC_MAX_LENGTH] + "..."
            convos[-1]["topic"] = topic
            self._save(user_id)

    def update_topic(self, user_id: str, summary: str):
        convos = self._store.get(user_id, [])
        if convos:
            convos[-1]["topic"] = summary
            convos[-1]["summarized"] = True
            self._save(user_id)

    def needs_summary(self, user_id: str) -> bool:
        """True if active conversation needs an LLM-generated topic summary."""
        convos = self._store.get(user_id, [])
        if not convos:
            return False
        active = convos[-1]
        return len(active["history"]) >= 2 and not active.get("summarized", False)

    def list_topics(self, user_id: str) -> list[str]:
        convos = self._store.get(user_id, [])
        topics = []
        for c in convos:
            if c["history"]:
                topics.append(c["topic"] or "(kein Thema)")
        return topics

    def format_topic_list(self, user_id: str) -> str:
        topics = self.list_topics(user_id)
        if not topics:
            return ""
        lines = []
        for i, topic in enumerate(topics, 1):
            lines.append(f"{i}. {topic}")
        return ". ".join(lines)

    def select_conversation(self, user_id: str, transcript: str) -> bool:
        """Select a past conversation by number from transcript."""
        num = self._parse_number(transcript)
        if num is None:
            return False
        convos = self._store.get(user_id, [])
        with_history = [i for i, c in enumerate(convos) if c["history"]]
        if num < 1 or num > len(with_history):
            return False
        selected_idx = with_history[num - 1]
        selected = convos[selected_idx]
        convos.pop(selected_idx)
        convos.append(selected)
        self._save(user_id)
        return True

    def get_history_for_llm(self, user_id: str, max_tokens: int | None = None) -> list:
        """Return newest messages that fit within token budget."""
        history = self.get_active_history(user_id)
        if not history:
            return []
        budget = max_tokens or self.max_context_tokens
        total_tokens = 0
        cutoff = len(history)
        for i in range(len(history) - 1, -1, -1):
            msg_tokens = self._estimate_tokens_single(history[i])
            if total_tokens + msg_tokens > budget and i < len(history) - 1:
                cutoff = i + 1
                break
            total_tokens += msg_tokens
        else:
            cutoff = 0

        trimmed = history[cutoff:]
        if len(trimmed) < len(history):
            log.info("Token trimming: %d/%d messages (est. %d tokens)",
                     len(trimmed), len(history), total_tokens)
        else:
            log.debug("History for LLM: %d messages, est. %d tokens (budget: %d)",
                      len(trimmed), total_tokens, budget)
        return trimmed

    @staticmethod
    def _estimate_tokens_single(message: dict) -> int:
        content = message.get("content", "")
        return int(len(content) / CHARS_PER_TOKEN) + 4

    @staticmethod
    def _estimate_tokens(messages: list) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += int(len(content) / CHARS_PER_TOKEN) + 4
        return total

    def delete_all(self, user_id: str):
        """Delete all conversations and files for a user."""
        self._store[user_id] = []
        if self._data_dir:
            filepath = os.path.join(self._data_dir, f"{user_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
                log.info("Deleted conversation file: %s", filepath)
        log.info("All conversations deleted for %s", user_id)

    def trim_active_history(self, user_id: str, max_turns: int):
        convos = self._store.get(user_id, [])
        if convos:
            history = convos[-1]["history"]
            if len(history) > max_turns:
                convos[-1]["history"] = history[-max_turns:]
                self._save(user_id)

    @staticmethod
    def _parse_number(transcript: str) -> int | None:
        """Extract a number (1-10) from transcript. Supports digits and German words."""
        text = transcript.lower().strip()
        text = re.sub(r'[.,!?;:\-\'"]', ' ', text)
        match = re.search(r'\b(\d{1,2})\b', text)
        if match:
            return int(match.group(1))
        for word, num in _WORD_TO_NUM.items():
            if word in text.split():
                return num
        return None
