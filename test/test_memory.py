"""Tests for memory system: SQLite store, vector store, extractor, handlers, context."""

import json
import os
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from app.memory.sqlite_store import MemorySQLite
from app.memory.extractor import MemoryExtractor, MEMORY_CONTEXT_HEADER
from app.memory.context import build_memory_context
from app.intent.router import IntentRouter
from app.intent.handlers import handle_memory_store, handle_memory_query
from app.search.llm_search import build_search_prompt, build_nosearch_prompt


# --- SQLite Store ---

class TestMemorySQLite:
    def test_add_and_retrieve(self, tmp_path):
        db = MemorySQLite(str(tmp_path / "test.db"))
        mid = db.add("joerg", "Isa", "hat montags Yoga", "schedule", "Isa hat montags Yoga")
        assert mid is not None
        results = db.search_by_subject("Isa")
        assert len(results) == 1
        assert results[0]["fact"] == "hat montags Yoga"
        assert results[0]["category"] == "schedule"
        db.close()

    def test_search_by_subject_case_insensitive(self, tmp_path):
        db = MemorySQLite(str(tmp_path / "test.db"))
        db.add("joerg", "Isa", "hat montags Yoga", "schedule")
        results = db.search_by_subject("isa")
        assert len(results) == 1
        db.close()

    def test_search_by_text(self, tmp_path):
        db = MemorySQLite(str(tmp_path / "test.db"))
        db.add("joerg", "Isa", "hat montags Yoga", "schedule")
        db.add("joerg", "Jörg", "trinkt gerne Cappuccino", "preference")
        results = db.search_by_text("Yoga")
        assert len(results) == 1
        assert results[0]["subject"] == "Isa"
        db.close()

    def test_get_all_for_user(self, tmp_path):
        db = MemorySQLite(str(tmp_path / "test.db"))
        db.add("joerg", "Isa", "hat Yoga", "schedule")
        db.add("joerg", "Katze", "heißt Mimi", "family")
        db.add("isa", "Jörg", "mag Kaffee", "preference")
        results = db.get_all_for_user("joerg")
        assert len(results) == 2
        db.close()

    def test_delete(self, tmp_path):
        db = MemorySQLite(str(tmp_path / "test.db"))
        mid = db.add("joerg", "Isa", "hat Yoga", "schedule")
        db.delete(mid)
        results = db.search_by_subject("Isa")
        assert len(results) == 0
        db.close()

    def test_update(self, tmp_path):
        db = MemorySQLite(str(tmp_path / "test.db"))
        mid = db.add("joerg", "Isa", "hat montags Yoga", "schedule")
        db.update(mid, "hat dienstags Yoga")
        results = db.search_by_subject("Isa")
        assert results[0]["fact"] == "hat dienstags Yoga"
        assert results[0]["updated_at"] != results[0]["created_at"]
        db.close()

    def test_deduplication(self, tmp_path):
        db = MemorySQLite(str(tmp_path / "test.db"))
        mid1 = db.add("joerg", "Isa", "hat montags Yoga", "schedule")
        mid2 = db.add("joerg", "Isa", "hat montags Yoga", "schedule")
        assert mid1 is not None
        assert mid2 is None  # duplicate
        results = db.search_by_subject("Isa")
        assert len(results) == 1
        db.close()

    def test_empty_search(self, tmp_path):
        db = MemorySQLite(str(tmp_path / "test.db"))
        results = db.search_by_subject("Nobody")
        assert results == []
        results = db.search_by_text("nonexistent")
        assert results == []
        db.close()

    def test_get_all(self, tmp_path):
        db = MemorySQLite(str(tmp_path / "test.db"))
        db.add("joerg", "Isa", "hat Yoga", "schedule")
        db.add("isa", "Jörg", "mag Kaffee", "preference")
        results = db.get_all()
        assert len(results) == 2
        db.close()

    def test_creates_directory(self, tmp_path):
        db_path = str(tmp_path / "subdir" / "deep" / "test.db")
        db = MemorySQLite(db_path)
        assert os.path.exists(os.path.dirname(db_path))
        db.close()


# --- Vector Store (mocked) ---

class TestMemoryVectorStore:
    def test_disabled_without_packages(self):
        """Test that vector store reports disabled when packages unavailable."""
        with patch("app.memory.vector_store._LANCEDB_AVAILABLE", False):
            from app.memory.vector_store import MemoryVectorStore
            store = MemoryVectorStore("/tmp/test_lance")
            assert store.enabled is False

    def test_search_returns_empty_when_disabled(self):
        with patch("app.memory.vector_store._LANCEDB_AVAILABLE", False):
            from app.memory.vector_store import MemoryVectorStore
            store = MemoryVectorStore("/tmp/test_lance")
            assert store.search("test") == []

    def test_add_noop_when_disabled(self):
        with patch("app.memory.vector_store._LANCEDB_AVAILABLE", False):
            from app.memory.vector_store import MemoryVectorStore
            store = MemoryVectorStore("/tmp/test_lance")
            store.add(1, "joerg", "test")  # should not raise

    def test_lazy_model_loading(self):
        """Model should not be loaded at init time."""
        with patch("app.memory.vector_store._LANCEDB_AVAILABLE", True), \
             patch("app.memory.vector_store._ST_AVAILABLE", True):
            from app.memory.vector_store import MemoryVectorStore
            store = MemoryVectorStore("/tmp/test_lance")
            assert store._model is None  # not loaded yet


# --- Memory Extractor ---

class TestMemoryExtractor:
    def _make_extractor(self):
        mock_sqlite = MagicMock()
        mock_sqlite.add.return_value = 42
        mock_vector = MagicMock()
        mock_vector.enabled = True
        mock_vector.search.return_value = []
        return MemoryExtractor(mock_sqlite, mock_vector), mock_sqlite, mock_vector

    def test_extract_facts_from_message(self):
        extractor, mock_sqlite, mock_vector = self._make_extractor()
        mock_llm = MagicMock()
        mock_llm.complete_sync.return_value = json.dumps([
            {"subject": "Isa", "fact": "hat montags Yoga", "category": "schedule"}
        ])

        stored = extractor.extract_and_store("joerg", "Isa hat montags Yoga", "", mock_llm)
        assert len(stored) == 1
        assert stored[0]["subject"] == "Isa"
        assert stored[0]["fact"] == "hat montags Yoga"
        mock_sqlite.add.assert_called_once()
        mock_vector.add.assert_called_once()

    def test_no_facts_extracted(self):
        extractor, mock_sqlite, _ = self._make_extractor()
        mock_llm = MagicMock()
        mock_llm.complete_sync.return_value = "KEINE"

        stored = extractor.extract_and_store("joerg", "Wie wird das Wetter?", "", mock_llm)
        assert stored == []
        mock_sqlite.add.assert_not_called()

    def test_malformed_llm_response(self):
        extractor, mock_sqlite, _ = self._make_extractor()
        mock_llm = MagicMock()
        mock_llm.complete_sync.return_value = "Das ist keine JSON Antwort blabla"

        stored = extractor.extract_and_store("joerg", "test", "", mock_llm)
        assert stored == []
        mock_sqlite.add.assert_not_called()

    def test_json_in_code_block(self):
        extractor, mock_sqlite, _ = self._make_extractor()
        mock_llm = MagicMock()
        mock_llm.complete_sync.return_value = '```json\n[{"subject": "Isa", "fact": "mag Pizza", "category": "preference"}]\n```'

        stored = extractor.extract_and_store("joerg", "Isa mag Pizza", "", mock_llm)
        assert len(stored) == 1

    def test_llm_failure_handled(self):
        extractor, _, _ = self._make_extractor()
        mock_llm = MagicMock()
        mock_llm.complete_sync.side_effect = Exception("LLM down")

        stored = extractor.extract_and_store("joerg", "test", "", mock_llm)
        assert stored == []

    def test_retrieve_relevant_with_results(self):
        extractor, _, mock_vector = self._make_extractor()
        mock_vector.search.return_value = [
            {"id": 1, "user_id": "joerg", "text": "Isa: hat montags Yoga", "score": 0.1},
        ]

        context = extractor.retrieve_relevant("Wann hat Isa Sport?", "joerg")
        assert MEMORY_CONTEXT_HEADER in context
        assert "Isa: hat montags Yoga" in context

    def test_retrieve_relevant_empty(self):
        extractor, mock_sqlite, mock_vector = self._make_extractor()
        mock_vector.search.return_value = []
        mock_sqlite.search_by_text.return_value = []

        context = extractor.retrieve_relevant("Wie wird das Wetter?", "joerg")
        assert context == ""

    def test_deduplication_during_extraction(self):
        extractor, mock_sqlite, _ = self._make_extractor()
        # First call returns id, second returns None (duplicate)
        mock_sqlite.add.side_effect = [42, None]

        mock_llm = MagicMock()
        mock_llm.complete_sync.return_value = json.dumps([
            {"subject": "Isa", "fact": "hat Yoga", "category": "schedule"},
            {"subject": "Isa", "fact": "hat Yoga", "category": "schedule"},
        ])

        stored = extractor.extract_and_store("joerg", "test", "", mock_llm)
        assert len(stored) == 1  # second was deduplicated

    def test_invalid_category_defaults_to_general(self):
        extractor, mock_sqlite, _ = self._make_extractor()
        mock_llm = MagicMock()
        mock_llm.complete_sync.return_value = json.dumps([
            {"subject": "Test", "fact": "something", "category": "invalid_cat"}
        ])

        stored = extractor.extract_and_store("joerg", "test", "", mock_llm)
        assert len(stored) == 1
        # Check that the category was fixed to "general"
        call_args = mock_sqlite.add.call_args
        assert call_args.kwargs.get("category", call_args[0][3] if len(call_args[0]) > 3 else None) == "general"

    def test_enabled_property(self):
        extractor = MemoryExtractor(MagicMock(), MagicMock())
        assert extractor.enabled is True

        extractor_disabled = MemoryExtractor(None, MagicMock())
        assert extractor_disabled.enabled is False


# --- Memory Context ---

class TestBuildMemoryContext:
    def test_with_text_field(self):
        memories = [
            {"text": "Isa: hat montags Yoga"},
            {"text": "Jörg: trinkt Cappuccino"},
        ]
        result = build_memory_context(memories)
        assert "Bekannte Informationen" in result
        assert "Isa: hat montags Yoga" in result
        assert "Jörg: trinkt Cappuccino" in result

    def test_with_subject_fact_fields(self):
        memories = [
            {"subject": "Isa", "fact": "hat montags Yoga"},
        ]
        result = build_memory_context(memories)
        assert "Isa: hat montags Yoga" in result

    def test_empty_list(self):
        assert build_memory_context([]) == ""

    def test_none_text(self):
        memories = [{"text": ""}]
        assert build_memory_context(memories) == ""


# --- Intent Routing for Memory ---

class TestMemoryIntentRouting:
    @pytest.fixture
    def router(self):
        return IntentRouter()

    def test_merk_dir(self, router):
        assert router.classify("Merk dir, Isa hat montags Yoga") == "memory_store"

    def test_merke_dir(self, router):
        assert router.classify("Merke dir bitte meinen Geburtstag") == "memory_store"

    def test_vergiss_nicht(self, router):
        assert router.classify("Vergiss nicht, morgen kommt der Handwerker") == "memory_store"

    def test_erinnere_dich(self, router):
        assert router.classify("Erinnere dich daran") == "memory_store"

    def test_speicher(self, router):
        assert router.classify("Speicher das bitte ab") == "memory_store"

    def test_was_weisst_du(self, router):
        assert router.classify("Was weißt du über Isa?") == "memory_query"

    def test_was_weisst_du_ss(self, router):
        assert router.classify("Was weisst du über mich?") == "memory_query"

    def test_erinnerst_du_dich(self, router):
        assert router.classify("Erinnerst du dich an meinen Geburtstag?") == "memory_query"

    def test_was_hast_du_gemerkt(self, router):
        assert router.classify("Was hast du dir gemerkt?") == "memory_query"


# --- Memory Handlers ---

class TestHandleMemoryStore:
    def test_stores_and_confirms(self):
        mock_extractor = MagicMock()
        mock_extractor.enabled = True
        mock_extractor.extract_and_store.return_value = [
            {"subject": "Isa", "fact": "hat montags Yoga", "category": "schedule"}
        ]
        mock_llm = MagicMock()

        result = handle_memory_store(
            "Merk dir, Isa hat montags Yoga",
            extractor=mock_extractor, llm=mock_llm, user_id="joerg",
        )
        assert "merke mir" in result.lower()
        assert "Isa" in result

    def test_no_extractor(self):
        result = handle_memory_store("Merk dir was", extractor=None)
        assert "nicht aktiviert" in result

    def test_disabled_extractor(self):
        mock_ext = MagicMock()
        mock_ext.enabled = False
        result = handle_memory_store("Merk dir was", extractor=mock_ext)
        assert "nicht aktiviert" in result

    def test_no_llm(self):
        mock_ext = MagicMock()
        mock_ext.enabled = True
        result = handle_memory_store("Merk dir was", extractor=mock_ext, llm=None)
        assert "nicht erreichbar" in result

    def test_no_facts_found(self):
        mock_ext = MagicMock()
        mock_ext.enabled = True
        mock_ext.extract_and_store.return_value = []
        mock_llm = MagicMock()
        result = handle_memory_store("Merk dir was", extractor=mock_ext, llm=mock_llm, user_id="joerg")
        assert "keine konkreten Fakten" in result


class TestHandleMemoryQuery:
    def test_found_memories(self):
        mock_ext = MagicMock()
        mock_ext.enabled = True
        mock_ext.retrieve_relevant.return_value = (
            "Bekannte Informationen über den Haushalt:\n- Isa: hat montags Yoga"
        )

        result = handle_memory_query(
            "Was weißt du über Isa?",
            extractor=mock_ext, user_id="joerg",
        )
        assert "Yoga" in result

    def test_no_memories(self):
        mock_ext = MagicMock()
        mock_ext.enabled = True
        mock_ext.retrieve_relevant.return_value = ""

        result = handle_memory_query("Was weißt du über Katzen?", extractor=mock_ext, user_id="joerg")
        assert "nichts gespeichert" in result

    def test_no_extractor(self):
        result = handle_memory_query("Was weißt du?", extractor=None)
        assert "nicht aktiviert" in result


# --- Prompt integration ---

class TestMemoryContextInPrompts:
    def test_search_prompt_with_memory(self):
        prompt = build_search_prompt(
            "Ergebnis: sonnig",
            display_name="Jörg",
            memory_context="Bekannte Informationen:\n- Jörg mag Sonne",
        )
        assert "Bekannte Informationen" in prompt
        assert "Jörg mag Sonne" in prompt
        assert "sonnig" in prompt

    def test_nosearch_prompt_with_memory(self):
        prompt = build_nosearch_prompt(
            display_name="Jörg",
            memory_context="Bekannte Informationen:\n- Jörg mag Kaffee",
        )
        assert "Bekannte Informationen" in prompt
        assert "Jörg mag Kaffee" in prompt

    def test_empty_memory_no_extra_text(self):
        prompt_with = build_search_prompt("results", memory_context="")
        prompt_without = build_search_prompt("results")
        assert prompt_with == prompt_without

    def test_nosearch_empty_memory(self):
        prompt_with = build_nosearch_prompt(memory_context="")
        prompt_without = build_nosearch_prompt()
        assert prompt_with == prompt_without
