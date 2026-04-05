# Phase 6 — Memory: SQLite + LanceDB

## Goal
Give Nabu long-term memory so it remembers personal facts about users and
their household across conversations and restarts.

## Architecture

```
User conversation
  ├── Memory Retrieval (before LLM call)
  │   └── LanceDB semantic search → relevant facts injected into prompt
  │
  ├── LLM generates response (grounded in memories + search results)
  │
  └── Memory Extraction (after LLM response)
      └── LLM extracts facts → SQLite + LanceDB storage

Explicit commands:
  ├── "Merk dir..." → memory_store intent → extract + store
  └── "Was weißt du über..." → memory_query intent → search + respond
```

## Components

### SQLite Store (`app/memory/sqlite_store.py`)
- Structured fact storage: subject, fact, category, timestamps
- Case-insensitive subject lookup + LIKE text search
- Automatic deduplication (same user + subject + fact = skip)
- Categories: schedule, preference, family, health, general

### Vector Store (`app/memory/vector_store.py`)
- LanceDB embedded vector database (file-based, no server)
- sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, German support)
- Lazy model loading (~120MB RAM, loads on first memory operation)
- CPU-only embeddings (avoids GPU contention with Whisper)

### Memory Extractor (`app/memory/extractor.py`)
- Proactive: after each general_chat turn, LLM extracts personal facts
- Explicit: "Merk dir..." commands trigger immediate extraction
- Parses LLM JSON response, handles code blocks and malformed output
- Deduplication at storage layer prevents duplicate facts

### Memory Context (`app/memory/context.py`)
- Formats retrieved memories as "Bekannte Informationen" block
- Injected into both search-augmented and no-search prompts

### Intent Routing
- `memory_store`: "Merk dir", "Merke dir", "Vergiss nicht", "Erinnere dich", "Speicher"
- `memory_query`: "Was weißt du über", "Erinnerst du dich", "Was hast du gemerkt"

## Config
```yaml
memory:
  enabled: true
  db_path: "data/memory/nabu_memory.db"
  vector_path: "data/memory/lancedb"
  embedding_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  extract_after_turns: true
  max_context_memories: 5
```

## Data Layout
```
data/memory/
  nabu_memory.db          # SQLite: facts, subjects, categories
  lancedb/                # LanceDB: vector embeddings for semantic search
```

## Test Coverage (216 tests total, all passing)
- 10 SQLite store tests (CRUD, deduplication, directory creation)
- 4 vector store tests (disabled mode, lazy loading)
- 10 extractor tests (extraction, parsing, deduplication, error handling)
- 4 context formatting tests
- 9 intent routing tests (memory_store + memory_query patterns)
- 8 handler tests (store, query, edge cases)
- 4 prompt integration tests
- 167 existing Phase 1-5 tests (no regressions)

## Latency Impact
- Memory retrieval: ~50-100ms (vector search, CPU)
- Memory extraction: ~1-2s (LLM call, async after response)
- Embedding model first load: ~2-3s (lazy, only on first use)
- Steady state: negligible impact on response time
