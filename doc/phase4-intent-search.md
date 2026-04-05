# Phase 4 — Intent Routing + Web Search (Always-Verify)

## Goal
Add intent classification so Nabu handles different request types smartly,
and always back up general_chat answers with Tavily web search to prevent hallucinations.

## Architecture

```
User question
  ├── Intent Router (keyword matching)
  │   ├── time_date    → local clock, no LLM needed (~0ms)
  │   ├── home_control → stub (Phase 5: HA integration)
  │   ├── web_search   → Tavily search → LLM summarizes results
  │   ├── system       → direct answers or short LLM call
  │   └── general_chat → Tavily search (always) → LLM with search results
  │
  └── TTS speaks response
```

## Always-Verify Flow (general_chat)
Every general_chat question gets backed up with Tavily search:

1. User asks a question
2. Tavily searches the web for the question (~1-2s)
3. If results found: LLM answers grounded in search results
4. If search fails: LLM answers honestly, prompted to say "nicht sicher" when uncertain

This prevents hallucinations for factual questions (election results, weather,
sports scores, prices) while still allowing the LLM to answer conversational
questions naturally when search results aren't relevant.

## Components

### Intent Router (`app/intent/router.py`)
- Keyword-based regex matching, checked in priority order
- time_date > home_control > web_search > system > general_chat (fallback)
- Configurable patterns in config.yaml, with sensible German defaults

### Intent Handlers (`app/intent/handlers.py`)
- `handle_time_date` — Python datetime, German formatting, no LLM
- `handle_home_control` — stub for now
- `handle_web_search` — Tavily → LLM summary (for keyword-matched queries)
- `handle_system` — direct answers for identity questions, LLM for capabilities

### Tavily Client (`app/search/tavily.py`)
- HTTP client using httpx (synchronous)
- Returns formatted results: Tavily answer + numbered source excerpts
- Handles errors gracefully, returns None on failure

### Search-Augmented Prompts (`app/search/llm_search.py`)
- `build_search_prompt()` — includes search results in system prompt
- `build_nosearch_prompt()` — tells LLM to be honest about uncertainty
- ~1,000 tokens for typical search results (fits easily in 28k budget)

## Config Added
```yaml
web_search:
  api_key: "tvly-dev-..."
  max_results: 3
  search_depth: "basic"
  timeout: 8.0
```

## Test Coverage (170 tests, all passing)
- 52 intent router tests (all intents, edge cases, config override)
- 12 handler tests (time_date, system, stubs)
- 7 search extraction + prompt tests
- 6 Tavily client tests (mocked)
- 5 web search handler tests (mocked)
- 1 live Tavily API test
- 87 existing Phase 1-3 tests (no regressions)

## Latency Budget
- time_date: ~500ms (TTS only, no LLM)
- web_search: ~3-4s (Tavily ~1.5s + LLM ~1.5s + TTS ~0.5s)
- general_chat: ~3-4s (Tavily ~1.5s + LLM ~1.5s + TTS ~0.5s)
- system (identity): ~500ms (direct answer, no LLM)
