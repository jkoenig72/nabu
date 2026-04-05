# Phase 2 — LLM Integration

## Goal
Wire Gemma 3 12B-IT (via vLLM) between STT and TTS so Nabu can converse.
Loop: listen → transcribe → LLM → speak response.

## LLM Server
- URL: `http://192.168.10.11:8000/v1/chat/completions`
- Model: `google/gemma-3-12b-it` (vLLM, OpenAI-compatible)
- Context: 32K tokens
- Latency: ~170ms for 20 tokens, ~1.3s for 100 tokens

## Components

### LLM Client (`app/llm/client.py`)
- `LLMClient` with async httpx + sync wrappers
- Persistent event loop for connection pool reuse
- `complete_sync(system_prompt, messages) → str`
- `health_check_sync() → bool`
- `LLMError` exception for all failure modes

### Conversation History
- Simple list of `{"role": "user/assistant", "content": str}` dicts
- Trimmed to last 6 entries (3 exchanges)
- Lives in memory only (persistence in Phase 5)

### Error Handling
- LLM unreachable: speaks "Der Sprachserver ist gerade nicht erreichbar"
- Removes failed user turn from history
- Continues listen loop

## Config Added
```yaml
llm:
  url, model, max_tokens (256), temperature (0.7), timeout (10s)
  system_prompt: German conversational assistant persona
  max_history_turns: 6
```

## Test Coverage
- 9 mocked unit tests (init, health check, completion, error handling)
- 2 live network tests against vLLM server
- All passing
