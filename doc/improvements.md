# Nabu -- Technical Assessment and Improvements

## System Summary

Three-machine architecture over LAN:

- Jetson Orin NX 16GB -- audio I/O, STT (Whisper large-v3 on CUDA), VAD (Silero), orchestration
- PC 1 (RTX 4070) -- LLM inference (Qwen3.5-27B via llama.cpp, port 8000)
- PC 2 (RTX 4070) -- TTS (Qwen3-TTS via HAL9000 server, port 8091)
- External: Home Assistant (REST API), Tavily (web search, only cloud dependency)

Data flow:

```
Mic -> Silero VAD -> faster-whisper -> transcript
  -> wake word check (substring match)
  -> speaker ID (alias match)
  -> intent router (regex, 9 intents)
  -> handler (local or LLM call)
  -> TTS (HAL streaming or Piper fallback)
  -> speaker
```

252 tests passing. 6 development phases complete.


## Current Issues

### Design

- main.py orchestrator is a single ~420-line function with nested loops and a long if/elif intent dispatch chain
- Intent routing is regex-only -- cannot handle rephrased commands or ambiguity
- home_control intent routes to a stub inside conversation but the same command works from wake phrase via shortcuts
- No retry or reconnection when LLM or TTS server goes down mid-session
- Async LLMClient wrapper with no async callers -- adds complexity with zero benefit

### Maintainability

- History append pattern (append user + append assistant + save) duplicated 6 times in main.py
- Four identical normalize() functions across detector.py, speaker.py, router.py, shortcuts.py
- No shared utility module

### Performance

- Embedding model (sentence-transformers) cold starts at ~30s on first memory query, mid-conversation
- Whisper large-v3 runs on every ambient sound for wake word detection -- most expensive idle-loop operation
- LLM and TTS run sequentially -- user waits for full LLM completion before TTS begins


## Improvements

### High value, low effort

- Extract intent dispatch from main.py into a dict mapping intent to handler with a generic history wrapper
- Create shared normalize() utility, import from four files
- Replace async LLMClient with sync httpx.Client, drop the event loop
- Pre-load embedding model at startup when memory is enabled

### Medium effort, high impact

- Use Whisper small or tiny for wake word loop, large-v3 only for conversation commands
- Stream LLM tokens to sentence boundary detection to HAL TTS streaming -- user hears first sentence while LLM still generates
- Wire shortcut handler into the conversation loop for home_control intent


## Model Upgrades

### Drop-in replacements

| Current | Upgrade | Benefit |
|---------|---------|---------|
| Whisper large-v3 | Whisper large-v3-turbo | 2x faster, same accuracy |
| Qwen3.5-27B Q8_0 | Qwen3.5-27B Q4_K_M | 40% less VRAM, negligible quality loss |

### Inference improvements

- llama.cpp speculative decoding with a small draft model (Qwen3.5-3B) for 1.5-2x throughput
- KV cache quantization (FP8) to reduce memory for long conversations
- vLLM with continuous batching if multi-user support is needed

### Architecture change

- Tool-calling LLM instead of regex router -- give the LLM a tool list (time, search, memory, home_control) and let it decide. Qwen3.5 supports this natively. Trade-off: +1s latency per turn but handles arbitrary phrasing.


## Library Upgrades

| Current | Alternative | Benefit |
|---------|------------|---------|
| Manual regex intent router | sentence-transformers zero-shot (already installed) | Semantic matching without regex maintenance |
| asyncio event loop wrapper | httpx.Client (sync) | Remove async complexity |
| JSON file conversations | SQLite (already used for memory) | Atomic writes, no tmp-file pattern |
| Linear interpolation resampling | soxr | Higher quality 48kHz to 16kHz |
| Tavily (cloud) | SearXNG (self-hosted) | Eliminates last cloud dependency |


## Replacement Candidates

| Alternative | Pros | Cons |
|------------|------|------|
| Home Assistant Voice Pipeline | Integrated HA, large community | Less flexible LLM integration |
| openedai-speech + Whisper.cpp | Lighter weight, C++ speed | No conversation memory, less features |
| Rhasspy 3 | Designed for local voice | Alpha quality, uncertain future |

None of these replace what Nabu does. The custom orchestrator is the right choice for full control over LLM prompts, memory, and multi-machine architecture.


## Priority Ranking

Ranked by impact relative to effort:

1. LLM streaming to sentence split to TTS streaming -- biggest UX improvement
2. Small Whisper model for wake loop -- biggest idle-loop performance win
3. Extract intent dispatch from main.py -- biggest maintainability win
4. Whisper large-v3-turbo -- free speed, drop-in
5. Pre-load embedding model at startup -- eliminates surprise 30s stall
6. Replace async LLMClient with sync -- removes complexity for zero cost
7. SearXNG for self-hosted search -- removes last cloud dependency
