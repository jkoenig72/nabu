# Nabu

A privacy-first local voice assistant that replaces cloud-based assistants like Alexa.
Everything runs on your own hardware -- no data leaves your home network except web search queries.

## Architecture

Three machines on the local network:

```
Jetson Orin NX 16GB (192.168.11.98)     RTX 4070 PC (192.168.10.11)
hostname: nabu                           LLM Server
+----------------------------+          +----------------------------+
|  USB Microphone            |          |  Qwen3.5-27B-GGUF (Q8_0)  |
|  |                         |  HTTP    |  llama.cpp / vLLM          |
|  Silero VAD (neural, CPU)  | -------> |  OpenAI-compatible API     |
|  |                         |          |  :8000                     |
|  faster-whisper large-v3   |          +----------------------------+
|  (CUDA, float16)           |
|  |                         |          TTS PC (192.168.10.6)
|  Intent Router (regex)     |          +----------------------------+
|  |                         |  HTTP    |  Qwen3-TTS-0.6B (HAL9000) |
|  Handlers / LLM call       | -------> |  Streaming PCM, 24kHz     |
|  |                         |          |  :8091                     |
|  HAL9000 TTS (streaming)   |          +----------------------------+
|  Piper TTS (local fallback)|
|  |                         |          Home Assistant (192.168.10.22)
|  USB Speaker               |          +----------------------------+
+----------------------------+  HTTP    |  REST API :8123            |
                               -------> |  Lights, TV, switches      |
                                        +----------------------------+
```

## Features

| Feature | How it works |
|---------|-------------|
| Wake word | "OK Nabu" detected via Whisper transcript matching (fuzzy, handles mishearings) |
| Speaker ID | Verbal identification: "Joerg hier" / "Isa hier" |
| Conversations | Per-user history, resume past topics, JSON persistence, token-aware trimming |
| Time and Date | Direct from system clock, German formatting, no LLM needed |
| Web Search | Explicit search queries routed to Tavily + LLM summary via intent detection |
| Memory | Remembers personal facts (schedules, preferences, family) via SQLite + LanceDB |
| Home Control | Voice shortcuts for lights and TV via Home Assistant REST API |
| Volume Control | "Lauter" / "Leiser" adjusts speaker volume within conversation |
| TTS | HAL9000 (Qwen3-TTS) with streaming playback, automatic Piper fallback |
| Streaming | LLM tokens stream into sentences, each sentence plays via TTS immediately |
| VAD | Silero neural VAD (speech vs noise), energy-based fallback |

## Quick Start

### Prerequisites

- Jetson Orin NX 16GB with JetPack 6.0 / R36.3.0
- Python venv at `/home/fritz/nabu-venv` with dependencies installed
- ctranslate2 built from source at `/home/fritz/ct2-install/`
- USB audio device (mic + speaker)
- LLM server running Qwen3.5-27B on 192.168.10.11:8000
- TTS server running HAL9000 (Qwen3-TTS) on 192.168.10.6:8091
- Home Assistant at 192.168.10.22:8123 (optional)

### Start

```bash
cd /home/fritz/nabu-dev
./run.sh
```

The `run.sh` script sets `LD_LIBRARY_PATH` for ctranslate2 CUDA, enables all 8 CPUs, configures PulseAudio, and launches Nabu. On startup it loads:
- Whisper large-v3 on CUDA (first load downloads ~3GB model)
- Silero VAD model (~2MB, from torch hub)
- Piper TTS model (local fallback)
- Checks HAL TTS server reachability
- Checks LLM server reachability
- Loads conversation history and memory from disk

You will see:

```
Ready. Listening for 'OK, Nabu!'. Press Ctrl+C to exit.
```

### Stop

Press `Ctrl+C` in the terminal. Conversations are auto-saved to disk. The LLM client, TTS client, and SQLite connection are closed cleanly.

## Usage

1. Say "OK Nabu" (or "Okay Nabu") to wake it up
2. Identify yourself: "Joerg hier" or "Isa hier"
3. If you have past conversations, Nabu asks: continue or new topic?
4. After the acknowledgment beep, ask your question
5. A short beep after each response signals Nabu is ready for the next command
6. Say "Tschuess" or "Bye" to end the conversation

### Example Commands

| What you say | What happens |
|-------------|-------------|
| "Wie spaet ist es?" | Returns time from system clock |
| "Welcher Tag ist heute?" | Returns date in German |
| "Wie wird das Wetter morgen?" | Web search + LLM summary |
| "Merk dir, Isa hat montags Yoga" | Stores fact in memory |
| "Was weisst du ueber Isa?" | Searches memory, returns known facts |
| "Erzaehl mir einen Witz" | General chat via LLM |
| "OK Nabu, Abendbeleuchtung ein" | Shortcut: toggles HA switch directly |
| "OK Nabu, Fernseher aus" | Shortcut: toggles HA input_boolean directly |
| "Lauter" / "Leiser" | Adjusts speaker volume |
| "Loesche alle Konversationen" | Deletes all history and memories (with confirmation) |

## Configuration

All settings are in `app/config.yaml`:

- **audio** -- input/output device names, sample rate, Silero VAD settings and threshold
- **stt** -- Whisper model size (large-v3), CUDA device, compute type, language
- **tts.hal** -- HAL9000 TTS server URL, voice (ref1/ref2/ref3), language, timeout
- **tts** (top-level) -- Piper fallback model path, speaker voice, speech rate
- **llm** -- LLM server URL, model name, temperature, token limits, system prompt
- **wake** -- wake phrases, speaker aliases with display names, acknowledgment beep
- **homeassistant** -- HA URL, bearer token, shortcuts with entity IDs and patterns
- **memory** -- SQLite/LanceDB paths, embedding model, extraction toggle
- **web_search** -- Tavily API key, result count, timeout
- **conversation** -- idle timeout (30s), max duration (300s)

## Project Structure

```
nabu-dev/
├── app/
│   ├── main.py              # Main loop: wake -> listen -> route -> respond -> speak
│   ├── logging_setup.py     # Unified logging: console + daily rotating file (7-day retention)
│   ├── config.py            # YAML config loader, device index resolver
│   ├── config.yaml          # All settings
│   ├── audio/
│   │   ├── capture.py       # Microphone input with Silero VAD (neural) + energy fallback
│   │   └── playback.py      # Speaker output with volume control and resampling
│   ├── stt/
│   │   └── whisper_stt.py   # faster-whisper large-v3 GPU transcription
│   ├── tts/
│   │   ├── hal_tts.py       # HAL9000 TTS client (streaming PCM from remote Qwen3-TTS)
│   │   ├── nabu_tts.py      # TTS wrapper: HAL primary, Piper fallback, streaming support
│   │   └── piper_tts.py     # Piper TTS (local CPU fallback)
│   ├── llm/
│   │   ├── client.py        # LLM HTTP client (non-streaming + SSE streaming)
│   │   └── sentence_splitter.py  # Sentence boundary detection for streaming output
│   ├── wake/
│   │   ├── detector.py      # Wake word detection (fuzzy transcript matching)
│   │   ├── speaker.py       # Verbal speaker identification
│   │   └── conversations.py # Per-user conversation persistence + token trimming
│   ├── intent/
│   │   ├── router.py        # Regex-based intent classification (9 intents + fallback)
│   │   └── handlers.py      # Intent-specific response handlers
│   ├── search/
│   │   ├── tavily.py        # Tavily web search client
│   │   └── llm_search.py    # Search-augmented and no-search prompt templates
│   ├── homeassistant/
│   │   ├── client.py        # Home Assistant REST API client
│   │   └── shortcuts.py     # Quick voice command shortcuts (bypass conversation)
│   └── memory/
│       ├── sqlite_store.py  # SQLite fact storage with deduplication
│       ├── vector_store.py  # LanceDB semantic search (lazy-loaded embeddings)
│       ├── extractor.py     # LLM-based fact extraction from conversations
│       └── context.py       # Memory context formatting for prompts
├── test/                    # 285+ tests (pytest)
│   ├── test_audio.py        # RMS, VAD, Silero VAD, playback tests
│   ├── test_sentence_splitter.py  # Sentence boundary detection tests
│   ├── test_streaming.py    # LLM streaming + pipeline integration tests
│   ├── test_hal_tts.py      # HAL TTS unit tests (mocked)
│   ├── test_hal_tts_live.py # HAL TTS live streaming test (hardware+network)
│   ├── test_tts.py          # Piper TTS tests
│   ├── test_conversations.py
│   ├── test_homeassistant.py
│   ├── test_integration.py
│   ├── test_intent.py
│   ├── test_llm.py
│   ├── test_memory.py
│   ├── test_search.py
│   ├── test_stt.py
│   └── test_wake.py
├── doc/                     # Phase design documents
├── data/
│   ├── conversations/       # Per-user JSON conversation files
│   ├── memory/              # SQLite DB + LanceDB vector index
│   └── nabu.log             # Debug log (daily rotation, 7-day retention)
├── run.sh                   # Launcher (LD_LIBRARY_PATH, CPU enable, PulseAudio)
└── pytest.ini               # Test markers: hardware, slow, network
```

## Running Tests

```bash
cd /home/fritz/nabu-dev
source /home/fritz/nabu-venv/bin/activate
export LD_LIBRARY_PATH=/home/fritz/ct2-install/lib:${LD_LIBRARY_PATH}

# All unit tests (no hardware/network/slow model loading required)
python -m pytest -v -m "not hardware and not network and not slow"

# Include live HAL TTS streaming test (requires HAL server + USB speaker)
python -m pytest -v -m "hardware and network"

# Full suite including model-loading tests
python -m pytest -v
```

Test markers defined in `pytest.ini`:
- `hardware` -- requires physical USB audio devices
- `network` -- requires network access to LLM/TTS servers
- `slow` -- loads ML models (Whisper, Piper)

## Data Storage

All data stays local:

| Data | Location | Format |
|------|----------|--------|
| Conversations | `data/conversations/{user_id}.json` | JSON per user |
| Facts | `data/memory/nabu_memory.db` | SQLite |
| Embeddings | `data/memory/lancedb/` | LanceDB (Arrow) |
| Log | `data/nabu.log` | Text (daily rotation, 7-day retention) |
| Config | `app/config.yaml` | YAML |

## Key Technical Details

- **ctranslate2** is built from source with CUDA (PyPI wheel is CPU-only on aarch64). Installed at `/home/fritz/ct2-install/`. The `LD_LIBRARY_PATH` must include `/home/fritz/ct2-install/lib` or Whisper will not use the GPU.
- **Silero VAD** buffers audio samples across chunks to handle the case where the USB device runs at 48kHz but Silero needs 512 samples at 16kHz (1024 samples at 48kHz resamples to only 341 at 16kHz).
- **HAL TTS streaming** opens an HTTP stream to the TTS server, receives PCM chunks, and feeds them to sounddevice OutputStream in real-time. Audio starts playing before generation completes.
- **Piper TTS** is always loaded as a fallback. If the HAL server becomes unreachable (startup or mid-conversation), Piper takes over transparently.
- **Streaming pipeline** for general conversation: LLM tokens arrive via SSE, are split into sentences at boundary markers (handling abbreviations and decimals), and each sentence is immediately streamed to the TTS speaker. First audio typically arrives within 0.5-1 seconds instead of 4-9 seconds.
- The **embedding model** (paraphrase-multilingual-MiniLM-L12-v2) is lazy-loaded on first memory operation. First load takes ~30 seconds.

## Privacy

- All speech processing (VAD, STT) runs locally on the Jetson GPU
- LLM inference runs on your own PC at 192.168.10.11, not in the cloud
- TTS runs on your own PC at 192.168.10.6, not in the cloud
- Web search (Tavily) is the only external network call
- No telemetry, no cloud accounts, no data collection
