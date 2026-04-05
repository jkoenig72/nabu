# Nabu Logging

## Overview

Nabu uses Python's standard `logging` module with two output targets:

| Target | Level | Purpose |
|--------|-------|---------|
| **Console** (stdout) | INFO | Operational status — what Nabu is doing |
| **File** (`data/nabu.log`) | DEBUG | Full trace — every decision, timing, request/response |

The log file is recreated on each startup (mode `"w"`), so it only contains the current session.

## Usage

```bash
# Normal mode: INFO to console, DEBUG to file
./run.sh

# Verbose mode: DEBUG to both console and file
./run.sh --verbose
# or
./run.sh -v

# Watch the debug log in real time (from another terminal)
tail -f /home/fritz/nabu-dev/data/nabu.log
```

## Log Format

```
HH:MM:SS [module.name] LEVEL message
```

Example:
```
11:23:45 [app.audio.capture] DEBUG Recording: device=25, rate=48000→16000, threshold=0.0150, silence=0.8s, max=3.0s
11:23:46 [app.audio.capture] DEBUG Speech detected: RMS=0.0312 > threshold=0.0150
11:23:47 [app.audio.capture] DEBUG Recorded 48000 samples (1.00s at 48000 Hz)
11:23:47 [app.audio.capture] DEBUG Resampled to 16000 samples (1.00s at 16000 Hz)
11:23:47 [app.stt.whisper_stt] DEBUG STT: transcribing 16000 samples (1.00s)
11:23:47 [app.stt.whisper_stt] DEBUG STT: 'Ok Nabu Jörg hier' [de] in 0.09s
11:23:47 [app.wake.detector] DEBUG Wake check: normalized='ok nabu jörg hier' → MATCH
11:23:47 [app.wake.speaker] DEBUG Speaker matched: 'jörg' → joerg (Jörg)
11:23:47 [nabu] INFO Wake word detected! Speaker: Jörg
```

## What Each Module Logs

### Audio Capture (`app.audio.capture`)

| Level | What |
|-------|------|
| INFO | Device sample rate mismatch (resampling active) |
| DEBUG | Recording parameters (device, rates, threshold, durations) |
| DEBUG | Speech detection trigger (RMS vs threshold) |
| DEBUG | Sample counts, durations, resampling details |
| DEBUG | "No speech detected" when returning empty |

### Audio Playback (`app.audio.playback`)

| Level | What |
|-------|------|
| DEBUG | Output device native rate detection |
| DEBUG | WAV playback size |
| WARNING | No working output sample rate found |

### STT — Whisper (`app.stt.whisper_stt`)

| Level | What |
|-------|------|
| INFO | Model loaded (size, device, compute type) |
| DEBUG | Input sample count and duration |
| DEBUG | Transcription result, language, and processing time |

### TTS — Piper (`app.tts.piper_tts`)

| Level | What |
|-------|------|
| INFO | Model loaded (path) |
| DEBUG | Input text (first 80 chars) and character count |
| DEBUG | Output audio duration, synthesis time, realtime factor |

### Wake Word Detector (`app.wake.detector`)

| Level | What |
|-------|------|
| DEBUG | Configured wake phrases |
| DEBUG | Normalized transcript and match result (MATCH / no match) |

### Speaker Parser (`app.wake.speaker`)

| Level | What |
|-------|------|
| DEBUG | Configured alias map |
| DEBUG | Matched alias and resolved user, or "no speaker found" |

### Conversations (`app.wake.conversations`)

| Level | What |
|-------|------|
| INFO | Conversations loaded (count, turns, user) |
| INFO | Token trimming applied (messages kept vs total) |
| DEBUG | JSON file path, size, save operations |
| DEBUG | History size and token estimates for LLM |
| WARNING | Failed to load corrupt JSON file |

### Intent Router (`app.intent.router`)

| Level | What |
|-------|------|
| DEBUG | Matched intent, pattern, and input text |
| DEBUG | Fallback to general_chat |

### Intent Handlers (`app.intent.handlers`)

| Level | What |
|-------|------|
| DEBUG | Handler entry with command text and user |
| ERROR | Handler failures (memory store, memory query) |

### LLM Client (`app.llm.client`)

| Level | What |
|-------|------|
| INFO | Server reachable (health check) |
| DEBUG | Request: message count, max_tokens, temperature |
| DEBUG | System prompt (first 120 chars) |
| DEBUG | Last user message (first 120 chars) |
| DEBUG | Response: time, prompt tokens, completion tokens |
| DEBUG | Answer text (first 150 chars) |
| ERROR | Timeout with duration |
| WARNING | Empty choices or None content |

### Tavily Web Search (`app.search.tavily`)

| Level | What |
|-------|------|
| DEBUG | Request: query, search depth, max results |
| DEBUG | Response: time, result count, answer length |
| DEBUG | Answer preview (first 150 chars) |
| ERROR | Timeout with duration |
| WARNING | API key not configured |

### Search Prompt Builder (`app.search.llm_search`)

| Level | What |
|-------|------|
| DEBUG | Prompt type (search-augmented or no-search) |
| DEBUG | Prompt size, memory context presence, user name |

### Memory SQLite (`app.memory.sqlite_store`)

| Level | What |
|-------|------|
| INFO | Stored memory (id, subject, fact) |
| DEBUG | Duplicate memory skipped |
| DEBUG | Search results (subject search, text search) with counts |

### Memory Vector Store (`app.memory.vector_store`)

| Level | What |
|-------|------|
| INFO | Embedding model loading and loaded |
| INFO | LanceDB table created |
| DEBUG | Add operation (id, user, text) |
| DEBUG | Search query, user filter, row count, result count |

### Memory Extractor (`app.memory.extractor`)

| Level | What |
|-------|------|
| DEBUG | Extraction input (user, message) |
| DEBUG | LLM extraction response |
| DEBUG | Search results with scores |
| DEBUG | "No results" for failed searches |
| WARNING | LLM call failure, vector store failure |

### Main Loop (`nabu`)

| Level | What |
|-------|------|
| INFO | Startup progress (loading models, server checks) |
| INFO | Wake word detected with speaker |
| INFO | Intent classification result |
| INFO | Memory extraction count |
| DEBUG | Audio capture details (duration, samples, RMS) |
| DEBUG | "Too short" skips |
| DEBUG | Wake check transcript |
| DEBUG | Speaker ID responses |
| DEBUG | Topic choices |
| DEBUG | Command STT results |
| DEBUG | Listening state transitions |

## Configuration

Logging is configured in `app/main.py` at module level:

```python
_LOG_FILE = "data/nabu.log"
_VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv

# Console: INFO (or DEBUG with --verbose)
# File: always DEBUG
```

### Changing the log file path

Edit `_LOG_FILE` in `app/main.py`:
```python
_LOG_FILE = "data/nabu.log"  # change this path
```

### Adjusting log levels per module

To silence a noisy module, add to `app/main.py` after the `logging.basicConfig()` call:
```python
logging.getLogger("app.audio.capture").setLevel(logging.INFO)  # suppress capture DEBUG
logging.getLogger("httpx").setLevel(logging.WARNING)           # suppress HTTP request logs
```

## Troubleshooting with Logs

### Wake word not detected

Look for the wake check cycle in the log:
```
grep "Wake check\|Speech detected\|Too short\|No speech" data/nabu.log
```

If you see no "Speech detected" lines, the VAD threshold may be too high. If you see STT results but no "MATCH", check what Whisper is hearing vs the configured wake phrases.

### LLM slow or failing

```
grep "LLM request\|LLM response\|LLM timeout" data/nabu.log
```

Shows request/response timing and token counts.

### Memory not working

```
grep "Memory\|VectorDB\|SQLite" data/nabu.log
```

Shows extraction attempts, storage operations, and search results.

### Audio issues

```
grep "Recording\|Speech detected\|Resampled\|device" data/nabu.log
```

Shows device selection, sample rates, and VAD decisions.
