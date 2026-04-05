# NABU — Local Voice Assistant System Design

## 1. Overview

A privacy-first, locally-running voice assistant that replaces cloud-dependent assistants like Alexa. Runs on a Jetson Orin Nano 8GB (audio + orchestration) with LLM inference offloaded to a backend PC (RTX 4070 12GB). Integrates with an existing Home Assistant installation via REST API. Supports multi-user voice identification (Joerg, Isabel, Guest) and German as the primary language.

**Target latency:** ≤ 5 seconds from end-of-speech to start of audio response.

---

## 2. Hardware & Network Topology

| Component | Role | Specs |
|-----------|------|-------|
| Jetson Orin Nano 8GB | Edge device: audio I/O, STT, TTS, orchestrator, memory | 40 TOPS, 8GB shared RAM/VRAM |
| Backend PC | Primary LLM inference | RTX 4070 12GB, connected via LAN |
| Home Assistant server | Smart home control | Existing installation, REST API |
| USB Microphone | Audio input | Any USB mic with decent SNR |
| USB Speaker | Audio output | Any USB speaker/DAC |

All devices communicate over the local network. The Jetson connects to the PC's vLLM server and to Home Assistant via HTTP.

---

## 3. Component Design

### 3.1 Wake Word Detection — openWakeWord
- **Runs on:** Jetson CPU (always-on, ~2% CPU)
- **Model:** Custom wake word or pre-trained ("Hey Nabu", "Hallo Nabu")
- **Output:** Triggers audio capture pipeline
- **Latency contribution:** ~50ms

### 3.2 Speech-to-Text — faster-whisper
- **Runs on:** Jetson GPU
- **Model:** `whisper-medium` with INT8 quantization via CTranslate2
- **Why medium:** With HA and fallback LLM removed from Jetson, there is enough VRAM headroom. Medium INT8 uses ~3GB VRAM and delivers significantly better German accuracy than small.
- **Fallback:** `whisper-small` INT8 (~1GB VRAM) if medium proves too tight
- **Output:** Transcript text + detected language (de/en)
- **Latency contribution:** ~1–2s (medium)

### 3.3 Speaker Identification — wespeaker (standalone)
- **Runs on:** Jetson CPU
- **Approach:** Use `wespeaker` directly (not through pyannote, which is heavier and slower on CPU). Pre-enrolled voice embeddings for Joerg and Isabel stored in SQLite. At wake time, extract a short embedding from the utterance audio and compare via cosine similarity. Fall back to "Guest" if confidence is below threshold.
- **Enrollment:** One-time setup where each user speaks 5–10 phrases to build a reference embedding
- **Output:** `user_id` string: `joerg`, `isabel`, or `guest`
- **Latency contribution:** ~200–500ms (runs in parallel with STT, so mostly hidden)
- **Note:** Full pyannote diarization pipeline can be 2-7x slower on CPU. Using wespeaker directly for simple 2-speaker ID is much faster.

### 3.4 LLM Inference — Gemma 3 12B-IT via vLLM

**Primary (Backend PC — RTX 4070 12GB):**
- Model: `google/gemma-3-12b-it`
- Server: vLLM at `http://192.168.10.11:8000`
- Context window: 32K tokens (keep requests tight for speed)
- API: OpenAI-compatible `/v1/chat/completions`

**No Jetson fallback.** The 8GB shared RAM/VRAM is too constrained for a useful local LLM alongside Whisper medium. If the PC is unreachable, Nabu announces degraded mode and handles only cached/simple responses.

**Failover logic:**
```
1. Health check PC every 30s (GET /health)
2. If healthy → route to PC
3. If unhealthy → announce "Der Hauptrechner ist nicht erreichbar"
4. Continue handling system/memory queries locally (no LLM needed)
```

### 3.5 Text-to-Speech — Piper TTS
- **Runs on:** Jetson CPU
- **Voice:** `de_DE-thorsten-high` (German male) or `de_DE-kerstin-high` (German female)
- Can switch voice per user preference (stored in profile)
- **Output:** WAV audio → USB speaker
- **Latency contribution:** ~300–500ms for a typical sentence
- **Streaming:** Piper supports sentence-level streaming — start playing the first sentence while generating the rest

### 3.6 Python Orchestrator

The brain of the system. A single async Python application.

**Tech stack:**
- Python 3.11+
- `asyncio` for concurrency
- `httpx` for async HTTP calls (to LLM server, HA, Tavily)
- `sounddevice` or `pyaudio` for audio I/O
- No heavy framework needed — just a well-structured async app

**Core loop (pseudocode):**
```python
async def handle_utterance(audio: AudioSegment):
    # Run STT and speaker ID in parallel
    transcript, user_id = await asyncio.gather(
        stt.transcribe(audio),
        speaker_id.identify(audio)
    )

    # Load user context
    user_profile = db.get_user_profile(user_id)
    recent_turns = db.get_recent_turns(user_id, limit=5)
    relevant_memories = vector_db.search(transcript, user_id, limit=3)

    # Route intent
    intent = await router.classify(transcript, user_profile)

    # Build context package based on intent
    context = build_context(intent, user_profile, recent_turns, relevant_memories)

    # Execute intent-specific actions
    if intent == "home_control":
        ha_entities = db.get_ha_entity_cache()
        context.add(ha_entities)
    elif intent == "web_search":
        await tts.speak("Moment, ich schaue nach...")
        search_results = await tavily.search(transcript)
        context.add(search_results)

    # Generate response
    response = await llm.complete(
        system_prompt=prompts[intent].system,
        messages=context.to_messages(),
        max_tokens=prompts[intent].max_tokens
    )

    # Execute actions if needed (HA service calls)
    if intent == "home_control" and response.has_action:
        await ha.call_service(response.action)

    # Speak response
    await tts.speak(response.text)

    # Async: log turn + extract memories
    asyncio.create_task(memory.process_turn(user_id, transcript, response))
```

---

## 4. Intent Router — Two-Stage Design

### Stage 1: Lightweight Classification

A small, fast classifier that maps the user utterance to one of these intents:

| Intent | Examples | Tools Available |
|--------|----------|-----------------|
| `home_control` | "Mach das Licht an", "Wie warm ist es im Wohnzimmer?" | HA REST API |
| `web_search` | "Was ist das Wetter morgen?", "Wer hat gestern Bayern gespielt?" | Tavily API |
| `general_chat` | "Erzähl mir einen Witz", "Was ist Photosynthese?" | None (pure LLM) |
| `memory_query` | "Was hatte ich dir letzte Woche über den Urlaub gesagt?" | LanceDB search |
| `memory_store` | "Merk dir dass Isabel Montags Yoga hat" | SQLite write |
| `system` | "Wechsel auf Englisch", "Wer bin ich?" | Local config |

**Implementation options (from simple to complex):**

1. **Keyword matching + embedding similarity** — A dict of keywords per intent plus a small sentence-transformer for fuzzy matching. Fast (~20ms), no GPU needed, but needs maintenance.

2. **Single LLM call with constrained output** — Send the transcript to the LLM with a classification prompt that returns only the intent label. Uses ~30 tokens output. ~200ms on the 4070. More flexible but adds a round-trip.

3. **Small local classifier model** — Fine-tuned DistilBERT or similar on your intent categories. Best accuracy, but requires training data.

**Recommendation:** Start with option 1 (keyword + embeddings) for speed. Graduate to option 2 if accuracy is insufficient. The keyword approach handles 80% of cases and keeps latency minimal.

### Stage 2: Intent-Specific Prompt

Once intent is classified, the orchestrator loads the matching prompt template and assembles the full context.

---

## 5. Prompt Management — YAML Config

```yaml
# prompts.yaml
global:
  persona: |
    Du bist NABU, ein hilfreicher Sprachassistent im Zuhause von
    Joerg und Isabel. Du sprichst Deutsch, bist freundlich und
    antwortest kurz und präzise, da deine Antworten vorgelesen werden.
    Aktuelle Zeit: {current_time}
    Aktueller Benutzer: {user_name}

intents:
  home_control:
    system_prompt: |
      {persona}
      Du steuerst das Smart Home über Home Assistant.
      Verfügbare Geräte und deren Status:
      {ha_entities}

      Wenn der Benutzer ein Gerät steuern will, antworte mit:
      1. Einer kurzen Bestätigung für den Benutzer
      2. Einem JSON-Block mit der Aktion:
      ```action
      {{"service": "light.turn_on", "entity_id": "light.wohnzimmer", "data": {{}}}}
      ```
    max_tokens: 150
    temperature: 0.1

  web_search:
    system_prompt: |
      {persona}
      Beantworte die Frage basierend auf diesen aktuellen Suchergebnissen:
      {search_results}

      Fasse die Antwort in 2-3 Sätzen zusammen. Nenne die Quelle nur wenn
      der Benutzer danach fragt.
    max_tokens: 300
    temperature: 0.3

  general_chat:
    system_prompt: |
      {persona}
      Benutzer-Profil: {user_profile}
      Relevante Erinnerungen: {memories}

      Antworte natürlich und hilfreich. Halte dich kurz (1-3 Sätze),
      es sei denn der Benutzer bittet um eine ausführliche Erklärung.
    max_tokens: 200
    temperature: 0.7

  memory_query:
    system_prompt: |
      {persona}
      Der Benutzer fragt nach etwas, das er dir früher erzählt hat.
      Hier sind relevante Einträge aus dem Gedächtnis:
      {memories}

      Wenn du die Information findest, gib sie wieder. Wenn nicht,
      sag ehrlich dass du dich nicht erinnern kannst.
    max_tokens: 200
    temperature: 0.3

  memory_store:
    system_prompt: |
      {persona}
      Der Benutzer möchte, dass du dir etwas merkst. Extrahiere den
      Fakt und bestätige kurz.

      Antworte mit:
      1. Einer kurzen Bestätigung
      2. Dem extrahierten Fakt als JSON:
      ```memory
      {{"fact": "Isabel hat montags Yoga", "category": "routine", "user": "isabel"}}
      ```
    max_tokens: 100
    temperature: 0.1
```

---

## 6. Memory Architecture

### 6.1 SQLite — Structured Data

```sql
-- User profiles
CREATE TABLE users (
    user_id     TEXT PRIMARY KEY,  -- 'joerg', 'isabel', 'guest'
    display_name TEXT NOT NULL,
    language    TEXT DEFAULT 'de',
    tts_voice   TEXT DEFAULT 'de_DE-thorsten-high',
    preferences JSON DEFAULT '{}',
    voice_embedding BLOB,          -- reference embedding for speaker ID
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Conversation log
CREATE TABLE conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT REFERENCES users(user_id),
    transcript  TEXT NOT NULL,
    response    TEXT NOT NULL,
    intent      TEXT NOT NULL,
    language    TEXT DEFAULT 'de',
    latency_ms  INTEGER,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Explicit facts / memories
CREATE TABLE memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT REFERENCES users(user_id),
    fact        TEXT NOT NULL,
    category    TEXT,  -- 'preference', 'routine', 'person', 'event', 'general'
    source      TEXT,  -- 'explicit' (user asked to remember) or 'extracted' (proactive)
    confidence  REAL DEFAULT 1.0,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at  TIMESTAMP  -- optional, for time-bound facts
);
CREATE INDEX idx_memories_user ON memories(user_id);
CREATE INDEX idx_memories_category ON memories(user_id, category);

-- HA entity cache
CREATE TABLE ha_entities (
    entity_id   TEXT PRIMARY KEY,
    friendly_name TEXT,
    domain      TEXT,     -- 'light', 'switch', 'climate', 'sensor', ...
    state       TEXT,
    attributes  JSON,
    last_synced TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 6.2 LanceDB — Semantic Memory

```python
import lancedb
from sentence_transformers import SentenceTransformer

# Use a small multilingual model for embeddings
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# ~120MB, supports German well, runs on CPU

db = lancedb.connect("./memory.lance")

# Create table for semantic search over memories + conversations
memory_table = db.create_table("semantic_memory", [
    {"vector": embedder.encode("init"), "text": "", "user_id": "", "type": "", "created_at": ""}
])

# Search example
def search_memory(query: str, user_id: str, limit: int = 3):
    query_vec = embedder.encode(query)
    results = memory_table.search(query_vec) \
        .where(f"user_id = '{user_id}'") \
        .limit(limit) \
        .to_list()
    return results
```

### 6.3 Proactive Memory Extraction

After each conversation turn, an async background task analyzes whether the exchange contained memorable information:

```python
MEMORY_EXTRACTION_PROMPT = """
Analysiere diesen Dialog und extrahiere Fakten, die es wert sind,
sich zu merken. Nur Fakten über den Benutzer, seine Vorlieben,
Routinen, Personen oder wichtige Ereignisse.

Dialog:
User ({user_name}): {transcript}
Assistant: {response}

Antworte mit einer JSON-Liste der Fakten, oder einer leeren Liste []:
```json
[{{"fact": "...", "category": "preference|routine|person|event|general"}}]
```
"""

async def process_turn(user_id, transcript, response):
    # 1. Log conversation
    db.log_conversation(user_id, transcript, response)

    # 2. Extract memories (async, non-blocking)
    facts = await llm.complete(
        MEMORY_EXTRACTION_PROMPT.format(...),
        max_tokens=200
    )

    # 3. Store extracted facts
    for fact in facts:
        db.insert_memory(user_id, fact["fact"], fact["category"], source="extracted")
        vector_db.insert(fact["fact"], user_id)
```

---

## 7. Home Assistant Integration

### 7.1 Entity Sync

On startup and every 5 minutes, fetch all relevant entities:

```python
async def sync_ha_entities():
    """Fetch HA entities and cache in SQLite."""
    entities = await httpx.get(
        f"{HA_URL}/api/states",
        headers={"Authorization": f"Bearer {HA_TOKEN}"}
    )

    # Filter to controllable/useful entities
    relevant_domains = {"light", "switch", "climate", "cover",
                        "media_player", "sensor", "binary_sensor",
                        "automation", "scene"}

    for entity in entities.json():
        domain = entity["entity_id"].split(".")[0]
        if domain in relevant_domains:
            db.upsert_ha_entity(
                entity_id=entity["entity_id"],
                friendly_name=entity["attributes"].get("friendly_name"),
                domain=domain,
                state=entity["state"],
                attributes=entity["attributes"]
            )
```

### 7.2 Service Calls

When the LLM outputs an action block, parse and execute it:

```python
ALLOWED_SERVICES = {
    "light.turn_on", "light.turn_off", "light.toggle",
    "switch.turn_on", "switch.turn_off", "switch.toggle",
    "climate.set_temperature", "climate.set_hvac_mode",
    "cover.open_cover", "cover.close_cover",
    "media_player.media_play", "media_player.media_pause",
    "media_player.volume_set",
    "scene.turn_on",
    "automation.trigger",
}

async def call_ha_service(action: dict):
    """Execute a Home Assistant service call (whitelisted only)."""
    service = action["service"]  # e.g. "light.turn_on"

    if service not in ALLOWED_SERVICES:
        return False, f"Service {service} is not allowed"

    domain, service_name = service.split(".")

    # Validate entity_id exists in cache
    if not db.entity_exists(action["entity_id"]):
        return False, f"Unknown entity {action['entity_id']}"

    response = await httpx.post(
        f"{HA_URL}/api/services/{domain}/{service_name}",
        headers={"Authorization": f"Bearer {HA_TOKEN}"},
        json={
            "entity_id": action["entity_id"],
            **action.get("data", {})
        }
    )
    return response.status_code == 200, None
```

### 7.3 Entity List for LLM Context

Format cached entities as a compact string for the system prompt:

```
Verfügbare Geräte:
- light.wohnzimmer (Wohnzimmer Licht) — an, brightness: 80%
- light.schlafzimmer (Schlafzimmer Licht) — aus
- climate.wohnzimmer (Wohnzimmer Thermostat) — 21.5°C, target: 22°C
- switch.kaffeemaschine (Kaffeemaschine) — aus
- sensor.aussentemperatur (Außentemperatur) — 8.3°C
```

---

## 8. Web Search Integration — Tavily

```python
import httpx

TAVILY_API_KEY = "tvly-..."

async def web_search(query: str, max_results: int = 3) -> str:
    """Search the web via Tavily and return LLM-ready results."""
    response = await httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",      # fast
            "include_answer": True,        # get a pre-summarized answer
            "include_raw_content": False,
            "max_results": max_results
        }
    )
    data = response.json()

    # Build context string
    context = f"Tavily Antwort: {data.get('answer', 'keine direkte Antwort')}\n\n"
    for r in data.get("results", []):
        context += f"- {r['title']}: {r['content'][:200]}\n"

    return context
```

---

## 9. Latency Budget

Target: **≤ 5 seconds** from end-of-speech to first audio output.

| Step | Component | Time | Notes |
|------|-----------|------|-------|
| 1 | Wake word → start capture | ~50ms | CPU, always ready |
| 2 | STT (faster-whisper medium INT8) | 1000–2000ms | Jetson GPU |
| 2' | Speaker ID (parallel with STT) | 200–500ms | Jetson CPU, hidden behind STT |
| 3 | Intent routing | 20–200ms | Keyword: 20ms / LLM: 200ms |
| 4a | Memory retrieval | ~50ms | SQLite + LanceDB local |
| 4b | Web search (if needed) | 500–1500ms | Tavily API, user gets "Moment..." |
| 5 | LLM generation (Gemma 3 12B) | 1500–2500ms | vLLM on 4070, 100 tokens |
| 6 | TTS (Piper, first sentence) | 300–500ms | Streaming: play while generating |
| **Total (fast path, no web)** | | **~3.0–5.0s** | **Within budget** |
| **Total (web search path)** | | **~4.0–6.5s** | **Marginal, user informed** |

### Speed Optimizations

1. **Pipeline STT → Router while STT still finishing:** Start intent classification on partial transcripts
2. **TTS streaming:** Begin audio playback on the first sentence while the LLM generates the rest
3. **Keep Whisper warm:** Don't unload the model between requests (~3GB stays in VRAM)
4. **HA entity pre-fetch:** Cache is always ready, no network call during request
5. **Connection pooling:** Reuse HTTP connections to LLM server and HA

---

## 10. Configuration

### 10.1 Main Config File

```yaml
# config.yaml
nabu:
  language: de
  wake_word: "hey_nabu"
  log_level: info

audio:
  input_device: "USB Audio"   # or device index
  output_device: "USB Audio"
  sample_rate: 16000
  vad_threshold: 0.5          # voice activity detection

stt:
  model: "medium"
  device: "cuda"
  compute_type: "int8"
  language: "de"              # force German, or "auto" for detection

tts:
  model: "de_DE-thorsten-high"
  speed: 1.0

speaker_id:
  enabled: true
  model: "wespeaker-voxceleb-resnet34"
  threshold: 0.65             # cosine similarity threshold
  fallback_user: "guest"

llm:
  primary:
    url: "http://192.168.10.11:8000/v1/chat/completions"
    model: "google/gemma-3-12b-it"
    health_check_interval: 30

home_assistant:
  url: "http://192.168.1.50:8123"
  token: "${HA_LONG_LIVED_TOKEN}"    # from environment variable
  sync_interval: 300                  # seconds

web_search:
  provider: tavily
  api_key: "${TAVILY_API_KEY}"
  max_results: 3
  search_depth: basic

memory:
  sqlite_path: "./data/nabu.db"
  lancedb_path: "./data/memory.lance"
  embedding_model: "paraphrase-multilingual-MiniLM-L12-v2"
  proactive_extraction: true
  max_context_turns: 5
  max_semantic_results: 3

users:
  joerg:
    display_name: "Jörg"
    language: de
    tts_voice: "de_DE-thorsten-high"
  isabel:
    display_name: "Isabel"
    language: de
    tts_voice: "de_DE-kerstin-high"
  guest:
    display_name: "Gast"
    language: de
    tts_voice: "de_DE-thorsten-high"
```

---

## 11. Project Structure

```
nabu/
├── config.yaml                  # Main configuration
├── prompts.yaml                 # All prompt templates
├── requirements.txt
├── data/
│   ├── nabu.db                 # SQLite database
│   └── memory.lance/           # LanceDB vector store
├── src/
│   ├── __init__.py
│   ├── main.py                 # Entry point, async event loop
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── capture.py          # Mic input + VAD
│   │   ├── playback.py         # Speaker output
│   │   └── wake_word.py        # openWakeWord integration
│   ├── stt/
│   │   ├── __init__.py
│   │   └── whisper.py          # faster-whisper wrapper
│   ├── tts/
│   │   ├── __init__.py
│   │   └── piper.py            # Piper TTS wrapper
│   ├── speaker_id/
│   │   ├── __init__.py
│   │   └── identifier.py       # Voice identification
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py           # OpenAI-compatible client with failover
│   │   └── prompt_manager.py   # YAML prompt loader + template engine
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── router.py           # Intent classification
│   │   ├── pipeline.py         # Main orchestration loop
│   │   └── context_builder.py  # Assembles context per intent
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── sqlite_store.py     # SQLite operations
│   │   ├── vector_store.py     # LanceDB operations
│   │   └── extractor.py        # Proactive memory extraction
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── home_assistant.py   # HA REST API client
│   │   └── tavily.py           # Web search client
│   └── utils/
│       ├── __init__.py
│       └── config.py           # YAML config loader
└── scripts/
    ├── enroll_voice.py         # Speaker enrollment utility
    ├── sync_ha_entities.py     # Manual HA sync
    └── test_latency.py         # End-to-end latency benchmark
```

---

## 12. Implementation Order

**Phase 1 — Audio + STT + TTS (Get sound working)**
1. Set up USB mic/speaker on Jetson, test with `sounddevice`
2. Install and test faster-whisper with German audio
3. Install and test Piper TTS with German voice
4. Build basic loop: mic → Whisper → print → Piper → speaker

**Phase 2 — LLM Integration (Make it talk back)**
5. Verify Gemma 3 12B-IT on the backend PC via vLLM
6. Build the LLM client with health check
7. Wire STT → LLM → TTS into a working conversation

**Phase 3 — Wake Word + Speaker ID (Make it hands-free)**
8. Integrate openWakeWord, train custom "Hey Nabu"
9. Add speaker identification, enroll Joerg and Isabel
10. Personalize greetings and responses per user

**Phase 4 — Intent Routing + HA (Make it useful)**
11. Build the intent router (start with keyword matching)
12. Integrate Home Assistant REST API
13. Implement HA entity sync + cached entity list
14. Test full home control flow

**Phase 5 — Memory (Make it smart)**
15. Set up SQLite schema + conversation logging
16. Set up LanceDB + embedding model
17. Implement proactive memory extraction
18. Add memory retrieval to context building

**Phase 6 — Web Search (Make it knowledgeable)**
19. Integrate Tavily API
20. Add "Moment..." feedback before search
21. Test web search flow end-to-end

**Phase 7 — Polish**
22. Prompt tuning and iteration
23. Latency optimization (streaming TTS, pipeline overlap)
24. Error handling and graceful degradation
25. Systemd service for auto-start on boot

---

## 13. Key Trade-offs & Decisions

| Decision | Choice | Alternative | Rationale |
|----------|--------|-------------|-----------|
| LLM location | 4070 PC primary | Jetson only | Much faster inference, 12GB VRAM handles 12B model easily |
| LLM model | Gemma 3 12B-IT | Qwen 7B, smaller models | Best available on 12GB VRAM; strong multilingual/German |
| LLM server | vLLM | llama.cpp | Already running, OpenAI-compatible API |
| Jetson fallback LLM | None | Qwen 1.5B | 8GB too tight with Whisper medium; not worth the complexity |
| STT model | Whisper medium INT8 | Whisper small | German accuracy matters; VRAM available after removing HA + fallback LLM |
| Vector DB | LanceDB | ChromaDB, Qdrant | Embedded (no server), tiny footprint, fast |
| Intent routing | Keyword + embeddings | LLM classification | Saves 200ms per request, good enough for v1 |
| HA integration | REST API to existing | Run HA on Jetson | Frees ~694 MiB RAM on Jetson, existing setup works |
| HA service calls | Whitelisted services only | Let LLM call anything | Security: prevent hallucinated destructive actions |
| TTS engine | Piper | Coqui, eSpeak | Best German voices, lightweight, proven |
| Prompt config | YAML files | Database, code | Easy to edit, version control, no restart needed |
| Web search | Tavily | Brave, SearXNG | Pre-summarized results save tokens in small context |

---

## 14. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| 4070 PC is off/sleeping | No LLM available | Announce degraded mode; handle system/memory queries locally without LLM |
| Whisper German accuracy | Misunderstood commands | Use medium model; add common corrections dict for household terms |
| Speaker ID false match | Wrong user profile | Require confidence > 0.65; announce detected user ("Hallo Jörg!") so user can correct |
| LLM hallucinates HA actions | Wrong device controlled | Whitelist allowed services; validate entity_id against cache before executing |
| Memory grows unbounded | Slow retrieval, disk full | Monthly cleanup of low-confidence extracted memories; cap conversation log at 90 days |
| Network between devices fails | No LLM, no HA | STT + TTS still work locally; announce connectivity issues |

---

## 15. Future Extensions

- **Camera + face recognition** for visual user identification (complements voice ID)
- **Multi-room satellites** — Raspberry Pi + mic/speaker nodes that stream audio to the Jetson
- **Music playback** via HA media_player integration
- **Calendar integration** — "Was steht morgen an?"
- **Proactive notifications** — "Jörg, es wird gleich regnen, die Fenster sind noch offen"
- **Fine-tuned wake word** per user — "Hey Nabu" (Joerg) vs "Hallo Nabu" (Isabel)
- **Conversation continuity** — follow-up questions without re-triggering wake word
