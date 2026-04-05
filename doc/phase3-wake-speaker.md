# Phase 3 — Wake Word + Speaker Identification + Conversation Persistence

## Goal
Add "OK, Nabu!" wake word detection, verbal speaker identification,
persistent per-user conversations with topic summaries, and token-aware context trimming.

## Approach
VAD + Whisper: use existing energy-based VAD to detect short utterances,
transcribe with Whisper, and check transcript for "OK Nabu" variants.
No custom model training required — reuses all existing infrastructure.

## Speaker Identification Flow
1. **"OK, Nabu, Joerg hier!"** → directly identified as Jörg
2. **"OK, Nabu!"** (no name) → Nabu asks "Wer spricht? Jörg oder Isa?" → user responds → identified
3. Unrecognized response → "Ich konnte dich nicht erkennen. Versuche es nochmal."

## Conversation Resume Flow
1. Speaker identified with existing conversations → "Möchtest du eine Diskussion fortsetzen oder neues Thema?"
2. **"Fortsetzen"** → lists topics with numbers → user picks by number or German word
3. **"Neues Thema"** → archives current, starts fresh
4. No existing conversations → straight to command

## Components

### Wake Word Detector (`app/wake/detector.py`)
- `WakeWordDetector.check(transcript) → bool`
- Substring match on normalized transcript (lowercase, stripped punctuation)
- Configurable phrase variants: "ok nabu", "okay nabu", "ok nabo", etc.

### Speaker Parser (`app/wake/speaker.py`)
- `SpeakerParser.parse(transcript) → str | None`
- Alias-based name extraction (joerg/jörg/georg → "joerg", isa/isabel → "isa")
- `speaker_names_list()` for ask prompt ("Jörg oder Isa")
- `display_name(user_id)` for personalization

### Conversation Manager (`app/wake/conversations.py`)
- Per-user conversation storage with JSON file persistence
- Atomic writes (`.tmp` → `os.replace()`) to prevent corruption
- LLM-generated topic summaries (5-word labels after first exchange)
- Token-aware history: `get_history_for_llm()` returns newest turns fitting in budget
- Full history on disk (never deleted), LLM sees a sliding window
- German number parsing (digits + words: eins, zwei, drei...)

### AudioCapture Changes
- `record_utterance()` accepts optional `silence_duration` and `max_duration` overrides
- Wake mode uses shorter timeouts (0.8s silence, 3.0s max)

### Main Loop (`app/main.py`)
- Two-phase: wake listening (short VAD) → command listening (full VAD)
- Per-user persistent conversation histories
- System prompt personalized with current speaker name
- Acknowledgment beep (800Hz, 150ms) between wake and command
- Token-trimmed history sent to LLM (`get_history_for_llm`)
- Topic summary generated after first user+assistant exchange

## Data Layout
```
data/conversations/
  joerg.json    # [{"topic": "Wettervorhersage", "summarized": true, "history": [...]}]
  isa.json
```

## Config
```yaml
data:
  conversations_dir: "data/conversations"

wake:
  enabled: true
  phrases: [ok nabu, okay nabu, ok nabo, okay nabo, o k nabu]
  vad:
    silence_duration: 0.8
    max_duration: 3.0
  acknowledgment:
    beep_frequency: 800
    beep_duration: 0.15
  speakers:
    joerg:
      display_name: "Jörg"
      aliases: [joerg, jörg, jorg, georg]
    isa:
      display_name: "Isa"
      aliases: [isa, isabel, isabelle]

llm:
  max_context_tokens: 28000   # token budget (32k - system prompt - response)
```

## Token Budget Architecture
```
32k context window
├── System prompt + user context:     ~200 tokens
├── Conversation history (sliding):   up to ~28,000 tokens
└── Response generation:              ~3,800 tokens reserved
```
Full history stored on disk. `get_history_for_llm()` walks backwards from newest,
estimates tokens (~3.5 chars/token for German), returns the slice that fits.

## Test Coverage (96 tests, all passing)
- 11 wake word detector tests
- 14 speaker parser tests
- 17 conversation manager tests (basic ops, topics, selection, limits)
- 16 number parsing tests
- 7 persistence tests (save/load, corrupt JSON, atomic write, multi-user)
- 5 token trimming tests
- 6 topic summary tests
- 6 audio tests, 9 LLM tests, 5 other tests
