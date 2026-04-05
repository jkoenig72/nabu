# Phase 5 — Home Assistant Integration (Proof of Concept)

## Goal

Control Home Assistant devices via voice shortcuts — no speaker authentication
required. Say "OK Nabu, Abendbeleuchtung einschalten" and the lights turn on.

## Architecture

```
"OK Nabu, Abendbeleuchtung einschalten"
  ↓
Wake word detected + shortcut matched in same transcript
  ↓
ShortcutHandler → HAClient REST API → HA toggles switch
  ↓
"Abendbeleuchtung ist eingeschaltet." (TTS)
  ↓
Back to wake word listening (no conversation loop)
```

Shortcuts bypass speaker identification and the conversation loop entirely.
They execute immediately from the wake word transcript.

## Components

### HAClient (`app/homeassistant/client.py`)

Minimal REST API client for Home Assistant.

- `call_service(domain, service, entity_id)` — POST to `/api/services/{domain}/{service}`
- `get_state(entity_id)` — GET from `/api/states/{entity_id}`
- Authentication via Long-Lived Access Token

### ShortcutHandler (`app/homeassistant/shortcuts.py`)

Matches voice commands against configured regex patterns.

- Parses the full wake-word transcript (e.g., "Okay Nabu Abendbeleuchtung einschalten")
- Detects on/off from keywords: "ein/an" → turn_on, "aus/abschalt" → turn_off
- Returns response text for TTS, or None if no match (falls through to normal flow)

## Configuration

```yaml
homeassistant:
  url: "http://192.168.10.22:8123"
  token: "<long-lived-access-token>"
  timeout: 5.0
  shortcuts:
    - name: "Abendbeleuchtung"
      entity_id: "switch.evening_lights"
      domain: "switch"
      patterns:
        - "abendbeleuchtung.{0,10}ein"
        - "abendbeleuchtung.{0,10}an"
        - "abendbeleuchtung.{0,10}aus"
      response_on: "Abendbeleuchtung ist eingeschaltet."
      response_off: "Abendbeleuchtung ist ausgeschaltet."
```

### Adding More Shortcuts

Add entries to `shortcuts` in `config.yaml`:

```yaml
    - name: "Kaffeemaschine"
      entity_id: "switch.coffee_machine"
      domain: "switch"
      patterns:
        - "kaffee.{0,10}ein"
        - "kaffee.{0,10}an"
        - "kaffee.{0,10}aus"
      response_on: "Kaffeemaschine läuft."
      response_off: "Kaffeemaschine ist aus."
```

Patterns are regex, matched against the normalized (lowercase, punctuation removed)
wake word transcript.

## Usage

```
User:  "OK Nabu, Abendbeleuchtung einschalten"
Nabu:  "Abendbeleuchtung ist eingeschaltet."
       (back to listening — no auth, no conversation)

User:  "OK Nabu, Abendbeleuchtung ausschalten"
Nabu:  "Abendbeleuchtung ist ausgeschaltet."
```

## Test Coverage

- 8 HAClient tests (enabled/disabled, service calls, state queries, error handling)
- 8 ShortcutHandler tests (ein/aus/an, no match, disabled, failure, punctuation)
- 240 total tests passing

## Debug Logging

```
grep "Shortcut\|HA call\|HA response" data/nabu.log
```
