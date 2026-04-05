# Phase 1 — Audio + STT + TTS

## Goal
Get a working end-to-end loop: microphone → Whisper STT → print transcript → Piper TTS → speaker.
No wake word, no LLM, no intent routing yet — just audio in/out with transcription.

## Hardware Detected

| Device | Card | Type | Use |
|--------|------|------|-----|
| UM10 USB mic | card 0 | Capture only | Primary microphone |
| USB Composite Device | card 1 | Capture + Playback | Primary speaker output |
| HDMI (HDA) | card 2 | Playback only | Not used |
| Tegra APE | card 3 | Capture + Playback | Not used (internal) |

## Components

### 1. Audio Capture (`app/audio/capture.py`)
- Uses `sounddevice` with InputStream
- 16kHz mono (Whisper's expected format)
- Simple energy-based VAD: start recording on energy above threshold, stop after silence
- Returns `numpy` array of float32 samples
- Configurable: device index, sample rate, silence threshold, max duration

### 2. Audio Playback (`app/audio/playback.py`)
- Uses `sounddevice` with play/wait
- Accepts WAV bytes or numpy array
- Configurable: device index, sample rate

### 3. STT — faster-whisper (`app/stt/whisper.py`)
- Model: `medium` with INT8 on CUDA (fall back to `small` if OOM)
- CTranslate2 backend via `faster-whisper`
- Input: numpy float32 array at 16kHz
- Output: transcript string + language code
- Model stays loaded between calls (warm)

### 4. TTS — Piper (`app/tts/piper.py`)
- Uses `piper-tts` Python package (or subprocess to `piper` binary)
- Voice: `de_DE-thorsten-high`
- Input: text string
- Output: WAV bytes (int16, 22050Hz)
- Auto-downloads voice model on first use

### 5. Integration (`app/main.py`)
- Simple loop: press Enter → record → transcribe → speak back
- Phase 1 just echoes the transcript as TTS (no LLM)

## Python Dependencies
```
sounddevice
numpy
faster-whisper
piper-tts
pytest
```

## System Dependencies
```
sudo apt install -y portaudio19-dev
```

## Test Plan
- `test/test_audio.py` — device listing, short recording returns numpy array
- `test/test_stt.py` — whisper loads, transcribes a known WAV file
- `test/test_tts.py` — piper generates WAV bytes from text
- `test/test_integration.py` — end-to-end with a pre-recorded WAV
