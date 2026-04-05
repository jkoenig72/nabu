# Nabu Voice Assistant — Migration Handoff

## Date: 2026-04-05

## Overview

Nabu is a fully local, privacy-first voice assistant. Development has been migrated from an 8GB Jetson Orin Nano (192.168.11.167) to a 16GB Jetson Orin NX (192.168.11.98). This document captures the full state so a new Claude Code session on the 16GB machine can continue development.

---

## Architecture

```
┌─────────────────────────────────┐
│  Jetson Orin NX 16GB            │  192.168.11.98 (hostname: nabu)
│  - Audio I/O (USB Composite)    │
│  - faster-whisper STT (CUDA)    │
│  - Piper TTS (CPU)              │
│  - openWakeWord (CPU)           │
│  - wespeaker (CPU)              │
│  - Python orchestrator          │
└────────────┬────────────────────┘
             │ HTTP (OpenAI-compatible API)
             ▼
┌─────────────────────────────────┐
│  Backend PC (RTX 4070 12GB)     │  192.168.10.11:8000
│  - Gemma 3 12B-IT via vLLM      │
│  - 32K context                   │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│  Home Assistant                  │  Separate server, REST API
│  (NOT on Jetson)                 │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│  Tavily                          │  Web search (only cloud dependency)
└─────────────────────────────────┘
```

---

## New Machine: 192.168.11.98

### Hardware
- **Device:** Seeed reComputer Mini J4012
- **SoC:** Jetson Orin NX 16GB
- **CPU:** 4-core aarch64 (Cortex-A78AE)
- **RAM:** 16GB (14GB available at idle)
- **Storage:** 937GB NVMe (858GB free)

### Software
- **OS:** Ubuntu 22.04 (Jammy)
- **Kernel:** 5.15.136-tegra (original R36.3.0 kernel — DO NOT UPGRADE)
- **JetPack:** R36.3.0 (base), with r36.4 apt repo enabled for userspace libs
- **CUDA Toolkit:** 12.2.140 (system-installed)
- **cuDNN:** 9.3.0 (installed from r36.4 repo)
- **NVIDIA Driver:** 540.3.0 (integrated Tegra)
- **Python:** 3.10.12

### Login
- **User:** fritz
- **Password:** arkon77
- **Sudo password:** arkon77

---

## Directory Layout on 192.168.11.98

```
/home/fritz/
├── nabu-dev/              # Main project code
│   ├── app/               # Application source
│   │   ├── main.py        # Entry point
│   │   ├── config.py      # Configuration loader
│   │   ├── config.yaml    # Runtime configuration
│   │   ├── audio/         # Audio capture (VAD)
│   │   ├── stt/           # Speech-to-text (faster-whisper)
│   │   ├── tts/           # Text-to-speech (Piper)
│   │   ├── wake/          # Wake word detection (openWakeWord)
│   │   ├── llm/           # LLM client (vLLM on backend PC)
│   │   ├── intent/        # Intent processing
│   │   ├── homeassistant/ # Home Assistant REST API client
│   │   ├── search/        # Tavily web search
│   │   └── memory/        # Memory/RAG (lancedb)
│   ├── test/              # Tests (pytest)
│   ├── doc/               # Documentation
│   ├── data/              # Runtime data
│   ├── run.sh             # Launch script (sets LD_LIBRARY_PATH)
│   ├── pytest.ini         # Test configuration
│   └── README.md
│
├── nabu-venv/             # Python virtual environment
│
├── ct2-install/           # Custom-built CTranslate2 with CUDA
│   └── lib/
│       └── libctranslate2.so  # CUDA-enabled shared library
│
├── ha-hal/                # Home Assistant / Wyoming ecosystem
│   ├── wyoming/
│   │   ├── piper/         # Piper TTS voice models
│   │   │   ├── de_DE-thorsten_emotional-medium.onnx  # Primary German voice
│   │   │   ├── en_US-lessac-medium.onnx              # Primary English voice
│   │   │   └── ... (many more voices)
│   │   └── whisper/       # Whisper models (faster-whisper format)
│   ├── docker-compose.yml
│   └── satellite/
│
├── .asoundrc              # ALSA audio device configuration
├── .claude/               # Claude Code config and memory
├── .claude.json           # Claude Code session state
├── bin/                   # Utility scripts
└── logs/                  # Application logs
```

---

## How to Run Nabu

```bash
cd ~/nabu-dev
./run.sh
```

The `run.sh` script sets the critical `LD_LIBRARY_PATH` for the custom CTranslate2 build:
```bash
export LD_LIBRARY_PATH=/home/fritz/ct2-install/lib:${LD_LIBRARY_PATH}
```

This is **required** for faster-whisper to use CUDA. Without it, ctranslate2 loads the PyPI CPU-only wheel instead.

---

## Python Environment

### Virtual Environment
- **Location:** `/home/fritz/nabu-venv/`
- **Python:** 3.10.12
- **Activation:** `source ~/nabu-venv/bin/activate`

### Key Packages and Versions

| Package | Version | Purpose | Notes |
|---|---|---|---|
| faster-whisper | 1.2.1 | Speech-to-text | Uses ctranslate2 CUDA backend |
| ctranslate2 | 4.7.1 | CUDA inference engine | PyPI wheel installed, but `_ext.so` replaced with custom CUDA-enabled build from ct2-install |
| piper-tts | 1.4.1 | Text-to-speech | thorsten_emotional-medium voice, speaker_id=4 (neutral) |
| openwakeword | 0.6.0 | Wake word detection | CPU-based |
| sounddevice | 0.5.5 | Audio I/O | Requires libportaudio2 system package |
| onnxruntime | 1.20.1 | ONNX inference | **Must stay at 1.20.1** — version 1.23.2 crashes on Orin NX CPU |
| lancedb | 0.30.2 | Vector database | For memory/RAG |
| sentence-transformers | 5.3.0 | Text embeddings | For memory/RAG |
| torch | 2.11.0+cu130 | ML framework | CUDA **not** functional (driver too old) — Nabu doesn't use torch CUDA, only ctranslate2 |
| numpy | 2.2.6 | Numerical computing | |
| pydantic | 2.12.5 | Data validation | |
| httpx | 0.28.1 | HTTP client | For vLLM and HA API calls |
| pytest | 9.0.2 | Testing | |

### CTranslate2 CUDA — How It Works

The PyPI ctranslate2 wheel is CPU-only on aarch64. To get CUDA:
1. CTranslate2 was built from source on the 8GB Jetson (build artifacts at `/home/fritz/ct2-install/`)
2. The compiled `libctranslate2.so` lives in `/home/fritz/ct2-install/lib/`
3. The Python extension `_ext.cpython-310-aarch64-linux-gnu.so` was copied from the source build into the pip-installed package at `/home/fritz/nabu-venv/lib/python3.10/site-packages/ctranslate2/`
4. At runtime, `LD_LIBRARY_PATH=/home/fritz/ct2-install/lib` must be set (run.sh does this)
5. This gives full CUDA with float16, int8, bfloat16 compute types

**If you ever need to rebuild ctranslate2**, the source is at `/home/fritz/CTranslate2/` on the old 8GB machine (192.168.11.167). The build directory was `/home/fritz/ct2-build/`.

---

## CUDA Verification

To verify CUDA is working:
```bash
LD_LIBRARY_PATH=/home/fritz/ct2-install/lib:$LD_LIBRARY_PATH \
  ~/nabu-venv/bin/python -c "
import ctranslate2
print('CUDA types:', ctranslate2.get_supported_compute_types('cuda'))
"
```
Expected output: `{'float16', 'int8_float32', 'int8_float16', 'bfloat16', 'int8_bfloat16', 'float32', 'int8'}`

---

## Audio Configuration

- **Audio device:** USB Composite Device (card 1) for both input and output
- **UM10 mic (card 0):** Not visible to PortAudio
- **Config:** `~/.asoundrc` defines ALSA device routing
- **Whisper performance:** ~0.1s transcription for 3s audio on GPU (float16)

---

## Project Status

### Phase 1 — COMPLETE
Audio capture (VAD) → Whisper STT (CUDA) → Piper TTS → speaker playback.
14/14 tests passing (on old machine — need to verify on new machine).

### Remaining Phases
- Phase 2+: LLM integration, intent processing, Home Assistant control, memory/RAG, web search (Tavily)

---

## CRITICAL WARNINGS

### ⚠️ NEVER upgrade kernel/bootloader on Jetson via apt
```
DO NOT RUN:
  apt-get dist-upgrade
  apt-get install nvidia-l4t-kernel*
  apt-get install nvidia-l4t-bootloader*
  apt-get install nvidia-l4t-display-kernel*
```
A `dist-upgrade` on this exact machine bricked it and required a full reflash. Only upgrade **userspace** libraries (cuDNN, TensorRT, CUDA toolkit) with targeted `apt-get install <package-name>`.

### ⚠️ NEVER add cma= to extlinux.conf
This breaks GPU access on 8GB Jetsons. May also affect the 16GB.

### ⚠️ onnxruntime version
Must stay at **1.20.1**. Version 1.23.2 crashes with `Unknown CPU vendor` assertion failure on Orin NX.

### ⚠️ ctranslate2 — don't pip upgrade
If you `pip install --upgrade ctranslate2`, it will replace the custom CUDA `_ext.so` with the CPU-only PyPI version. You'd need to re-copy the custom build.

---

## .bashrc Additions

The following lines were added to `~/.bashrc` on 192.168.11.98:
```bash
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
```

---

## APT Repository Note

The NVIDIA apt source was changed from r36.3 to r36.4 to get cuDNN 9:
```
/etc/apt/sources.list.d/nvidia-l4t-apt-source.list
  deb https://repo.download.nvidia.com/jetson/common r36.4 main
  deb https://repo.download.nvidia.com/jetson/t234 r36.4 main
  deb https://repo.download.nvidia.com/jetson/ffmpeg r36.4 main
```
This means `apt-get update` will show ~85 upgradable packages (kernel, bootloader, etc). **DO NOT upgrade them.** Only install specific userspace packages.

---

## Old Machine (for reference only)

- **IP:** 192.168.11.167
- **Device:** Jetson Orin Nano 8GB
- **JetPack:** R36.4.x (fully upgraded userspace)
- **Role:** Previous dev machine, still has CTranslate2 source build at `/home/fritz/CTranslate2/` and `/home/fritz/ct2-build/`
- **Status:** Still operational, can be used as reference

---

## Tests

```bash
cd ~/nabu-dev
source ~/nabu-venv/bin/activate
export LD_LIBRARY_PATH=/home/fritz/ct2-install/lib:$LD_LIBRARY_PATH
pytest
```

Phase 1 had 14/14 tests passing. Should verify on the new machine after migration.

---

## Next Steps

1. SSH into 192.168.11.98 and run Claude Code directly
2. Verify tests pass: `cd ~/nabu-dev && ./run.sh` (or run pytest)
3. Check audio device is connected and detected
4. Continue with Phase 2+ development (LLM integration, HA control, etc.)
5. With 16GB RAM, there's now room for a local fallback LLM on the Jetson if desired
