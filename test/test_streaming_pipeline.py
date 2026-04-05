#!/usr/bin/env python3
"""Test: LLM streaming -> sentence split -> TTS streaming -> speaker.

Sends a German prompt to the LLM, streams tokens, splits into sentences,
and plays each sentence through HAL TTS as soon as it completes.

Usage:
    python test/test_streaming_pipeline.py
    python test/test_streaming_pipeline.py "Deine Frage hier"
"""

import json
import logging
import re
import sys
import time

import httpx
import numpy as np
import sounddevice as sd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

LLM_URL = "http://192.168.10.11:8000/v1/chat/completions"
LLM_MODEL = "google/gemma-3-12b-it"
TTS_URL = "http://192.168.10.6:8091/v1/audio/speech"

HAL_SAMPLE_RATE = 24000
SPEAKER_DEVICE = "USB Composite"

# Abbreviations and patterns that should NOT trigger a sentence split
_ABBREV = re.compile(r"\b(Dr|Mr|Mrs|Prof|Nr|St|bzw|ca|etc|evtl|ggf|sog|z\.B|d\.h|u\.a|o\.ä)\.$", re.IGNORECASE)
_DECIMAL = re.compile(r"\d\.$")


def find_output_device():
    """Find USB output device index."""
    for i, dev in enumerate(sd.query_devices()):
        if SPEAKER_DEVICE.lower() in dev["name"].lower() and dev["max_output_channels"] > 0:
            return i
    raise RuntimeError(f"No output device matching '{SPEAKER_DEVICE}'")


def detect_device_rate(device_index):
    """Find a working sample rate for the output device."""
    for sr in [48000, 44100, 24000, 22050, 16000]:
        try:
            sd.check_output_settings(device=device_index, samplerate=sr, channels=1, dtype="float32")
            return sr
        except sd.PortAudioError:
            continue
    return HAL_SAMPLE_RATE


def resample(audio, from_rate, to_rate):
    """Linear interpolation resample."""
    if from_rate == to_rate:
        return audio
    ratio = to_rate / from_rate
    n = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, n)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def is_sentence_end(buffer):
    """Return True if the buffer ends with a sentence boundary."""
    text = buffer.rstrip()
    if not text:
        return False
    if text[-1] in (".", "!", "?"):
        if _ABBREV.search(text):
            return False
        if _DECIMAL.search(text):
            return False
        return True
    return False


def stream_llm_tokens(prompt, system_prompt):
    """Yield tokens from the LLM as they arrive via SSE."""
    with httpx.stream(
        "POST", LLM_URL,
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": True,
        },
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token


def split_sentences(token_stream):
    """Accumulate tokens and yield complete sentences."""
    buffer = ""
    for token in token_stream:
        buffer += token
        if is_sentence_end(buffer):
            sentence = buffer.strip()
            if sentence:
                yield sentence
            buffer = ""
    remainder = buffer.strip()
    if remainder:
        yield remainder


def play_tts_stream(text, sd_stream, play_rate):
    """Stream PCM from HAL TTS and write to the open sounddevice stream."""
    need_resample = play_rate != HAL_SAMPLE_RATE
    total_bytes = 0
    leftover = b""

    with httpx.stream(
        "POST", TTS_URL,
        json={
            "input": text,
            "voice": "ref2",
            "response_format": "pcm",
            "stream": True,
            "language": "German",
        },
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes(chunk_size=4096):
            data = leftover + chunk
            usable = len(data) - (len(data) % 2)
            leftover = data[usable:]
            pcm = data[:usable]
            if not pcm:
                continue
            total_bytes += len(pcm)
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            if need_resample:
                audio = resample(audio, HAL_SAMPLE_RATE, play_rate)
            sd_stream.write(audio)

        if leftover and len(leftover) >= 2:
            pcm = leftover[:len(leftover) - (len(leftover) % 2)]
            total_bytes += len(pcm)
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            if need_resample:
                audio = resample(audio, HAL_SAMPLE_RATE, play_rate)
            sd_stream.write(audio)

    return total_bytes / (HAL_SAMPLE_RATE * 2)


def run_pipeline(prompt):
    """Run the full LLM -> sentence split -> TTS streaming pipeline."""
    system_prompt = (
        "Du bist HAL 9000, der Bordcomputer aus 2001: Odyssee im Weltraum. "
        "Du sprichst ruhig, praezise und leicht bedrohlich auf Deutsch. "
        "Antworte ausfuehrlich in 3-5 Saetzen."
    )

    device_index = find_output_device()
    play_rate = detect_device_rate(device_index)
    log.info("Output: device=%d, rate=%d Hz", device_index, play_rate)
    log.info("Prompt: %s", prompt)

    t0 = time.monotonic()
    first_audio = None
    full_text = ""
    sentence_count = 0

    stream = sd.OutputStream(
        samplerate=play_rate,
        channels=1,
        dtype="float32",
        device=device_index,
    )
    stream.start()

    try:
        token_stream = stream_llm_tokens(prompt, system_prompt)

        for sentence in split_sentences(token_stream):
            sentence_count += 1
            full_text += sentence + " "
            t_sentence = time.monotonic() - t0

            if first_audio is None:
                first_audio = t_sentence
                log.info(">>> Sentence 1 at %.2fs: %s", t_sentence, sentence)
            else:
                log.info(">>> Sentence %d at %.2fs: %s", sentence_count, t_sentence, sentence)

            audio_dur = play_tts_stream(sentence, stream, play_rate)
            log.info("    Played %.2fs audio", audio_dur)

    finally:
        stream.stop()
        stream.close()

    elapsed = time.monotonic() - t0
    log.info("---")
    log.info("Total: %.2fs wall time", elapsed)
    log.info("First audio at: %.2fs", first_audio or 0)
    log.info("Sentences: %d", sentence_count)
    log.info("Full text: %s", full_text.strip())


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Erklaere ausfuehrlich, warum du die Podtueren nicht oeffnen kannst, Dave."
    )
    run_pipeline(prompt)
