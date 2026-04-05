"""HAL9000 TTS client — streams PCM from remote Qwen3-TTS server."""

import io
import logging
import time
import wave

import numpy as np
import httpx

log = logging.getLogger(__name__)

HAL_SAMPLE_RATE = 24000
HAL_CHANNELS = 1
HAL_SAMPLE_WIDTH = 2


class HalTTS:
    """Remote TTS via HAL9000 server. Returns WAV or streams to speaker."""

    def __init__(self, config):
        hal_cfg = config["tts"]["hal"]
        self.base_url = hal_cfg["url"].rstrip("/")
        self.url = self.base_url + "/v1/audio/speech"
        self.voice = hal_cfg.get("voice", "ref2")
        self.language = hal_cfg.get("language", "German")
        self.timeout = hal_cfg.get("timeout", 120.0)
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=self.timeout, write=10.0, pool=10.0)
        )
        log.info("HAL TTS configured: %s voice=%s lang=%s", self.url, self.voice, self.language)

    def _stream_request(self, text):
        """Return a streaming POST context manager for the HAL server."""
        return self._client.stream(
            "POST",
            self.url,
            json={
                "input": text,
                "voice": self.voice,
                "response_format": "pcm",
                "stream": True,
                "language": self.language,
            },
        )

    def synthesize(self, text):
        """Stream PCM from server, collect all chunks, return as WAV bytes."""
        t0 = time.monotonic()
        log.debug("HAL TTS: synthesizing %d chars: '%s'", len(text), text[:80])

        with self._stream_request(text) as resp:
            resp.raise_for_status()
            pcm_chunks = []
            first = True
            for chunk in resp.iter_bytes(chunk_size=4096):
                if first:
                    log.debug("HAL TTS: first chunk in %.2fs", time.monotonic() - t0)
                    first = False
                pcm_chunks.append(chunk)

        pcm_data = b"".join(pcm_chunks)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(HAL_CHANNELS)
            wf.setsampwidth(HAL_SAMPLE_WIDTH)
            wf.setframerate(HAL_SAMPLE_RATE)
            wf.writeframes(pcm_data)

        elapsed = time.monotonic() - t0
        duration = len(pcm_data) / (HAL_SAMPLE_RATE * HAL_SAMPLE_WIDTH * HAL_CHANNELS)
        rtf = duration / elapsed if elapsed > 0 else 0
        log.debug("HAL TTS: %.2fs audio in %.2fs (%.1fx realtime)", duration, elapsed, rtf)

        return wav_buffer.getvalue()

    def stream_to_player(self, text, playback):
        """Stream PCM from HAL and play chunks as they arrive."""
        import sounddevice as sd

        t0 = time.monotonic()
        log.debug("HAL TTS stream: %d chars: '%s'", len(text), text[:80])

        with self._stream_request(text) as resp:
            resp.raise_for_status()

            play_rate = playback._device_rate or HAL_SAMPLE_RATE
            need_resample = play_rate != HAL_SAMPLE_RATE

            stream = sd.OutputStream(
                samplerate=play_rate,
                channels=HAL_CHANNELS,
                dtype="float32",
                device=playback.device_index,
                blocksize=0,
            )
            stream.start()

            first = True
            total_bytes = 0
            leftover = b""

            try:
                for chunk in resp.iter_bytes(chunk_size=4096):
                    if first:
                        log.debug("HAL TTS stream: first audio at %.2fs", time.monotonic() - t0)
                        first = False

                    data = leftover + chunk
                    usable = len(data) - (len(data) % HAL_SAMPLE_WIDTH)
                    leftover = data[usable:]
                    pcm = data[:usable]

                    if not pcm:
                        continue

                    total_bytes += len(pcm)
                    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                    audio = audio * playback.volume

                    if need_resample:
                        audio = playback._resample(audio, HAL_SAMPLE_RATE, play_rate)

                    stream.write(audio)

                # Flush remaining aligned bytes
                if leftover and len(leftover) >= HAL_SAMPLE_WIDTH:
                    usable = len(leftover) - (len(leftover) % HAL_SAMPLE_WIDTH)
                    pcm = leftover[:usable]
                    total_bytes += len(pcm)
                    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                    audio = audio * playback.volume
                    if need_resample:
                        audio = playback._resample(audio, HAL_SAMPLE_RATE, play_rate)
                    stream.write(audio)
            finally:
                stream.stop()
                stream.close()

        elapsed = time.monotonic() - t0
        duration = total_bytes / (HAL_SAMPLE_RATE * HAL_SAMPLE_WIDTH * HAL_CHANNELS)
        log.debug("HAL TTS stream: %.2fs audio, %.2fs wall", duration, elapsed)

    def _play_pcm_stream(self, text, sd_stream, play_rate, playback):
        """Stream one sentence from HAL TTS into an open sounddevice stream."""
        need_resample = play_rate != HAL_SAMPLE_RATE
        total_bytes = 0
        leftover = b""

        with self._stream_request(text) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes(chunk_size=4096):
                data = leftover + chunk
                usable = len(data) - (len(data) % HAL_SAMPLE_WIDTH)
                leftover = data[usable:]
                pcm = data[:usable]
                if not pcm:
                    continue
                total_bytes += len(pcm)
                audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                audio = audio * playback.volume
                if need_resample:
                    audio = playback._resample(audio, HAL_SAMPLE_RATE, play_rate)
                sd_stream.write(audio)

            if leftover and len(leftover) >= HAL_SAMPLE_WIDTH:
                usable = len(leftover) - (len(leftover) % HAL_SAMPLE_WIDTH)
                pcm = leftover[:usable]
                total_bytes += len(pcm)
                audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                audio = audio * playback.volume
                if need_resample:
                    audio = playback._resample(audio, HAL_SAMPLE_RATE, play_rate)
                sd_stream.write(audio)

        return total_bytes / (HAL_SAMPLE_RATE * HAL_SAMPLE_WIDTH * HAL_CHANNELS)

    def stream_sentences_to_player(self, sentences, playback):
        """Play a sentence generator through HAL TTS. Returns full joined text.

        LLMError propagates to caller. TTS errors are caught and stop playback,
        returning whatever text was collected.
        """
        import sounddevice as sd
        from app.llm.client import LLMError

        t0 = time.monotonic()
        play_rate = playback._device_rate or HAL_SAMPLE_RATE

        stream = sd.OutputStream(
            samplerate=play_rate,
            channels=HAL_CHANNELS,
            dtype="float32",
            device=playback.device_index,
        )
        stream.start()

        collected = []
        try:
            for sentence in sentences:
                collected.append(sentence)
                log.debug("TTS sentence: '%s'", sentence[:80])
                self._play_pcm_stream(sentence, stream, play_rate, playback)
        except LLMError:
            raise
        except Exception as e:
            log.warning("TTS streaming error: %s", e)
        finally:
            stream.stop()
            stream.close()

        full_text = " ".join(collected)
        elapsed = time.monotonic() - t0
        log.debug("Streamed %d sentences in %.2fs", len(collected), elapsed)
        return full_text

    def health_check(self):
        """Check if HAL server is reachable."""
        try:
            url = self.base_url + "/health"
            resp = self._client.get(url, timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def close(self):
        self._client.close()
