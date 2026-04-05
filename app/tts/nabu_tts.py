"""TTS wrapper: HAL9000 primary with streaming, Piper local fallback."""

import logging

from app.tts.hal_tts import HalTTS
from app.tts.piper_tts import PiperTTS

log = logging.getLogger(__name__)


class NabuTTS:
    """TTS with automatic fallback: HAL9000 (remote) -> Piper (local)."""

    def __init__(self, config):
        tts_cfg = config.get("tts", {})
        hal_cfg = tts_cfg.get("hal")

        self._hal = None
        self._piper = None
        self._hal_available = False

        log.info("Loading Piper TTS (fallback)...")
        self._piper = PiperTTS(config)

        if hal_cfg and hal_cfg.get("enabled", True):
            try:
                self._hal = HalTTS(config)
                if self._hal.health_check():
                    self._hal_available = True
                    log.info("HAL TTS server reachable — using as primary TTS (streaming)")
                else:
                    log.warning("HAL TTS server not reachable — using Piper as primary")
            except Exception as e:
                log.warning("HAL TTS init failed: %s — using Piper as primary", e)
        else:
            log.info("HAL TTS not configured — using Piper only")

    @property
    def active_engine(self):
        return "hal" if self._hal_available else "piper"

    def synthesize(self, text):
        """Convert text to WAV bytes, with automatic fallback."""
        if self._hal_available:
            try:
                return self._hal.synthesize(text)
            except Exception as e:
                log.warning("HAL TTS failed (%s), falling back to Piper", e)
                self._hal_available = False

        return self._piper.synthesize(text)

    def speak(self, text, playback):
        """Synthesize and play. Streams with HAL, full WAV with Piper."""
        if self._hal_available:
            try:
                self._hal.stream_to_player(text, playback)
                return
            except Exception as e:
                log.warning("HAL TTS streaming failed (%s), falling back to Piper", e)
                self._hal_available = False

        wav_bytes = self._piper.synthesize(text)
        playback.play_wav_bytes(wav_bytes)

    def speak_streamed(self, sentences, playback):
        """Play a sentence generator via HAL TTS streaming. Returns full text."""
        return self._hal.stream_sentences_to_player(sentences, playback)

    def close(self):
        if self._hal:
            self._hal.close()
