import io
import logging
import wave

from piper import PiperVoice
from piper.config import SynthesisConfig

log = logging.getLogger(__name__)


class PiperTTS:
    def __init__(self, config):
        tts_cfg = config["tts"]
        self.voice = PiperVoice.load(
            tts_cfg["model_path"],
            config_path=tts_cfg["config_path"],
        )
        self.syn_config = SynthesisConfig(
            speaker_id=tts_cfg.get("speaker_id"),
            length_scale=tts_cfg.get("length_scale", 1.0),
        )
        log.info("Piper TTS loaded: %s", tts_cfg["model_path"])

    def synthesize(self, text):
        """Return WAV bytes (int16 mono) for the given text."""
        import time
        t0 = time.monotonic()
        log.debug("TTS: synthesizing %d chars: '%s'", len(text), text[:80])

        all_audio = b""
        sample_rate = 22050

        for chunk in self.voice.synthesize(text, syn_config=self.syn_config):
            all_audio += chunk.audio_int16_bytes
            sample_rate = chunk.sample_rate

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(all_audio)

        elapsed = time.monotonic() - t0
        duration = len(all_audio) / (sample_rate * 2)
        log.debug("TTS: %.2fs audio in %.2fs (%.1fx realtime)", duration, elapsed, duration / elapsed if elapsed > 0 else 0)
        return wav_buffer.getvalue()
