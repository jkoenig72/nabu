import logging

from faster_whisper import WhisperModel

log = logging.getLogger(__name__)


class WhisperSTT:
    def __init__(self, config):
        stt_cfg = config["stt"]
        device = stt_cfg["device"]
        compute_type = stt_cfg["compute_type"]
        model_size = stt_cfg["model_size"]
        cache_dir = stt_cfg.get("model_cache_dir")

        # Try loading from HF cache, fall back to model name
        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=cache_dir,
            )
        except Exception:
            log.warning("Failed to load model '%s' from cache, trying default download", model_size)
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )

        self.language = stt_cfg.get("language")
        self.beam_size = stt_cfg.get("beam_size", 5)
        log.info("Whisper model '%s' loaded on %s (%s)", model_size, device, compute_type)

    def transcribe(self, audio):
        """Return (transcript, language) for a float32 audio array."""
        import time
        t0 = time.monotonic()
        log.debug("STT: transcribing %d samples (%.2fs)", len(audio), len(audio) / 16000)
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
        )
        transcript = " ".join(seg.text.strip() for seg in segments)
        elapsed = time.monotonic() - t0
        log.debug("STT: '%s' [%s] in %.2fs", transcript, info.language, elapsed)
        return transcript, info.language
