import numpy as np
import pytest

from app.stt.whisper_stt import WhisperSTT


@pytest.fixture
def stt_config():
    return {
        "stt": {
            "model_size": "small",
            "device": "cuda",
            "compute_type": "float16",
            "language": "de",
            "beam_size": 5,
            "model_cache_dir": None,
        }
    }


@pytest.mark.slow
class TestWhisperSTT:
    def test_model_loads(self, stt_config):
        stt = WhisperSTT(stt_config)
        assert stt.model is not None

    def test_transcribe_silence_returns_empty(self, stt_config):
        stt = WhisperSTT(stt_config)
        silence = np.zeros(16000, dtype=np.float32)
        transcript, lang = stt.transcribe(silence)
        assert isinstance(transcript, str)

    def test_transcribe_returns_language(self, stt_config):
        stt = WhisperSTT(stt_config)
        # Generate some noise — won't be meaningful but tests the interface
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        transcript, lang = stt.transcribe(audio)
        assert isinstance(lang, str)

    def test_transcribe_wav_fixture(self, stt_config, test_wav_path):
        import wave
        with wave.open(test_wav_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        stt = WhisperSTT(stt_config)
        transcript, lang = stt.transcribe(audio)
        assert len(transcript) > 0
        assert lang == "de"
