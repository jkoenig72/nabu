import io
import wave

import pytest

from app.tts.piper_tts import PiperTTS


@pytest.fixture
def tts_config():
    return {
        "tts": {
            "model_path": "/home/fritz/ha-hal/wyoming/piper/de_DE-thorsten_emotional-medium.onnx",
            "config_path": "/home/fritz/ha-hal/wyoming/piper/de_DE-thorsten_emotional-medium.onnx.json",
            "speaker_id": 4,
            "length_scale": 1.0,
            "output_sample_rate": 22050,
        }
    }


@pytest.mark.slow
class TestPiperTTS:
    def test_model_loads(self, tts_config):
        tts = PiperTTS(tts_config)
        assert tts.voice is not None

    def test_synthesize_returns_wav(self, tts_config):
        tts = PiperTTS(tts_config)
        wav_bytes = tts.synthesize("Hallo Welt")
        assert len(wav_bytes) > 44  # WAV header is 44 bytes
        assert wav_bytes[:4] == b"RIFF"

    def test_synthesize_wav_parseable(self, tts_config):
        tts = PiperTTS(tts_config)
        wav_bytes = tts.synthesize("Das ist ein Test.")
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getframerate() == 22050
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() > 0

    def test_synthesize_german_sentence(self, tts_config):
        tts = PiperTTS(tts_config)
        wav_bytes = tts.synthesize("Guten Morgen, wie geht es dir?")
        assert len(wav_bytes) > 1000  # should be a meaningful amount of audio
