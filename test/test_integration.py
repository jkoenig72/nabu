import io
import wave

import numpy as np
import pytest

from app.stt.whisper_stt import WhisperSTT
from app.tts.piper_tts import PiperTTS


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
class TestIntegration:
    def test_tts_produces_valid_audio(self, tts_config):
        """Synthesize text and verify the output is valid WAV audio."""
        tts = PiperTTS(tts_config)
        wav_bytes = tts.synthesize("Hallo, das ist ein Integrationstest.")

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        assert len(audio) > 0
        assert np.max(np.abs(audio)) > 0.01  # not silence

    def test_stt_then_tts_roundtrip(self, stt_config, tts_config, test_wav_path):
        """Load a WAV, transcribe it, synthesize the transcript, verify output."""
        with wave.open(test_wav_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        stt = WhisperSTT(stt_config)
        transcript, lang = stt.transcribe(audio)
        assert len(transcript) > 0

        tts = PiperTTS(tts_config)
        wav_bytes = tts.synthesize(transcript)
        assert len(wav_bytes) > 44
        assert wav_bytes[:4] == b"RIFF"


@pytest.mark.hardware
class TestLiveIntegration:
    def test_live_record_transcribe_speak(self, config):
        """Full loop with real hardware. Run manually on Jetson."""
        from app.audio.capture import AudioCapture
        from app.audio.playback import AudioPlayback

        capture = AudioCapture(config)
        playback = AudioPlayback(config)
        stt = WhisperSTT(config)
        tts = PiperTTS(config)

        print("Speak now (3 seconds)...")
        audio = capture.record_utterance()
        transcript, lang = stt.transcribe(audio)
        print(f"[{lang}] {transcript}")

        if transcript.strip():
            wav_bytes = tts.synthesize(transcript)
            playback.play_wav_bytes(wav_bytes)
