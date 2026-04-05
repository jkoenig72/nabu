"""Tests for HAL9000 TTS and NabuTTS (fallback wrapper)."""

import io
import wave
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.tts.hal_tts import HalTTS, HAL_SAMPLE_RATE, HAL_CHANNELS, HAL_SAMPLE_WIDTH
from app.tts.nabu_tts import NabuTTS


def _make_wav_bytes(num_samples=4800, sample_rate=HAL_SAMPLE_RATE):
    """Create minimal valid WAV bytes for testing."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(HAL_CHANNELS)
        wf.setsampwidth(HAL_SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)
    return buf.getvalue()


def _make_pcm_bytes(num_samples=4800):
    """Create raw PCM bytes (what the streaming API returns)."""
    return b"\x00\x00" * num_samples


class _FakeStreamResponse:
    """Mock for httpx streaming response context manager."""
    def __init__(self, pcm_data):
        self._pcm_data = pcm_data
        self.status_code = 200

    def raise_for_status(self):
        pass

    def iter_bytes(self, chunk_size=4096):
        for i in range(0, len(self._pcm_data), chunk_size):
            yield self._pcm_data[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture
def hal_config():
    return {
        "tts": {
            "hal": {
                "enabled": True,
                "url": "http://192.168.10.6:8091",
                "voice": "ref2",
                "language": "German",
                "timeout": 30.0,
            },
            "model_path": "/home/fritz/ha-hal/wyoming/piper/de_DE-thorsten-high.onnx",
            "config_path": "/home/fritz/ha-hal/wyoming/piper/de_DE-thorsten-high.onnx.json",
            "speaker_id": 0,
            "length_scale": 1.0,
            "output_sample_rate": 22050,
        }
    }


@pytest.fixture
def piper_only_config():
    return {
        "tts": {
            "model_path": "/home/fritz/ha-hal/wyoming/piper/de_DE-thorsten-high.onnx",
            "config_path": "/home/fritz/ha-hal/wyoming/piper/de_DE-thorsten-high.onnx.json",
            "speaker_id": 0,
            "length_scale": 1.0,
            "output_sample_rate": 22050,
        }
    }


class TestHalTTS:
    def test_init_sets_url(self, hal_config):
        with patch("app.tts.hal_tts.httpx.Client"):
            tts = HalTTS(hal_config)
        assert tts.url == "http://192.168.10.6:8091/v1/audio/speech"
        assert tts.voice == "ref2"
        assert tts.language == "German"

    def test_synthesize_returns_wav(self, hal_config):
        pcm_data = _make_pcm_bytes()
        fake_resp = _FakeStreamResponse(pcm_data)

        with patch("app.tts.hal_tts.httpx.Client") as MockClient:
            MockClient.return_value.stream.return_value = fake_resp
            tts = HalTTS(hal_config)
            result = tts.synthesize("Hallo Welt")

        assert result[:4] == b"RIFF"
        assert len(result) > 44

    def test_synthesize_sends_correct_payload(self, hal_config):
        pcm_data = _make_pcm_bytes()
        fake_resp = _FakeStreamResponse(pcm_data)

        with patch("app.tts.hal_tts.httpx.Client") as MockClient:
            MockClient.return_value.stream.return_value = fake_resp
            tts = HalTTS(hal_config)
            tts.synthesize("Test text")

        call_args = MockClient.return_value.stream.call_args
        assert call_args.args[0] == "POST"
        payload = call_args.kwargs["json"]
        assert payload["input"] == "Test text"
        assert payload["voice"] == "ref2"
        assert payload["response_format"] == "pcm"
        assert payload["stream"] is True
        assert payload["language"] == "German"

    def test_health_check_success(self, hal_config):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("app.tts.hal_tts.httpx.Client") as MockClient:
            MockClient.return_value.get.return_value = mock_resp
            tts = HalTTS(hal_config)
            assert tts.health_check() is True

    def test_health_check_failure(self, hal_config):
        with patch("app.tts.hal_tts.httpx.Client") as MockClient:
            MockClient.return_value.get.side_effect = httpx.ConnectError("down")
            tts = HalTTS(hal_config)
            assert tts.health_check() is False


class TestNabuTTS:
    @patch("app.tts.nabu_tts.PiperTTS")
    @patch("app.tts.nabu_tts.HalTTS")
    def test_uses_hal_when_available(self, MockHal, MockPiper, hal_config):
        wav_data = _make_wav_bytes()
        MockHal.return_value.health_check.return_value = True
        MockHal.return_value.synthesize.return_value = wav_data

        tts = NabuTTS(hal_config)
        assert tts.active_engine == "hal"
        result = tts.synthesize("test")
        assert result == wav_data
        MockHal.return_value.synthesize.assert_called_once_with("test")
        MockPiper.return_value.synthesize.assert_not_called()

    @patch("app.tts.nabu_tts.PiperTTS")
    @patch("app.tts.nabu_tts.HalTTS")
    def test_falls_back_to_piper_when_hal_unreachable(self, MockHal, MockPiper, hal_config):
        wav_data = _make_wav_bytes(sample_rate=22050)
        MockHal.return_value.health_check.return_value = False
        MockPiper.return_value.synthesize.return_value = wav_data

        tts = NabuTTS(hal_config)
        assert tts.active_engine == "piper"
        result = tts.synthesize("test")
        assert result == wav_data
        MockPiper.return_value.synthesize.assert_called_once_with("test")

    @patch("app.tts.nabu_tts.PiperTTS")
    @patch("app.tts.nabu_tts.HalTTS")
    def test_falls_back_on_runtime_error(self, MockHal, MockPiper, hal_config):
        wav_data = _make_wav_bytes(sample_rate=22050)
        MockHal.return_value.health_check.return_value = True
        MockHal.return_value.synthesize.side_effect = httpx.ConnectError("lost")
        MockPiper.return_value.synthesize.return_value = wav_data

        tts = NabuTTS(hal_config)
        assert tts.active_engine == "hal"
        result = tts.synthesize("test")
        assert result == wav_data
        assert tts.active_engine == "piper"  # switched after failure

    @patch("app.tts.nabu_tts.PiperTTS")
    def test_piper_only_when_no_hal_config(self, MockPiper, piper_only_config):
        wav_data = _make_wav_bytes(sample_rate=22050)
        MockPiper.return_value.synthesize.return_value = wav_data

        tts = NabuTTS(piper_only_config)
        assert tts.active_engine == "piper"
        result = tts.synthesize("test")
        assert result == wav_data


# --- Network tests (only on nabu with HAL server reachable) ---


@pytest.mark.network
class TestHalTTSLive:
    """Live tests against the real HAL server. Run with: pytest -m network"""

    def test_health(self, hal_config):
        with patch("app.tts.hal_tts.httpx.Client", httpx.Client):
            tts = HalTTS(hal_config)
            assert tts.health_check() is True

    def test_synthesize_german(self, hal_config):
        with patch("app.tts.hal_tts.httpx.Client", httpx.Client):
            tts = HalTTS(hal_config)
            wav_bytes = tts.synthesize("Hallo, ich bin Nabu.")
            assert wav_bytes[:4] == b"RIFF"
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                assert wf.getframerate() == HAL_SAMPLE_RATE
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2
                assert wf.getnframes() > 0

    def test_synthesize_longer_text(self, hal_config):
        with patch("app.tts.hal_tts.httpx.Client", httpx.Client):
            tts = HalTTS(hal_config)
            wav_bytes = tts.synthesize(
                "Das Wetter heute ist sonnig mit Temperaturen um die zwanzig Grad. "
                "Ein perfekter Tag für einen Spaziergang im Park."
            )
            assert len(wav_bytes) > 5000
