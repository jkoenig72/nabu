import queue
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from app.audio.capture import AudioCapture


@pytest.fixture
def capture_config():
    return {
        "audio": {
            "input_device_name": "USB Composite",
            "output_device_name": "USB Composite",
            "sample_rate": 16000,
            "channels": 1,
            "dtype": "float32",
            "vad": {
                "use_silero": False,  # unit tests use energy-based VAD
                "silero_threshold": 0.5,
                "energy_threshold": 0.015,
                "silence_duration": 0.2,
                "max_duration": 2.0,
                "pre_speech_buffer": 0.1,
            },
        }
    }


class TestRMS:
    def test_silence_is_zero(self):
        chunk = np.zeros(1024, dtype=np.float32)
        assert AudioCapture._compute_rms(chunk) == pytest.approx(0.0)

    def test_known_signal(self):
        chunk = np.ones(1024, dtype=np.float32) * 0.5
        assert AudioCapture._compute_rms(chunk) == pytest.approx(0.5, abs=1e-6)

    def test_sine_wave(self):
        t = np.linspace(0, 1, 1024, dtype=np.float32)
        chunk = np.sin(2 * np.pi * 440 * t)
        rms = AudioCapture._compute_rms(chunk)
        assert 0.5 < rms < 1.0  # RMS of sine is ~0.707


class TestVAD:
    def _make_chunk(self, amplitude, size=1024):
        return np.ones((size, 1), dtype=np.float32) * amplitude

    @patch("app.audio.capture.sd.check_input_settings")
    @patch("app.audio.capture.resolve_device_index", return_value=0)
    def test_detects_speech_and_stops_on_silence(self, mock_resolve, mock_check, capture_config):
        capture = AudioCapture(capture_config)
        assert not capture.use_silero  # energy-based in unit tests

        # Simulate: 2 silent chunks, 3 loud chunks, 5 silent chunks (exceeds silence_duration)
        chunks = (
            [self._make_chunk(0.001)] * 2    # silence (waiting)
            + [self._make_chunk(0.1)] * 3    # speech
            + [self._make_chunk(0.001)] * 5  # silence (triggers stop)
        )

        audio_queue = queue.Queue()
        for c in chunks:
            audio_queue.put(c)

        with patch("app.audio.capture.sd.InputStream") as mock_stream:
            mock_stream.return_value.__enter__ = MagicMock(return_value=None)
            mock_stream.return_value.__exit__ = MagicMock(return_value=False)

            with patch.object(queue, "Queue", return_value=audio_queue):
                result = capture.record_utterance()

        assert len(result) > 0
        assert result.dtype == np.float32

    @patch("app.audio.capture.sd.check_input_settings")
    @patch("app.audio.capture.resolve_device_index", return_value=0)
    def test_max_duration_stops_recording(self, mock_resolve, mock_check, capture_config):
        capture_config["audio"]["vad"]["max_duration"] = 0.2  # very short
        capture = AudioCapture(capture_config)

        # All loud chunks — should stop at max_duration
        chunks = [self._make_chunk(0.1)] * 20

        audio_queue = queue.Queue()
        for c in chunks:
            audio_queue.put(c)

        with patch("app.audio.capture.sd.InputStream") as mock_stream:
            mock_stream.return_value.__enter__ = MagicMock(return_value=None)
            mock_stream.return_value.__exit__ = MagicMock(return_value=False)

            with patch.object(queue, "Queue", return_value=audio_queue):
                result = capture.record_utterance()

        assert len(result) > 0
        max_samples = int(0.2 * 16000)
        assert len(result) <= max_samples + 1024  # allow one extra chunk


class TestSileroVAD:
    """Tests for Silero VAD integration (mocked model)."""

    @patch("app.audio.capture._load_silero_vad")
    @patch("app.audio.capture.sd.check_input_settings")
    @patch("app.audio.capture.resolve_device_index", return_value=0)
    def test_silero_enabled(self, mock_resolve, mock_check, mock_load, capture_config):
        mock_model = MagicMock()
        mock_load.return_value = (mock_model, None)
        capture_config["audio"]["vad"]["use_silero"] = True

        capture = AudioCapture(capture_config)
        assert capture.use_silero

    @patch("app.audio.capture._load_silero_vad")
    @patch("app.audio.capture.sd.check_input_settings")
    @patch("app.audio.capture.resolve_device_index", return_value=0)
    def test_silero_fallback_on_failure(self, mock_resolve, mock_check, mock_load, capture_config):
        mock_load.return_value = (None, RuntimeError("no torch"))
        capture_config["audio"]["vad"]["use_silero"] = True

        capture = AudioCapture(capture_config)
        assert not capture.use_silero  # fell back to energy

    @patch("app.audio.capture._load_silero_vad")
    @patch("app.audio.capture.sd.check_input_settings")
    @patch("app.audio.capture.resolve_device_index", return_value=0)
    def test_silero_detects_speech(self, mock_resolve, mock_check, mock_load, capture_config):
        """Silero VAD detects speech and stops on silence using mocked model."""
        mock_model = MagicMock()
        mock_model.reset_states = MagicMock()
        mock_load.return_value = (mock_model, None)
        capture_config["audio"]["vad"]["use_silero"] = True

        capture = AudioCapture(capture_config)

        # Mock Silero: return low prob for silence, high for speech
        call_count = [0]
        speech_pattern = [0.01, 0.01, 0.9, 0.9, 0.9, 0.01, 0.01, 0.01, 0.01, 0.01]

        def mock_call(tensor, sr):
            idx = min(call_count[0], len(speech_pattern) - 1)
            call_count[0] += 1
            return MagicMock(item=lambda: speech_pattern[idx])

        mock_model.side_effect = mock_call

        # Create chunks (need enough samples for Silero windows)
        def make_chunk(size=1024):
            return np.ones((size, 1), dtype=np.float32) * 0.05

        chunks = [make_chunk()] * 10
        audio_queue = queue.Queue()
        for c in chunks:
            audio_queue.put(c)

        with patch("app.audio.capture.sd.InputStream") as mock_stream:
            mock_stream.return_value.__enter__ = MagicMock(return_value=None)
            mock_stream.return_value.__exit__ = MagicMock(return_value=False)

            with patch.object(queue, "Queue", return_value=audio_queue):
                result = capture.record_utterance()

        assert len(result) > 0
        mock_model.reset_states.assert_called_once()


class TestPlayback:
    def test_play_wav_bytes(self):
        import io
        import wave
        from app.audio.playback import AudioPlayback

        # Create a minimal WAV
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(np.zeros(22050, dtype=np.int16).tobytes())

        wav_bytes = buf.getvalue()

        with patch("app.audio.playback.resolve_device_index", return_value=0), \
             patch("app.audio.playback.sd.check_output_settings"), \
             patch("app.audio.playback.sd.query_devices", return_value={"max_output_channels": 2}):
            playback = AudioPlayback({"audio": {"output_device_name": "test"}})

        with patch("app.audio.playback.sd.play") as mock_play, \
             patch("app.audio.playback.sd.wait"):
            playback.play_wav_bytes(wav_bytes)
            mock_play.assert_called_once()
            args = mock_play.call_args
            assert args[1]["samplerate"] in (22050, 48000)
