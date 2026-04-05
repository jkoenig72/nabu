import io
import logging
import wave

import numpy as np
import sounddevice as sd

from app.config import resolve_device_index

log = logging.getLogger(__name__)


class AudioPlayback:
    VOLUME_STEP = 0.1
    VOLUME_MIN = 0.1
    VOLUME_MAX = 1.0
    VOLUME_DEFAULT = 0.5

    def __init__(self, config):
        self.device_index = resolve_device_index(
            config["audio"]["output_device_name"], "output"
        )
        self._device_rate = self._detect_device_rate()
        self._volume = self.VOLUME_DEFAULT

    @property
    def volume(self) -> float:
        return self._volume

    def volume_up(self) -> float:
        """Increase volume by one step. Returns new volume."""
        self._volume = min(self.VOLUME_MAX, round(self._volume + self.VOLUME_STEP, 2))
        log.info("Volume up → %.0f%%", self._volume * 100)
        return self._volume

    def volume_down(self) -> float:
        """Decrease volume by one step. Returns new volume."""
        self._volume = max(self.VOLUME_MIN, round(self._volume - self.VOLUME_STEP, 2))
        log.info("Volume down → %.0f%%", self._volume * 100)
        return self._volume

    def _detect_device_rate(self) -> int | None:
        """Find a working output sample rate, or None to use source rate."""
        info = sd.query_devices(self.device_index)
        max_ch = max(1, info.get("max_output_channels", 2))
        for sr in [48000, 44100, 22050, 16000]:
            try:
                sd.check_output_settings(
                    device=self.device_index, samplerate=sr,
                    channels=min(2, max_ch), dtype="float32",
                )
                log.debug("Output device %d: native rate=%d Hz", self.device_index, sr)
                return sr
            except sd.PortAudioError:
                continue
        log.warning("Output device %d: no working sample rate found", self.device_index)
        return None

    def play_wav_bytes(self, wav_bytes):
        """Play WAV data (bytes). Blocks until playback finishes."""
        log.debug("Playing WAV: %d bytes", len(wav_bytes))
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if sample_width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        if channels > 1:
            audio = audio.reshape(-1, channels)

        play_rate = sr
        if self._device_rate and sr != self._device_rate:
            audio = self._resample(audio, sr, self._device_rate)
            play_rate = self._device_rate

        sd.play(audio * self._volume, samplerate=play_rate, device=self.device_index)
        sd.wait()

    def play_array(self, audio, sample_rate):
        """Play a numpy float32 array. Blocks until playback finishes."""
        play_rate = sample_rate
        if self._device_rate and sample_rate != self._device_rate:
            audio = self._resample(audio, sample_rate, self._device_rate)
            play_rate = self._device_rate

        sd.play(audio * self._volume, samplerate=play_rate, device=self.device_index)
        sd.wait()

    @staticmethod
    def _resample(audio, from_rate, to_rate):
        """Resample audio using linear interpolation."""
        ratio = to_rate / from_rate
        if audio.ndim == 1:
            n_samples = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, n_samples)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        else:
            n_samples = int(audio.shape[0] * ratio)
            indices = np.linspace(0, audio.shape[0] - 1, n_samples)
            result = np.zeros((n_samples, audio.shape[1]), dtype=np.float32)
            for ch in range(audio.shape[1]):
                result[:, ch] = np.interp(indices, np.arange(audio.shape[0]), audio[:, ch])
            return result
