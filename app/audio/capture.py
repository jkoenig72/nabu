import collections
import logging
import queue

import numpy as np
import sounddevice as sd

from app.config import resolve_device_index

log = logging.getLogger(__name__)


def _load_silero_vad():
    """Load Silero VAD model on CPU. Returns (model, None) or (None, error)."""
    try:
        import torch
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            verbose=False,
        )
        log.info("Silero VAD loaded (neural speech detection)")
        return model, None
    except Exception as e:
        log.warning("Silero VAD not available (%s), using energy-based VAD", e)
        return None, e


class AudioCapture:
    CHUNK_SIZE = 1024
    SILERO_WINDOW = 512       # Silero expects 512 samples at 16kHz
    SILERO_SAMPLE_RATE = 16000

    def __init__(self, config):
        audio_cfg = config["audio"]
        vad_cfg = audio_cfg["vad"]

        self.device_index = resolve_device_index(audio_cfg["input_device_name"], "input")
        self.target_sample_rate = audio_cfg["sample_rate"]
        self.channels = audio_cfg["channels"]

        self.device_sample_rate = self._detect_device_rate()
        self._need_resample = self.device_sample_rate != self.target_sample_rate

        if self._need_resample:
            log.info("Audio: recording at %d Hz, resampling to %d Hz",
                     self.device_sample_rate, self.target_sample_rate)

        self.energy_threshold = vad_cfg["energy_threshold"]
        self.silence_duration = vad_cfg["silence_duration"]
        self.max_duration = vad_cfg["max_duration"]
        self.pre_speech_buffer_sec = vad_cfg["pre_speech_buffer"]

        self.silero_threshold = vad_cfg.get("silero_threshold", 0.5)
        use_silero = vad_cfg.get("use_silero", True)

        self._silero_model = None
        self._silero_buffer = np.array([], dtype=np.float32)
        if use_silero:
            self._silero_model, _ = _load_silero_vad()

    @property
    def use_silero(self):
        return self._silero_model is not None

    def _detect_device_rate(self) -> int:
        """Find a working sample rate for the device."""
        candidates = [self.target_sample_rate, 48000, 44100, 22050, 8000]
        for sr in candidates:
            try:
                sd.check_input_settings(
                    device=self.device_index, samplerate=sr,
                    channels=self.channels, dtype="float32",
                )
                return sr
            except sd.PortAudioError:
                continue
        info = sd.query_devices(self.device_index)
        return int(info["default_samplerate"])

    def _is_speech(self, audio_chunk):
        """Return True if speech detected. Uses Silero or energy fallback."""
        if self._silero_model is not None:
            return self._is_speech_silero(audio_chunk)
        return self._is_speech_energy(audio_chunk)

    def _is_speech_energy(self, audio_chunk):
        """Fallback VAD: RMS above threshold."""
        rms = self._compute_rms(audio_chunk)
        return rms > self.energy_threshold

    def _is_speech_silero(self, audio_chunk):
        """Neural VAD via Silero. Buffers across chunks for 48kHz devices."""
        import torch

        audio = audio_chunk.flatten()

        if self.device_sample_rate != self.SILERO_SAMPLE_RATE:
            audio = self._resample_audio(audio, self.device_sample_rate, self.SILERO_SAMPLE_RATE)

        self._silero_buffer = np.concatenate([self._silero_buffer, audio])

        speech_detected = False
        consumed = 0
        for start in range(0, len(self._silero_buffer) - self.SILERO_WINDOW + 1, self.SILERO_WINDOW):
            window = self._silero_buffer[start:start + self.SILERO_WINDOW]
            tensor = torch.from_numpy(window)
            prob = self._silero_model(tensor, self.SILERO_SAMPLE_RATE).item()
            consumed = start + self.SILERO_WINDOW
            if prob > self.silero_threshold:
                speech_detected = True
                break
        if consumed > 0:
            self._silero_buffer = self._silero_buffer[consumed:]

        return speech_detected

    def record_utterance(self, silence_duration=None, max_duration=None):
        """Record speech, return float32 array at target_sample_rate. Blocks."""
        silence_dur = silence_duration if silence_duration is not None else self.silence_duration
        max_dur = max_duration if max_duration is not None else self.max_duration

        audio_queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            audio_queue.put(indata.copy())

        chunk_duration = self.CHUNK_SIZE / self.device_sample_rate
        ring_size = max(1, int(self.pre_speech_buffer_sec / chunk_duration))
        ring_buffer = collections.deque(maxlen=ring_size)

        recorded_chunks = []
        silence_chunks = 0
        silence_chunks_limit = int(silence_dur / chunk_duration)
        max_chunks = int(max_dur / chunk_duration)
        recording = False
        total_chunks = 0

        vad_type = "silero" if self.use_silero else "energy"
        log.debug("Recording: device=%s, rate=%d→%d, vad=%s, silence=%.1fs, max=%.1fs",
                  self.device_index, self.device_sample_rate, self.target_sample_rate,
                  vad_type, silence_dur, max_dur)

        with sd.InputStream(
            device=self.device_index,
            samplerate=self.device_sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.CHUNK_SIZE,
            callback=callback,
        ):
            while True:
                try:
                    chunk = audio_queue.get(timeout=5.0)
                except queue.Empty:
                    log.warning("Audio queue timeout — device may be disconnected")
                    break
                speech = self._is_speech(chunk)

                if not recording:
                    ring_buffer.append(chunk)
                    if speech:
                        log.debug("Speech detected (vad=%s)", vad_type)
                        recording = True
                        recorded_chunks.extend(ring_buffer)
                        ring_buffer.clear()
                        silence_chunks = 0
                        total_chunks = len(recorded_chunks)
                else:
                    recorded_chunks.append(chunk)
                    total_chunks += 1

                    if not speech:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0

                    if silence_chunks >= silence_chunks_limit:
                        break
                    if total_chunks >= max_chunks:
                        break

        # Reset Silero state between utterances for clean next detection
        if self._silero_model is not None:
            self._silero_model.reset_states()
            self._silero_buffer = np.array([], dtype=np.float32)

        if not recorded_chunks:
            log.debug("No speech detected, returning empty")
            return np.array([], dtype=np.float32)

        audio = np.concatenate(recorded_chunks).flatten()
        log.debug("Recorded %d samples (%.2fs at %d Hz)",
                  len(audio), len(audio) / self.device_sample_rate, self.device_sample_rate)

        if self._need_resample:
            audio = self._resample_audio(audio, self.device_sample_rate, self.target_sample_rate)
            log.debug("Resampled to %d samples (%.2fs at %d Hz)",
                      len(audio), len(audio) / self.target_sample_rate, self.target_sample_rate)

        return audio

    @staticmethod
    def _resample_audio(audio, from_rate, to_rate):
        """Resample audio using linear interpolation."""
        ratio = to_rate / from_rate
        n_samples = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, n_samples)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    @staticmethod
    def _compute_rms(audio_chunk):
        """Compute root-mean-square energy of a chunk."""
        return float(np.sqrt(np.mean(audio_chunk ** 2)))
