import os

import numpy as np
import pytest

from app.config import load_config


@pytest.fixture
def config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "app", "config.yaml")
    return load_config(config_path)


@pytest.fixture
def sample_audio():
    """1 second of 440Hz sine wave at 16kHz, float32."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    return 0.3 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def silence_audio():
    """1 second of silence at 16kHz."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def test_wav_path():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "test_german.wav")
    if not os.path.exists(path):
        pytest.skip("Test fixture test_german.wav not found")
    return path
