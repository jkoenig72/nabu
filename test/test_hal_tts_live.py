"""Live end-to-end test: HAL TTS streaming → USB speaker on nabu."""

import pytest

from app.tts.hal_tts import HalTTS
from app.audio.playback import AudioPlayback


HAL_CONFIG = {
    "tts": {
        "hal": {
            "enabled": True,
            "url": "http://192.168.10.6:8091",
            "voice": "ref2",
            "language": "German",
            "timeout": 120.0,
        },
    },
    "audio": {
        "output_device_name": "USB Composite",
    },
}


@pytest.mark.hardware
@pytest.mark.network
def test_hal_tts_streaming_to_speaker():
    """Stream German TTS from HAL server directly to USB speaker."""
    tts = HalTTS(HAL_CONFIG)
    assert tts.health_check(), "HAL TTS server not reachable at 192.168.10.6:8091"

    playback = AudioPlayback(HAL_CONFIG)
    text = "Isabel, habe ich Dir heute schon gesagt, dass ich Dich liebe?"
    print(f"\n  Streaming: {text}")

    tts.stream_to_player(text, playback)
    print("  Done!")
