import os
import yaml
import sounddevice as sd


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(config_path=None):
    path = config_path or os.environ.get("NABU_CONFIG", DEFAULT_CONFIG_PATH)
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device_index(device_name, kind):
    """Find a sounddevice device index by substring match. Raises RuntimeError if not found."""
    devices = sd.query_devices()
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"

    for i, dev in enumerate(devices):
        if device_name.lower() in dev["name"].lower() and dev[channel_key] > 0:
            return i

    available = [
        f"  [{i}] {d['name']} (in={d['max_input_channels']}, out={d['max_output_channels']})"
        for i, d in enumerate(devices)
    ]
    raise RuntimeError(
        f"No {kind} device matching '{device_name}'. Available:\n" + "\n".join(available)
    )
