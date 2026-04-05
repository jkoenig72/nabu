#!/bin/bash
# Launch Nabu with correct library paths
# Usage: ./run.sh           (normal mode - DEBUG to console and log file)
#        ./run.sh --verbose  (same, kept for compatibility)
export LD_LIBRARY_PATH=/home/fritz/ct2-install/lib:${LD_LIBRARY_PATH}
cd /home/fritz/nabu-dev

# Ensure all CPUs are online (onnxruntime crashes if possible > online)
for i in 4 5 6 7; do
    if [ -f /sys/devices/system/cpu/cpu${i}/online ] && [ "$(cat /sys/devices/system/cpu/cpu${i}/online)" = "0" ]; then
        echo 1 | sudo tee /sys/devices/system/cpu/cpu${i}/online > /dev/null 2>&1
    fi
done

# Ensure USB mic is the default PulseAudio source
USB_MIC=$(pactl list sources short 2>/dev/null | grep -i "usb.*composite.*mono" | awk '{print $2}')
if [ -n "$USB_MIC" ]; then
    pactl set-default-source "$USB_MIC" 2>/dev/null
    echo "[run.sh] PulseAudio input: $USB_MIC"
fi

exec /home/fritz/nabu-venv/bin/python -m app.main "$@"
