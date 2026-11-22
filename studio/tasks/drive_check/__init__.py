"""Task: camera and motor check (preview + hold-to-drive buttons).

Useful before every run (wheels lifted): is the camera picture right, do both
motors turn the right way. It is also the simplest task of the platform and
exercises the whole chain: runner, stream, heartbeat, e-stop.

Python 3.6 compatible (runs on the Jetson Nano).
"""

# Declaration read by the web process (light: no torch here)
TASK = {
    "id": "drive_check",
    "title": "Camera and motor check",
    "description": "Live camera picture and hold-to-drive buttons. Lift the wheels first.",
    "status": "ready",
    "order": 5,
    "roles": ["car", "pc"],
    "devices": ["camera", "motors"],
    "params": [
        {"name": "max_power", "type": "float", "min": 0.05, "max": 0.6, "step": 0.05, "default": 0.25,
         "label": "Max motor power", "group": "Drive", "live": True,
         "help": "Motor value used by the buttons (0..1)."},
        {"name": "hold_time", "type": "float", "min": 0.1, "max": 1.0, "step": 0.05, "default": 0.3,
         "label": "Hold time (s)", "group": "Drive", "live": True,
         "help": "Motors stop when no drive command arrived for this long (release the button = stop)."},
        {"name": "jpeg_overlay", "type": "bool", "default": True, "label": "Draw motor values on the picture",
         "group": "Display", "live": True},
    ],
    "runtime": "studio.tasks.drive_check.runtime:Runtime",
}
