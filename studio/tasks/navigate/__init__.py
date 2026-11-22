"""Task: obstacle avoidance + click a target to drive to (TASKS M3, in development).

Python 3.6 compatible (runs on the Jetson Nano).
"""

# Declaration read by the web process (light: no torch here)
TASK = {
    "id": "navigate",
    "title": "Avoid obstacles and drive to a clicked target",
    "description": "Click a point in the picture; the robot drives there and avoids obstacles. "
                   "Every click and emergency stop becomes a training label (interactive imitation learning).",
    "status": "dev",
    "order": 20,
    "roles": ["car"],
    "devices": ["camera", "motors"],
    "params": [],
    "runtime": "studio.tasks.navigate.runtime:Runtime",
}
