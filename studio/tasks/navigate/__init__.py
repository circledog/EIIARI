"""Task: obstacle avoidance + click a target to drive to (TASKS M3).

Interactive imitation learning: clicks, emergency stops and undisturbed
driving become labels in project4's format; a network trained on them (on
the PC) can then replace the rule-based planner ("learned" mode).

Python 3.6 compatible (runs on the Jetson Nano).
"""

# Declaration read by the web process (light: no torch here)
TASK = {
    "id": "navigate",
    "title": "Avoid obstacles and drive to a clicked target",
    "description": "Click a point in the picture; the robot drives there and avoids obstacles. "
                   "Every click and emergency stop becomes a training label (interactive imitation learning).",
    "status": "ready",
    "order": 20,
    "roles": ["car", "pc"],
    "devices": ["camera", "motors"],
    "params": [
        {"name": "mode", "type": "choice", "default": "rule", "label": "Mode", "group": "Mode", "live": True,
         "choices": [["rule", "Rule-based (free space + tracker)"], ["learned", "Learned model"]]},
        {"name": "model_dir", "type": "choice", "default": "auto", "label": "Model folder", "group": "Mode",
         "choices": [["auto", "Auto (car: models_trt, PC: models)"], ["trt", "models_trt (TensorRT)"],
                     ["torch", "models (PyTorch)"]]},
        {"name": "sectors", "type": "int", "min": 5, "max": 15, "default": 9, "label": "Sectors",
         "group": "Free space"},
        {"name": "horizon", "type": "float", "min": 0.2, "max": 0.8, "step": 0.05, "default": 0.45,
         "label": "Horizon (fraction of height)", "group": "Free space", "live": True,
         "help": "Only the picture below this line is analysed."},
        {"name": "floor_threshold", "type": "float", "min": 0.05, "max": 0.9, "step": 0.05, "default": 0.35,
         "label": "Floor similarity threshold", "group": "Free space", "live": True},
        {"name": "tape_is_floor", "type": "bool", "default": True, "label": "White tape counts as floor",
         "group": "Free space", "live": True},
        {"name": "free_threshold", "type": "float", "min": 0.1, "max": 0.9, "step": 0.05, "default": 0.35,
         "label": "Free distance to drive into", "group": "Planner", "live": True},
        {"name": "stop_distance", "type": "float", "min": 0.0, "max": 0.5, "step": 0.02, "default": 0.12,
         "label": "Stop distance (obstacle ahead)", "group": "Planner", "live": True,
         "help": "Last protection, independent of any model: stop when the sectors ahead are blocked closer."},
        {"name": "arrive_v", "type": "float", "min": 0.6, "max": 1.0, "step": 0.01, "default": 0.85,
         "label": "Arrived when the target is this low", "group": "Planner", "live": True},
        {"name": "record_free", "type": "bool", "default": True, "label": "Record 'free' pictures while driving",
         "group": "Labels", "live": True},
        {"name": "max_speed", "type": "float", "min": 0.0, "max": 0.5, "step": 0.01, "default": 0.15,
         "label": "Max speed", "group": "Control", "live": True},
        {"name": "min_speed", "type": "float", "min": 0.0, "max": 0.5, "step": 0.01, "default": 0.08,
         "label": "Min speed", "group": "Control", "live": True},
        {"name": "kp", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.18, "label": "kp",
         "group": "Control", "live": True},
        {"name": "kd", "type": "float", "min": 0.0, "max": 0.1, "step": 0.001, "default": 0.008,
         "label": "kd (per second)", "group": "Control", "live": True},
        {"name": "turn_slowdown", "type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.65,
         "label": "Turn slowdown", "group": "Control", "live": True},
        {"name": "max_output", "type": "float", "min": 0.1, "max": 1.0, "step": 0.05, "default": 0.3,
         "label": "Motor limit", "group": "Control", "live": True},
        {"name": "safety_timeout", "type": "float", "min": 0.2, "max": 2.0, "step": 0.1, "default": 0.5,
         "label": "Loop / camera timeout (s)", "group": "Safety", "live": True},
    ],
    "runtime": "studio.tasks.navigate.runtime:Runtime",
}
