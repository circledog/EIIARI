"""Task: lane following (project4 wrapped in studio, TASKS M2).

Drive page (this task's runner): choose a model -> sanity check -> live
picture -> start / stop driving, every control option of ``live_demo.ipynb``,
CSV recording. Extra pages (``web.py``): datasets, training (PC), model
library and TensorRT conversion (car).

The algorithms stay in project4/ and are imported, not copied.

Python 3.6 compatible (runs on the Jetson Nano).
"""

# Declaration read by the web process (light: no torch here)
TASK = {
    "id": "lane_follow",
    "title": "Lane following",
    "description": "Follow the taped road with a trained model (project4). Datasets, training, "
                   "model library, TensorRT conversion and driving.",
    "status": "ready",
    "order": 10,
    "roles": ["car", "pc"],
    "devices": ["camera", "motors"],
    # Control defaults equal project4/control.py DEFAULTS (checked by a test)
    "params": [
        {"name": "model_dir", "type": "choice", "default": "auto", "label": "Model folder", "group": "Model",
         "choices": [["auto", "Auto (car: models_trt, PC: models)"], ["trt", "models_trt (TensorRT)"],
                     ["torch", "models (PyTorch)"]]},
        {"name": "flush_on_switch", "type": "bool", "default": True, "label": "Flush camera after loading",
         "group": "Model", "help": "Drop frames queued while a model loads (jetbot source restarts its pipeline)."},
        {"name": "torch_threads", "type": "int", "min": 0, "max": 8, "default": 0, "label": "Torch CPU threads",
         "group": "Model", "help": "0 = torch default. 1 can make the control period steadier (see HANDOFF M0)."},
        {"name": "formula", "type": "choice", "default": "teammate", "label": "Steering formula", "group": "Control",
         "live": True, "choices": [["teammate", "Teammate formula"], ["slides", "Slides formula"]]},
        {"name": "max_speed", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.20,
         "label": "Max speed", "group": "Control", "live": True},
        {"name": "min_speed", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.10,
         "label": "Min speed", "group": "Control", "live": True},
        {"name": "kp", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.18, "label": "kp",
         "group": "Control", "live": True},
        {"name": "ki", "type": "float", "min": 0.0, "max": 0.5, "step": 0.005, "default": 0.0, "label": "ki",
         "group": "Control", "live": True},
        {"name": "kd", "type": "float", "min": 0.0, "max": 0.1, "step": 0.001, "default": 0.008,
         "label": "kd (per second)", "group": "Control", "live": True},
        {"name": "bias", "type": "float", "min": -0.3, "max": 0.3, "step": 0.01, "default": 0.0,
         "label": "Steering bias", "group": "Control", "live": True},
        {"name": "turn_slowdown", "type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.65,
         "label": "Turn slowdown", "group": "Control", "live": True},
        {"name": "angle_alpha", "type": "float", "min": 0.05, "max": 1.0, "step": 0.05, "default": 0.35,
         "label": "Angle smoothing alpha", "group": "Control", "live": True},
        {"name": "slew_per_second", "type": "float", "min": 0.0, "max": 3.0, "step": 0.1, "default": 0.6,
         "label": "Motor slew per second", "group": "Control", "live": True},
        {"name": "max_output", "type": "float", "min": 0.1, "max": 1.0, "step": 0.05, "default": 0.4,
         "label": "Motor limit", "group": "Control", "live": True},
        {"name": "allow_reverse", "type": "bool", "default": False, "label": "Inner wheel may reverse",
         "group": "Control", "live": True},
        {"name": "safety_timeout", "type": "float", "min": 0.2, "max": 2.0, "step": 0.1, "default": 0.5,
         "label": "Inference / camera timeout (s)", "group": "Safety", "live": True,
         "help": "Stop when no successful control step happened for this long (model or camera stalled)."},
    ],
    "runtime": "studio.tasks.lane_follow.runtime:Runtime",
}
