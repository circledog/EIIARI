"""Task: search for an object and build a map (TASKS M4).

Register photos of the object, start the search: the robot explores with
frontier exploration and M3's obstacle avoidance, estimates its position from
the floor (visual odometry, no wheel encoders), builds a 2-D occupancy map,
stops when the object is confirmed and records everything for a 3-D map
built after the run.

Python 3.6 compatible (runs on the Jetson Nano).
"""

# Declaration read by the web process (light: no torch here)
TASK = {
    "id": "search",
    "title": "Object search and 3D map",
    "description": "Upload photos of an object; the robot explores, builds a map and marks the object, "
                   "the start point and its route.",
    "status": "ready",
    "order": 30,
    "roles": ["car", "pc"],
    "devices": ["camera", "motors"],
    "params": [
        {"name": "target", "type": "string", "default": "", "max_length": 40, "label": "Target name",
         "group": "Search", "help": "Name of a registered target (see 'Targets' on this page)."},
        {"name": "time_limit_s", "type": "int", "min": 30, "max": 3600, "default": 600, "label": "Time limit (s)",
         "group": "Search", "live": True},
        {"name": "confirm", "type": "int", "min": 1, "max": 10, "default": 3, "label": "Hits in a row to confirm",
         "group": "Search", "live": True},
        {"name": "hfov_deg", "type": "float", "min": 40.0, "max": 180.0, "step": 1.0, "default": 120.0,
         "label": "Camera horizontal field of view (deg)", "group": "Camera",
         "help": "Ignored when a calibration file is given."},
        {"name": "cam_height", "type": "float", "min": 0.03, "max": 0.5, "step": 0.005, "default": 0.10,
         "label": "Camera height above the floor (m)", "group": "Camera", "help": "Sets the map's scale: measure it."},
        {"name": "cam_pitch", "type": "float", "min": 0.0, "max": 60.0, "step": 1.0, "default": 20.0,
         "label": "Camera tilt down (deg)", "group": "Camera"},
        {"name": "calibration", "type": "string", "default": "", "max_length": 200, "label": "Calibration file",
         "group": "Camera", "help": "JSON from python3 -m studio.tasks.search.calibrate (fisheye K and D)."},
        {"name": "map_size", "type": "float", "min": 4.0, "max": 30.0, "step": 1.0, "default": 12.0,
         "label": "Map size (m)", "group": "Map"},
        {"name": "grid_res", "type": "float", "min": 0.02, "max": 0.2, "step": 0.01, "default": 0.05,
         "label": "Map cell size (m)", "group": "Map"},
        {"name": "keyframe_dist", "type": "float", "min": 0.05, "max": 1.0, "step": 0.05, "default": 0.15,
         "label": "Keyframe every (m)", "group": "Map"},
        {"name": "horizon", "type": "float", "min": 0.2, "max": 0.8, "step": 0.05, "default": 0.45,
         "label": "Horizon (fraction of height)", "group": "Free space", "live": True},
        {"name": "floor_threshold", "type": "float", "min": 0.05, "max": 0.9, "step": 0.05, "default": 0.35,
         "label": "Floor similarity threshold", "group": "Free space", "live": True},
        {"name": "free_threshold", "type": "float", "min": 0.05, "max": 2.0, "step": 0.05, "default": 0.35,
         "label": "Free distance to drive into (m)", "group": "Free space", "live": True},
        {"name": "stop_distance", "type": "float", "min": 0.05, "max": 1.0, "step": 0.01, "default": 0.15,
         "label": "Stop distance (m, obstacle ahead)", "group": "Free space", "live": True},
        {"name": "max_speed", "type": "float", "min": 0.0, "max": 0.5, "step": 0.01, "default": 0.12,
         "label": "Max speed", "group": "Control"},
        {"name": "min_speed", "type": "float", "min": 0.0, "max": 0.5, "step": 0.01, "default": 0.07,
         "label": "Min speed", "group": "Control"},
        {"name": "turn_power", "type": "float", "min": 0.05, "max": 0.5, "step": 0.01, "default": 0.12,
         "label": "Turn-on-the-spot power", "group": "Control", "live": True},
        {"name": "kp", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.18, "label": "kp",
         "group": "Control"},
        {"name": "kd", "type": "float", "min": 0.0, "max": 0.1, "step": 0.001, "default": 0.008,
         "label": "kd (per second)", "group": "Control"},
        {"name": "max_output", "type": "float", "min": 0.1, "max": 1.0, "step": 0.05, "default": 0.3,
         "label": "Motor limit", "group": "Control"},
        {"name": "safety_timeout", "type": "float", "min": 0.2, "max": 2.0, "step": 0.1, "default": 0.5,
         "label": "Loop / camera timeout (s)", "group": "Safety", "live": True},
    ],
    "runtime": "studio.tasks.search.runtime:Runtime",
}
