"""Task: search for an object and build a 3D map (TASKS M4, in development).

Python 3.6 compatible (runs on the Jetson Nano).
"""

# Declaration read by the web process (light: no torch here)
TASK = {
    "id": "search",
    "title": "Object search and 3D map",
    "description": "Upload photos of an object; the robot explores, builds a map and marks the object, "
                   "the start point and its route.",
    "status": "dev",
    "order": 30,
    "roles": ["car"],
    "devices": ["camera", "motors"],
    "params": [],
    "runtime": "studio.tasks.search.runtime:Runtime",
}
