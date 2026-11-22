"""Task registry: every sub-package of ``studio/tasks/`` is a task.

A task package's ``__init__.py`` must stay light (it is imported by the web
process, which never loads torch) and define ``TASK``::

    TASK = {
        "id": "lane_follow",              # same as the package name
        "title": "Lane following",
        "description": "...",
        "status": "ready",                # or "dev": listed greyed out, cannot start
        "order": 10,                      # position on the home page
        "roles": ["car"],                 # where its runner may run ("car", "pc")
        "devices": ["camera", "motors"],  # what its runner holds exclusively
        "params": [...],                  # see studio/params.py
        "runtime": "studio.tasks.lane_follow.runtime:Runtime",  # runner-side class
    }

Optional:

* ``web.py`` in the package with ``HANDLERS = [(pattern, RequestHandler), ...]``
  for task-specific pages / APIs (light imports only).
* ``page.html`` in the package: replaces the generic task page body.

Every task automatically gets the safety parameter ``link_timeout``.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import importlib
import os
import pkgutil
import traceback

# Schema checks
from studio import params as params_mod

# Folder of this package (one sub-folder per task)
TASKS_DIR = os.path.dirname(os.path.abspath(__file__))
# Safety parameter added to every task
LINK_TIMEOUT_PARAM = {
    "name": "link_timeout", "type": "float", "min": 0.3, "max": 5.0, "step": 0.1, "default": 0.5,
    "label": "Link timeout (s)", "group": "Safety", "live": True,
    "help": "Stop the motors when no open page has answered for this long (page closed, Wi-Fi lost, "
            "web process crashed). Longer values tolerate a weak Wi-Fi but stop later.",
}
# Keys every TASK dict must have
REQUIRED = ("id", "title", "description", "status", "roles", "devices", "params", "runtime")


# Copy of a task declaration with defaults and the safety parameter added
def _normalise(task, package):
    # Shallow copy so the module's dict is not changed
    t = dict(task)
    # Every required key must be present
    missing = [k for k in REQUIRED if k not in t]
    # Missing keys make the task unusable
    if missing:
        # Report them
        raise ValueError("TASK is missing %s" % ", ".join(missing))
    # The id must match the folder name
    if t["id"] != package:
        # Report the mismatch
        raise ValueError("TASK id %r differs from its folder %r" % (t["id"], package))
    # Home page position (default: at the end)
    t.setdefault("order", 100)
    # Parameters plus the safety parameter (unless the task declares its own)
    plist = list(t["params"])
    # Add link_timeout once
    if "link_timeout" not in [p["name"] for p in plist]:
        # At the end of the form
        plist.append(dict(LINK_TIMEOUT_PARAM))
    # Store the extended list
    t["params"] = plist
    # Catch malformed declarations early
    params_mod.check_schema(plist)
    # Custom page template inside the package, if any
    page = os.path.join(TASKS_DIR, package, "page.html")
    # Absolute path or None
    t["page"] = page if os.path.exists(page) else None
    # Normalised declaration
    return t


# Import every task package
def discover():
    """Return ``(tasks, errors)``: tasks sorted by order, errors ``{package: traceback}``."""
    # Valid tasks and broken packages
    tasks, errors = [], {}
    # Every sub-package of studio/tasks
    for info in pkgutil.iter_modules([TASKS_DIR]):
        # Only packages (folders) are tasks; names starting with '_' are private
        if not info.ispkg or info.name.startswith("_"):
            # Skip it
            continue
        # A broken task must not break the whole server
        try:
            # Import the light declaration module
            mod = importlib.import_module("studio.tasks." + info.name)
            # Validate and complete the declaration
            tasks.append(_normalise(getattr(mod, "TASK"), info.name))
        # Keep the error for the home page
        except Exception:
            # Full traceback
            errors[info.name] = traceback.format_exc()
    # Home page order, then title
    tasks.sort(key=lambda t: (t["order"], t["title"]))
    # Results
    return tasks, errors


# Task-specific web handlers
def web_handlers(task_id):
    """``HANDLERS`` of ``studio.tasks.<id>.web`` or an empty list."""
    # Optional module
    try:
        # Import it
        mod = importlib.import_module("studio.tasks.%s.web" % task_id)
    # No web.py in this task
    except ImportError as e:
        # Only a missing web.py is fine; other import errors must surface
        if getattr(e, "name", None) == "studio.tasks.%s.web" % task_id:
            # No extra handlers
            return []
        # Real error inside web.py
        raise
    # Its handlers
    return list(getattr(mod, "HANDLERS", []))


# Load the runner-side class named by a task (runner process only)
def load_runtime(task):
    """Import ``"module:Class"`` from ``task['runtime']``."""
    # Module and class name
    module_name, _, class_name = task["runtime"].partition(":")
    # Import the module (may load torch)
    mod = importlib.import_module(module_name)
    # The class
    return getattr(mod, class_name)
