"""Task parameter schemas: validation and presets.

A task declares its parameters as a list of dicts::

    {"name": "max_speed", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01,
     "default": 0.2, "label": "Max speed", "group": "Control", "live": True,
     "help": "Top speed on straights"}

Types: ``float``, ``int``, ``bool``, ``choice`` (``choices``: list of
``[value, label]``) and ``string`` (``max_length``). ``live`` parameters may
be changed while the task runs; the others only before starting it.

The browser builds its form from the same schema (``static/form.js``) and
checks ranges there too, but the server always validates again here.

Presets are JSON files in ``STUDIO_HOME/presets/<task>/<name>.json``.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import json
import math
import os
import re

# Studio paths
from studio import config

# Valid parameter types
TYPES = ("float", "int", "bool", "choice", "string")
# Allowed preset names (also safe as file names)
PRESET_NAME = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")


# Check a schema once when a task is loaded (catches typos in task declarations)
def check_schema(schema):
    """Raise ``ValueError`` when a parameter declaration is malformed."""
    # Names seen so far
    seen = set()
    # Every declaration
    for p in schema:
        # Name and type are required
        name, kind = p.get("name"), p.get("type")
        # Missing or repeated name
        if not name or name in seen:
            # Report it
            raise ValueError("parameter without a name or declared twice: %r" % (name,))
        # Remember it
        seen.add(name)
        # Unknown type
        if kind not in TYPES:
            # Report it
            raise ValueError("parameter %s: unknown type %r" % (name, kind))
        # Number types need a range
        if kind in ("float", "int") and ("min" not in p or "max" not in p or p["min"] > p["max"]):
            # Report it
            raise ValueError("parameter %s: needs min <= max" % name)
        # Choices need a non-empty list
        if kind == "choice" and not p.get("choices"):
            # Report it
            raise ValueError("parameter %s: needs choices" % name)
        # The default itself must be valid
        _, err = convert(p, p.get("default"))
        # Invalid default
        if err:
            # Report it
            raise ValueError("parameter %s: bad default: %s" % (name, err))


# Convert and check one value
def convert(p, value):
    """Return ``(clean_value, None)`` or ``(None, error_message)``."""
    # Parameter type
    kind = p["type"]
    # Booleans: accept real booleans and the usual strings
    if kind == "bool":
        # Already a boolean
        if isinstance(value, bool):
            # Accept it
            return value, None
        # Common text forms (form posts)
        if isinstance(value, str) and value.lower() in ("true", "false", "1", "0", "on", "off"):
            # Map to a boolean
            return value.lower() in ("true", "1", "on"), None
        # Anything else is invalid
        return None, "must be true or false"
    # Numbers
    if kind in ("float", "int"):
        # Booleans are ints in Python; reject them explicitly
        if isinstance(value, bool):
            # Invalid
            return None, "must be a number"
        # Parse strings and numbers
        try:
            # Float first (accepts "3", "3.5", 3)
            number = float(value)
        # Not a number at all
        except (TypeError, ValueError):
            # Invalid
            return None, "must be a number"
        # NaN / inf are never valid settings
        if not math.isfinite(number):
            # Invalid
            return None, "must be a finite number"
        # Integers must not have a fraction
        if kind == "int":
            # Reject 2.5 for an int parameter
            if number != int(number):
                # Invalid
                return None, "must be a whole number"
            # Store as int
            number = int(number)
        # Range check (inclusive)
        if number < p["min"] or number > p["max"]:
            # Out of range
            return None, "must be between %s and %s" % (p["min"], p["max"])
        # Valid number
        return number, None
    # One of a list
    if kind == "choice":
        # Allowed values
        allowed = [c[0] for c in p["choices"]]
        # Value must be one of them
        if value not in allowed:
            # Invalid
            return None, "must be one of %s" % ", ".join(str(a) for a in allowed)
        # Valid choice
        return value, None
    # Text
    if not isinstance(value, str):
        # Invalid
        return None, "must be text"
    # Length limit (default 200)
    if len(value) > int(p.get("max_length", 200)):
        # Too long
        return None, "is too long"
    # Valid text
    return value, None


# Defaults of a schema
def defaults(schema):
    """``{name: default}`` for every parameter."""
    # Default of each declaration
    return dict((p["name"], p.get("default")) for p in schema)


# Validate a set of values against a schema
def validate(schema, values, partial=False, live_only=False):
    """Return ``(clean, errors)``.

    ``clean`` has every parameter (defaults for missing ones) unless
    ``partial``, then only the given ones. Unknown names are errors.
    ``live_only`` rejects parameters that cannot change while running.
    """
    # Declarations by name
    by_name = dict((p["name"], p) for p in schema)
    # Start from the defaults unless only a part is checked
    clean = {} if partial else defaults(schema)
    # name -> message
    errors = {}
    # Every given value
    for name, value in (values or {}).items():
        # Unknown parameter
        if name not in by_name:
            # Report it
            errors[name] = "unknown parameter"
            # Next one
            continue
        # Its declaration
        p = by_name[name]
        # Fixed parameter changed while running
        if live_only and not p.get("live"):
            # Report it
            errors[name] = "cannot be changed while the task is running"
            # Next one
            continue
        # Convert and range-check
        v, err = convert(p, value)
        # Invalid value
        if err:
            # Report it
            errors[name] = err
        # Valid value
        else:
            # Keep it
            clean[name] = v
    # Clean values and errors
    return clean, errors


# Folder with a task's presets
def _preset_dir(task_id):
    # STUDIO_HOME/presets/<task>
    return config.home_dir("presets", task_id)


# Names of a task's presets
def list_presets(task_id):
    """Sorted preset names of a task."""
    # File names without .json
    return sorted(os.path.splitext(n)[0] for n in os.listdir(_preset_dir(task_id)) if n.endswith(".json"))


# Save a preset (values are validated first)
def save_preset(task_id, schema, name, values):
    """Validate and store; returns ``(ok, errors)``."""
    # Name must be a safe file name
    if not PRESET_NAME.match(name or ""):
        # Reject
        return False, {"name": "use 1-40 letters, digits, '-' or '_'"}
    # Validate the values
    clean, errors = validate(schema, values)
    # Invalid values are not saved
    if errors:
        # Reject
        return False, errors
    # Write the file (ASCII JSON)
    with open(os.path.join(_preset_dir(task_id), name + ".json"), "w", encoding="utf-8") as f:
        # Pretty-printed values
        json.dump(clean, f, indent=2, sort_keys=True)
    # Saved
    return True, {}


# Load a preset (re-validated: the schema may have changed since it was saved)
def load_preset(task_id, schema, name):
    """Return ``(values, errors)``; values outside the current schema are dropped."""
    # Name must be a safe file name
    if not PRESET_NAME.match(name or ""):
        # Reject
        return None, {"name": "invalid preset name"}
    # Preset file
    p = os.path.join(_preset_dir(task_id), name + ".json")
    # Missing preset
    if not os.path.exists(p):
        # Reject
        return None, {"name": "no such preset"}
    # Read it
    with open(p, encoding="utf-8") as f:
        # Stored values
        stored = json.load(f)
    # Names the current schema still knows
    known = set(q["name"] for q in schema)
    # Validate only those (others were removed from the task)
    return validate(schema, dict((k, v) for k, v in stored.items() if k in known))


# Delete a preset
def delete_preset(task_id, name):
    """True when the preset existed and was removed."""
    # Name must be a safe file name
    if not PRESET_NAME.match(name or ""):
        # Nothing removed
        return False
    # Preset file
    p = os.path.join(_preset_dir(task_id), name + ".json")
    # Remove it if present
    if os.path.exists(p):
        # Delete the file
        os.remove(p)
        # Removed
        return True
    # Did not exist
    return False
