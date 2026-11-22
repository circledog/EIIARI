"""Parameter schemas: validation on the server, presets, task registry."""
import unittest

# Test helpers (temporary STUDIO_HOME)
from _base import StudioTestCase
# Modules under test
from studio import params, tasks

# Schema covering every type
SCHEMA = [
    {"name": "speed", "type": "float", "min": 0.0, "max": 1.0, "default": 0.2, "live": True},
    {"name": "epochs", "type": "int", "min": 1, "max": 100, "default": 10},
    {"name": "flip", "type": "bool", "default": False},
    {"name": "net", "type": "choice", "choices": [["resnet18", "ResNet-18"], ["mobilenet_v2", "MobileNetV2"]],
     "default": "resnet18"},
    {"name": "note", "type": "string", "default": "", "max_length": 5},
]


# Validation rules
class ValidateTest(unittest.TestCase):
    # Defaults fill missing values; good values pass and are converted
    def test_good_values(self):
        # Validate a few values given as strings (form posts)
        clean, errors = params.validate(SCHEMA, {"speed": "0.5", "epochs": "20", "flip": "on"})
        # No errors
        self.assertEqual(errors, {})
        # Converted values and defaults
        self.assertEqual(clean, {"speed": 0.5, "epochs": 20, "flip": True, "net": "resnet18", "note": ""})

    # Every kind of bad value is refused
    def test_bad_values(self):
        # Bad value for every parameter plus an unknown one
        _, errors = params.validate(SCHEMA, {"speed": 1.5, "epochs": 2.5, "flip": "maybe", "net": "vgg",
                                             "note": "toolong", "hack": 1})
        # All six are reported
        self.assertEqual(set(errors), {"speed", "epochs", "flip", "net", "note", "hack"})
        # NaN / inf / booleans are not numbers
        for bad in (float("nan"), float("inf"), True, "abc", None):
            # Refused
            self.assertIn("speed", params.validate(SCHEMA, {"speed": bad})[1], bad)

    # While running only live parameters may change
    def test_live_only(self):
        # Change one live and one fixed parameter
        clean, errors = params.validate(SCHEMA, {"speed": 0.3, "epochs": 5}, partial=True, live_only=True)
        # Live one accepted
        self.assertEqual(clean, {"speed": 0.3})
        # Fixed one refused
        self.assertIn("epochs", errors)

    # Malformed declarations are caught when a task loads
    def test_check_schema(self):
        # Every malformed declaration
        for bad in ([{"name": "a", "type": "float", "min": 1, "max": 0, "default": 0}],
                    [{"name": "a", "type": "nope", "default": 0}],
                    [{"name": "a", "type": "int", "min": 0, "max": 5, "default": 9}],
                    [{"name": "a", "type": "bool", "default": True}, {"name": "a", "type": "bool", "default": True}]):
            # Raises
            with self.assertRaises(ValueError):
                # Check it
                params.check_schema(bad)

    # Presets: save, list, load, delete; bad names and values refused
    def test_presets(self):
        # Save
        ok, errors = params.save_preset("t_test", SCHEMA, "fast", {"speed": 0.9})
        # Saved
        self.assertTrue(ok, errors)
        # Listed
        self.assertIn("fast", params.list_presets("t_test"))
        # Loaded with defaults for the rest
        values, errors = params.load_preset("t_test", SCHEMA, "fast")
        # Stored value
        self.assertEqual(values["speed"], 0.9)
        # Path tricks are refused
        self.assertFalse(params.save_preset("t_test", SCHEMA, "../evil", {})[0])
        # Invalid values are not saved
        self.assertFalse(params.save_preset("t_test", SCHEMA, "bad", {"speed": 7})[0])
        # Deleted
        self.assertTrue(params.delete_preset("t_test", "fast"))
        # Gone
        self.assertNotIn("fast", params.list_presets("t_test"))


# Task registry
class RegistryTest(unittest.TestCase):
    # Tasks load, the safety parameter is added with the 5 s default
    def test_discover(self):
        # Load all tasks
        found, errors = tasks.discover()
        # No broken task package
        self.assertEqual(errors, {})
        # id -> task
        by_id = dict((t["id"], t) for t in found)
        # The four tasks
        self.assertTrue({"drive_check", "navigate", "search"} <= set(by_id))
        # Development tasks are marked
        self.assertEqual(by_id["search"]["status"], "dev")
        # Safety parameter added to every task
        link = [p for p in by_id["drive_check"]["params"] if p["name"] == "link_timeout"][0]
        # Default chosen by the user: 5 s
        self.assertEqual(link["default"], 5.0)


# The same rules through the HTTP API
class ParamsHttpTest(StudioTestCase):
    # Server refuses out-of-range values on validate and start
    def test_server_validates(self):
        # Log in
        self.login()
        # Out of range
        status, body = self.api("POST", "/api/tasks/drive_check/validate", {"values": {"max_power": 5}})
        # Reported
        self.assertIn("max_power", body["errors"])
        # Start with an invalid value
        status, body = self.api("POST", "/api/tasks/drive_check/start", {"values": {"hold_time": -1}})
        # Refused before anything starts
        self.assertEqual(status, 400)
        # No runner was created
        self.assertIsNone(self.studio.manager.runner)
        # A development task cannot start
        self.assertEqual(self.api("POST", "/api/tasks/search/start", {"values": {}})[0], 409)

    # Presets through the API
    def test_presets_http(self):
        # Log in
        self.login()
        # Save
        status, body = self.api("POST", "/api/tasks/drive_check/presets", {"name": "p1", "values": {"max_power": 0.4}})
        # Saved
        self.assertEqual(status, 200)
        # Load
        status, body = self.api("GET", "/api/tasks/drive_check/presets/p1")
        # Value back
        self.assertEqual(body["values"]["max_power"], 0.4)
        # Delete
        self.assertTrue(self.api("DELETE", "/api/tasks/drive_check/presets/p1")[1]["deleted"])


# Run directly
if __name__ == "__main__":
    # unittest entry point
    unittest.main()
