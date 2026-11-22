"""Motors and sensors for a runner process.

* ``open_robot()``: ``jetbot.Robot`` when available, otherwise ``FakeRobot``
  (values are only recorded), so every task also runs on a PC.
* ``open_sensors(settings)``: a ``SensorHub`` with the camera; other sensors
  only have their interface for now (see ``studio/sensors.py``).

Imported only in runner processes (the camera code loads torch).

Python 3.6 compatible (runs on the Jetson Nano).
"""
import sys

# studio paths (project4 folder)
from studio import PROJECT4_DIR
# Sensor interfaces
from studio.sensors import SensorHub, CameraSensor

# Make project4's modules importable (camera, drive, control, ...)
if PROJECT4_DIR not in sys.path:
    # Put project4/ first on the import path
    sys.path.insert(0, PROJECT4_DIR)


# Motor stand-in with the jetbot attribute name
class FakeMotor(object):
    """Holds the last value written."""

    # Start at zero
    def __init__(self):
        # Motor value in [-1, 1]
        self.value = 0.0


# Robot stand-in with the jetbot interface used by the tasks
class FakeRobot(object):
    """``left_motor`` / ``right_motor`` / ``stop()`` without hardware."""

    # Two motors at zero
    def __init__(self):
        # Left and right motor
        self.left_motor, self.right_motor = FakeMotor(), FakeMotor()

    # Both motors to zero
    def stop(self):
        """Set both motors to zero."""
        # Zero both
        self.left_motor.value = self.right_motor.value = 0.0


# Open the motor driver
def open_robot(allow_fake=True):
    """Return ``(robot, info)``; info is ``{'kind': 'jetbot'|'fake', 'error': ...}``."""
    # Try the real robot first
    try:
        from jetbot import Robot
        # Real motor driver
        robot = Robot()
        # Make sure it starts standing still
        robot.stop()
        # Real hardware
        return robot, {"kind": "jetbot", "error": None}
    # No jetbot package or no motor driver
    except Exception as e:
        # Fakes not allowed (e.g. a strict car setup): re-raise
        if not allow_fake:
            # Propagate the reason
            raise
        # Values are only recorded
        return FakeRobot(), {"kind": "fake", "error": repr(e)[:200]}


# Open the sensors configured in the settings
def open_sensors(settings):
    """``SensorHub`` with the camera (if it opens) and nothing else yet."""
    # Empty hub
    hub = SensorHub()
    # project4 camera sources
    import camera as cam_mod
    # The camera may be missing or held by another program (e.g. a notebook)
    try:
        # Open the configured source
        source = cam_mod.open_camera(settings.get("camera_source", "jetbot"),
                                     folder=settings.get("camera_folder") or None)
        # Register it
        hub.add(CameraSensor(source, settings.get("camera_source", "jetbot")))
    # Remember why there is no camera
    except Exception as e:
        # Shown on the task page
        hub.errors["camera"] = repr(e)[:300]
    # The hub
    return hub
