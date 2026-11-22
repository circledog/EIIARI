"""Sensor interfaces.

The robot only has a camera today (user, 2026-11-22). The other kinds are
declared here so tasks can already ask for them (``hub.get('odometry')``
returns ``None`` until a driver exists) and a future driver only has to
subclass ``Sensor``:

==============  ==============================================================
kind            ``read()`` value
==============  ==============================================================
``camera``      BGR uint8 frame (HxWx3), same object while no new frame
``odometry``    ``{'left_ticks': int, 'right_ticks': int}`` (wheel encoders)
``imu``         ``{'gyro': (x, y, z) rad/s, 'accel': (x, y, z) m/s^2}``
``range``       distance in metres (ultrasonic / ToF / lidar beam), or None
==============  ==============================================================

Every ``read()`` returns ``(timestamp, value)`` (``time.monotonic()``) or
``None`` when nothing has been measured yet.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import time

# Sensor kinds a task may ask for
KINDS = ("camera", "odometry", "imu", "range")


# Base class of every sensor driver
class Sensor(object):
    """Subclass and implement ``read()`` (and ``close()`` if needed)."""

    # Kind of the sensor (one of KINDS)
    kind = None

    # Name shown on the page
    def __init__(self, name):
        # Human-readable name
        self.name = name

    # Newest measurement
    def read(self):
        """``(timestamp, value)`` or None."""
        # Drivers override this
        raise NotImplementedError

    # Release the hardware
    def close(self):
        """Release the device (default: nothing to do)."""
        # Nothing by default
        pass

    # Short description for the page
    def describe(self):
        """``{'kind', 'name'}``."""
        # Kind and name
        return {"kind": self.kind, "name": self.name}


# Camera wrapper around a project4/camera.py source
class CameraSensor(Sensor):
    """Wraps an object with non-blocking ``read()`` / ``flush()`` / ``stop()``."""

    # Kind of this sensor
    kind = "camera"

    # Keep the source
    def __init__(self, source, name="camera"):
        # Name shown on the page
        Sensor.__init__(self, name)
        # project4 camera source
        self.source = source
        # Last frame object and when it was first seen
        self._last, self._t = None, None

    # Newest frame with the time it was first seen
    def read(self):
        """``(timestamp, frame)`` or None before the first frame."""
        # Newest frame (same object while nothing new arrived)
        frame = self.source.read()
        # No frame yet
        if frame is None:
            # Nothing measured
            return None
        # A new frame object: remember when it appeared
        if frame is not self._last:
            # New timestamp
            self._last, self._t = frame, time.monotonic()
        # Frame and its time
        return self._t, frame

    # Drop queued frames
    def flush(self):
        """Pass through to the source."""
        # Source's flush
        return self.source.flush()

    # Stop the camera
    def close(self):
        """Release the camera."""
        # Stop the source
        self.source.stop()


# Collection of the sensors a runner opened
class SensorHub(object):
    """``get(kind)`` returns the first sensor of that kind or None."""

    # Empty hub
    def __init__(self):
        # Registered sensors in order
        self.sensors = []
        # kind -> reason it could not be opened
        self.errors = {}

    # Register a sensor
    def add(self, sensor):
        """Add ``sensor`` (a ``Sensor``)."""
        # Only known kinds
        if sensor.kind not in KINDS:
            # Programming error in a driver
            raise ValueError("unknown sensor kind %r" % sensor.kind)
        # Keep it
        self.sensors.append(sensor)

    # First sensor of a kind
    def get(self, kind):
        """The sensor or None (not installed / failed to open)."""
        # Search in order
        for s in self.sensors:
            # Same kind
            if s.kind == kind:
                # Found
                return s
        # None of that kind
        return None

    # Description for the page
    def describe(self):
        """Open sensors, errors and the kinds that are not installed."""
        # Kinds that are open
        present = set(s.kind for s in self.sensors)
        # Everything the page shows
        return {"sensors": [s.describe() for s in self.sensors], "errors": dict(self.errors),
                "missing": [k for k in KINDS if k not in present and k not in self.errors]}

    # Close every sensor
    def close(self):
        """Close all sensors (errors are ignored)."""
        # Every sensor
        for s in self.sensors:
            # One failure must not keep the others open
            try:
                # Release it
                s.close()
            # Ignore
            except Exception:
                # Next one
                pass
