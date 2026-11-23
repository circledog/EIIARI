"""CameraLoopRuntime: the safety stops shared by the navigate and search tasks."""
import time
import unittest

import numpy as np

# Test helpers (temporary STUDIO_HOME, import path)
import _base  # noqa: F401
# Module under test
from studio.camloop import CameraLoopRuntime
# Fake motors
from studio.hardware import FakeRobot


# Camera that produces a new frame object per read until frozen
class FakeCamera(object):
    # Running
    def __init__(self):
        # When set, read() returns the same frame forever
        self.frozen = False
        # Current frame
        self.frame = np.zeros((8, 8, 3), np.uint8)

    # Newest frame
    def read(self):
        # New object unless frozen
        if not self.frozen:
            # Fresh frame
            self.frame = np.zeros((8, 8, 3), np.uint8)
        # Current one
        return self.frame


# Sensor wrapper with a .source
class FakeSensor(object):
    # Wrap the camera
    def __init__(self, cam):
        # Frame source
        self.source = cam


# Minimal runner context
class FakeCtx(object):
    # Parts the base class uses
    def __init__(self, cam):
        # Parameters
        self.params = {"safety_timeout": 0.2}
        # Sensors
        self.sensors = {"camera": FakeSensor(cam)}
        # Motors
        self.robot = FakeRobot()
        # Task declaration
        self.task = {"id": "test"}
        # Emitted events
        self.events = []

    # Record events
    def emit(self, kind, **data):
        # Store kind and data
        self.events.append(dict(data, kind=kind))


# Runtime that drives with a configurable step
class Drive(CameraLoopRuntime):
    # Event name
    STOPPED_EVENT = "t.stopped"

    # Step behaviour
    def __init__(self, ctx):
        # Base
        CameraLoopRuntime.__init__(self, ctx)
        # 'ok', 'hang' (no last_ok update) or 'raise'
        self.mode = "ok"
        # Safety-stop hook calls
        self.hooked = []

    # One frame
    def step(self, frame):
        # Fail on purpose
        if self.mode == "raise":
            # Error
            raise RuntimeError("boom")
        # Good step: motors on, watchdog fed
        if self.mode == "ok":
            # Under the lock
            with self.lock:
                # Only while driving
                if self.driving:
                    # Motors
                    self.ctx.robot.left_motor.value = self.ctx.robot.right_motor.value = 0.2
                    # Fed
                    self.last_ok = time.monotonic()
        # A little work
        time.sleep(0.005)
        # Part times
        return {"work": 0.005}

    # Hook
    def on_safety_stop(self, code):
        # Record it
        self.hooked.append(code)

    # Start driving
    def go(self):
        # Under the lock
        with self.lock:
            # Fed now
            self.last_ok = time.monotonic()
            # Driving
            self.driving = True


# The three safety stops and the statistics
class CameraLoopTest(unittest.TestCase):

    # Running runtime
    def setUp(self):
        # Camera
        self.cam = FakeCamera()
        # Runtime
        self.rt = Drive(FakeCtx(self.cam))
        # Threads
        self.rt.start()

    # Stop threads
    def tearDown(self):
        # End
        self.rt.close()

    # Wait for a stop code
    def wait_stop(self, code, limit=2.0):
        # Deadline
        t0 = time.monotonic()
        # Poll
        while time.monotonic() - t0 < limit:
            # Stopped with that code
            if any(e["kind"] == "t.stopped" and e["code"] == code for e in self.rt.ctx.events):
                # Seconds it took
                return time.monotonic() - t0
            # Short wait
            time.sleep(0.01)
        # Not seen
        self.fail("no %s stop; events %r" % (code, self.rt.ctx.events))

    # Frozen camera while driving -> camera_stalled
    def test_camera_stall_stops(self):
        # Drive
        self.rt.go()
        # Let it run
        time.sleep(0.1)
        # Freeze
        self.cam.frozen = True
        # Stops within about safety_timeout
        self.assertLess(self.wait_stop("camera_stalled"), 0.6)
        # Motors off, hook called
        self.assertFalse(self.rt.driving)
        # Hook
        self.assertIn("camera_stalled", self.rt.hooked)

    # Step that never feeds last_ok -> watchdog
    def test_watchdog_stops(self):
        # Hang mode
        self.rt.mode = "hang"
        # Drive
        self.rt.go()
        # Stops
        self.assertLess(self.wait_stop("watchdog"), 0.6)
        # Hook
        self.assertIn("watchdog", self.rt.hooked)

    # Raising step -> loop_error, loop survives
    def test_loop_error_stops_and_survives(self):
        # Drive
        self.rt.go()
        # Fail
        self.rt.mode = "raise"
        # Stops
        self.wait_stop("loop_error")
        # Error event for the page
        self.assertTrue(any(e["kind"] == "error" and e["where"] == "test loop" for e in self.rt.ctx.events))
        # Back to normal: the thread is still alive and publishes statistics
        self.rt.mode = "ok"
        # Old statistics forgotten
        self.rt.stats = {}
        # Wait for a new window
        time.sleep(1.3)
        # Fresh statistics with the part time
        self.assertIn("work_ms", self.rt.stats)
        # Loop running at a sensible rate
        self.assertGreater(self.rt.stats["fps"], 20)

    # No stop while things are fine
    def test_no_false_stop(self):
        # Drive
        self.rt.go()
        # A while
        time.sleep(0.6)
        # Still driving
        self.assertTrue(self.rt.driving)
        # Motors on
        self.assertEqual(self.rt.ctx.robot.left_motor.value, 0.2)


# Run directly
if __name__ == "__main__":
    # unittest main
    unittest.main()
