"""Runner side of the camera and motor check.

* A camera thread puts every new frame into the display slot.
* ``drive`` sets the motors for ``hold_time`` seconds (dead-man: the page
  repeats the command while a button is held; releasing it stops the motors).
* ``stop`` stops at once.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import threading
import time

import cv2

# Base class of task runtimes
from studio.runner import TaskRuntime


# Camera preview + hold-to-drive
class Runtime(TaskRuntime):
    """See the module docstring."""

    # Only `drive` moves the robot
    MOTION_COMMANDS = ("drive",)

    # Prepare state; threads start in start()
    def __init__(self, ctx):
        # Base: context and display slot
        TaskRuntime.__init__(self, ctx)
        # Camera sensor (None when it could not be opened)
        self.camera = ctx.sensors.get("camera")
        # Current motor values and when they expire
        self.left = self.right = 0.0
        # Expiry time of the last drive command
        self.until = 0.0
        # Lock around motor changes
        self.lock = threading.Lock()
        # Set by close()
        self.done = threading.Event()
        # Frames seen and the measured camera FPS
        self.frames, self.fps = 0, 0.0

    # Start the camera and dead-man threads
    def start(self):
        # Both threads
        for target in (self._camera_loop, self._deadman):
            # Daemon thread
            t = threading.Thread(target=target, name=target.__name__)
            # Do not block shutdown
            t.daemon = True
            # Run it
            t.start()

    # Camera thread: new frames into the slot
    def _camera_loop(self):
        # No camera: nothing to show
        if self.camera is None:
            # Done
            return
        # Last frame object and the FPS window
        last, n, t0 = None, 0, time.monotonic()
        # Until close()
        while not self.done.is_set():
            # Newest frame (non-blocking)
            got = self.camera.read()
            # A new frame object
            if got is not None and got[1] is not last:
                # Remember it
                last = got[1]
                # Hand it to the display
                self.slot.put(last)
                # Count it
                self.frames += 1
                # And for the FPS
                n += 1
            # Once per second: FPS
            if time.monotonic() - t0 >= 1.0:
                # Frames per second
                self.fps, n, t0 = n / (time.monotonic() - t0), 0, time.monotonic()
            # Poll about 100 times per second
            time.sleep(0.01)

    # Dead-man thread: stop the motors when the last drive command expired
    def _deadman(self):
        # Until close()
        while not self.done.wait(0.02):
            # Check under the lock
            with self.lock:
                # Moving and expired
                if (self.left or self.right) and time.monotonic() > self.until:
                    # Stop
                    self._set(0.0, 0.0)

    # Write the motors (caller holds the lock)
    def _set(self, left, right):
        # Remember them
        self.left, self.right = left, right
        # Send them
        self.ctx.robot.left_motor.value, self.ctx.robot.right_motor.value = left, right

    # Commands from the page
    def command(self, name, args):
        # Drive for hold_time seconds
        if name == "drive":
            # Power limit
            p = float(self.ctx.params["max_power"])
            # Requested values in [-1, 1], scaled by the limit
            left = max(-1.0, min(1.0, float(args.get("left", 0.0)))) * p
            # Right motor
            right = max(-1.0, min(1.0, float(args.get("right", 0.0)))) * p
            # Set them with a fresh expiry
            with self.lock:
                # Motors
                self._set(left, right)
                # Expiry
                self.until = time.monotonic() + float(self.ctx.params["hold_time"])
            # Values actually used
            return {"left": left, "right": right}
        # Stop now
        if name == "stop":
            # Stop
            self.stop_motion("button")
            # Done
            return {"left": 0.0, "right": 0.0}
        # Anything else
        raise ValueError("unknown command %r" % name)

    # Safety stop
    def stop_motion(self, code, detail=None):
        # Motors to zero, expiry cleared
        with self.lock:
            # Zero
            self._set(0.0, 0.0)
            # Expired
            self.until = 0.0

    # Moving while a motor is non-zero
    def is_moving(self):
        # Any motor value
        return bool(self.left or self.right)

    # Frame -> JPEG with the motor values drawn on it
    def render(self, frame, quality):
        # Draw on a copy
        img = frame.copy() if self.ctx.params.get("jpeg_overlay", True) else frame
        # Overlay
        if self.ctx.params.get("jpeg_overlay", True):
            # Motor values at the top
            cv2.putText(img, "L %+.2f  R %+.2f" % (self.left, self.right), (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1)
        # Encode
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        # JPEG and metadata
        return (bytes(buf) if ok else b""), {"left": self.left, "right": self.right}

    # State for the page
    def telemetry(self):
        # Camera, motors, hardware
        return {"camera_fps": round(self.fps, 1), "frames": self.frames, "left": self.left, "right": self.right,
                "robot": self.ctx.robot_info["kind"], "camera": self.camera is not None,
                "camera_error": self.ctx.sensors.errors.get("camera")}

    # End the threads
    def close(self):
        # Stop them
        self.done.set()
