"""Camera-driven control loop shared by task runtimes that drive by themselves.

``CameraLoopRuntime`` (navigate, search) owns the parts every such task
needs and that must behave the same everywhere:

* loop thread: read the newest camera frame, call ``step(frame)`` (the task's
  perception + planning + motor command), measure periods;
* stall detection: no new frame for ``safety_timeout`` seconds while driving
  -> stop;
* watchdog thread: no successful step (``last_ok``) for ``safety_timeout``
  seconds while driving -> stop (catches a hung step);
* ``stop_drive``: motors off under the lock, one event for the page;
* per-second statistics (FPS, period p50 / p95, mean time of each part
  ``step`` reports).

A subclass sets ``STOPPED_EVENT``, implements ``step(frame)`` (returns a
dict ``part -> seconds``; sets ``self.last_ok`` after each motor command) and
may override ``on_safety_stop(code)``.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import threading
import time
import traceback

# Base class of task runtimes (also puts project4/ on the import path)
from studio.runner import TaskRuntime
# Rolling period percentiles (project4's drive loop uses the same)
from drive import PeriodStats  # noqa: E402


# Loop / watchdog / stop logic for camera-driven tasks
class CameraLoopRuntime(TaskRuntime):
    """See the module docstring."""

    # Event kind sent by stop_drive (pages listen to it)
    STOPPED_EVENT = "task.stopped"

    # Camera, lock, driving state, statistics
    def __init__(self, ctx):
        # Base: context and display slot
        TaskRuntime.__init__(self, ctx)
        # Camera sensor (None without a camera)
        sensor = ctx.sensors.get("camera")
        # Its frame source
        self.camera = sensor.source if sensor is not None else None
        # Guards `driving` and the motor commands
        self.lock = threading.RLock()
        # Whether motor commands are being sent; time of the last good step
        self.driving, self.last_ok = False, 0.0
        # Newest frame and when it arrived
        self.frame, self.frame_t = None, 0.0
        # Per-second statistics and rolling periods
        self.stats, self.periods = {}, PeriodStats()
        # Set by close()
        self.done = threading.Event()

    # Seconds without frame / successful step that stop the robot
    def safety_timeout(self):
        """The task's ``safety_timeout`` parameter."""
        # Current value (live parameter)
        return float(self.ctx.params["safety_timeout"])

    # Motors off
    def stop_drive(self, code, fault=False, detail=None):
        """Stop the motors; ``STOPPED_EVENT`` for the page."""
        # Atomic with respect to motor commands
        with self.lock:
            # Remember whether it was driving
            was = self.driving
            # No more motor commands
            self.driving = False
            # Both motors to zero
            self.ctx.robot.stop()
        # Tell the page
        self.ctx.emit(self.STOPPED_EVENT, code=code, fault=fault, detail=detail, was_driving=was)

    # Hook after a stop by the loop's own safety checks
    def on_safety_stop(self, code):
        """Called after watchdog / camera_stalled / loop_error stops."""
        # Nothing by default
        pass

    # A safety stop: motors off plus the hook
    def _safety_stop(self, code, detail=None):
        # Motors off
        self.stop_drive(code, fault=True, detail=detail)
        # Task-specific state
        self.on_safety_stop(code)

    # One frame of work (task-specific)
    def step(self, frame):
        """Perception, planning, motor command; returns ``{part: seconds}``."""
        # Subclasses implement it
        raise NotImplementedError

    # Start the loop and the watchdog
    def start(self):
        """Background threads (daemon)."""
        # Both threads
        for target in (self._loop, self._watchdog):
            # Named after the method
            t = threading.Thread(target=target, name=target.__name__.strip("_"))
            # Do not block interpreter shutdown
            t.daemon = True
            # Run it
            t.start()

    # Watchdog thread: stop when steps stop succeeding
    def _watchdog(self):
        # Until close()
        while not self.done.wait(0.02):
            # Read the state atomically
            with self.lock:
                # Driving but no good step for too long
                late = self.driving and time.monotonic() - self.last_ok > self.safety_timeout()
            # Stop
            if late:
                # Safety stop
                self._safety_stop("watchdog")

    # Loop thread
    def _loop(self):
        # Previous frame object, previous step time, part sums, window start, steps in the window
        last, prev, acc, win, n = None, None, {}, time.monotonic(), 0
        # Until close()
        while not self.done.is_set():
            # A failing step must not end the thread
            try:
                # No camera: idle
                if self.camera is None:
                    # Wait
                    time.sleep(0.1)
                    # Again
                    continue
                # Newest frame
                frame = self.camera.read()
                # Now
                now = time.monotonic()
                # Nothing new yet
                if frame is None or frame is last:
                    # Camera stalled while driving
                    if self.driving and now - self.frame_t > self.safety_timeout():
                        # Safety stop
                        self._safety_stop("camera_stalled")
                    # Short wait
                    time.sleep(0.003)
                    # Again
                    continue
                # Remember the frame
                last, self.frame, self.frame_t = frame, frame, now
                # The task's work
                timing = self.step(frame)
                # Period since the previous step
                if prev is not None:
                    # Record it
                    self.periods.add(now - prev)
                # This step's time
                prev = now
                # One more step in the window
                n += 1
                # Sum each part
                for k, v in timing.items():
                    # Add it
                    acc[k] = acc.get(k, 0.0) + v
                # Once per second
                if now - win >= 1.0:
                    # Rates, percentiles and mean part times
                    stats = {"fps": n / (now - win), "period_p50_ms": self.periods.percentile(50),
                             "period_p95_ms": self.periods.percentile(95)}
                    # Mean of each part in ms
                    stats.update((k + "_ms", v * 1000.0 / n) for k, v in acc.items())
                    # Publish
                    self.stats = stats
                    # New window
                    acc, n, win = {}, 0, now
            # Stop and report
            except Exception as e:
                # Safety stop with the reason
                self._safety_stop("loop_error", detail={"error": repr(e)})
                # Traceback for the page
                self.ctx.emit("error", where="%s loop" % self.ctx.task["id"], error=repr(e),
                              traceback=traceback.format_exc())
                # Do not spin on a persistent error
                time.sleep(0.5)

    # Whether motor commands are being sent
    def is_moving(self):
        """True while driving."""
        # The flag
        return bool(self.driving)

    # End the threads
    def close(self):
        """Motors off, then end the loop and watchdog."""
        # Motors off
        self.stop_drive("shutdown")
        # End the threads
        self.done.set()
