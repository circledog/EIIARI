"""Inference + control loop decoupled from the display.

Two threads that never wait for each other:

* ``DriveLoop`` (highest priority): read the newest camera frame, mark the
  lanes, run the network, run the controller, send the motors, write the CSV
  row. It never touches a widget or a socket. Every step's result
  ``(frame, prediction, control values)`` is ``put`` into a ``LatestSlot``.
* ``DisplayWorker`` (lowest priority): waits for the newest result, draws,
  JPEG-encodes and hands it to a ``render`` callback (notebook widget, web
  stream). When rendering is slow it lowers its FPS and JPEG quality and
  simply skips results; it never builds a backlog and never delays control.

Rare events (model loaded, robot stopped, errors) go into ``DriveLoop.events``
and are shown by the display side, so even those never block the loop.

The loop knows nothing about ipywidgets: ``live_demo.ipynb`` and the
``studio`` web platform both drive the robot through this module. Messages and
event codes are English; the UI translates them.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import collections
import csv
import datetime
import os
import threading
import time
import traceback

import numpy as np
import torch

# Camera frame -> network input tensor
import camera as cam_mod
# Newest-item mailbox between the threads
from latest_slot import LatestSlot
# Lane marking: the preprocessing the models were trained with
from preprocessing import apply_frame as mark_lanes

# Public names of this module
__all__ = ["DriveLoop", "DisplayWorker", "PeriodStats", "CSV_HEADER", "STOP_CODES"]

# Columns of the drive log (one row per control step)
CSV_HEADER = ["t", "model", "driving", "x", "y", "raw_angle", "angle",
              "p", "i", "d", "steering", "speed", "left", "right", "status", "latency_ms",
              "formula", "kp", "ki", "kd", "bias", "min_speed", "max_speed", "turn_slowdown",
              "angle_alpha", "slew_per_second", "max_output", "reverse",
              "period_ms", "read_ms", "prep_ms", "infer_ms", "control_ms",
              "display_fps", "jpeg_quality", "render_ms"]
# Every reason the loop can stop the robot for (the UI maps them to its own language)
STOP_CODES = ("manual", "model_switch", "camera_stalled", "watchdog", "inference_error", "loop_error",
              "loop_ended", "closed", "benchmark", "invalid_prediction", "out_of_range", "target_too_close")
# Controller values written to the CSV, in order
_CTRL_KEYS = ("x", "y", "raw_angle", "angle", "p", "i", "d", "steering", "speed", "left", "right")
# Maximum number of pending UI events (older ones are dropped)
_MAX_EVENTS = 200


# Rolling window of durations with percentiles
class PeriodStats(object):
    """Keeps the last ``size`` durations (seconds); percentiles in milliseconds."""

    # Empty window
    def __init__(self, size=300):
        # Fixed-size window: old values fall out automatically
        self.values = collections.deque(maxlen=size)

    # Add one duration
    def add(self, seconds):
        # Append to the window
        self.values.append(seconds)

    # Percentile of the window in milliseconds (None when empty)
    def percentile(self, q):
        # Nothing measured yet
        if not self.values:
            # No value
            return None
        # numpy percentile, converted to milliseconds
        return float(np.percentile(np.asarray(self.values), q)) * 1000.0


# Background inference + control loop with its own watchdog
class DriveLoop(object):
    """Camera -> lane marking -> network -> controller -> motors, in one thread.

    ``camera`` has a non-blocking ``read()`` (newest frame; the same object
    while no new frame arrived) plus ``flush()`` / ``stop()``; ``robot`` has
    ``left_motor.value``, ``right_motor.value`` and ``stop()``.
    ``load_fn(path)`` returns a ``loader.LoadedModel``-like object and
    ``check_fn(model)`` a ``sanity.check``-like dict; both default to the
    project's real functions and are replaceable for tests.
    """

    # Store the parts; nothing runs until start()
    def __init__(self, camera, robot, controller, device="cpu", load_fn=None, check_fn=None,
                 log_dir="runs", flush_on_switch=True, timeout=0.5):
        # Default model loader: the project's loader on the chosen device
        if load_fn is None:
            # Imported here so tests with a fake loader do not need it
            from loader import load_model
            # Bind the device
            load_fn = lambda path: load_model(path, device)
        # Default sanity check: compare with the outputs stored at training time
        if check_fn is None:
            # Imported here for the same reason
            import sanity
            # The real check
            check_fn = sanity.check
        # Frame source (None = no camera)
        self.camera = camera
        # Motor driver (real or fake)
        self.robot = robot
        # Steering controller; its attributes may be changed at any time
        self.controller = controller
        # Device the network runs on
        self.device = torch.device(device)
        # Model loader and sanity check
        self.load_fn, self.check_fn = load_fn, check_fn
        # Folder for the CSV logs
        self.log_dir = log_dir
        # Drop frames queued during a model switch
        self.flush_on_switch = flush_on_switch
        # Stop when no successful control step happened for this long while driving (seconds)
        self.timeout = timeout
        # Newest result for the display side
        self.slot = LatestSlot()
        # Rare events for the UI: dicts with 'kind' and details
        self.events = collections.deque(maxlen=_MAX_EVENTS)
        # Loop statistics, refreshed once per second
        self.stats = {}
        # Rolling window of control periods (time between two control steps)
        self.periods = PeriodStats()
        # Display statistics, written by the DisplayWorker (shown and logged here)
        self.display_stats = {}
        # Lock around driving state changes and motor writes
        self.lock = threading.RLock()
        # Lock around the CSV file
        self._rec_lock = threading.Lock()
        # Current model, its sanity result and whether to mark lanes for it
        self.model, self.sanity, self.lanes = None, None, True
        # Model path requested by the UI, loaded by the loop thread
        self.pending = None
        # Driving flag and time of the last successful control step
        self.driving, self.last_ok = False, 0.0
        # Last frame object seen and when it arrived
        self.last_frame, self.frame_time = None, 0.0
        # Pause request (benchmark) and the loop's answer
        self._pause, self.paused = False, False
        # Set by close(): all threads end
        self.closed = False
        # Open CSV file, its writer and path
        self._csv, self._writer, self.csv_path = None, None, None
        # Traceback of the last unexpected loop error
        self.loop_error = None
        # Loop and watchdog threads
        self._threads = []

    # ----------------------------------------------------------------- events
    # Queue an event for the UI
    def emit(self, kind, **data):
        """Append ``{'kind': kind, 't': time, ...}`` to ``events``."""
        # Event kind plus its details
        data["kind"], data["t"] = kind, time.time()
        # Thread-safe append; the oldest event is dropped when full
        self.events.append(data)

    # Take every pending event
    def drain_events(self):
        """Pop and return all pending events, oldest first."""
        # Collected events
        out = []
        # Pop until empty (popleft is thread-safe)
        while True:
            # Take the oldest event
            try:
                # Remove it from the queue
                out.append(self.events.popleft())
            # Queue empty
            except IndexError:
                # Done
                return out

    # ---------------------------------------------------------------- driving
    # Start driving if everything needed is there
    def start_drive(self):
        """Returns ``(True, None)`` or ``(False, reason_code)``."""
        # No frames, nothing to steer by
        if self.camera is None:
            # Refuse
            return False, "no_camera"
        # No model loaded yet
        if self.model is None:
            # Refuse
            return False, "no_model"
        # The sanity comparison failed: the converted model is not trustworthy
        if self.sanity and self.sanity.get("status") == "failed":
            # Refuse
            return False, "sanity_failed"
        # One state change at a time
        with self.lock:
            # Fresh controller state: motors ramp up from 0
            self.controller.reset()
            # The watchdog counts from now
            self.last_ok = time.monotonic()
            # Motors may receive values
            self.driving = True
        # Tell the UI
        self.emit("drive_started")
        # Started
        return True, None

    # Stop driving and the motors right away
    def stop_drive(self, code="manual", detail=None, fault=False):
        """Stop the motors; ``code`` is one of ``STOP_CODES``."""
        # One state change at a time
        with self.lock:
            # Whether this call actually stopped a driving robot
            was_driving = self.driving
            # No more motor commands
            self.driving = False
            # Stop both motors
            self.robot.stop()
        # Tell the UI (also when already standing, so the reason is shown)
        self.emit("drive_stopped", code=code, detail=detail, fault=fault, was_driving=was_driving)

    # Ask the loop thread to load a model
    def request_model(self, path):
        """The loop thread loads, checks and activates ``path`` at its next step."""
        # Picked up by the loop
        self.pending = path

    # -------------------------------------------------------------- recording
    # Open or close the CSV drive log
    def set_recording(self, on):
        """Start (``on=True``) or stop the CSV log; returns the file path or None."""
        # One change at a time, never while the loop writes a row
        with self._rec_lock:
            # Switched on and no file open yet
            if on and self._csv is None:
                # Create the log folder if needed
                os.makedirs(self.log_dir, exist_ok=True)
                # File name with the current date and time
                path = os.path.join(self.log_dir, "drive_%s.csv" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
                # Open the file (UTF-8: the robot's container has an ASCII locale)
                self._csv = open(path, "w", newline="", encoding="utf-8")
                # CSV writer for it
                self._writer = csv.writer(self._csv)
                # Header row
                self._writer.writerow(CSV_HEADER)
                # Remember the path
                self.csv_path = path
                # Tell the UI
                self.emit("recording_started", path=path)
            # Switched off while a file is open
            elif not on and self._csv is not None:
                # Close the file
                self._csv.close()
                # Forget the file and the writer
                self._csv = self._writer = None
                # Tell the UI
                self.emit("recording_stopped", path=self.csv_path)
            # Current file, if any
            return self.csv_path if self._csv is not None else None

    # Write one CSV row if recording
    def _log_row(self, m, status, latency_ms, timing):
        # Never at the same time as opening / closing the file
        with self._rec_lock:
            # Not recording
            if self._writer is None:
                # Nothing to do
                return
            # Short name for the controller
            c = self.controller
            # Display statistics written by the display thread
            d = self.display_stats
            # Time, model, driving flag, readings, status, latency, settings and timing
            self._writer.writerow(
                ["%.3f" % time.time(), os.path.basename(m.path), int(self.driving)]
                + ["%.4f" % c.last[k] for k in _CTRL_KEYS]
                + [status, "%.2f" % latency_ms, c.formula, c.kp, c.ki, c.kd, c.bias, c.min_speed,
                   c.max_speed, c.turn_slowdown, c.angle_alpha, c.slew_per_second, c.max_output,
                   int(c.allow_reverse)]
                + ["%.2f" % (timing[k] * 1000.0) if timing.get(k) is not None else ""
                   for k in ("period", "read", "prep", "infer", "control")]
                + ["%.1f" % d["fps"] if d.get("fps") is not None else "",
                   d.get("quality", ""),
                   "%.1f" % d["render_ms"] if d.get("render_ms") is not None else ""])

    # ----------------------------------------------------------------- models
    # Load a model inside the loop thread, check it and make it the current one
    def _switch_model(self, path):
        # Never drive with a model that is being replaced
        self.stop_drive("model_switch")
        # No model and no check result while loading
        self.model, self.sanity = None, None
        # Tell the UI loading started
        self.emit("model_loading", path=path)
        # Loading, checking and warm-up can fail (missing file, SHA256 mismatch, no TensorRT on a PC, ...)
        try:
            # Load the checkpoint
            m = self.load_fn(path)
            # Lane marking only for models trained with it (train.py --no-lanes stores method 'none')
            lanes = ((m.meta or {}).get("preprocessing") or {}).get("method") != "none"
            # Compare with the outputs stored at training time
            result = self.check_fn(m)
            # Warm-up: the first TRT / CUDA inferences are slow, so do them before the flush
            warm = self.camera.read() if self.camera is not None else None
            # Only possible once the camera has a frame
            if warm is not None:
                # A few warm-up runs
                for _ in range(3):
                    # Same input path as the real loop
                    m(cam_mod.preprocess(mark_lanes(warm) if lanes else warm, self.device, m.dtype))
        # Report and keep running without a model
        except Exception as e:
            # Tell the UI
            self.emit("model_failed", path=path, error=repr(e), traceback=traceback.format_exc())
            # No model
            return
        # Throw away frames queued during loading so the picture is live again
        if self.flush_on_switch and self.camera is not None:
            # Tell the UI
            self.emit("flushing", path=path)
            # Empty the camera queue
            self.camera.flush()
        # Preprocessing and check result of the new model, set before the model itself
        self.lanes, self.sanity = lanes, result
        # Start using the new model: inference begins right away
        self.model = m
        # Forget the previous controller state
        self.controller.reset()
        # Tell the UI
        self.emit("model_loaded", path=path, model=m, sanity=result)

    # ------------------------------------------------------------------- loop
    # Inference + control loop body (runs in its own thread)
    def _loop(self):
        # Control period window (time between two control steps)
        periods = self.periods
        # Per-second accumulators: frame count and summed stage times
        acc = {"n": 0, "read": 0.0, "prep": 0.0, "infer": 0.0, "control": 0.0}
        # Start of the current one-second window
        win_t = time.monotonic()
        # Time of the previous control step
        prev_step = None
        # Run until close()
        while not self.closed:
            # One iteration; an unexpected error must not kill the loop thread
            try:
                # The benchmark needs the GPU / camera alone
                if self._pause:
                    # Tell the benchmark the loop is idle
                    self.paused = True
                    # The pause must not count as one long control period
                    prev_step = None
                    # Wait
                    time.sleep(0.02)
                    # Check again
                    continue
                # Not paused
                self.paused = False
                # A model was requested by the UI
                if self.pending is not None:
                    # Take the request and clear it
                    path, self.pending = self.pending, None
                    # Load, check and activate it
                    self._switch_model(path)
                    # Restart the period measurement
                    prev_step = None
                # No camera: nothing to infer
                if self.camera is None:
                    # Idle
                    time.sleep(0.1)
                    # Check again
                    continue
                # Start of the read stage
                t_read = time.monotonic()
                # Newest camera frame (non-blocking)
                frame = self.camera.read()
                # Current time
                now = time.monotonic()
                # No frame yet, or the same frame as last time (camera has not delivered a new one)
                if frame is None or frame is self.last_frame:
                    # The camera stalled for too long while driving
                    if self.driving and now - self.frame_time > self.timeout:
                        # Stop with the reason
                        self.stop_drive("camera_stalled", detail={"seconds": self.timeout}, fault=True)
                    # Wait a little
                    time.sleep(0.003)
                    # Try again
                    continue
                # Remember the frame and when it arrived
                self.last_frame, self.frame_time = frame, now
                # Read stage done
                t_prep = time.monotonic()
                # Preprocessing: paint the lane lines red, unless the model was trained without it
                frame = mark_lanes(frame) if self.lanes else frame
                # Current model
                m = self.model
                # No model selected yet: publish the lane-marked picture only
                if m is None:
                    # Picture without prediction
                    self.slot.put({"frame": frame, "xy": None, "ctrl": None, "left": 0.0, "right": 0.0,
                                   "driving": False, "status": "no_model", "model": None, "t": time.time()})
                    # Next frame
                    continue
                # Network input tensor
                x = cam_mod.preprocess(frame, self.device, m.dtype)
                # Preprocessing stage done
                t_infer = time.monotonic()
                # Inference can fail (e.g. CUDA / TensorRT errors)
                try:
                    # Run the network and get the prediction as a flat float numpy array
                    xy = m(x).detach().float().cpu().numpy().flatten()
                    # Wait for the GPU so the timing is correct
                    if self.device.type == "cuda":
                        # Block until all GPU work is done
                        torch.cuda.synchronize()
                # Stop, unload and report the error
                except Exception as e:
                    # Stop driving
                    self.stop_drive("inference_error", detail={"error": repr(e)}, fault=True)
                    # Unload the model; choose it again after fixing the problem
                    self.model = None
                    # Tell the UI
                    self.emit("inference_error", error=repr(e), traceback=traceback.format_exc())
                    # Next frame
                    continue
                # Inference stage done
                t_control = time.monotonic()
                # One control step
                left, right, status = self.controller.update(xy, t_control)
                # Motor commands only while driving (checked and sent under the lock)
                with self.lock:
                    # Driving state at this moment
                    driving = self.driving
                    # Driving and the controller is happy: send the motor values
                    if driving and status == "ok":
                        # Send them
                        self.robot.left_motor.value, self.robot.right_motor.value = left, right
                        # The watchdog sees a successful step
                        self.last_ok = time.monotonic()
                    # Standing: keep the motors at zero
                    elif not driving:
                        # Stop both motors
                        self.robot.stop()
                # Driving but the controller refused (bad prediction / target too close)
                if driving and status != "ok":
                    # Stop with the reason
                    self.stop_drive(status, fault=True)
                # Time of this control step
                step = time.monotonic()
                # Time since the previous control step
                period = None if prev_step is None else step - prev_step
                # Record it
                if period is not None:
                    # Into the rolling window
                    periods.add(period)
                # Remember this step
                prev_step = step
                # Stage times of this step in seconds (control is finished after the CSV row)
                timing = {"period": period, "read": t_prep - t_read, "prep": t_infer - t_prep,
                          "infer": t_control - t_infer}
                # Preprocessing + inference latency in milliseconds (same meaning as before)
                latency_ms = (t_control - t_prep) * 1000.0
                # Control values of this step (a copy, the controller overwrites its dict)
                ctrl = dict(self.controller.last)
                # Hand the result to the display: overwrite, never queue
                self.slot.put({"frame": frame, "xy": xy, "ctrl": ctrl,
                               "left": left if self.driving else 0.0, "right": right if self.driving else 0.0,
                               "driving": self.driving, "status": status, "model": os.path.basename(m.path),
                               "latency_ms": latency_ms, "t": time.time()})
                # Control stage: controller + motors + publish (CSV row below)
                timing["control"] = time.monotonic() - t_control
                # Write one log row while recording
                self._log_row(m, status, latency_ms, timing)
                # Accumulate the stage times of this second
                acc["n"] += 1
                # Sum of each stage
                for key in ("read", "prep", "infer", "control"):
                    # Add this step's time
                    acc[key] += timing[key]
                # Once per second: publish the statistics
                if step - win_t >= 1.0:
                    # Steps per second and mean stage times in milliseconds
                    n = max(1, acc["n"])
                    # New statistics dict (replaced as a whole, so readers never see half an update)
                    self.stats = {"fps": acc["n"] / (step - win_t),
                                  "period_p50_ms": periods.percentile(50),
                                  "period_p95_ms": periods.percentile(95),
                                  "read_ms": acc["read"] * 1000.0 / n, "prep_ms": acc["prep"] * 1000.0 / n,
                                  "infer_ms": acc["infer"] * 1000.0 / n, "control_ms": acc["control"] * 1000.0 / n}
                    # Start a new window
                    acc = {"n": 0, "read": 0.0, "prep": 0.0, "infer": 0.0, "control": 0.0}
                    # Window start
                    win_t = step
            # Report the error, stop the robot and keep the loop alive
            except Exception as e:
                # Keep it for inspection
                self.loop_error = traceback.format_exc()
                # Never keep driving after an unexpected error
                self.stop_drive("loop_error", detail={"error": repr(e)}, fault=True)
                # Tell the UI
                self.emit("loop_error", error=repr(e), traceback=self.loop_error)
                # Do not spin on a repeating error
                time.sleep(0.5)
        # Leave the robot in a safe state when the loop ends
        self.stop_drive("loop_ended")
        # Close the log
        self.set_recording(False)

    # Stops the robot when the loop or the camera stalls while driving
    def _watchdog(self):
        # Run until close()
        while not self.closed:
            # Check 50 times per second
            time.sleep(0.02)
            # Read the state consistently
            with self.lock:
                # Driving and no successful control step for too long
                late = self.driving and time.monotonic() - self.last_ok > self.timeout
            # Stop with the reason
            if late:
                # Fault stop
                self.stop_drive("watchdog", detail={"seconds": self.timeout}, fault=True)

    # ------------------------------------------------------------- lifecycle
    # Start the loop and watchdog threads
    def start(self):
        """Start the background threads (call once)."""
        # Loop thread, then watchdog thread
        for target, name in ((self._loop, "drive-loop"), (self._watchdog, "drive-watchdog")):
            # Daemon thread so a crashed notebook never keeps the process alive
            t = threading.Thread(target=target, name=name)
            # Do not block interpreter shutdown
            t.daemon = True
            # Run it
            t.start()
            # Keep it for close()
            self._threads.append(t)
        # Allow chaining
        return self

    # Pause the loop (benchmark) and wait until it is idle
    def pause(self, timeout=10.0):
        """Stop the robot and block until the loop is idle; raises on timeout."""
        # Never drive while paused
        self.stop_drive("benchmark")
        # Ask the loop to pause
        self._pause = True
        # Give up after `timeout` seconds
        t0 = time.monotonic()
        # Wait for the loop's answer
        while not self.paused:
            # The loop thread did not answer
            if time.monotonic() - t0 > timeout:
                # Resume and report
                self._pause = False
                # Explain
                raise RuntimeError("the drive loop did not pause within %.0f s" % timeout)
            # Short wait
            time.sleep(0.01)

    # Let the loop run again
    def resume(self):
        """Undo ``pause()``."""
        # Clear the pause request
        self._pause = False

    # Stop the robot, then end the threads
    def close(self, timeout=1.0):
        """Stop motors first, then end the loop and watchdog; True if both ended in time."""
        # Motors off before anything else
        self.stop_drive("closed")
        # Ask the threads to end
        self.closed = True
        # Wake the display side waiting for results
        self.slot.close()
        # Overall deadline for joining
        deadline = time.monotonic() + timeout
        # Wait for each thread
        for t in self._threads:
            # Remaining time
            t.join(max(0.0, deadline - time.monotonic()))
        # Close the log in case the loop could not
        self.set_recording(False)
        # Make extra sure the motors are off
        self.robot.stop()
        # True when every thread ended
        return not any(t.is_alive() for t in self._threads)


# Display thread: newest result -> render callback, adapting FPS and JPEG quality
class DisplayWorker(object):
    """Calls ``render(item, quality)`` with the newest ``DriveLoop`` result.

    * Never more than ``max_fps`` renders per second; results produced in
      between are skipped (only the newest is ever shown).
    * When one render takes longer than its time budget (1 / current FPS),
      FPS and JPEG quality are lowered; after a second of fast renders they
      are raised step by step again.
    * ``on_events(events)`` is called with the loop's pending events before
      each render, so status changes are shown from this thread too.
    * ``stats`` (dict) is updated in place: fps, quality, render_ms, skipped.
    """

    # Store the callbacks and the limits
    def __init__(self, slot, render, events_fn=None, on_events=None, max_fps=15.0, min_fps=1.0,
                 max_quality=80, min_quality=40, stats=None):
        # Mailbox with the newest result
        self.slot = slot
        # render(item, quality): draw + encode + show
        self.render = render
        # Function returning pending events (DriveLoop.drain_events) and their handler
        self.events_fn, self.on_events = events_fn, on_events
        # FPS limits
        self.max_fps, self.min_fps = float(max_fps), float(min_fps)
        # JPEG quality limits
        self.max_quality, self.min_quality = int(max_quality), int(min_quality)
        # Current target FPS and quality (start at the best)
        self.fps, self.quality = self.max_fps, self.max_quality
        # Statistics shared with the loop (DriveLoop.display_stats)
        self.stats = stats if stats is not None else {}
        # Traceback of the last render error
        self.error = None
        # Set by stop()
        self._stop = False
        # The display thread
        self._thread = None

    # Handle pending events (exceptions are kept, never raised)
    def _events(self):
        # Nothing to do without an event source and a handler
        if self.events_fn is None or self.on_events is None:
            # Done
            return
        # Pending events
        events = self.events_fn()
        # Only call the handler when there is something to show
        if events:
            # A broken handler must not stop the display
            try:
                # Show them
                self.on_events(events)
            # Remember the error
            except Exception:
                # Keep the traceback for inspection
                self.error = traceback.format_exc()

    # Adapt FPS and quality to how long the last render took
    def _adapt(self, seconds, good):
        """Returns the updated count of fast renders in a row."""
        # Time budget of one render at the current FPS
        budget = 1.0 / self.fps
        # Too slow: back off quickly
        if seconds > budget:
            # Lower FPS by 30 %, not below the minimum
            self.fps = max(self.min_fps, self.fps * 0.7)
            # Lower quality by 10, not below the minimum
            self.quality = max(self.min_quality, self.quality - 10)
            # Start counting fast renders again
            return 0
        # Clearly fast enough: count it
        if seconds < 0.5 * budget:
            # One more fast render
            good += 1
            # About one second of fast renders: recover a step
            if good >= self.fps:
                # One more frame per second, up to the maximum
                self.fps = min(self.max_fps, self.fps + 1.0)
                # Better quality, up to the maximum
                self.quality = min(self.max_quality, self.quality + 5)
                # Count again
                good = 0
        # Updated count
        return good

    # Display thread body
    def _run(self):
        # Sequence number of the last rendered result
        seq = 0
        # Next allowed render time
        next_t = 0.0
        # Fast renders in a row
        good = 0
        # Renders and skipped results in the current one-second window
        n, skipped, win_t = 0, 0, time.monotonic()
        # Until stop() or the slot is closed
        while not self._stop and not self.slot.closed:
            # Show status changes first
            self._events()
            # Wait for a newer result (short timeout so events and stop are handled)
            got = self.slot.wait_newer(seq, timeout=0.1)
            # Nothing new yet
            if got is None:
                # Check again
                continue
            # Respect the FPS limit: wait until the next render is allowed
            wait = next_t - time.monotonic()
            # Too early
            if wait > 0:
                # Sleep (results arriving meanwhile overwrite each other in the slot)
                time.sleep(wait)
                # Take the newest result after waiting
                got = self.slot.get()
            # Sequence number and result to render
            new_seq, item = got
            # Results that were never shown
            skipped += max(0, new_seq - seq - 1)
            # Remember what is shown now
            seq = new_seq
            # Time the render
            t0 = time.monotonic()
            # A render error must not kill the display thread
            try:
                # Draw, encode and show
                self.render(item, self.quality)
            # Remember the error and slow down
            except Exception:
                # Keep the traceback
                self.error = traceback.format_exc()
                # Do not spin on a repeating error
                time.sleep(0.5)
            # Render duration
            dt = time.monotonic() - t0
            # Adapt FPS and quality
            good = self._adapt(dt, good)
            # Next render no earlier than one period after this one started
            next_t = t0 + 1.0 / self.fps
            # Count this render
            n += 1
            # Last render time in milliseconds and current settings
            self.stats.update({"render_ms": dt * 1000.0, "quality": self.quality, "target_fps": self.fps})
            # Once per second: actual FPS and skipped results
            if time.monotonic() - win_t >= 1.0:
                # Window length
                span = time.monotonic() - win_t
                # Actual renders per second and results skipped per second
                self.stats.update({"fps": n / span, "skipped_per_s": skipped / span})
                # New window
                n, skipped, win_t = 0, 0, time.monotonic()
        # Show the last events (e.g. the final stop)
        self._events()

    # Start the display thread
    def start(self):
        """Start the thread (call once)."""
        # Daemon thread
        self._thread = threading.Thread(target=self._run, name="display")
        # Do not block interpreter shutdown
        self._thread.daemon = True
        # Run it
        self._thread.start()
        # Allow chaining
        return self

    # End the display thread
    def stop(self, timeout=1.0):
        """Ask the thread to end and wait; True if it ended in time."""
        # Ask it to end
        self._stop = True
        # Nothing started
        if self._thread is None:
            # Nothing to wait for
            return True
        # Wait (a render in progress finishes first)
        self._thread.join(timeout)
        # Whether it ended
        return not self._thread.is_alive()
