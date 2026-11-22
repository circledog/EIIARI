"""Control loop vs display: a slow display must not slow down control (TASKS M0).

Fake camera (new frame every 1/30 s), fake model (5 ms, releases the GIL like
TensorRT) and fake motors; the display's render callback is made slow on
purpose (0.5 s per frame, like a stalled Wi-Fi / widget).
"""
import gc
import time
import tracemalloc
import unittest

import numpy as np
import torch

# Shared test helpers (sets the import path, synthetic track frames)
from _helpers import track_frame
# Camera module: ThreadedReader under test
import camera as cam_mod
# Steering controller used by the loop
from control import SteeringController
# Modules under test
from drive import DriveLoop, DisplayWorker

# Camera frame rate of the fake camera
CAMERA_FPS = 30.0
# One intra-op thread: torch's thread pool on tiny tensors adds timing jitter
# (p95 58 ms vs 39 ms on a 12-core PC) that has nothing to do with the display
torch.set_num_threads(1)


# Camera whose read() returns a new frame object every 1/CAMERA_FPS seconds
class FakeCamera(object):
    # Pre-render a few frames and start the clock
    def __init__(self):
        # Different synthetic track pictures
        self.frames = [track_frame(seed=i) for i in range(8)]
        # Clock start
        self.t0 = time.monotonic()
        # Frame index and object returned last
        self.index, self.frame = None, None
        # When True, no new frames arrive (camera stall)
        self.frozen = False

    # Newest frame; the same object until the next frame time
    def read(self):
        # Index of the frame that is "live" now
        i = int((time.monotonic() - self.t0) * CAMERA_FPS)
        # A new frame arrived (unless frozen)
        if i != self.index and not self.frozen:
            # Fresh copy so every frame is a new object
            self.index, self.frame = i, self.frames[i % len(self.frames)].copy()
        # Current frame
        return self.frame

    # Nothing queued
    def flush(self):
        # Always fresh
        return True

    # Nothing to release
    def stop(self):
        # No hardware
        pass


# Motor stand-in
class FakeMotor(object):
    # Last value written
    value = 0.0


# Robot stand-in with two motors
class FakeRobot(object):
    # Two motors at zero
    def __init__(self):
        # Left and right motor
        self.left_motor, self.right_motor = FakeMotor(), FakeMotor()

    # Both motors to zero
    def stop(self):
        # Zero both
        self.left_motor.value = self.right_motor.value = 0.0


# Model stand-in: fixed prediction after a short sleep
class FakeModel(object):
    # Prediction and delay
    def __init__(self, xy=(0.1, -0.3), delay=0.005):
        # Output the controller accepts (target ahead, slightly right)
        self.xy = torch.tensor([list(xy)], dtype=torch.float32)
        # Inference time; sleeping releases the GIL like TensorRT does
        self.delay = delay
        # Attributes the loop reads
        self.dtype, self.meta, self.path = torch.float32, {}, "fake.pth"

    # Run "inference"
    def __call__(self, x):
        # Simulated inference time
        time.sleep(self.delay)
        # Fixed prediction
        return self.xy


# Build a loop with fakes and a loaded model
def make_loop(model=None, timeout=0.5):
    # The model the fake loader returns
    model = model or FakeModel()
    # Loop with a fake loader and an always-passing sanity check
    loop = DriveLoop(FakeCamera(), FakeRobot(), SteeringController(), device="cpu",
                     load_fn=lambda path: model, check_fn=lambda m: {"status": "passed", "message": ""},
                     timeout=timeout)
    # Start the threads
    loop.start()
    # Ask for the (fake) model
    loop.request_model("fake.pth")
    # Wait until it is active
    t0 = time.monotonic()
    # At most 5 s
    while loop.model is None and time.monotonic() - t0 < 5:
        # Poll
        time.sleep(0.01)
    # Ready loop
    return loop


# Render callback that draws and encodes like the notebook, plus an optional delay
def make_render(delay, counter):
    # The callback
    def render(item, quality):
        # Draw prediction and steering like the notebook
        img = cam_mod.draw_overlay(item["frame"], item["xy"], (item["ctrl"] or {}).get("steering"))
        # JPEG-encode with the current quality
        cam_mod.to_jpeg(img, quality)
        # Simulated slow widget / network
        time.sleep(delay)
        # Count renders
        counter.append(time.monotonic())
    # Return it
    return render


# Run a display worker for `seconds` and return (loop period p95 ms, renders, display stats)
def measure(loop, delay, seconds):
    # Render timestamps
    renders = []
    # Display thread sharing the loop's statistics dict
    disp = DisplayWorker(loop.slot, make_render(delay, renders), loop.drain_events, lambda ev: None,
                         stats=loop.display_stats).start()
    # Let it settle for a moment (the display adapts its FPS)
    time.sleep(1.0)
    # Only count periods of this phase
    loop.periods.values.clear()
    # Renders before the window
    n0 = len(renders)
    # Measurement window
    time.sleep(seconds)
    # p95 of this phase's control periods
    p95 = loop.periods.percentile(95)
    # Renders in the window
    n = len(renders) - n0
    # Stop the display thread
    disp.stop(2.0)
    # Results
    return p95, n, dict(loop.display_stats)


# Tests of the decoupled loop
class DecouplingTest(unittest.TestCase):
    # Slow display: control period unchanged, display FPS drops, memory flat
    def test_slow_display_does_not_slow_control(self):
        # Loop with a fake model
        loop = make_loop()
        # Measure with a fast display first (fresh period window afterwards)
        try:
            # Fast display: no extra delay
            p95_fast, n_fast, _ = measure(loop, 0.0, 4)
            # Slow display: 0.5 s per render
            p95_slow, n_slow, dstats = measure(loop, 0.5, 4)
            # Memory tracking in a separate phase (tracemalloc itself slows the loop down)
            tracemalloc.start()
            # Slow display, first part
            measure(loop, 0.5, 1)
            # Collect cyclic garbage first: it is freed later anyway and is not a backlog
            gc.collect()
            # Memory now
            mem_mid = tracemalloc.get_traced_memory()[0]
            # Keep running slowly a bit longer
            measure(loop, 0.5, 2)
            # Same before the second reading
            gc.collect()
            # Memory after more slow rendering
            mem_end = tracemalloc.get_traced_memory()[0]
            # Stop tracking
            tracemalloc.stop()
        # Always shut down
        finally:
            # End the threads
            loop.close()
        # Print the numbers for the HANDOFF record
        print("\n[decoupling] period p95 fast %.1f ms, slow %.1f ms; renders fast %d / slow %d; "
              "display %s; memory growth %.0f kB"
              % (p95_fast, p95_slow, n_fast, n_slow,
                 {k: round(v, 1) for k, v in dstats.items()}, (mem_end - mem_mid) / 1024.0))
        # Control period about one camera frame (33 ms) in both cases
        self.assertLess(p95_fast, 1000.0 / CAMERA_FPS * 1.5)
        # A slow display changes the control period p95 by less than 10 % (+1 ms timer slack)
        self.assertLess(p95_slow, p95_fast * 1.10 + 1.0)
        # The display really slowed down: about 2 renders per second instead of 15
        self.assertLess(n_slow, 0.25 * n_fast)
        # Adaptive quality went down while slow
        self.assertLess(dstats["quality"], 80)
        # No backlog: memory does not grow while the display lags
        self.assertLess(mem_end - mem_mid, 2 * 1024 * 1024)

    # close(): both threads end within 1 s and the motors are zero (TASKS M0-6)
    def test_close_stops_everything(self):
        # Loop with a fake model
        loop = make_loop()
        # Display thread
        disp = DisplayWorker(loop.slot, make_render(0.0, []), loop.drain_events, lambda ev: None).start()
        # Start driving
        ok, why = loop.start_drive()
        # Driving was allowed
        self.assertTrue(ok, why)
        # Let it drive
        time.sleep(0.5)
        # Motors received values
        self.assertGreater(loop.robot.left_motor.value + loop.robot.right_motor.value, 0.0)
        # Start the timer
        t0 = time.monotonic()
        # Inference + motors first
        ended = loop.close(timeout=1.0)
        # Then the display
        ended_disp = disp.stop(timeout=1.0)
        # Both ended in time
        self.assertTrue(ended and ended_disp)
        # Within one second in total
        self.assertLess(time.monotonic() - t0, 1.0)
        # Motors are zero
        self.assertEqual((loop.robot.left_motor.value, loop.robot.right_motor.value), (0.0, 0.0))

    # Inference that hangs while driving: the watchdog stops the robot
    def test_watchdog_stops_on_slow_inference(self):
        # Model that takes 1.2 s per step (longer than the 0.5 s timeout)
        slow = FakeModel(delay=0.005)
        # Loop with it
        loop = make_loop(slow)
        # Drive
        try:
            # Start driving
            self.assertTrue(loop.start_drive()[0])
            # Let a few normal steps run
            time.sleep(0.3)
            # Now inference hangs
            slow.delay = 1.2
            # Wait for the watchdog
            time.sleep(1.0)
            # Stop events so far
            stops = [e for e in loop.drain_events() if e["kind"] == "drive_stopped"]
        # Always shut down
        finally:
            # End the threads (waits for the slow step)
            loop.close(timeout=2.0)
        # The robot was stopped
        self.assertFalse(loop.driving)
        # By the watchdog, as a fault
        self.assertTrue(any(e["code"] == "watchdog" and e["fault"] for e in stops), stops)

    # Camera that delivers no new frames while driving: stop
    def test_camera_stall_stops(self):
        # Loop with a fake model
        loop = make_loop()
        # Drive
        try:
            # Start driving
            self.assertTrue(loop.start_drive()[0])
            # Normal driving for a moment
            time.sleep(0.2)
            # Camera stalls
            loop.camera.frozen = True
            # Longer than the timeout
            time.sleep(0.9)
            # Stop events so far
            codes = [e["code"] for e in loop.drain_events() if e["kind"] == "drive_stopped"]
        # Always shut down
        finally:
            # End the threads
            loop.close()
        # Stopped by the camera check or the watchdog, whichever came first
        self.assertTrue(set(codes) & {"camera_stalled", "watchdog"}, codes)
        # Not driving any more
        self.assertFalse(loop.driving)

    # Pause / resume (benchmark) works and stops the robot
    def test_pause_resume(self):
        # Loop with a fake model
        loop = make_loop()
        # Run the scenario
        try:
            # Start driving
            loop.start_drive()
            # Pause: must stop the robot and become idle
            loop.pause(timeout=2.0)
            # Idle and standing
            self.assertTrue(loop.paused)
            # Not driving
            self.assertFalse(loop.driving)
            # Results stop while paused
            seq = loop.slot.get()[0]
            # Wait a bit
            time.sleep(0.2)
            # No new results
            self.assertEqual(loop.slot.get()[0], seq)
            # Resume
            loop.resume()
            # New results arrive again
            self.assertIsNotNone(loop.slot.wait_newer(seq, timeout=1.0))
        # Always shut down
        finally:
            # End the threads
            loop.close()

    # CSV rows contain the timing columns
    def test_csv_has_timing(self):
        # Temporary folder for the log
        import tempfile
        # Folder removed afterwards
        with tempfile.TemporaryDirectory() as tmp:
            # Loop logging into it
            loop = make_loop()
            # Log folder
            loop.log_dir = tmp
            # Record for a moment
            try:
                # Start recording
                path = loop.set_recording(True)
                # Some steps
                time.sleep(0.5)
            # Always shut down
            finally:
                # Closes the file too
                loop.close()
            # Read the log back
            import csv
            # Open it
            with open(path, encoding="utf-8") as f:
                # All rows
                rows = list(csv.DictReader(f))
        # Several rows were written
        self.assertGreater(len(rows), 5)
        # Inference time recorded (fake model sleeps 5 ms)
        self.assertGreaterEqual(float(rows[-1]["infer_ms"]), 4.0)
        # Control period recorded after the first row
        self.assertTrue(rows[-1]["period_ms"])


# Non-blocking camera reader (TASKS M0-5)
class ThreadedReaderTest(unittest.TestCase):
    # read() returns at once although grab() blocks for 50 ms
    def test_read_does_not_block(self):
        # Blocking source: every grab takes 50 ms
        class Slow(object):
            # Counter for distinct frames
            n = 0
            # Blocking read
            def grab(self):
                # Simulated exposure / transfer time
                time.sleep(0.05)
                # One more frame
                self.n += 1
                # New frame object
                return np.full((4, 4, 3), self.n % 255, np.uint8)
            # Nothing to release
            def stop(self):
                # No hardware
                pass
        # Wrap it
        reader = cam_mod.ThreadedReader(Slow())
        # Use and always stop it
        try:
            # Wait for the first frame
            self.assertTrue(reader.flush(timeout=1.0))
            # Longest read() duration
            worst = 0.0
            # Many reads in a row
            for _ in range(200):
                # Time one read
                t0 = time.perf_counter()
                # Newest frame
                frame = reader.read()
                # Track the worst case
                worst = max(worst, time.perf_counter() - t0)
            # A frame was returned
            self.assertIsNotNone(frame)
            # Same object when nothing new arrived (the loop relies on this)
            self.assertIs(reader.read(), reader.read())
            # Never blocked for 1 ms
            self.assertLess(worst, 0.001)
            # flush() waits for a frame captured afterwards
            before = reader.read()
            # Wait for a fresh one
            self.assertTrue(reader.flush(timeout=1.0))
            # Different frame now
            self.assertIsNot(reader.read(), before)
        # Always stop the thread
        finally:
            # End it
            reader.stop()


# Run the tests directly
if __name__ == "__main__":
    # unittest entry point
    unittest.main()
