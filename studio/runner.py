"""Runner process: one task, owning the camera and the motors.

Started by the web process (``studio/runlink.py``)::

    python3 -m studio.runner --port P --token T --task lane_follow

It connects back to the web process on 127.0.0.1:P, proves the token,
receives the validated parameters, opens the hardware and runs the task's
``TaskRuntime`` subclass. Threads:

* reader (main thread): heartbeats, e-stop, parameter changes, frame requests;
  never blocks, so heartbeats are always processed on time;
* command worker: task commands one at a time (they may be slow, e.g. load a model);
* frame worker: answers ``want_frame`` with the newest result, JPEG-encoded at the
  requested quality -- only on request, so frames never queue (backpressure);
* telemetry: task state and events to the web process 5 times per second;
* link watchdog (``studio/safety.py``).

Python 3.6 compatible (runs on the Jetson Nano).
"""
import argparse
import collections
import os
import queue
import signal
import sys
import threading
import time
import traceback

import numpy as np

# studio paths
from studio import PROJECT4_DIR, config
# Wire format
from studio.ipc import Connection
# Link watchdog
from studio.safety import LinkWatchdog
# Task registry
from studio import tasks as tasks_mod

# Make project4's modules importable (latest_slot, camera, drive, ...)
if PROJECT4_DIR not in sys.path:
    # Put project4/ first on the import path
    sys.path.insert(0, PROJECT4_DIR)
# Newest-item mailbox (torch-free)
from latest_slot import LatestSlot  # noqa: E402

# Telemetry messages per second
TELEMETRY_HZ = 5.0
# Maximum queued events (oldest dropped)
MAX_EVENTS = 200


# Turn numpy values (and containers of them) into plain JSON types
def jsonable(value):
    """Recursively convert numpy scalars / arrays, tuples and sets for ``json.dumps``."""
    # Dicts: convert keys to str and values recursively
    if isinstance(value, dict):
        # New dict
        return dict((str(k), jsonable(v)) for k, v in value.items())
    # Sequences: lists
    if isinstance(value, (list, tuple, set)):
        # New list
        return [jsonable(v) for v in value]
    # numpy arrays: lists
    if isinstance(value, np.ndarray):
        # Nested lists
        return jsonable(value.tolist())
    # numpy scalars: Python scalars
    if isinstance(value, np.generic):
        # Python value
        return value.item()
    # Non-finite floats are not valid JSON
    if isinstance(value, float) and not np.isfinite(value):
        # Represent as null
        return None
    # Anything else as is
    return value


# Everything a task runtime gets from the runner
class Context(object):
    """Task declaration, parameters, settings, hardware and an event function."""

    # Store the parts
    def __init__(self, task, params, settings, robot, robot_info, sensors, emit):
        # Normalised TASK declaration
        self.task = task
        # Validated parameter values (updated in place for live changes)
        self.params = params
        # Server settings (camera source, role, ...)
        self.settings = settings
        # 'car' or 'pc'
        self.role = settings.get("role", "pc")
        # Motor driver (real or fake) and its description
        self.robot, self.robot_info = robot, robot_info
        # Sensors (camera, ...)
        self.sensors = sensors
        # emit(kind, **data): event for the page
        self.emit = emit

    # Folder for files this task writes
    def data_dir(self, *parts):
        """``STUDIO_HOME/<sub>/<task id>/...`` folder, created when missing."""
        # Under the studio home
        return config.home_dir(*(parts + (self.task["id"],)))


# Base class of every task's runner-side logic
class TaskRuntime(object):
    """Override what the task needs. All methods are called from runner threads.

    * ``slot``: put display items here (newest only); ``render`` turns one into JPEG.
    * ``MOTION_COMMANDS``: commands that make the robot move; refused while the
      link to an open page is down.
    * ``stop_motion`` must stop the motors immediately and never raise.
    """

    # Commands that start motion (refused without a live page)
    MOTION_COMMANDS = ()

    # Keep the context and create the display mailbox
    def __init__(self, ctx):
        # Runner context
        self.ctx = ctx
        # Newest display item
        self.slot = LatestSlot()

    # Start background work (camera loop, ...)
    def start(self):
        """Called once after construction."""
        # Nothing by default
        pass

    # Live parameter change (already validated)
    def update_params(self, values):
        """Apply changed ``live`` parameters."""
        # Store them
        self.ctx.params.update(values)

    # Task command from the page
    def command(self, name, args):
        """Return a JSON-able result; raise to report an error."""
        # No commands by default
        raise ValueError("unknown command %r" % name)

    # Stop the motors right now
    def stop_motion(self, code, detail=None):
        """Stop the motors (safety); default: robot.stop()."""
        # Stop both motors
        self.ctx.robot.stop()

    # Whether the robot may currently be moving
    def is_moving(self):
        """True while motor commands are being sent."""
        # Not moving by default
        return False

    # One display item -> JPEG bytes and metadata
    def render(self, item, quality):
        """Return ``(jpeg_bytes, meta_dict)``; default: item is a BGR frame."""
        import cv2
        # Encode with the requested quality
        ok, buf = cv2.imencode(".jpg", item, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        # Bytes and no metadata
        return (bytes(buf) if ok else b""), {}

    # State shown on the page
    def telemetry(self):
        """JSON-able dict, sent 5 times per second."""
        # Nothing by default
        return {}

    # Release resources
    def close(self):
        """Called once at shutdown, after ``stop_motion``."""
        # Nothing by default
        pass


# The runner's message loop and worker threads
class Runner(object):
    """Glue between the connection, the task runtime and the safety logic."""

    # Store the connection and prepare the queues
    def __init__(self, conn):
        # Connection to the web process
        self.conn = conn
        # Events for the page
        self.events = collections.deque(maxlen=MAX_EVENTS)
        # Commands waiting for the command worker
        self.commands = queue.Queue()
        # Pending frame request (after_seq, quality) and its condition
        self.want, self.want_cond = None, threading.Condition()
        # Set when the runner should end
        self.done = threading.Event()
        # Task runtime, hardware and watchdog (created in setup)
        self.runtime = self.sensors = self.robot = self.watchdog = None

    # Queue an event for the page
    def emit(self, kind, **data):
        """``{'kind', 't', ...}`` into the event queue."""
        # Kind and time
        data["kind"], data["t"] = kind, time.time()
        # Thread-safe append
        self.events.append(data)

    # Send a message, ignoring a closed connection
    def send(self, header, payload=b""):
        """Send; returns False when the connection is gone."""
        # The web process may have died
        try:
            # Send it
            self.conn.send(jsonable(header), payload)
            # Sent
            return True
        # Connection closed
        except (OSError, EOFError):
            # The reader notices the closed socket and shuts down
            return False

    # Stop the motors through the runtime, never raising
    def safe_stop(self, code, detail=None):
        """Call ``runtime.stop_motion``; falls back to ``robot.stop()``."""
        # The runtime's own stop
        try:
            # Task-specific stop (also stops its control loop)
            if self.runtime is not None:
                # Stop
                self.runtime.stop_motion(code, detail)
        # A broken runtime must not prevent the stop
        except Exception:
            # Record it
            self.emit("error", where="stop_motion", traceback=traceback.format_exc())
        # Always stop the motors directly too
        try:
            # Both motors to zero
            if self.robot is not None:
                # Stop
                self.robot.stop()
        # Nothing more we can do here (web process runs motor_guard if we die)
        except Exception:
            # Record it
            self.emit("error", where="robot.stop", traceback=traceback.format_exc())

    # Link watchdog callback
    def _link_lost(self, seconds):
        # Only matters while moving
        if self.runtime is not None and self.runtime.is_moving():
            # Stop the motors
            self.safe_stop("link_lost", {"seconds": round(seconds, 2)})
            # Tell the page (it will see it once it is back)
            self.emit("safety_stop", code="link_lost", seconds=round(seconds, 2))

    # Open the hardware and create the runtime
    def setup(self, start):
        """``start`` is the web process's 'start' message."""
        # Hardware helpers (load the camera code / torch)
        from studio import hardware
        # Task declaration
        task = start["task"]
        # Server settings plus role
        settings = start.get("settings") or {}
        # Motors (fake on a PC)
        self.robot, robot_info = hardware.open_robot()
        # Camera and other sensors
        self.sensors = hardware.open_sensors(settings)
        # Context for the runtime
        ctx = Context(task, dict(start.get("params") or {}), settings, self.robot, robot_info,
                      self.sensors, self.emit)
        # Task's runtime class
        cls = tasks_mod.load_runtime(task)
        # Create it
        self.runtime = cls(ctx)
        # Stop the motors when the page link is lost
        self.watchdog = LinkWatchdog(self._link_lost, timeout=float(ctx.params.get("link_timeout", 5.0)))
        # Start the task's background work
        self.runtime.start()
        # Start the watchdog
        self.watchdog.start()
        # Information for the page
        return {"robot": robot_info, "sensors": self.sensors.describe(), "pid": os.getpid()}

    # Command worker thread
    def _command_worker(self):
        # Until the runner ends
        while not self.done.is_set():
            # Next command (short timeout so `done` is checked)
            try:
                # Wait for one
                msg = self.commands.get(timeout=0.2)
            # None waiting
            except queue.Empty:
                # Check `done` again
                continue
            # Command name and arguments
            name, args = msg.get("name"), msg.get("args") or {}
            # Motion needs an open page answering heartbeats
            if name in self.runtime.MOTION_COMMANDS and not self.watchdog.alive():
                # Refuse
                self.send({"type": "reply", "id": msg.get("id"), "ok": False, "error": "link_down"})
                # Next command
                continue
            # Run it; errors are reported, not raised
            try:
                # Task command
                result = self.runtime.command(name, args)
                # Success
                self.send({"type": "reply", "id": msg.get("id"), "ok": True, "result": result})
            # Report the error
            except Exception as e:
                # Failure with the reason
                self.send({"type": "reply", "id": msg.get("id"), "ok": False, "error": repr(e),
                           "traceback": traceback.format_exc()})

    # Frame worker thread: answer frame requests with the newest item
    def _frame_worker(self):
        # Until the runner ends
        while not self.done.is_set():
            # Wait for a request
            with self.want_cond:
                # Sleep until a request arrives or the runner ends
                self.want_cond.wait_for(lambda: self.want is not None or self.done.is_set(), 0.5)
                # Take the request
                want, self.want = self.want, None
            # No request (timeout or shutting down)
            if want is None:
                # Check again
                continue
            # Requested sequence and quality
            after, quality = want
            # Wait for an item newer than `after`
            got = None
            # Keep waiting while running and no newer request replaced this one
            while got is None and not self.done.is_set():
                # Short waits so shutdown is noticed
                got = self.runtime.slot.wait_newer(after, timeout=0.5)
            # Shutting down
            if got is None:
                # End
                break
            # Newest item
            seq, item = got
            # Rendering may fail (task bug): report, do not die
            try:
                # JPEG and metadata
                jpeg, meta = self.runtime.render(item, quality)
            # Report and send an empty frame so the web process asks again
            except Exception:
                # Record it
                self.emit("error", where="render", traceback=traceback.format_exc())
                # Empty frame
                jpeg, meta = b"", {"error": "render failed"}
            # Send it
            self.send({"type": "frame", "seq": seq, "meta": meta}, jpeg)

    # Telemetry thread
    def _telemetry_worker(self):
        # Seconds between telemetry messages
        period = 1.0 / TELEMETRY_HZ
        # Time of the last telemetry message
        last = 0.0
        # Until the runner ends
        while not self.done.wait(0.05):
            # Send pending events right away
            while True:
                # Take the oldest
                try:
                    # Remove it from the queue
                    ev = self.events.popleft()
                # Empty
                except IndexError:
                    # Done with events
                    break
                # Send it
                self.send({"type": "event", "event": ev})
            # Telemetry at its own rate
            if time.monotonic() - last >= period:
                # Remember the time
                last = time.monotonic()
                # Task state; a broken telemetry() must not stop the thread
                try:
                    # Collect it
                    data = self.runtime.telemetry()
                # Report the error instead
                except Exception as e:
                    # Error text
                    data = {"error": repr(e)}
                # Send it with the link state
                self.send({"type": "telemetry", "data": data, "link": self.watchdog.alive(),
                           "moving": bool(self.runtime.is_moving())})

    # Dispatch one message from the web process (reader thread; must not block)
    def handle(self, msg):
        """Returns False when the runner should end."""
        # Message type
        kind = msg.get("type")
        # Heartbeat from the page (forwarded by the web process)
        if kind == "heartbeat":
            # Link is alive
            self.watchdog.beat()
        # Motors off before an orderly shutdown (not an emergency)
        elif kind == "halt":
            # Stop
            self.safe_stop("task_stop")
        # Emergency stop
        elif kind == "estop":
            # Stop now
            self.safe_stop("estop")
            # Tell the page
            self.emit("safety_stop", code="estop")
        # Live parameter change (validated by the web process)
        elif kind == "params":
            # New values
            values = msg.get("values") or {}
            # Apply them
            self.runtime.update_params(values)
            # The watchdog timeout is a parameter too
            if "link_timeout" in values:
                # New timeout
                self.watchdog.timeout = float(values["link_timeout"])
        # Task command: run in the command worker
        elif kind == "command":
            # Queue it
            self.commands.put(msg)
        # Frame request: replace any older request
        elif kind == "want_frame":
            # Store and wake the frame worker
            with self.want_cond:
                # Newest request wins
                self.want = (int(msg.get("after", 0)), int(msg.get("quality", 70)))
                # Wake it
                self.want_cond.notify_all()
        # Orderly shutdown
        elif kind == "shutdown":
            # End the loop
            return False
        # Keep running
        return True

    # Main loop
    def run(self):
        """Reader loop until shutdown or the connection closes; returns the exit code."""
        # Worker threads
        for target in (self._command_worker, self._frame_worker, self._telemetry_worker):
            # Daemon threads
            t = threading.Thread(target=target, name=target.__name__.strip("_"))
            # Do not block interpreter shutdown
            t.daemon = True
            # Run it
            t.start()
        # Exit code
        code = 0
        # Read until told to stop
        try:
            # One message at a time
            while not self.done.is_set():
                # Next message (raises EOFError when the web process is gone)
                msg, _ = self.conn.recv()
                # Dispatch it
                if not self.handle(msg):
                    # Shutdown requested
                    break
        # Web process gone: stop immediately
        except (EOFError, OSError):
            # Stop the motors
            self.safe_stop("web_gone")
            # Not an orderly shutdown
            code = 2
        # Always clean up
        finally:
            # Stop the motors before anything else
            self.safe_stop("shutdown")
            # End the worker threads
            self.done.set()
            # Wake the frame worker
            with self.want_cond:
                # Wake it
                self.want_cond.notify_all()
            # Stop the watchdog
            if self.watchdog is not None:
                # End its thread
                self.watchdog.stop()
            # Release the task's resources
            try:
                # Task cleanup
                self.runtime.close()
            # Do not fail the shutdown
            except Exception:
                # Print to the runner log
                traceback.print_exc()
            # Release the camera
            self.sensors.close()
            # Last stop of the motors
            self.robot.stop()
            # Close the socket
            self.conn.close()
        # Exit code
        return code


# Command-line entry point
def main(argv=None):
    """Connect, set up the task, run until shutdown."""
    # Command-line options
    ap = argparse.ArgumentParser(description="studio task runner (started by the web process)")
    # Port of the web process
    ap.add_argument("--port", type=int, required=True)
    # Token proving we were started by that web process
    ap.add_argument("--token", required=True)
    # Task id
    ap.add_argument("--task", required=True)
    # Parse
    args = ap.parse_args(argv)
    # Connect back to the web process
    conn = Connection.connect(args.port)
    # Introduce ourselves
    conn.send({"type": "hello", "token": args.token, "task": args.task, "pid": os.getpid()})
    # Wait for the start message
    start, _ = conn.recv()
    # Anything else means the web process refused us
    if start.get("type") != "start":
        # Give up
        return 3
    # The runner
    runner = Runner(conn)
    # SIGTERM / SIGINT: stop the motors and end the loop
    def on_signal(signum, frame):
        # Stop now
        runner.safe_stop("signal")
        # End the loop
        runner.done.set()
        # Unblock the reader
        conn.close()
    # Register for both signals
    for sig in (signal.SIGTERM, signal.SIGINT):
        # Same handler
        signal.signal(sig, on_signal)
    # Hardware and runtime may fail (camera busy, import error, ...)
    try:
        # Set up the task
        info = runner.setup(start)
    # Report the failure and exit
    except Exception as e:
        # Tell the web process why
        runner.send({"type": "error", "error": repr(e), "traceback": traceback.format_exc()})
        # Motors were possibly opened: stop them
        runner.safe_stop("setup_failed")
        # Close the camera if it was opened
        if runner.sensors is not None:
            # Release it
            runner.sensors.close()
        # Failure
        return 1
    # Ready
    runner.send({"type": "ready", "info": info})
    # Run until shutdown
    return runner.run()


# Run as a module
if __name__ == "__main__":
    # Exit with the runner's code
    sys.exit(main())
