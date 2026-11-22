"""Web-process side of a runner: start it, talk to it, stop it.

``RunnerLink`` (one per running task):

* starts ``python3 -m studio.runner`` and waits for it to connect back on a
  random 127.0.0.1 port with the right token;
* forwards a heartbeat every 0.1 s **only while at least one open task page
  answers** (``presence``) -> the runner stops the motors when every page is
  gone or silent for ``link_timeout`` (TASKS M1-7);
* keeps only the newest frame; a frame is requested from the runner only when
  a page is ready for one (backpressure, TASKS M1-6);
* forwards commands, parameter changes and e-stops; collects events and
  telemetry for the pages;
* when the runner exits unexpectedly on the car, runs ``studio.motor_guard``.

Runs on tornado's IOLoop (tornado 5.1+ / 6.x, Python 3.6+).

Python 3.6 compatible (runs on the Jetson Nano).
"""
import collections
import itertools
import logging
import os
import secrets
import subprocess
import sys
import time

from tornado import gen, locks, netutil
from tornado.concurrent import Future
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.iostream import StreamClosedError
from tornado.tcpserver import TCPServer

# studio paths and settings
from studio import REPO_DIR, config
# Wire format
from studio.ipc import pack, read_message

# Server log
log = logging.getLogger("studio.runlink")
# Seconds to wait for the runner to connect and get ready (torch import on the Nano is slow)
START_TIMEOUT = 120.0
# Heartbeat interval in milliseconds
HEARTBEAT_MS = 100
# A page counts as present when its last heartbeat is at most this old (pages beat every 0.1 s)
PRESENCE_WINDOW = 0.3
# Maximum events kept for pages that (re)connect
MAX_EVENTS = 200


# Startup failure of a runner
class RunnerError(Exception):
    """The runner could not be started (message for the page)."""


# The newest frame received from the runner
class Frame(object):
    """Sequence number, metadata and JPEG bytes."""

    # Store the parts
    def __init__(self, seq, meta, jpeg):
        # Runner's sequence number of the display item
        self.seq = seq
        # Task metadata (prediction, motor values, ...)
        self.meta = meta
        # JPEG bytes
        self.jpeg = jpeg
        # Arrival time
        self.t = time.time()


# TCP server that hands the runner's connection to the link
class _AcceptServer(TCPServer):
    # Remember the link
    def __init__(self, link):
        # Plain TCP server
        TCPServer.__init__(self)
        # Link waiting for the connection
        self.link = link

    # A connection arrived
    def handle_stream(self, stream, address):
        # Only the first connection is used; later ones are closed
        if self.link.stream is not None:
            # Close the extra connection
            stream.close()
            # Done
            return
        # Hand it to the link
        self.link.stream = stream
        # Wake start()
        if not self.link._connected.done():
            # Connected
            self.link._connected.set_result(True)


# One running task
class RunnerLink(object):
    """See the module docstring. ``status``: starting, running, stopping, stopped, failed."""

    # Remember what to start; nothing runs until start()
    def __init__(self, task, params, settings, user=None, python=None, motor_guard=None):
        # Normalised TASK declaration
        self.task = task
        # Validated parameters (live changes update this dict)
        self.params = dict(params)
        # Server settings (camera source, role, ...)
        self.settings = dict(settings)
        # Who started it
        self.user = user
        # Python interpreter for the runner (same as the server by default)
        self.python = python or sys.executable
        # Function run when the runner died while possibly driving (None = default)
        self.motor_guard = motor_guard if motor_guard is not None else self._run_motor_guard
        # Life-cycle state
        self.status = "starting"
        # Error text when failed
        self.error = None
        # Start time
        self.started = time.time()
        # Runner process, its connection and the accept server
        self.proc, self.stream, self.server = None, None, None
        # Resolved when the runner connects
        self._connected = Future()
        # Information from the runner's 'ready' message
        self.info = {}
        # Newest frame and the condition pages wait on
        self.frame, self.frame_cond = None, locks.Condition()
        # Whether a frame request is outstanding, and when it was sent
        self._want_out, self._want_t = False, 0.0
        # Outstanding commands: id -> Future
        self._pending = {}
        # Command ids
        self._ids = itertools.count(1)
        # Recent events with increasing ids: (id, event)
        self.events = collections.deque(maxlen=MAX_EVENTS)
        # Event ids
        self._event_ids = itertools.count(1)
        # Newest telemetry message
        self.telemetry = {}
        # Functions called with every event / telemetry message (control sockets)
        self.listeners = set()
        # Page presence: client id -> time of its last heartbeat
        self.presence = {}
        # Periodic callbacks (heartbeat, process watch)
        self._timers = []
        # Runner log file
        self.log_path = config.home_file("logs", "runner-%s.log" % task["id"])

    # ------------------------------------------------------------- lifecycle
    # Start the runner and wait until it is ready
    async def start(self, timeout=START_TIMEOUT):
        """Raises ``RunnerError`` when the runner fails to start."""
        # Listen on a random local port
        sockets = netutil.bind_sockets(0, "127.0.0.1")
        # Accept server for the runner's connection
        self.server = _AcceptServer(self)
        # Serve on those sockets
        self.server.add_sockets(sockets)
        # The chosen port
        port = sockets[0].getsockname()[1]
        # Secret the runner must send back
        token = secrets.token_hex(16)
        # Rotate the runner log: keep the previous run as .1
        if os.path.exists(self.log_path):
            # Replace the older copy
            os.replace(self.log_path, self.log_path + ".1")
        # Runner output goes to its log file
        logf = open(self.log_path, "w", encoding="utf-8")
        # Unbuffered output so the log is current
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        # Start the runner from the repository root (so `studio` is importable)
        self.proc = subprocess.Popen([self.python, "-m", "studio.runner", "--port", str(port),
                                      "--token", token, "--task", self.task["id"]],
                                     cwd=REPO_DIR, stdout=logf, stderr=subprocess.STDOUT, env=env)
        # The child has its own copy of the file
        logf.close()
        # Everything below may fail; clean up on failure
        try:
            # Wait for the connection (or the runner dying)
            await self._wait(self._connected, timeout)
            # First message: hello with the token
            hello, _ = await self._read_with_timeout(timeout)
            # Wrong token or message: refuse
            if hello.get("type") != "hello" or hello.get("token") != token:
                # Not our runner
                raise RunnerError("runner sent a bad hello")
            # Send the task, parameters and settings
            self._send({"type": "start", "task": self.task, "params": self.params,
                        "settings": self.settings})
            # Wait for ready or error
            msg, _ = await self._read_with_timeout(timeout)
            # Setup failed in the runner (camera busy, import error, ...)
            if msg.get("type") == "error":
                # Report its message
                raise RunnerError("%s\n%s" % (msg.get("error"), msg.get("traceback", "")))
            # Anything but ready is a protocol error
            if msg.get("type") != "ready":
                # Report it
                raise RunnerError("unexpected message %r" % msg.get("type"))
        # Clean up and report
        except Exception as e:
            # Failed state with the reason
            self.status, self.error = "failed", str(e) if isinstance(e, RunnerError) else repr(e)
            # Stop the process
            await self._terminate()
            # Report to the caller
            raise RunnerError(self.error)
        # Runner information (robot, sensors, pid)
        self.info = msg.get("info") or {}
        # Running
        self.status = "running"
        # Read messages in the background
        IOLoop.current().spawn_callback(self._reader)
        # Heartbeats while a page is present
        self._timers.append(PeriodicCallback(self._heartbeat, HEARTBEAT_MS))
        # Watch for the process ending
        self._timers.append(PeriodicCallback(self._watch, 250))
        # Start both timers
        for t in self._timers:
            # Run it
            t.start()
        # Tell the pages
        self._event({"kind": "runner_ready", "info": self.info})

    # Stop the runner (orderly, then forcefully)
    async def stop(self, timeout=5.0):
        """Shutdown message, wait, terminate, kill."""
        # Already finished
        if self.status in ("stopped", "failed"):
            # Nothing to do
            return
        # Expected exit from now on (no motor guard)
        self.status = "stopping"
        # Stop the motors first (not an e-stop: tasks treat those as events / labels), then ask it to end
        self._send({"type": "halt"})
        # Orderly shutdown
        self._send({"type": "shutdown"})
        # Wait for it to exit
        deadline = time.monotonic() + timeout
        # Poll the process
        while self.proc.poll() is None and time.monotonic() < deadline:
            # Short wait
            await gen.sleep(0.05)
        # Force it if needed
        await self._terminate()
        # Stopped
        self.status = "stopped"
        # Tell the pages
        self._event({"kind": "runner_stopped", "returncode": self.proc.returncode})

    # Terminate / kill the process and close everything
    async def _terminate(self):
        # Timers off
        for t in self._timers:
            # Stop it
            t.stop()
        # Process still running: terminate (the runner stops the motors on SIGTERM)
        if self.proc is not None and self.proc.poll() is None:
            # Polite request
            self.proc.terminate()
            # Up to 3 s
            deadline = time.monotonic() + 3.0
            # Poll the process
            while self.proc.poll() is None and time.monotonic() < deadline:
                # Short wait
                await gen.sleep(0.05)
            # Still alive: kill
            if self.proc.poll() is None:
                # Forceful end
                self.proc.kill()
                # Reap it
                self.proc.wait()
        # Close the connection
        if self.stream is not None:
            # Close it
            self.stream.close()
        # Stop listening
        if self.server is not None:
            # Close the server sockets
            self.server.stop()
        # Fail outstanding commands
        for fut in self._pending.values():
            # Not resolved yet
            if not fut.done():
                # Reply with an error
                fut.set_result({"ok": False, "error": "runner stopped"})
        # Forget them
        self._pending.clear()
        # Wake pages waiting for frames
        self.frame_cond.notify_all()

    # Wait for a future, failing when the runner dies or time runs out
    async def _wait(self, fut, timeout):
        # Deadline
        deadline = time.monotonic() + timeout
        # Until the future resolves
        while not fut.done():
            # The runner exited
            if self.proc.poll() is not None:
                # Last lines of its log explain why
                raise RunnerError("runner exited with code %s:\n%s" % (self.proc.returncode, self.log_tail()))
            # Too slow
            if time.monotonic() > deadline:
                # Give up
                raise RunnerError("runner did not start within %.0f s" % timeout)
            # Check again soon
            await gen.sleep(0.1)
        # The result
        return fut.result()

    # Read one message with the startup deadline
    async def _read_with_timeout(self, timeout):
        # Read in the background
        fut = gen.convert_yielded(read_message(self.stream))
        # Wait for it (or the runner dying / the deadline)
        return await self._wait(fut, timeout)

    # Last lines of the runner log
    def log_tail(self, lines=30):
        """The runner log's last ``lines`` lines."""
        # The log may not exist yet
        try:
            # Read it
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                # Last lines
                return "".join(f.readlines()[-lines:])
        # No log
        except OSError:
            # Nothing to show
            return ""

    # ----------------------------------------------------------- messaging
    # Send one message (fire and forget)
    def _send(self, header, payload=b""):
        # Not connected
        if self.stream is None or self.stream.closed():
            # Dropped
            return False
        # Queue it on the stream
        try:
            # Write without waiting
            fut = self.stream.write(pack(header, payload))
            # Retrieve a later error so it is not logged as unhandled
            fut.add_done_callback(lambda f: f.exception())
            # Sent
            return True
        # Closed meanwhile
        except StreamClosedError:
            # Dropped
            return False

    # Reader loop
    async def _reader(self):
        # Until the connection closes
        while True:
            # Next message
            try:
                # Header and payload
                msg, payload = await read_message(self.stream)
            # Runner gone or garbage
            except (StreamClosedError, ValueError):
                # Stop reading (the process watcher handles the rest)
                return
            # Message type
            kind = msg.get("type")
            # Reply to a command
            if kind == "reply":
                # Matching future
                fut = self._pending.pop(msg.get("id"), None)
                # Resolve it
                if fut is not None and not fut.done():
                    # The reply message
                    fut.set_result(msg)
            # New frame
            elif kind == "frame":
                # Replace the newest frame
                self.frame = Frame(msg.get("seq", 0), msg.get("meta") or {}, payload)
                # The request is answered
                self._want_out = False
                # Wake waiting pages
                self.frame_cond.notify_all()
            # Event from the task / safety logic
            elif kind == "event":
                # Store and broadcast
                self._event(msg.get("event") or {})
            # Telemetry
            elif kind == "telemetry":
                # Keep the newest
                self.telemetry = msg
                # Broadcast
                self._broadcast({"type": "telemetry", "data": msg.get("data"), "link": msg.get("link"),
                                 "moving": msg.get("moving")})

    # Store an event and tell the listeners
    def _event(self, ev):
        # Next id
        eid = next(self._event_ids)
        # Keep it
        self.events.append((eid, ev))
        # Broadcast
        self._broadcast({"type": "event", "id": eid, "event": ev})

    # Call every listener with a message
    def _broadcast(self, msg):
        # Copy: listeners may unregister while iterating
        for fn in list(self.listeners):
            # A broken listener must not stop the others
            try:
                # Deliver
                fn(msg)
            # Log and continue
            except Exception:
                # Server log
                log.exception("listener failed")

    # Run a task command in the runner
    async def command(self, name, args=None, timeout=60.0):
        """Return the runner's reply dict (``ok``, ``result`` / ``error``)."""
        # Not running
        if self.status != "running":
            # Error reply
            return {"ok": False, "error": "task is not running"}
        # New id
        cid = next(self._ids)
        # Future resolved by the reader
        fut = Future()
        # Register it
        self._pending[cid] = fut
        # Send the command
        self._send({"type": "command", "id": cid, "name": name, "args": args or {}})
        # Wait for the reply
        try:
            # Reply or timeout
            return await gen.with_timeout(time.time() + timeout, fut)
        # No reply in time
        except gen.TimeoutError:
            # Forget the future
            self._pending.pop(cid, None)
            # Error reply
            return {"ok": False, "error": "no reply within %.0f s" % timeout}

    # Live parameter change (already validated)
    def set_params(self, values):
        """Forward ``values`` to the runner and remember them."""
        # Remember
        self.params.update(values)
        # Forward
        self._send({"type": "params", "values": values})

    # Emergency stop
    def estop(self):
        """Stop the motors now."""
        # Forward
        return self._send({"type": "estop"})

    # ------------------------------------------------------------- frames
    # Wait for a frame newer than `after`
    async def next_frame(self, after, quality=70, timeout=2.0):
        """The newest ``Frame`` with ``seq > after``, or None on timeout / not running."""
        # Deadline
        deadline = time.monotonic() + timeout
        # Until a newer frame exists
        while self.frame is None or self.frame.seq <= after:
            # Not running any more
            if self.status != "running":
                # Nothing
                return None
            # Ask the runner unless a request is outstanding (re-ask after 2 s in case it got lost)
            if not self._want_out or time.monotonic() - self._want_t > 2.0:
                # Newest sequence already here
                have = self.frame.seq if self.frame is not None else 0
                # Request something newer than that
                self._send({"type": "want_frame", "after": max(after, have), "quality": int(quality)})
                # Outstanding now
                self._want_out, self._want_t = True, time.monotonic()
            # Time left
            left = deadline - time.monotonic()
            # Out of time
            if left <= 0:
                # Nothing new
                return None
            # Wait for the reader to deliver a frame
            await self.frame_cond.wait(timeout=time.time() + min(left, 0.5))
        # The newest frame
        return self.frame

    # ----------------------------------------------------------- presence
    # A page answered (control socket heartbeat)
    def touch(self, client_id):
        """Record a heartbeat from a page."""
        # Time of the heartbeat
        self.presence[client_id] = time.monotonic()

    # A page closed its control socket
    def leave(self, client_id):
        """Forget the page; the runner stops after ``link_timeout`` if no page is left.

        No immediate stop: a flaky Wi-Fi also closes sockets, and the user chose
        to tolerate ``link_timeout`` (default 5 s) of silence.
        """
        # Forget it
        self.presence.pop(client_id, None)

    # Whether at least one page answered recently
    def present(self):
        """True when a page heartbeat arrived within ``PRESENCE_WINDOW``.

        Pages beat every 0.1 s; only the runner applies ``link_timeout``
        (otherwise the two timeouts would add up).
        """
        # Now
        now = time.monotonic()
        # Any recent heartbeat
        return any(now - t <= PRESENCE_WINDOW for t in self.presence.values())

    # Heartbeat timer
    def _heartbeat(self):
        # Only while a page is present: otherwise the runner's watchdog stops the motors
        if self.present():
            # Forward
            self._send({"type": "heartbeat"})

    # Process watch timer
    def _watch(self):
        # Still running
        if self.proc.poll() is None:
            # Nothing to do
            return
        # Expected exit
        if self.status in ("stopping", "stopped"):
            # Nothing to do
            return
        # Unexpected exit: failed
        self.status = "failed"
        # Reason with the end of the log
        self.error = "runner exited with code %s\n%s" % (self.proc.returncode, self.log_tail())
        # Server log
        log.error("runner %s died: %s", self.task["id"], self.error)
        # Timers off
        for t in self._timers:
            # Stop it
            t.stop()
        # The motors may still be running: stop them from a fresh process
        guard = self.motor_guard()
        # Tell the pages
        self._event({"kind": "runner_died", "returncode": self.proc.returncode, "motor_guard": guard,
                     "log": self.log_tail(15)})
        # Wake pages waiting for frames
        self.frame_cond.notify_all()

    # Default motor guard: only on the car (a PC has no motors)
    def _run_motor_guard(self):
        """Run ``python3 -m studio.motor_guard`` on the car; returns its output."""
        # Only the robot has motors
        if self.settings.get("role") != "car":
            # Nothing to stop
            return "not on the car"
        # Run it in a fresh process (short timeout)
        try:
            # Stop the motors
            out = subprocess.run([self.python, "-m", "studio.motor_guard"], cwd=REPO_DIR, timeout=20,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # Its message
            return out.stdout.decode("utf-8", "replace").strip()
        # Could not run it
        except Exception as e:
            # Reason
            return "motor guard failed: %r" % e

    # Summary for the pages
    def describe(self):
        """JSON-able state."""
        # Everything the pages show
        return {"task": self.task["id"], "title": self.task["title"], "status": self.status,
                "error": self.error, "started": self.started, "user": self.user, "info": self.info,
                "pid": self.proc.pid if self.proc else None, "present": self.present(),
                "params": self.params}
