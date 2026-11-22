"""WebSockets of a task page.

``/ws/stream/<task>`` (TASKS M1-6): the newest frame only, one at a time.
A frame is sent only after the browser has *displayed* the previous one and
answered ``{"type": "ack"}``, so nothing ever queues on a slow link: the
picture gets fewer FPS, but never older. The JPEG quality follows the
measured round trip (slow -> lower quality, fast -> back up).

Binary message: 4-byte big-endian header length, JSON header
(``seq``, ``meta``, ``t``, ``quality``), JPEG bytes.

``/ws/control/<task>``: small JSON messages.

* browser -> server: ``hb`` (every 0.1 s; keeps the motors allowed),
  ``estop``, ``command`` (``id``, ``name``, ``args``), ``params`` (``values``)
* server -> browser: ``state`` (2 per second), ``telemetry``, ``event``,
  ``reply``. Telemetry / state are skipped while the previous message is still
  being sent (no backlog); events and replies are always sent.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import json
import struct
import time
import uuid

from tornado import gen, locks
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.websocket import WebSocketHandler, WebSocketClosedError

# Login cookie
from studio.auth import CurrentUserMixin
# Parameter validation
from studio import params as params_mod

# JPEG quality limits and start value
Q_MIN, Q_MAX, Q_START = 30, 85, 70
# Round trip above which quality goes down / below which it goes up (seconds)
RTT_SLOW, RTT_FAST = 0.5, 0.15
# Seconds to wait for an ack before giving up on the connection
ACK_TIMEOUT = 15.0


# Shared parts of both sockets
class _TaskSocket(CurrentUserMixin, WebSocketHandler):
    # Store the shared studio object
    def initialize(self, studio):
        # Settings, tasks, manager
        self.studio = studio

    # Login is checked in open(); refuse silently before the upgrade otherwise
    def check_login(self, task_id):
        """True when logged in and the task exists; closes the socket otherwise."""
        # Not logged in
        if not self.current_user:
            # Close with an application code
            self.close(4001, "login required")
            # Refused
            return False
        # Unknown task
        if task_id not in self.studio.task_map:
            # Close with an application code
            self.close(4004, "unknown task")
            # Refused
            return False
        # Accepted
        return True

    # Send JSON, ignoring a closed socket
    def send_json(self, msg):
        """Returns the write future, or None when closed."""
        # Closed meanwhile
        try:
            # Send it
            return self.write_message(json.dumps(msg))
        # Closed
        except WebSocketClosedError:
            # Nothing sent
            return None


# Frame stream with ack-based backpressure
class StreamSocket(_TaskSocket):
    """See the module docstring."""

    # Connection opened
    def open(self, task_id):
        # Login and task check
        if not self.check_login(task_id):
            # Refused
            return
        # Task of this page
        self.task_id = task_id
        # Set by on_close
        self.closed = False
        # Set when the browser acknowledged the last frame
        self.ack = locks.Event()
        # Current JPEG quality
        self.quality = Q_START
        # Link whose frames are being sent (sequence numbers restart with a new runner)
        self.link, self.seq = None, 0
        # Send loop
        IOLoop.current().spawn_callback(self._loop)

    # Message from the browser
    def on_message(self, message):
        # Ignore anything that is not JSON
        try:
            # Parse it
            msg = json.loads(message)
        # Not JSON
        except ValueError:
            # Ignore
            return
        # Frame displayed
        if msg.get("type") == "ack":
            # Release the send loop
            self.ack.set()

    # Connection closed
    def on_close(self):
        # Stop the loop
        self.closed = True
        # Release a waiting loop
        if hasattr(self, "ack"):
            # Wake it
            self.ack.set()

    # Adjust the quality after a round trip
    def _adapt(self, rtt):
        # Slow: lower the quality
        if rtt > RTT_SLOW:
            # Step down
            self.quality = max(Q_MIN, self.quality - 10)
        # Fast: raise it again slowly
        elif rtt < RTT_FAST:
            # Step up
            self.quality = min(Q_MAX, self.quality + 5)

    # Send loop: one frame, wait for its ack, repeat
    async def _loop(self):
        # Until the browser leaves
        while not self.closed:
            # Running link of this task
            link = self.studio.manager.link_for(self.task_id)
            # Not running (yet)
            if link is None or link.status != "running":
                # Check again soon
                await gen.sleep(0.3)
                # Next round
                continue
            # A new runner restarts its sequence numbers
            if link is not self.link:
                # Start from zero
                self.link, self.seq = link, 0
            # Newest frame newer than the last one sent
            frame = await link.next_frame(self.seq, self.quality, timeout=1.0)
            # Nothing new (or the page left meanwhile)
            if frame is None or self.closed:
                # Next round
                continue
            # Header with the frame's metadata
            header = json.dumps({"seq": frame.seq, "meta": frame.meta, "t": frame.t,
                                 "quality": self.quality}).encode("ascii")
            # Wait for an ack of this frame
            self.ack.clear()
            # Round-trip start
            t0 = time.monotonic()
            # Send it
            try:
                # Length + header + JPEG
                await self.write_message(struct.pack(">I", len(header)) + header + frame.jpeg, binary=True)
            # Browser gone
            except WebSocketClosedError:
                # End the loop
                return
            # Remember what was sent
            self.seq = frame.seq
            # Wait until the browser displayed it
            try:
                # Ack or timeout
                await self.ack.wait(timeout=time.time() + ACK_TIMEOUT)
            # No ack for a long time: the browser is stuck; close
            except gen.TimeoutError:
                # Close the socket (the page reconnects)
                self.close(4008, "no ack")
                # End the loop
                return
            # Adapt the quality to the round trip
            self._adapt(time.monotonic() - t0)


# Control socket: heartbeat, e-stop, commands, parameters, telemetry
class ControlSocket(_TaskSocket):
    """See the module docstring."""

    # Connection opened
    def open(self, task_id):
        # Login and task check
        if not self.check_login(task_id):
            # Refused
            return
        # Task of this page and its declaration
        self.task_id, self.task = task_id, self.studio.task_map[task_id]
        # Unique id of this page for the presence table
        self.cid = uuid.uuid4().hex
        # Link we listen to
        self.link = None
        # Write future of the last state / telemetry message (skip while busy)
        self._busy = None
        # State pushes and link (re)attachment twice per second
        self.timer = PeriodicCallback(self._tick, 500)
        # Start it
        self.timer.start()
        # First state right away
        self._tick()

    # Whether the previous droppable message is still being sent
    def _writing(self):
        # A write future that has not finished yet
        return self._busy is not None and not self._busy.done()

    # Message from the runner link (event / telemetry)
    def _on_link(self, msg):
        # Telemetry may be dropped when the browser is slow
        if msg.get("type") == "telemetry":
            # Skip while busy
            if self._writing():
                # Dropped
                return
            # Send and remember the future
            self._busy = self.send_json(msg)
        # Events are always sent
        else:
            # Send
            self.send_json(msg)

    # Periodic: attach to the current link and push the state
    def _tick(self):
        # Running link of this task (None when not running)
        link = self.studio.manager.link_for(self.task_id)
        # The link changed (task started / stopped)
        if link is not self.link:
            # Stop listening to the old one
            if self.link is not None:
                # Unregister
                self.link.listeners.discard(self._on_link)
            # Listen to the new one
            if link is not None:
                # Register
                link.listeners.add(self._on_link)
                # Send its recent events so a reloaded page sees what happened
                for eid, ev in list(link.events):
                    # Same format as live events
                    self.send_json({"type": "event", "id": eid, "event": ev, "replay": True})
            # Remember it
            self.link = link
        # Skip while busy
        if self._writing():
            # Dropped
            return
        # Runner shown even when finished (error text)
        r = self.studio.manager.runner
        # State of this task
        state = r.describe() if r is not None and r.task["id"] == self.task_id else None
        # Another task holds the devices
        other = r.describe() if r is not None and r.task["id"] != self.task_id and \
            r.status in ("starting", "running", "stopping") else None
        # Send it
        self._busy = self.send_json({"type": "state", "runner": state, "other": other, "t": time.time()})

    # Message from the browser
    def on_message(self, message):
        # Ignore anything that is not JSON
        try:
            # Parse it
            msg = json.loads(message)
        # Not JSON
        except ValueError:
            # Ignore
            return
        # Message type
        kind = msg.get("type")
        # Heartbeat: this page is present
        if kind == "hb":
            # Record it on the running link
            if self.link is not None:
                # Present
                self.link.touch(self.cid)
        # Emergency stop: whatever task runs
        elif kind == "estop":
            # Running link of any task
            r = self.studio.manager.runner
            # Forward
            if r is not None:
                # Stop now
                r.estop()
        # Task command
        elif kind == "command":
            # Run it without blocking this handler
            IOLoop.current().spawn_callback(self._command, msg)
        # Live parameter change
        elif kind == "params":
            # Validate and forward
            self._params(msg)

    # Run a command and send the reply
    async def _command(self, msg):
        # Not running
        if self.link is None:
            # Error reply
            reply = {"ok": False, "error": "task is not running"}
        # Running
        else:
            # Ask the runner
            reply = await self.link.command(msg.get("name"), msg.get("args") or {})
        # Send the reply with the browser's id
        self.send_json({"type": "reply", "id": msg.get("id"), "ok": reply.get("ok"),
                        "result": reply.get("result"), "error": reply.get("error")})

    # Validate live parameters and forward them
    def _params(self, msg):
        # Only live parameters, range-checked again here
        clean, errors = params_mod.validate(self.task["params"], msg.get("values") or {}, partial=True,
                                            live_only=True)
        # Forward the valid ones to the runner
        if clean and self.link is not None:
            # Apply
            self.link.set_params(clean)
        # Tell the browser what was accepted / refused
        self.send_json({"type": "params_result", "id": msg.get("id"), "applied": clean, "errors": errors,
                        "running": self.link is not None})

    # Connection closed
    def on_close(self):
        # Opened successfully?
        if not hasattr(self, "timer"):
            # Nothing to clean up
            return
        # Stop the timer
        self.timer.stop()
        # Stop listening and leave the presence table
        if self.link is not None:
            # Unregister
            self.link.listeners.discard(self._on_link)
            # This page no longer keeps the motors allowed
            self.link.leave(self.cid)
