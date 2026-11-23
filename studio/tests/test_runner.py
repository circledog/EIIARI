"""Real runner processes (dummy camera, fake motors): stream backpressure and safety stops.

TASKS M1-6: no frame is sent before the previous one was acknowledged.
TASKS M1-7: link lost / e-stop / web process gone / runner killed all stop the motors.
TASKS M1-8: a second task is refused while one holds the devices.
"""
import json
import os
import signal
import struct
import time
import unittest

from tornado import gen
from tornado.ioloop import IOLoop
from tornado.testing import gen_test

# Test helpers (temporary STUDIO_HOME, login, WebSockets)
from _base import StudioTestCase
# Web-side runner link
from studio.runlink import RunnerLink

# Generous timeout: the runner imports torch / OpenCV
TIMEOUT = 120


# Read with timeouts without losing messages
class Reader(object):
    """A timed-out ``read_message()`` future stays registered in tornado and would
    swallow the next message; keep it and reuse it on the next read."""

    # Wrap a client connection
    def __init__(self, conn):
        # The connection
        self.conn = conn
        # Read future that timed out last time
        self.pending = None

    # Next message or gen.TimeoutError
    async def read(self, timeout):
        """Next message (None when closed); raises ``gen.TimeoutError``."""
        # Reuse the pending future, or start a new read
        fut = self.pending or self.conn.read_message()
        # Remember it in case this wait times out
        self.pending = fut
        # Wait for it
        msg = await gen.with_timeout(time.time() + timeout, fut)
        # Consumed
        self.pending = None
        # The message
        return msg


# Decode a binary stream message
def parse_frame(data):
    """Return ``(header, jpeg)``."""
    # Header length
    hlen = struct.unpack(">I", data[:4])[0]
    # Header and JPEG
    return json.loads(data[4:4 + hlen].decode()), data[4 + hlen:]


# Runner integration tests
class RunnerTest(StudioTestCase):
    # Start drive_check with given values and return the link
    async def start_task(self, **values):
        # Log in
        await self.login_async()
        # Start through the API (runs the real runner)
        status, body = await self._api_async("POST", "/api/tasks/drive_check/start", {"values": values})
        # Started
        self.assertEqual(status, 200, body)
        # Its link
        return self.studio.manager.runner

    # Async variant of api() for use inside gen_test
    async def _api_async(self, method, url, body=None):
        # Request with cookies and the XSRF header
        r = await self.http_client.fetch(self.get_url(url), method=method, raise_error=False,
                                         body=json.dumps(body or {}) if method == "POST" else None,
                                         headers={"Cookie": self.cookie_header(),
                                                  "X-XSRFToken": self.cookies.get("_xsrf", ""),
                                                  "Content-Type": "application/json"})
        # Status and JSON
        return r.code, json.loads(r.body.decode()) if r.body else {}

    # Read control messages until one matches
    async def read_until(self, reader, pred, timeout):
        """First message (dict) with ``pred(msg)`` true, or None on timeout (``reader``: Reader)."""
        # Deadline
        deadline = time.monotonic() + timeout
        # Until the deadline
        while time.monotonic() < deadline:
            # Next message with the remaining time
            try:
                # Wait for it
                raw = await reader.read(max(0.01, deadline - time.monotonic()))
            # No message in time
            except gen.TimeoutError:
                # Give up
                return None
            # Socket closed
            if raw is None:
                # Give up
                return None
            # Parse
            msg = json.loads(raw)
            # Match
            if pred(msg):
                # Found
                return msg
        # Timeout
        return None

    # Heartbeat loop for a control socket (until the flag list is emptied)
    async def heartbeat(self, conn, alive):
        # While the test wants heartbeats
        while alive:
            # Heartbeat
            conn.write_message(json.dumps({"type": "hb"}))
            # Every 0.1 s
            await gen.sleep(0.1)

    # One frame at a time: no second frame before the ack
    @gen_test(timeout=TIMEOUT)
    async def test_stream_backpressure(self):
        # Running task
        link = await self.start_task()
        # Stream socket
        conn = await self.ws("/ws/stream/drive_check")
        # Reader that never loses a message
        rd = Reader(conn)
        # First frame
        header, jpeg = parse_frame(await rd.read(10))
        # A JPEG
        self.assertEqual(jpeg[:2], b"\xff\xd8")
        # Sequence number
        self.assertGreater(header["seq"], 0)
        # Without an ack nothing more arrives (the dummy camera makes ~100 frames/s meanwhile)
        with self.assertRaises(gen.TimeoutError):
            # Wait 1.5 s
            await rd.read(1.5)
        # Acknowledge
        conn.write_message(json.dumps({"type": "ack", "seq": header["seq"]}))
        # The next (newest) frame comes now
        header2, _ = parse_frame(await rd.read(3))
        # Newer and skipping the frames that were never sent
        self.assertGreater(header2["seq"], header["seq"] + 1)
        # Close
        conn.close()
        # Stop the task
        await link.stop()

    # Motion needs a live page; link loss and e-stop stop the motors
    @gen_test(timeout=TIMEOUT)
    async def test_link_loss_and_estop(self):
        # Short link timeout, long hold so the dead-man is not what stops it
        link = await self.start_task(link_timeout=0.3, hold_time=1.0)
        # Control socket without heartbeats yet
        conn = await self.ws("/ws/control/drive_check")
        # Reader that never loses a message
        rd = Reader(conn)
        # Let it attach to the link
        await gen.sleep(0.7)
        # Drive without any heartbeat
        conn.write_message(json.dumps({"type": "command", "id": 1, "name": "drive", "args": {"left": 1, "right": 1}}))
        # Refused: no live page
        reply = await self.read_until(rd, lambda m: m.get("type") == "reply" and m.get("id") == 1, 5)
        # link_down
        self.assertEqual(reply["error"], "link_down")
        # Start heartbeats
        alive = [1]
        # Background loop
        IOLoop.current().spawn_callback(self.heartbeat, conn, alive)
        # Let the runner see them
        await gen.sleep(0.4)
        # Drive now
        conn.write_message(json.dumps({"type": "command", "id": 2, "name": "drive", "args": {"left": 1, "right": 1}}))
        # Accepted
        reply = await self.read_until(rd, lambda m: m.get("type") == "reply" and m.get("id") == 2, 5)
        # OK with the scaled value
        self.assertTrue(reply["ok"], reply)
        # Telemetry shows the robot moving
        self.assertIsNotNone(await self.read_until(rd, lambda m: m.get("type") == "telemetry" and m["moving"], 3))
        # Fresh drive command so the 1.0 s dead-man starts now
        conn.write_message(json.dumps({"type": "command", "id": 9, "name": "drive", "args": {"left": 1, "right": 1}}))
        # Wait for its reply
        await self.read_until(rd, lambda m: m.get("type") == "reply" and m.get("id") == 9, 5)
        # Stop the heartbeats (page silent / Wi-Fi gone)
        alive.pop()
        # Time of the last heartbeat
        t_last = time.monotonic()
        # The runner reports the safety stop
        ev = await self.read_until(rd, lambda m: m.get("type") == "event" and
                                   m["event"].get("kind") == "safety_stop", 3)
        # Seconds from the last heartbeat to the stop
        took = time.monotonic() - t_last
        # Print for the HANDOFF record
        print("\n[safety] link lost -> stop after %.2f s (link_timeout 0.3, hold_time 1.0)" % took)
        # Because of the link
        self.assertEqual(ev["event"]["code"], "link_lost")
        # Before the 1.0 s dead-man would have stopped it
        self.assertLess(took, 0.95)
        # Heartbeats again
        alive.append(1)
        # Background loop
        IOLoop.current().spawn_callback(self.heartbeat, conn, alive)
        # Let the runner see them
        await gen.sleep(0.4)
        # Drive again
        conn.write_message(json.dumps({"type": "command", "id": 3, "name": "drive", "args": {"left": 1, "right": 1}}))
        # Accepted
        await self.read_until(rd, lambda m: m.get("type") == "reply" and m.get("id") == 3, 5)
        # E-stop
        t0 = time.monotonic()
        # Through the control socket
        conn.write_message(json.dumps({"type": "estop"}))
        # Reported
        ev = await self.read_until(rd, lambda m: m.get("type") == "event" and
                                   m["event"].get("code") == "estop", 3)
        # Seconds to the stop
        took = time.monotonic() - t0
        # Print for the HANDOFF record
        print("[safety] e-stop -> stop reported after %.2f s" % took)
        # Fast
        self.assertLess(took, 0.5)
        # Telemetry: not moving
        self.assertIsNotNone(await self.read_until(rd, lambda m: m.get("type") == "telemetry" and not m["moving"], 2))
        # End the heartbeats
        alive.pop()
        # Close
        conn.close()
        # Stop the task
        await link.stop()

    # The web process disappears: the runner stops and exits by itself
    @gen_test(timeout=TIMEOUT)
    async def test_web_process_gone(self):
        # Running task
        link = await self.start_task()
        # Simulate the web process dying: its end of the connection closes
        link.stream.close()
        # Process of the runner
        t0 = time.monotonic()
        # Wait for it to exit
        while link.proc.poll() is None and time.monotonic() - t0 < 5:
            # Poll
            await gen.sleep(0.05)
        # Print for the HANDOFF record
        print("\n[safety] web connection closed -> runner exited after %.2f s with code %s"
              % (time.monotonic() - t0, link.proc.returncode))
        # Exited with the "web gone" code
        self.assertEqual(link.proc.returncode, 2)
        # Mark as expected so tearDown does not complain
        link.status = "stopped"

    # The runner dies: the web process runs the motor guard
    @gen_test(timeout=TIMEOUT)
    async def test_runner_killed_runs_motor_guard(self):
        # Calls of the fake motor guard
        calls = []
        # Factory that injects the fake guard
        self.studio.manager.link_factory = lambda *a, **k: RunnerLink(
            *a, motor_guard=lambda: calls.append(time.monotonic()) or "fake guard", **k)
        # Running task
        link = await self.start_task()
        # Kill it hard (no chance to stop the motors itself)
        os.kill(link.proc.pid, signal.SIGKILL)
        # Time of the kill
        t0 = time.monotonic()
        # Wait for the watcher
        while not calls and time.monotonic() - t0 < 5:
            # Poll
            await gen.sleep(0.05)
        # Guard was called
        self.assertEqual(len(calls), 1)
        # Print for the HANDOFF record
        print("\n[safety] runner SIGKILL -> motor guard after %.2f s" % (calls[0] - t0))
        # Failed state
        self.assertEqual(link.status, "failed")
        # Wait for the cleanup after the guard
        while link.guarding and time.monotonic() - t0 < 5:
            # Poll
            await gen.sleep(0.05)
        # Devices are free again: the task can be started anew
        self.assertTrue(self.studio.manager.device_free())
        # The pages were told, with the guard's output
        died = [ev for _, ev in link.events if ev.get("kind") == "runner_died"]
        # Exactly one report
        self.assertEqual(len(died), 1)
        # Guard output included
        self.assertEqual(died[0]["motor_guard"], "fake guard")

    # A hung runner has to be killed on stop: the motor guard runs as well
    @gen_test(timeout=TIMEOUT)
    async def test_hung_runner_killed_runs_motor_guard(self):
        # Calls of the fake motor guard
        calls = []
        # Factory that injects the fake guard
        self.studio.manager.link_factory = lambda *a, **k: RunnerLink(
            *a, motor_guard=lambda: calls.append(1) or "fake guard", **k)
        # Running task
        link = await self.start_task()
        # Freeze it: it cannot answer shutdown or SIGTERM any more
        os.kill(link.proc.pid, signal.SIGSTOP)
        # Orderly stop gives up and kills it
        await link.stop(timeout=0.3)
        # Killed
        self.assertEqual(link.proc.returncode, -signal.SIGKILL)
        # The guard ran once
        self.assertEqual(calls, [1])
        # Reported to the pages
        self.assertTrue(any(ev.get("kind") == "runner_killed" for _, ev in link.events))
        # Devices free
        self.assertTrue(self.studio.manager.device_free())

    # A second task is refused while one runs; stop frees the devices
    @gen_test(timeout=TIMEOUT)
    async def test_device_lock(self):
        # Running task
        await self.start_task()
        # Second start
        status, body = await self._api_async("POST", "/api/tasks/drive_check/start", {"values": {}})
        # Refused with the reason
        self.assertEqual(status, 409)
        # Names the task holding the devices
        self.assertIn("Camera and motor check", body["error"])
        # Stop it
        status, body = await self._api_async("POST", "/api/tasks/drive_check/stop")
        # Stopped
        self.assertTrue(body["stopped"])
        # Free again
        self.assertTrue(self.studio.manager.device_free())


# Runner.send with values JSON cannot encode (no process needed)
class RunnerSendTest(unittest.TestCase):

    # A bad telemetry value is replaced, reported, and the thread lives on
    def test_unencodable_value_is_replaced(self):
        # Real encoder (raises like the socket connection)
        from studio.ipc import pack
        # Class under test
        from studio.runner import Runner

        # Connection stand-in that encodes like the real one
        class FakeConn(object):
            # Sent headers
            sent = []

            # Encode and record
            def send(self, header, payload=b""):
                # Raises TypeError like the real connection
                pack(header, payload)
                # Record it
                self.sent.append(header)

        # Runner around it
        runner = Runner(FakeConn())
        # An object json cannot encode
        ok = runner.send({"type": "telemetry", "data": {"odd": object(), "n": 1}})
        # Still sent
        self.assertTrue(ok)
        # With a readable stand-in, other values unchanged
        self.assertIn("object", runner.conn.sent[-1]["data"]["odd"])
        # The plain value is kept
        self.assertEqual(runner.conn.sent[-1]["data"]["n"], 1)
        # The problem is reported to the page
        self.assertEqual(runner.events[-1]["kind"], "error")
