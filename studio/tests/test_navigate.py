"""Navigate task (TASKS M3): free space, tracker, planner, label store, runner."""
import json
import os
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np

from tornado import gen
from tornado.ioloop import IOLoop
from tornado.testing import gen_test

# Test helpers (temporary STUDIO_HOME, login, WebSockets)
from _base import StudioTestCase
# Message reader that never loses a message
from test_runner import Reader
# Modules under test
from studio.tasks.navigate.freespace import FreeSpace
# Tracker
from studio.tasks.navigate.tracker import Tracker
# Planner
from studio.tasks.navigate.planner import Planner
# Label store
from studio.tasks.navigate.dataset import LabelStore
# project4 on the import path
from studio import PROJECT4_DIR

# project4 importable (data_loader for the format check)
if PROJECT4_DIR not in sys.path:
    # Put it first
    sys.path.insert(0, PROJECT4_DIR)


# Synthetic floor picture, optionally with a box and tape lines
def scene(box=None, seed=0, tape=True, size=224):
    """Brownish noisy floor; ``box`` = (x0, y0, x1, y1) fractions of a blue obstacle."""
    # Reproducible noise
    rng = np.random.RandomState(seed)
    # Floor colour (BGR) with texture
    img = np.clip(rng.normal((60, 90, 130), 12, (size, size, 3)), 0, 255).astype(np.uint8)
    # White tape lines like the track
    if tape:
        # Left line
        cv2.line(img, (30, size - 1), (80, size // 2), (235, 235, 235), 5)
        # Right line
        cv2.line(img, (194, size - 1), (144, size // 2), (235, 235, 235), 5)
    # Obstacle
    if box is not None:
        # Box in pixels
        x0, y0, x1, y1 = [int(f * size) for f in box]
        # Blue-ish textured box
        img[y0:y1, x0:x1] = np.clip(rng.normal((170, 80, 40), 15, (y1 - y0, x1 - x0, 3)), 0, 255).astype(np.uint8)
    # Picture
    return img


# Free-space sectors
class FreeSpaceTest(unittest.TestCase):
    # Clear floor: every sector free; tape is not an obstacle
    def test_clear_floor(self):
        # Estimator fitted on a clear floor
        fs = FreeSpace(sectors=9)
        # Fit
        fs.fit(scene(seed=1))
        # Another clear picture
        scores, _ = fs.estimate(scene(seed=2))
        # Every sector free up to the horizon (more or less)
        self.assertGreater(scores.min(), 0.8, scores)

    # A box straight ahead blocks the middle sectors but not the sides
    def test_box_ahead(self):
        # Fitted estimator
        fs = FreeSpace(sectors=9)
        # Fit on clear floor
        fs.fit(scene(seed=1))
        # Box in the middle, lower part
        scores, _ = fs.estimate(scene(box=(0.38, 0.62, 0.62, 0.85), seed=3))
        # Middle sector blocked early
        self.assertLess(scores[4], 0.35, scores)
        # Outer sectors still free
        self.assertGreater(min(scores[0], scores[8]), 0.8, scores)

    # Accuracy of "obstacle ahead" on 20 synthetic pictures (TASKS M3-2 asks >= 90 %)
    def test_obstacle_ahead_accuracy(self):
        # Fitted estimator and planner (its stop rule decides "ahead blocked")
        fs, planner = FreeSpace(sectors=9), Planner()
        # Fit once
        fs.fit(scene(seed=100))
        # Correct answers
        correct = 0
        # 10 with a near box ahead, 10 without (some with a box at the side)
        for i in range(20):
            # Truth: obstacle right ahead for even i
            ahead = i % 2 == 0
            # Box ahead (near) or at the side / none
            box = (0.35, 0.7, 0.65, 0.97) if ahead else ((0.0, 0.6, 0.2, 0.9) if i % 4 == 1 else None)
            # Sector scores
            scores, _ = fs.estimate(scene(box=box, seed=i))
            # Planner's emergency rule
            _, status, _ = planner.step(None, scores)
            # Compare
            correct += int((status == "too_close") == ahead)
        # Print for the HANDOFF record
        print("\n[navigate] obstacle-ahead accuracy on 20 synthetic pictures: %d/20" % correct)
        # At least 18 of 20
        self.assertGreaterEqual(correct, 18)


# Click tracker
class TrackerTest(unittest.TestCase):
    # Wide textured scene the camera pans over
    @staticmethod
    def panorama(seed=0):
        # Random blobs on noise, 3 frames wide
        rng = np.random.RandomState(seed)
        # Noise background
        img = np.clip(rng.normal(110, 25, (224, 672, 3)), 0, 255).astype(np.uint8)
        # Coloured blobs
        for _ in range(120):
            # Random circle
            cv2.circle(img, (int(rng.randint(0, 672)), int(rng.randint(0, 224))), int(rng.randint(4, 18)),
                       tuple(int(c) for c in rng.randint(0, 255, 3)), -1)
        # Scene
        return img

    # The robot turns (the view pans by ~110 px = ~30 deg with a 62 deg lens): target stays tracked
    def test_follows_while_turning(self):
        # Scene
        pano = self.panorama()
        # Tracker
        tr = Tracker()
        # Start view at x = 224; click right of the centre (x = 168 px)
        tr.start(pano[:, 224:448], 0.75, 0.5)
        # Pan 4 px per frame to the right, 28 frames = 112 px
        for k in range(1, 29):
            # Current view
            view = pano[:, 224 + 4 * k:448 + 4 * k]
            # Track
            u, v, score, ok = tr.update(view)
            # Still tracked
            self.assertTrue(ok, (k, score))
        # The point moved left by 112 px in the view (168 - 112 = 56 px)
        self.assertAlmostEqual(u * 224, 168 - 112, delta=6)
        # Panning further pushes it out of the picture: lost within a few frames
        oks = [tr.update(pano[:, 224 + 4 * k:448 + 4 * k])[3] for k in range(29, 60)]
        # Lost at some point
        self.assertIn(False, oks)

    # A completely different picture: lost after lost_frames updates
    def test_lost(self):
        # Two unrelated scenes
        a, b = self.panorama(1)[:, :224], self.panorama(2)[:, :224]
        # Tracker
        tr = Tracker(lost_frames=5)
        # Start on a
        tr.start(a, 0.5, 0.5)
        # Results on b
        oks = [tr.update(b)[3] for _ in range(5)]
        # Lost at the fifth bad match
        self.assertEqual(oks, [True, True, True, True, False])


# Planner rules
class PlannerTest(unittest.TestCase):
    # Free target sector: go to the target; blocked: detour; blocked ahead: stop
    def test_rules(self):
        # Planner
        p = Planner(free_threshold=0.35, stop_distance=0.12)
        # Everything free, target on the right
        xy, status, sector = p.step(0.8, [1.0] * 9)
        # Towards the target (x > 0), point near the horizon (y < 0)
        self.assertEqual(status, "ok")
        # Right
        self.assertGreater(xy[0], 0.5)
        # Target sector blocked (but not straight ahead): detour to the nearest free sector
        scores = [1.0] * 9
        # Sectors 6..8 blocked
        scores[6] = scores[7] = scores[8] = 0.2
        # Plan
        xy, status, sector = p.step(0.85, scores)
        # Detour into sector 5
        self.assertEqual((status, sector), ("detour", 5))
        # Obstacle right ahead: stop no matter where the target is
        scores = [1.0] * 9
        # Middle blocked very close
        scores[4] = 0.05
        # Plan
        xy, status, _ = p.step(0.1, scores)
        # Stop
        self.assertEqual((xy, status), (None, "too_close"))
        # Less free space -> point lower in the picture (the controller slows down)
        y_far = p.step(None, [1.0] * 9)[0][1]
        # Medium free space
        y_near = p.step(None, [0.5] * 9)[0][1]
        # Lower = larger y
        self.assertGreater(y_near, y_far)

    # Planner output works with project4's controller
    def test_controller_accepts_points(self):
        # project4 controller
        from control import SteeringController
        # Controller
        c = SteeringController()
        # Target on the right, everything free
        xy, _, _ = Planner().step(0.8, [1.0] * 9)
        # A second of control steps (the slew limit ramps the motors up from 0)
        for k in range(20):
            # Control step every 50 ms
            left, right, status = c.update(xy, 0.05 * k)
        # Accepted, turning right (left wheel faster)
        self.assertEqual(status, "ok")
        # Left faster
        self.assertGreater(left, right)


# Label store
class LabelStoreTest(unittest.TestCase):
    # Files in dataset_xy format, readable by project4's data loader; quota refuses
    def test_format_and_quota(self):
        # Temporary folder
        tmp = tempfile.mkdtemp()
        # Store without quota
        store = LabelStore(tmp)
        # A click at (0.25, 0.5)
        path = store.click(scene(), 0.25, 0.5, mode="rule")
        # dataset_xy name with 224-pixel coordinates
        self.assertTrue(os.path.basename(path).startswith("xy_056_112_"))
        # Blocked and free samples (free rate-limited)
        store.blocked(scene())
        # First free sample
        self.assertIsNotNone(store.free(scene()))
        # Second right away is skipped
        self.assertIsNone(store.free(scene()))
        # Counts
        self.assertEqual(store.counts(), {"xy": 1, "blocked": 1, "free": 1})
        # Label lines
        with open(os.path.join(tmp, "labels.jsonl"), encoding="utf-8") as f:
            # Parse every line
            lines = [json.loads(l) for l in f]
        # Three lines
        self.assertEqual([l["kind"] for l in lines], ["xy", "blocked", "free"])
        # project4's dataset reads the xy folder (train.py --data <folder>/xy)
        from data_loader import XYDataset
        # Dataset of the clicks
        ds = XYDataset(os.path.join(tmp, "xy"))
        # One sample, label in normalised units
        image, target = ds[0]
        # Network input shape
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        # x = (56 - 112) / 112 = -0.5
        self.assertAlmostEqual(float(target[0]), -0.5, places=3)
        # Quota full: refused, nothing written
        full = LabelStore(tempfile.mkdtemp(), quota_ok=lambda mb: (False, "recordings quota full"))
        # Refused
        self.assertIsNone(full.click(scene(), 0.5, 0.5))
        # Reason kept
        self.assertIn("quota", full.refused)


# Folder of identical synthetic floor pictures with a textured target (folder camera)
SCENES = tempfile.mkdtemp(prefix="nav-scenes-")
# Three copies of the same scene (the folder camera changes picture twice a second)
for _i in range(3):
    # Floor with a checkerboard target in the upper middle
    _img = scene(seed=5)
    # Checkerboard patch as the clicked object
    _img[70:110, 150:190] = (np.indices((40, 40)).sum(axis=0) // 5 % 2 * 200 + 30)[..., None].astype(np.uint8)
    # Save it
    cv2.imwrite(os.path.join(SCENES, "s%d.jpg" % _i), _img)


# Real runner with the folder camera
class NavigateRunnerTest(StudioTestCase):
    # Folder camera with the static scenes
    CONFIG = {"camera_source": "folder", "camera_folder": SCENES}

    # Read control messages until one matches
    async def until(self, rd, pred, timeout):
        # Shared helper of the lane test
        from test_lane import LaneRunnerTest
        # Same implementation
        return await LaneRunnerTest.until(self, rd, pred, timeout)

    # Send a command and wait for its reply
    async def command(self, conn, rd, cid, name, args=None):
        # Send
        conn.write_message(json.dumps({"type": "command", "id": cid, "name": name, "args": args or {}}))
        # Reply
        return await self.until(rd, lambda m: m.get("type") == "reply" and m.get("id") == cid, 30)

    # Target, drive, e-stop label, arrival
    @gen_test(timeout=180)
    async def test_click_drive_estop_arrive(self):
        # Log in
        await self.login_async()
        # Start (slow folder camera: long loop timeout)
        r = await self.http_client.fetch(self.get_url("/api/tasks/navigate/start"), method="POST", raise_error=False,
                                         body=json.dumps({"values": {"safety_timeout": 2.0}}),
                                         headers={"Cookie": self.cookie_header(), "X-XSRFToken": self.cookies["_xsrf"],
                                                  "Content-Type": "application/json"})
        # Started
        self.assertEqual(r.code, 200, r.body)
        # Control socket
        conn = await self.ws("/ws/control/navigate")
        # Reader
        rd = Reader(conn)
        # Heartbeats
        alive = [1]
        # Loop
        async def hb():
            # While wanted
            while alive:
                # Heartbeat
                conn.write_message(json.dumps({"type": "hb"}))
                # Every 0.1 s
                await gen.sleep(0.1)
        # Run it
        IOLoop.current().spawn_callback(hb)
        # First frames and heartbeats
        await gen.sleep(1.5)
        # Driving without a target is refused
        reply = await self.command(conn, rd, 1, "start_drive")
        # no_target
        self.assertIn("no_target", reply["error"])
        # Click on the checkerboard (centre at 170/224, 90/224)
        reply = await self.command(conn, rd, 2, "set_target", {"u": 170 / 224.0, "v": 90 / 224.0})
        # Label stored
        self.assertTrue(reply["result"]["label"].startswith("xy_"), reply)
        # Drive
        reply = await self.command(conn, rd, 3, "start_drive")
        # Driving
        self.assertTrue(reply["ok"], reply)
        # Telemetry: driving, target tracked, a planned point
        tel = await self.until(rd, lambda m: m.get("type") == "telemetry" and m["data"]["driving"]
                               and (m["data"]["state"].get("target") or {}).get("ok"), 10)
        # Got it
        self.assertIsNotNone(tel)
        # Heading right (target right of the centre)
        self.assertGreater(tel["data"]["state"]["xy"][0], 0.0)
        # Emergency stop
        conn.write_message(json.dumps({"type": "estop"}))
        # Stop event with the e-stop code
        ev = await self.until(rd, lambda m: m.get("type") == "event" and m["event"]["kind"] == "nav.stopped"
                              and m["event"]["code"] == "estop", 5)
        # Stopped
        self.assertIsNotNone(ev)
        # Labels: one click, one blocked
        tel = await self.until(rd, lambda m: m.get("type") == "telemetry" and m["data"]["labels"]["blocked"] >= 1, 5)
        # Counts
        self.assertEqual(tel["data"]["labels"]["xy"], 1)
        # Target low in the picture counts as arrived
        await self.command(conn, rd, 4, "set_target", {"u": 170 / 224.0, "v": 0.95})
        # Drive
        await self.command(conn, rd, 5, "start_drive")
        # Stops as arrived (a normal stop)
        ev = await self.until(rd, lambda m: m.get("type") == "event" and m["event"]["kind"] == "nav.stopped"
                              and m["event"]["code"] == "arrived", 10)
        # Not a fault
        self.assertFalse(ev["event"]["fault"])
        # Label API sees the files
        r = await self.http_client.fetch(self.get_url("/api/navigate/labels"), headers={"Cookie": self.cookie_header()})
        # Two clicks, one blocked
        counts = json.loads(r.body.decode())["counts"]
        # Values
        self.assertEqual((counts["xy"], counts["blocked"]), (2, 1))
        # Zip download contains them
        r = await self.http_client.fetch(self.get_url("/api/navigate/labels.zip"), headers={"Cookie": self.cookie_header()})
        # Open the zip
        import io
        # zip reader
        import zipfile
        # Names inside
        names = zipfile.ZipFile(io.BytesIO(r.body)).namelist()
        # Two xy files under navigate_labels/xy
        self.assertEqual(len([n for n in names if n.startswith("navigate_labels/xy/xy_")]), 2)
        # End heartbeats
        alive.pop()
        # Close
        conn.close()
        # Stop the task
        await self.studio.manager.stop_task()


# Run directly
if __name__ == "__main__":
    # unittest entry point
    unittest.main()
