"""Search task (TASKS M4): geometry, odometry, map, detection, exploration, 3-D map, runner."""
import json
import math
import os
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
# Synthetic world, square drive, posters
from _world import World, square_path, poster, poster_scene
# Message reader
from test_runner import Reader
# Modules under test
from studio.tasks.search.camera_model import CameraModel
# Odometry
from studio.tasks.search.odometry import FloorOdometry
# Grid
from studio.tasks.search.grid_map import GridMap
# Detection
from studio.tasks.search.target import TargetModel, find, Detector
# Exploration
from studio.tasks.search.explore import choose_goal, steer_to
# Reconstruction
from studio.tasks.search import reconstruct as rec
# Free space (M3)
from studio.tasks.navigate.freespace import FreeSpace
# Job queue
from studio import jobs

# Camera of the synthetic tests: 120 deg, 10 cm high, tilted 20 deg down
CAM = CameraModel(size=224, hfov_deg=120, height=0.10, pitch_deg=20)


# Geometry and odometry
class GeometryTest(unittest.TestCase):
    # Pixel -> floor -> pixel round trip; rows below the horizon only
    def test_round_trip(self):
        # Pixels below the horizon
        u, v = np.array([20.0, 112.0, 200.0]), np.array([210.0, 150.0, 120.0])
        # Floor points
        x, y = CAM.ground_from_pixel(u, v)
        # Back to pixels
        u2, v2 = CAM.pixel_from_ground(x, y)
        # Same pixels
        np.testing.assert_allclose(np.stack([u2, v2]), np.stack([u, v]), atol=1e-6)
        # Above the horizon: no floor
        self.assertTrue(np.isnan(CAM.ground_from_pixel(112.0, 40.0)[0]))
        # The centre column farther up is farther away
        self.assertGreater(float(CAM.row_distance(120)), float(CAM.row_distance(200)))

    # 5 m square drive closes within 10 cm and 3 degrees (synthetic textured floor)
    def test_odometry_square(self):
        # World and odometry
        world, odo = World(CAM), FloorOdometry(CAM)
        # Drive
        for p in [(0, 0, 0)] + square_path():
            # One frame
            x, y, yaw, ok = odo.update(world.render(*p))
        # Closure error
        err = math.hypot(x, y)
        # Heading error
        yaw_err = abs(math.degrees(math.atan2(math.sin(yaw), math.cos(yaw))))
        # Print for the HANDOFF record
        print("\n[search] odometry 5 m square: closure %.3f m, heading %.1f deg, failed steps %d" % (err, yaw_err, odo.lost))
        # Within limits
        self.assertLess(err, 0.10)
        # Heading
        self.assertLess(yaw_err, 3.0)
        # No failed steps on a textured floor
        self.assertEqual(odo.lost, 0)

    # A blank floor gives no features: steps fail instead of inventing motion
    def test_blank_floor_fails_safely(self):
        # Odometry
        odo = FloorOdometry(CAM)
        # Uniform grey frames
        blank = np.full((224, 224, 3), 128, np.uint8)
        # Several frames
        for _ in range(5):
            # Update
            x, y, yaw, ok = odo.update(blank)
        # Pose unchanged
        self.assertEqual((x, y, yaw), (0.0, 0.0, 0.0))
        # Failures counted
        self.assertGreater(odo.lost, 0)


# Map and exploration
class MapTest(unittest.TestCase):
    # Driving builds free space with frontiers; a wall-like blocked sector adds occupied cells
    def test_grid(self):
        # Grid and free space
        grid, fs, world = GridMap(), FreeSpace(horizon=0.45), World(CAM)
        # First part of the square
        for p in square_path()[:60]:
            # Sectors of the frame
            scores, _ = fs.estimate(world.render(*p))
            # Update
            grid.update(p, scores, CAM, 0.45)
        # Explored area and frontiers
        st = grid.stats()
        # Some square metres explored
        self.assertGreater(st["explored_m2"], 0.5)
        # Frontiers exist
        self.assertTrue(grid.frontiers())
        # An obstacle seen a few times straight ahead (fresh grid) becomes occupied
        fresh = GridMap()
        # Three observations of a sector blocked at 60 % of the analysed height
        for _ in range(3):
            # Update
            fresh.update((0.0, 0.0, 0.0), [1.0] * 4 + [0.6] + [1.0] * 4, CAM, 0.45)
        # Occupied cells now
        self.assertGreater(fresh.stats()["occupied"], 0)
        # Picture for the page
        self.assertEqual(grid.image().shape[2], 3)

    # Goals: nearest big frontier, skipping visited ones; steering: view column or turn
    def test_explore_rules(self):
        # Two frontiers: small near, big far
        fronts = [(1.0, 0.0, 5), (0.0, 3.0, 100)]
        # Big one wins (100 / 4 > 5 / 2)
        self.assertEqual(choose_goal(fronts, (0, 0, 0)), (0.0, 3.0))
        # Visited big one: the other
        self.assertEqual(choose_goal(fronts, (0, 0, 0), visited=[(0.0, 3.0)]), (1.0, 0.0))
        # Goal straight ahead: centre column
        kind, u = steer_to((2.0, 0.0), (0, 0, 0), 120)
        # View, centre
        self.assertEqual(kind, "view")
        # Centre
        self.assertAlmostEqual(u, 0.5)
        # Goal to the left in view: left half
        self.assertLess(steer_to((2.0, 0.5), (0, 0, 0), 120)[1], 0.5)
        # Goal behind: turn
        self.assertEqual(steer_to((-2.0, 0.1), (0, 0, 0), 120), ("turn", 1))


# Target detection
class DetectionTest(unittest.TestCase):
    # Recall >= 80 % and false positives <= 5 % over three posters (TASKS M4-3: 30 + 30 pictures)
    def test_recall_and_false_positives(self):
        # Totals
        tp, fp, n = 0, 0, 0
        # Three different targets
        for seed in (1, 2, 3):
            # Model of one photo
            m = TargetModel.from_images([poster(seed)])
            # 30 pictures with the target
            tp += sum(find(m, poster_scene(poster(seed), s))["hit"] for s in range(30))
            # 30 pictures with other posters
            fp += sum(find(m, poster_scene(poster(100 + s), 500 + s))["hit"] for s in range(30))
            # Count
            n += 30
        # Print for the HANDOFF record
        print("\n[search] detection: recall %d/%d, false positives %d/%d" % (tp, n, fp, n))
        # Recall
        self.assertGreaterEqual(tp, 0.8 * n)
        # False positives
        self.assertLessEqual(fp, 0.05 * n)

    # A plain photo is flagged hard; confirmation needs hits in a row
    def test_hard_photo_and_confirmation(self):
        # Uniform photo: almost no features
        m = TargetModel.from_images([np.full((150, 200, 3), (40, 90, 200), np.uint8)])
        # Flagged
        self.assertTrue(m.describe()["hard"])
        # Detector for a real poster, 3 hits needed
        det = Detector(TargetModel.from_images([poster(2)]), confirm=3)
        # Results for three pictures with the poster
        found = [det.step(poster_scene(poster(2), s))["found"] for s in (1, 2, 4)]
        # Confirmed only at the third hit
        self.assertEqual(found, [False, False, True])


# 3-D reconstruction of a synthetic run
class ReconstructionTest(unittest.TestCase):
    # Floor-only world: triangulated points lie on the floor; files written, small
    def test_floor_world(self):
        # Session folder
        d = tempfile.mkdtemp()
        # Keyframe folder
        os.makedirs(os.path.join(d, "keyframes"))
        # Session settings
        with open(os.path.join(d, "session.json"), "w") as f:
            # Camera of the test
            json.dump({"camera": {"size": 224, "hfov_deg": 120, "height": 0.10, "pitch_deg": 20}}, f)
        # World and keyframes every 15 cm / 15 deg
        world, kfs, last = World(CAM), [], None
        # Drive
        for i, p in enumerate([(0, 0, 0)] + square_path()):
            # Far enough from the last keyframe
            if last is None or math.hypot(p[0] - last[0], p[1] - last[1]) > 0.15 or abs(p[2] - last[2]) > math.radians(15):
                # Save it
                cv2.imwrite(os.path.join(d, "keyframes", "%04d.jpg" % i), world.render(*p))
                # Pose
                kfs.append({"file": "%04d.jpg" % i, "pose": list(p)})
                # Remember
                last = p
        # Keyframe list
        with open(os.path.join(d, "keyframes.json"), "w") as f:
            # JSON
            json.dump(kfs, f)
        # Reconstruct
        s = rec.reconstruct(d, progress=lambda m: None)
        # Points of the PLY
        with open(os.path.join(d, "map.ply")) as f:
            # Body after the header
            pts = np.array([l.split()[:3] for l in f.read().split("end_header\n")[1].splitlines()], float)
        # Height of the triangulated points
        z = np.abs(pts[:, 2])
        # Print for the HANDOFF record
        print("\n[search] reconstruction: %s; |z| p95 %.3f m, max %.3f m" % (s, np.percentile(z, 95), z.max()))
        # Structure points exist and lie on the floor (it is the only surface)
        self.assertGreater(s["structure_points"], 100)
        # Within 4 cm (95 %)
        self.assertLess(np.percentile(z, 95), 0.04)
        # Viewer data
        self.assertTrue(os.path.exists(os.path.join(d, "cloud.json")))
        # Far below the 300 MB map budget
        self.assertLess(s["ply_mb"], 30)

    # Keyframes of a session whose runner died: only the line file exists
    def test_keyframes_line_file_fallback(self):
        # Function under test
        from studio.tasks.search.reconstruct import load_keyframes
        # Empty session folder
        d = tempfile.mkdtemp()
        # Nothing yet
        self.assertEqual(load_keyframes(d), [])
        # Two complete lines and a partial one (crash while writing)
        with open(os.path.join(d, "keyframes.jsonl"), "w") as f:
            # Content
            f.write('{"file": "00000.jpg", "pose": [0, 0, 0]}\n{"file": "00001.jpg", "pose": [0.1, 0, 0]}\n{"fi')
        # The complete ones
        self.assertEqual([k["file"] for k in load_keyframes(d)], ["00000.jpg", "00001.jpg"])
        # The final list wins once written
        with open(os.path.join(d, "keyframes.json"), "w") as f:
            # One entry
            json.dump([{"file": "00000.jpg", "pose": [0, 0, 0]}], f)
        # Read from keyframes.json
        self.assertEqual(len(load_keyframes(d)), 1)


# Folder of camera pictures showing the poster (folder camera of the runner test)
SCENES = tempfile.mkdtemp(prefix="search-scenes-")
# Three pictures of the target from different angles
for _i in range(3):
    # Save
    cv2.imwrite(os.path.join(SCENES, "s%d.jpg" % _i), poster_scene(poster(2), 40 + _i))


# Real runner: register a target, search, found, files, 3-D job, viewer
class SearchRunnerTest(StudioTestCase):
    # Folder camera with the poster scenes
    CONFIG = {"camera_source": "folder", "camera_folder": SCENES}

    # Read messages until one matches
    async def until(self, rd, pred, timeout):
        # Shared implementation
        from test_lane import LaneRunnerTest
        # Same helper
        return await LaneRunnerTest.until(self, rd, pred, timeout)

    # The whole flow
    @gen_test(timeout=240)
    async def test_search_flow(self):
        # Log in
        await self.login_async()
        # Upload two photos of the target (multipart)
        boundary, parts = "B0undary", []
        # Name field
        parts.append(('--%s\r\nContent-Disposition: form-data; name="name"\r\n\r\nposter2\r\n' % boundary).encode())
        # Second photo: the same object from another angle (perspective-tilted)
        h, w = poster(2).shape[:2]
        # Corners moved by up to 12 % of the width
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]]) + np.random.RandomState(3).uniform(
            -0.12, 0.12, (4, 2)).astype(np.float32) * w + np.float32([0.15 * w, 0.15 * h])
        # Warp onto a grey background
        tilted = cv2.warpPerspective(poster(2), cv2.getPerspectiveTransform(
            np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst), (int(1.3 * w), int(1.3 * h)), borderValue=(120, 120, 120))
        # Two photos
        for k, img in enumerate((poster(2), tilted)):
            # JPEG bytes
            data = cv2.imencode(".jpg", img)[1].tobytes()
            # File part
            parts.append(('--%s\r\nContent-Disposition: form-data; name="files"; filename="p%d.jpg"\r\n'
                          'Content-Type: image/jpeg\r\n\r\n' % (boundary, k)).encode() + data + b"\r\n")
        # Send
        r = await self.http_client.fetch(self.get_url("/api/search/targets"), method="POST",
                                         body=b"".join(parts) + ("--%s--\r\n" % boundary).encode(),
                                         headers={"Cookie": self.cookie_header(), "X-XSRFToken": self.cookies["_xsrf"],
                                                  "Content-Type": "multipart/form-data; boundary=%s" % boundary})
        # Registered with features and the cross-check
        t = json.loads(r.body.decode())["target"]
        # Two photos
        self.assertEqual(len(t["photos"]), 2)
        # Each photo is recognised by the other one
        self.assertEqual(t["cross_check"], [True, True])
        # Start the task (slow folder camera: long loop timeout; 2 hits confirm)
        r = await self.http_client.fetch(self.get_url("/api/tasks/search/start"), method="POST", raise_error=False,
                                         body=json.dumps({"values": {"safety_timeout": 2.0, "confirm": 2}}),
                                         headers={"Cookie": self.cookie_header(), "X-XSRFToken": self.cookies["_xsrf"],
                                                  "Content-Type": "application/json"})
        # Started
        self.assertEqual(r.code, 200, r.body)
        # Control socket with heartbeats
        conn = await self.ws("/ws/control/search")
        # Reader
        rd = Reader(conn)
        # Heartbeat flag
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
        # Attach
        await gen.sleep(1.0)
        # Unknown target refused
        conn.write_message(json.dumps({"type": "command", "id": 1, "name": "start_search", "args": {"target": "nope"}}))
        # Reply
        reply = await self.until(rd, lambda m: m.get("type") == "reply" and m.get("id") == 1, 20)
        # Refused with the reason
        self.assertIn("no such target", reply["error"])
        # Start the real search
        conn.write_message(json.dumps({"type": "command", "id": 2, "name": "start_search", "args": {"target": "poster2"}}))
        # Reply with the session id
        reply = await self.until(rd, lambda m: m.get("type") == "reply" and m.get("id") == 2, 20)
        # Started
        self.assertTrue(reply["ok"], reply)
        # Session id
        sid = reply["result"]["session"]
        # Found event (2 hits in a row on the poster pictures)
        ev = await self.until(rd, lambda m: m.get("type") == "event" and m["event"]["kind"] == "search.found", 20)
        # Found
        self.assertIsNotNone(ev, "target not found")
        # Telemetry state
        tel = await self.until(rd, lambda m: m.get("type") == "telemetry", 5)
        # Found state, not driving
        self.assertEqual((tel["data"]["state"], tel["data"]["driving"]), ("found", False))
        # End heartbeats and stop the task (writes the files)
        alive.pop()
        # Close
        conn.close()
        # Stop
        await self.studio.manager.stop_task()
        # Session listed as found
        r = await self.http_client.fetch(self.get_url("/api/search/sessions"), headers={"Cookie": self.cookie_header()})
        # Its row
        row = [s for s in json.loads(r.body.decode())["sessions"] if s["id"] == sid][0]
        # Found
        self.assertTrue(row["found"])
        # Files served
        for name in ("grid.png", "found.jpg", "trajectory.csv", "summary.json"):
            # Fetch
            r = await self.http_client.fetch(self.get_url("/api/search/sessions/%s/%s" % (sid, name)),
                                             headers={"Cookie": self.cookie_header()}, raise_error=False)
            # Present
            self.assertEqual(r.code, 200, name)
        # Viewer page renders
        r = await self.http_client.fetch(self.get_url("/search/map/" + sid), headers={"Cookie": self.cookie_header()})
        # OK
        self.assertEqual(r.code, 200)
        # 3-D job runs (keyframes of a robot that did not move: floor points only)
        r = await self.http_client.fetch(self.get_url("/api/search/sessions/%s/reconstruct" % sid), method="POST",
                                         body="{}", headers={"Cookie": self.cookie_header(),
                                                             "X-XSRFToken": self.cookies["_xsrf"]})
        # The job
        job = self.studio.manager.jobs.jobs[json.loads(r.body.decode())["job"]["id"]]
        # Wait for it
        deadline = time.monotonic() + 60
        # Tick until finished
        while job.status not in jobs.FINISHED and time.monotonic() < deadline:
            # Advance
            self.studio.manager.jobs.tick()
            # Wait
            await gen.sleep(0.2)
        # Done
        self.assertEqual(job.status, "done", self.studio.manager.jobs.read_log(job.id)[0])
        # Point cloud served
        r = await self.http_client.fetch(self.get_url("/api/search/sessions/%s/cloud.json" % sid),
                                         headers={"Cookie": self.cookie_header()})
        # Has points
        self.assertGreater(len(json.loads(r.body.decode())["points"]), 0)


# Run directly
if __name__ == "__main__":
    # unittest entry point
    unittest.main()
