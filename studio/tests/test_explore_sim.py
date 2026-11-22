"""Closed-loop exploration in simulation (TASKS M4-5: 3 m x 3 m arena, find the object >= 3/5 in 10 min).

The search runtime's real exploration step (``Runtime._explore``: frontier goal,
M3 planner with its stop rule, turning on the spot, project4 controller) drives
a simulated differential-drive robot. Sectors come from ray casting in a
ground-truth arena (walls + a box) through the same camera model, i.e. an ideal
floor detector; the pose is the true pose (odometry is tested separately).
The object counts as seen when it is in the central 80 % of the view, closer
than 1 m and not hidden by the box.
"""
import math
import tempfile
import unittest

import numpy as np

# Test helpers (temporary STUDIO_HOME, import path)
import _base  # noqa: F401
# Task declaration (parameter defaults)
from studio.tasks.search import TASK
# Runtime under test
from studio.tasks.search.runtime import Runtime
# Grid
from studio.tasks.search.grid_map import GridMap
# Fake motors and an empty sensor hub
from studio.hardware import FakeRobot
# Sensors
from studio.sensors import SensorHub

# Simulation step (camera rate)
DT = 1.0 / 15
# Speed of a wheel at motor value 1.0 (m/s) and wheel distance (m)
WHEEL_SPEED, WHEELBASE = 0.5, 0.12
# Robot radius for collisions (m)
RADIUS = 0.07


# Minimal runner context
class Ctx(object):
    # Defaults of the task, fake robot, no camera
    def __init__(self):
        # Parameter defaults
        self.params = dict((p["name"], p["default"]) for p in TASK["params"])
        # 10 minutes
        self.params["time_limit_s"] = 600
        # Hardware stand-ins
        self.robot, self.robot_info, self.sensors = FakeRobot(), {"kind": "fake"}, SensorHub()
        # Settings
        self.settings, self.role = {}, "pc"

    # Events are ignored
    def emit(self, kind, **data):
        # Nothing
        pass

    # Temporary folders
    def data_dir(self, *parts):
        # New temporary folder
        return tempfile.mkdtemp()


# Ground-truth arena: 1 cm raster, True = obstacle
class Arena(object):
    # 3 m x 3 m room centred on the origin with a box
    def __init__(self, box):
        # Raster of 4 m x 4 m (walls outside the room)
        self.res, self.n = 0.01, 400
        # Everything is wall
        self.occ = np.ones((self.n, self.n), bool)
        # Free room: |x|, |y| < 1.5 m
        self.occ[50:350, 50:350] = False
        # The box (x0, y0, x1, y1) in metres
        c0, r0 = self.cell(box[0], box[3])
        # Other corner
        c1, r1 = self.cell(box[2], box[1])
        # Mark it
        self.occ[r0:r1, c0:c1] = True

    # Metres -> raster indices
    def cell(self, x, y):
        """Column and row (rows grow towards -y)."""
        # Origin in the middle
        return int(x / self.res + self.n / 2), int(-y / self.res + self.n / 2)

    # Obstacle at points (arrays)
    def blocked(self, x, y):
        """Boolean array; outside the raster counts as obstacle."""
        # Indices
        c, r = (x / self.res + self.n / 2).astype(int), (-y / self.res + self.n / 2).astype(int)
        # Inside the raster
        ok = (c >= 0) & (c < self.n) & (r >= 0) & (r < self.n)
        # Result
        out = np.ones_like(x, bool)
        # Look up
        out[ok] = self.occ[r[ok], c[ok]]
        # Obstacles
        return out


# Ideal sector scores from the arena through the camera model
class Sectors(object):
    # Precompute the floor points of every (sector, row)
    def __init__(self, rt):
        # Camera and free-space settings of the runtime
        cam, fs = rt.cam, rt.free
        # Rows from the bottom up to the horizon of the analysis
        top = int(fs.horizon * cam.size)
        # Rows, bottom first
        self.rows = np.arange(cam.size - 1, top - 1, -1)
        # Centre columns of the sectors
        cols = (np.arange(fs.sectors) + 0.5) * cam.size / fs.sectors
        # Floor points (sectors x rows), robot frame
        self.gx, self.gy = cam.ground_from_pixel(cols[:, None] + 0 * self.rows[None, :], self.rows[None, :] + 0.0)
        # Analysed height
        self.span = float(cam.size - top)

    # Scores at a pose
    def scores(self, arena, pose):
        """Free distance per sector as fractions (like FreeSpace.estimate)."""
        # Pose
        x, y, yaw = pose
        # Rotation
        c, s = math.cos(yaw), math.sin(yaw)
        # World points
        wx, wy = x + c * self.gx - s * self.gy, y + s * self.gx + c * self.gy
        # Obstacle at each (sector, row); NaN rows (beyond the horizon) count as free
        b = arena.blocked(np.nan_to_num(wx, nan=x), np.nan_to_num(wy, nan=y)) & np.isfinite(wx)
        # First blocked row per sector (or all rows free)
        first = np.where(b.any(axis=1), b.argmax(axis=1), len(self.rows))
        # Fraction of the analysed height
        return first / self.span


# One simulated search
def run(seed):
    """Return (found, seconds, collisions, explored m2)."""
    # Random layout
    rng = np.random.RandomState(seed)
    # Box somewhere in the room (not on the start)
    bx, by = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
    # Keep it away from the start
    if math.hypot(bx, by) < 0.5:
        # Push it out
        bx, by = bx + 0.6 * np.sign(bx or 1), by
    # Arena with a 0.3 m box
    arena = Arena((bx - 0.15, by - 0.15, bx + 0.15, by + 0.15))
    # Object on a random wall, 2 cm in front of it
    side, t = rng.randint(4), rng.uniform(-1.2, 1.2)
    # Its position
    target = [(1.48, t), (-1.48, t), (t, 1.48), (t, -1.48)][side]
    # Runtime with the defaults
    rt = Runtime(Ctx())
    # Simulated clock
    clock = [0.0]
    # Use it
    rt.clock = lambda: clock[0]
    # Session parts without files
    rt.session, rt.grid = {"started": 0.0, "found": None, "dir": tempfile.mkdtemp()}, GridMap(side=6.0)
    # No files in the simulation
    rt._save_session = lambda: None
    # Exploring
    rt.state, rt.driving, rt._explore_t0 = "exploring", True, 0.0
    # Ideal sectors
    sectors = Sectors(rt)
    # Start pose: centre, random heading
    x, y, yaw = 0.0, 0.0, rng.uniform(-math.pi, math.pi)
    # Collisions
    hits = 0
    # Up to 10 minutes
    while clock[0] < 600 and rt.state == "exploring":
        # Pose
        pose = (x, y, yaw)
        # Sector scores
        sc = sectors.scores(arena, pose)
        # Map update
        rt.grid.update(pose, sc, rt.cam, rt.free.horizon)
        # Object in view, near and not hidden?
        dx, dy = target[0] - x, target[1] - y
        # Bearing relative to the heading
        b = math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw))
        # Distance
        d = math.hypot(dx, dy)
        # Line of sight through the box
        ts = np.linspace(0, 1, 50)
        # Seen
        if abs(b) < math.radians(rt.ctx.params["hfov_deg"]) / 2 * 0.8 and d < 1.0 and \
                not arena.blocked(x + ts[:-2] * dx, y + ts[:-2] * dy).any():
            # Found
            return True, clock[0], hits, rt.grid.stats()["explored_m2"]
        # One exploration step (sets the motors)
        rt._explore(pose, sc)
        # Wheel speeds
        vl, vr = rt.ctx.robot.left_motor.value * WHEEL_SPEED, rt.ctx.robot.right_motor.value * WHEEL_SPEED
        # Forward and turning speed
        v, w = (vl + vr) / 2.0, (vr - vl) / WHEELBASE
        # New pose
        nx, ny, nyaw = x + v * math.cos(yaw) * DT, y + v * math.sin(yaw) * DT, yaw + w * DT
        # Collision check on a circle around the new position
        ang = np.linspace(0, 2 * math.pi, 12, endpoint=False)
        # Any obstacle within the robot radius
        if arena.blocked(nx + RADIUS * np.cos(ang), ny + RADIUS * np.sin(ang)).any() and v > 0:
            # Count it, do not move forward (turning is still applied)
            hits += 1
            # Only the rotation
            nx, ny = x, y
        # Advance
        x, y, yaw = nx, ny, nyaw
        # Clock
        clock[0] += DT
    # Not found
    return False, clock[0], hits, rt.grid.stats()["explored_m2"]


# The acceptance rule of TASKS M4-5 in simulation
class ExploreSimTest(unittest.TestCase):
    # Five layouts: found at least 3 times within 10 minutes, no collisions
    def test_five_runs(self):
        # Results
        results = [run(seed) for seed in range(5)]
        # Print for the HANDOFF record
        print("\n[explore-sim] " + "; ".join("found=%s t=%.0fs collisions=%d explored=%.1fm2" % r for r in results))
        # At least 3 of 5 found
        self.assertGreaterEqual(sum(r[0] for r in results), 3)
        # Never drove into an obstacle
        self.assertEqual(sum(r[2] for r in results), 0)


# Run directly
if __name__ == "__main__":
    # unittest entry point
    unittest.main()
