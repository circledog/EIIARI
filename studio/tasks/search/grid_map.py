"""2-D occupancy grid built from the free-space sectors and the odometry pose (TASKS M4-4).

Each frame, every free-space sector gives "floor is free up to here" along
its viewing direction. The sector's free distance (a row in the picture) is
turned into metres with the camera model; the cells along the ray become more
free, the cell where the floor ends becomes more occupied (log-odds).

Cells: unknown (log-odds 0), free (< -0.5), occupied (> 0.5). Frontiers are
free cells next to unknown ones -- where exploration should go next.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import math

import cv2
import numpy as np

# Log-odds added for a free / occupied observation, and the clamp
L_FREE, L_OCC, L_MAX = -0.4, 0.85, 4.0


# Occupancy grid centred on the start point
class GridMap(object):
    """``res`` metres per cell, ``side`` metres wide, origin (start) in the middle."""

    # Empty grid
    def __init__(self, side=12.0, res=0.05):
        # Cell size and count
        self.res, self.n = float(res), int(round(side / res))
        # Log-odds per cell
        self.logodds = np.zeros((self.n, self.n), np.float32)
        # Number of update() calls (frames)
        self.updates = 0
        # Trajectory as cells, a new entry only when the cell changes (bounded by the distance driven)
        self.track = []

    # World metres -> cell (column, row); rows grow towards -y
    def cell(self, x, y):
        """Integer cell indices (may be outside the grid)."""
        # Column from x, row from y
        return int(math.floor(x / self.res + self.n / 2.0)), int(math.floor(-y / self.res + self.n / 2.0))

    # Cell centre -> world metres
    def world(self, col, row):
        """Metres of a cell centre."""
        # Inverse of cell()
        return (col + 0.5 - self.n / 2.0) * self.res, -(row + 0.5 - self.n / 2.0) * self.res

    # Inside the grid
    def inside(self, col, row):
        """True for valid indices."""
        # Bounds check
        return 0 <= col < self.n and 0 <= row < self.n

    # Apply one frame's observations (arrays of columns / rows; duplicates allowed)
    def _apply(self, free_cols, free_rows, hit_cols, hit_rows):
        # Flat indices of the cells inside the grid, each cell once
        def flat(cols, rows):
            # Inside only
            ok = (cols >= 0) & (cols < self.n) & (rows >= 0) & (rows < self.n)
            # Unique flat indices
            return np.unique(rows[ok] * self.n + cols[ok])
        # Hit cells
        hits = flat(hit_cols, hit_rows)
        # A cell hit in this frame is never also marked free by a neighbouring ray
        free = np.setdiff1d(flat(free_cols, free_rows), hits, assume_unique=True)
        # Flat view of the grid
        g = self.logodds.reshape(-1)
        # More free, clamped
        g[free] = np.maximum(-L_MAX, g[free] + L_FREE)
        # More occupied, clamped
        g[hits] = np.minimum(L_MAX, g[hits] + L_OCC)

    # Update from one frame's sectors
    def update(self, pose, scores, camera, horizon_frac, max_range=1.5):
        """``pose`` (x, y, yaw); ``scores`` free distance per sector (fractions of the analysed height)."""
        # Pose
        x, y, yaw = pose
        # Count the frame
        self.updates += 1
        # Robot cell
        c0 = self.cell(x, y)
        # Extend the trajectory when the robot entered a new cell
        if not self.track or self.track[-1] != c0:
            # New cell
            self.track.append(c0)
        # Picture size and first analysed row
        h = camera.size
        # Top row of the analysed area
        top = horizon_frac * h
        # Sector scores as an array
        sc = np.asarray(scores, np.float64)
        # Number of sectors
        n = len(sc)
        # Centre column of each sector
        u = (np.arange(n) + 0.5) * camera.size / n
        # Row where each sector's floor ends (not above the horizon)
        v = np.maximum(h - sc * (h - top), camera.horizon_row() + 2)
        # Floor points there (robot frame), all sectors at once
        gx, gy = camera.ground_from_pixel(u, v)
        # Rays that hit the floor
        finite = np.isfinite(gx) & np.isfinite(gy)
        # Distance; NaN: infinitely far
        dist = np.where(finite, np.hypot(np.where(finite, gx, 0.0), np.where(finite, gy, 0.0)), np.inf)
        # The floor ended inside the analysed area and within range
        hit = (sc < 0.98) & (dist < max_range)
        # Limit the rays to max_range
        scale = np.where(dist > 0, np.minimum(1.0, max_range / np.where(dist > 0, dist, 1.0)), 0.0)
        # End points in the robot frame (no floor: straight ahead at max_range)
        ex = np.where(finite, np.where(finite, gx, 0.0) * scale, max_range)
        # Sideways part
        ey = np.where(finite, np.where(finite, gy, 0.0) * scale, 0.0)
        # Rotation of the robot
        c, s = math.cos(yaw), math.sin(yaw)
        # End cells (columns / rows) in the world
        c1 = np.floor((x + c * ex - s * ey) / self.res + self.n / 2.0).astype(np.int64)
        # Rows
        r1 = np.floor(-(y + s * ex + c * ey) / self.res + self.n / 2.0).astype(np.int64)
        # Steps of each ray along its longer axis (at least 1)
        steps = np.maximum(np.maximum(np.abs(c1 - c0[0]), np.abs(r1 - c0[1])), 1)
        # Ray index and step k for every cell before the end
        ray = np.repeat(np.arange(n), steps)
        # Step within its ray
        k = np.arange(ray.size) - np.repeat(np.cumsum(steps) - steps, steps)
        # Fraction along the ray
        t = k / steps[ray].astype(np.float64)
        # Free cells: rounded points along the segment, end cell excluded
        fc = np.round(c0[0] + (c1[ray] - c0[0]) * t).astype(np.int64)
        # Rows
        fr = np.round(c0[1] + (r1[ray] - c0[1]) * t).astype(np.int64)
        # Apply the whole frame at once
        self._apply(fc, fr, c1[hit], r1[hit])

    # Frontier clusters
    def frontiers(self, min_cells=4):
        """List of ``(x, y, cells)`` world centres of frontier clusters, largest first."""
        # Free and unknown masks
        free = (self.logodds < -0.5).astype(np.uint8)
        # Unknown cells
        unknown = (np.abs(self.logodds) < 0.1).astype(np.uint8)
        # Free cells with an unknown neighbour
        near_unknown = cv2.dilate(unknown, np.ones((3, 3), np.uint8))
        # Frontier mask
        front = free & near_unknown
        # Connected clusters
        count, labels, stats, cents = cv2.connectedComponentsWithStats(front, connectivity=8)
        # Clusters (label 0 is the background)
        out = []
        # Every cluster
        for k in range(1, count):
            # Big enough
            if stats[k, cv2.CC_STAT_AREA] >= min_cells:
                # World centre
                wx, wy = self.world(cents[k][0], cents[k][1])
                # Keep it
                out.append((wx, wy, int(stats[k, cv2.CC_STAT_AREA])))
        # Largest first
        return sorted(out, key=lambda f: -f[2])

    # Picture of the grid (unknown grey, free white, occupied black) with the trajectory
    def image(self, scale=2):
        """BGR picture for the page."""
        # Grey levels from the log-odds
        p = 1.0 - 1.0 / (1.0 + np.exp(self.logodds))
        # 0 = occupied, 255 = free
        grey = (255 * (1.0 - p)).astype(np.uint8)
        # Colour image
        img = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        # Trajectory in blue (one call for the whole polyline)
        if len(self.track) > 1:
            # Draw it
            cv2.polylines(img, [np.int32(self.track).reshape(-1, 1, 2)], False, (200, 80, 0), 1)
        # Start point in green
        cv2.circle(img, self.cell(0.0, 0.0), 3, (0, 180, 0), -1)
        # Enlarge
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # Counts of free / occupied / unknown cells
    def stats(self):
        """Cell counts and the explored area in square metres."""
        # Free cells
        free = int((self.logodds < -0.5).sum())
        # Occupied cells
        occ = int((self.logodds > 0.5).sum())
        # Counts
        return {"free": free, "occupied": occ, "explored_m2": round((free + occ) * self.res ** 2, 2)}
