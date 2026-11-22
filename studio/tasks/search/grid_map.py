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
        # Trajectory in metres
        self.path = []

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

    # Cells of one ray: the free ones and the end cell
    def ray_cells(self, x0, y0, x1, y1):
        """``(free_cells, end_cell)`` of the segment (cells as (col, row))."""
        # Cells of both ends
        c0, c1 = self.cell(x0, y0), self.cell(x1, y1)
        # Number of steps along the longer axis
        n = max(abs(c1[0] - c0[0]), abs(c1[1] - c0[1]), 1)
        # Every cell before the end
        free = set((int(round(c0[0] + (c1[0] - c0[0]) * k / float(n))), int(round(c0[1] + (c1[1] - c0[1]) * k / float(n))))
                   for k in range(n))
        # Free cells and the end
        return free, c1

    # Apply one frame's observations
    def _apply(self, free, hits):
        # A cell hit in this frame is never also marked free by a neighbouring ray
        for c in free - hits:
            # Inside only
            if self.inside(*c):
                # More free, clamped
                self.logodds[c[1], c[0]] = max(-L_MAX, self.logodds[c[1], c[0]] + L_FREE)
        # Hit cells
        for c in hits:
            # Inside only
            if self.inside(*c):
                # More occupied, clamped
                self.logodds[c[1], c[0]] = min(L_MAX, self.logodds[c[1], c[0]] + L_OCC)

    # Mark one ray (single observation)
    def ray(self, x0, y0, x1, y1, hit):
        """Free along the segment; occupied at its end when ``hit``."""
        # Cells of the ray
        free, end = self.ray_cells(x0, y0, x1, y1)
        # Apply
        self._apply(free, {end} if hit else set())

    # Update from one frame's sectors
    def update(self, pose, scores, camera, horizon_frac, max_range=1.5):
        """``pose`` (x, y, yaw); ``scores`` free distance per sector (fractions of the analysed height)."""
        # Pose
        x, y, yaw = pose
        # Remember the trajectory
        self.path.append((x, y))
        # Picture size and first analysed row
        h = camera.size
        # Top row of the analysed area
        top = horizon_frac * h
        # Rotation of the robot
        c, s = math.cos(yaw), math.sin(yaw)
        # Number of sectors
        n = len(scores)
        # Free and hit cells of this frame
        free, hits = set(), set()
        # Every sector
        for i, sc in enumerate(scores):
            # Centre column of the sector
            u = (i + 0.5) * camera.size / n
            # Row where its floor ends (just above it is the obstacle)
            v = h - sc * (h - top)
            # Floor point there (robot frame)
            gx, gy = camera.ground_from_pixel(u, max(v, camera.horizon_row() + 2))
            # Distance; NaN or too far: limit to max_range, no hit
            dist = math.hypot(float(gx), float(gy)) if np.isfinite(gx) else float("inf")
            # The floor ended inside the analysed area and within range
            hit = sc < 0.98 and dist < max_range
            # Limit the ray
            scale = min(1.0, max_range / dist) if dist > 0 else 0.0
            # End point in the robot frame
            ex, ey = float(gx) * scale if np.isfinite(gx) else max_range, float(gy) * scale if np.isfinite(gy) else 0.0
            # World end point
            wx, wy = x + c * ex - s * ey, y + s * ex + c * ey
            # Cells of the ray
            f, end = self.ray_cells(x, y, wx, wy)
            # Collect the free ones
            free |= f
            # And the hit
            if hit:
                # Occupied end
                hits.add(end)
        # Apply the whole frame at once
        self._apply(free, hits)

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
        # Trajectory in blue
        pts = [self.cell(x, y) for x, y in self.path]
        # Draw it
        for a, b in zip(pts, pts[1:]):
            # Segment
            cv2.line(img, a, b, (200, 80, 0), 1)
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
