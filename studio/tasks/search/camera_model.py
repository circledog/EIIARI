"""Camera geometry for a robot on a flat floor.

Robot frame: x forward, y left, z up; the camera sits ``height`` metres above
the floor, looks forward and is tilted down by ``pitch_deg``. A pinhole
model with a horizontal field of view ``hfov_deg`` (or a calibration file
from ``calibrate.py`` with K and fisheye distortion D -- the JetBot's
IMX219-160 lens is a fisheye and should be calibrated).

Because the floor is a plane, image pixels below the horizon map 1:1 to floor
points: ``ground_from_pixel`` and the bird's-eye-view warp ``bev()`` give
metric positions without any depth sensor (used by odometry and the map).

Python 3.6 compatible (runs on the Jetson Nano).
"""
import json
import math

import cv2
import numpy as np


# Pinhole (optionally fisheye-undistorted) camera above a floor
class CameraModel(object):
    """See the module docstring."""

    # Build K and the camera pose
    def __init__(self, size=224, hfov_deg=120.0, height=0.10, pitch_deg=20.0, K=None, D=None):
        # Image side in pixels (square frames)
        self.size = int(size)
        # Camera height above the floor (m)
        self.height = float(height)
        # Downward tilt (rad)
        self.pitch = math.radians(float(pitch_deg))
        # Fisheye distortion (None = pinhole frames)
        self.D = np.asarray(D, np.float64).reshape(4, 1) if D is not None else None
        # Intrinsics from a calibration, or from the field of view
        if K is not None:
            # Calibrated matrix
            self.K = np.asarray(K, np.float64).reshape(3, 3)
        # From the field of view
        else:
            # Focal length in pixels
            f = (self.size / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
            # Square pixels, centred principal point
            self.K = np.array([[f, 0, self.size / 2.0], [0, f, self.size / 2.0], [0, 0, 1]], np.float64)
        # Rotation robot -> camera (camera: x right, y down, z forward), tilted down by pitch
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        # Axes of the camera expressed in the robot frame: right = -y, down, forward
        r_level = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], np.float64)
        # Tilt about the camera's x axis (looking down)
        tilt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)
        # Robot -> camera rotation
        self.R = tilt.dot(r_level)
        # Undistortion maps for fisheye frames
        self._maps = None
        # Pinhole matrix used after undistortion
        if self.D is not None:
            # Same K for the undistorted image (keeps the size)
            self._maps = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K,
                                                             (self.size, self.size), cv2.CV_16SC2)

    # Load from a calibration file (calibrate.py) plus mounting values
    @classmethod
    def from_file(cls, path, height, pitch_deg):
        """``{'size', 'K', 'D'}`` JSON -> CameraModel."""
        # Read it
        with open(path, encoding="utf-8") as f:
            # Parsed
            c = json.load(f)
        # Model
        return cls(size=c["size"], height=height, pitch_deg=pitch_deg, K=c["K"], D=c.get("D"))

    # Remove fisheye distortion (no-op for pinhole frames)
    def undistort(self, frame):
        """Pinhole version of ``frame``."""
        # Pinhole frames need nothing
        if self._maps is None:
            # As is
            return frame
        # Remap
        return cv2.remap(frame, self._maps[0], self._maps[1], cv2.INTER_LINEAR)

    # Image row of the horizon (rows below it see the floor)
    def horizon_row(self):
        """Row where the floor meets the horizon (may be negative when tilted far down)."""
        # Horizon direction in the camera frame: robot forward vector
        d = self.R.dot([1.0, 0.0, 0.0])
        # Projected row
        return self.K[1, 1] * d[1] / d[2] + self.K[1, 2]

    # Floor point (robot frame, metres) seen at pixels (u, v)
    def ground_from_pixel(self, u, v):
        """Arrays ``(x, y)``; NaN for pixels at or above the horizon."""
        # Pixel arrays of the same shape (broadcast)
        u, v = np.broadcast_arrays(np.asarray(u, np.float64), np.asarray(v, np.float64))
        # Rays in the camera frame
        rays = np.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], np.ones_like(u)], -1)
        # Rays in the robot frame
        r = rays.dot(self.R)
        # Distance along the ray to the floor (camera at z = height, floor at z = 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Ray parameter
            t = -self.height / r[..., 2]
        # Only rays going down hit the floor
        t = np.where(r[..., 2] < -1e-6, t, np.nan)
        # Floor coordinates
        return t * r[..., 0], t * r[..., 1]

    # Pixel of a floor point (robot frame, metres)
    def pixel_from_ground(self, x, y):
        """Arrays ``(u, v)``; NaN for points behind the camera."""
        # Points relative to the camera
        p = np.stack([np.asarray(x, np.float64), np.asarray(y, np.float64),
                      -self.height * np.ones_like(np.asarray(x, np.float64))], -1)
        # Camera frame
        c = p.dot(self.R.T)
        # In front of the camera only
        z = np.where(c[..., 2] > 1e-6, c[..., 2], np.nan)
        # Projection
        return self.K[0, 0] * c[..., 0] / z + self.K[0, 2], self.K[1, 1] * c[..., 1] / z + self.K[1, 2]

    # Homography image -> bird's-eye view
    def bev_homography(self, x_range=(0.15, 0.75), y_half=0.35, res=0.005):
        """3x3 matrix and BEV size for the floor rectangle (metres) at ``res`` m/pixel.

        BEV pixel (col, row): col grows to the robot's right (-y), row grows towards the robot (-x).
        """
        # BEV size
        w, h = int(round(2 * y_half / res)), int(round((x_range[1] - x_range[0]) / res))
        # Four floor corners: far-left, far-right, near-right, near-left
        xs = np.array([x_range[1], x_range[1], x_range[0], x_range[0]])
        # Their y
        ys = np.array([y_half, -y_half, -y_half, y_half])
        # Their pixels in the image
        u, v = self.pixel_from_ground(xs, ys)
        # Corners in the BEV
        dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        # Homography
        H = cv2.getPerspectiveTransform(np.float32(np.stack([u, v], -1)), dst)
        # Matrix and size
        return H, (w, h)

    # Bird's-eye view of the floor in front of the robot
    def bev(self, frame, x_range=(0.15, 0.75), y_half=0.35, res=0.005):
        """Undistorted, warped floor image (see ``bev_homography``)."""
        # Homography and size (cached per parameters)
        key = (x_range, y_half, res)
        # Build once
        if getattr(self, "_bev_key", None) != key:
            # Cache
            self._bev_key, (self._bev_H, self._bev_size) = key, self.bev_homography(x_range, y_half, res)
        # Warp
        return cv2.warpPerspective(self.undistort(frame), self._bev_H, self._bev_size, flags=cv2.INTER_LINEAR)

    # Sector score (free fraction of the analysed rows) -> metres ahead
    def score_distance(self, horizon_frac):
        """Function ``score -> metres`` for the planner; a fully free sector is infinitely far."""
        # First analysed row
        top = horizon_frac * self.size
        # Lowest row that still sees the floor
        floor_top = self.horizon_row() + 1.0
        # The function
        def distance(score):
            # Free up to the top of the analysed area
            if score >= 0.98:
                # Far
                return float("inf")
            # Row where the floor ends
            v = max(floor_top, self.size - score * (self.size - top))
            # Forward distance of that row (centre column; conservative for side sectors)
            d = float(self.row_distance(v))
            # NaN would mean above the horizon: far
            return d if d == d else float("inf")
        # The function
        return distance

    # Floor distance of an image row in the middle column
    def row_distance(self, v):
        """Forward distance (m) of the floor point at row ``v`` in the centre column (NaN above the horizon)."""
        # Ground point of the centre column
        x, _ = self.ground_from_pixel(self.K[0, 2], v)
        # Forward distance
        return x
