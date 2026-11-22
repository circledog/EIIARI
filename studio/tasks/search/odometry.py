"""Visual odometry for a robot on a flat floor (TASKS M4-1 / M4-4).

Each frame's floor in front of the robot is warped to a metric bird's-eye
view (``CameraModel.bev``). Between two views the robot moved by a 2-D
rotation + translation, so ORB features of the two views are matched and a
rigid transform is fitted with RANSAC (``cv2.estimateAffinePartial2D``).
The camera height fixes the scale (TASKS: "use the floor plane for the
scale") -- no wheel encoders needed (the robot has none).

Default floor window 0.04-0.40 m ahead, 4 mm/pixel: on the synthetic 5 m
square (tests) it closes with 3 cm / 1.3 deg error; farther windows sit at the
horizon of the wide, tilted camera and drift more (0.15-0.75 m: 20 cm / 12 deg).

When too few features match (blank floor, motion blur) the step is skipped
and ``lost`` is counted; the pose then stays where it was.

Pose: ``(x, y, yaw)`` in metres / radians in the start frame (x forward at the
start, y left).

Python 3.6 compatible (runs on the Jetson Nano).
"""
import math

import cv2
import numpy as np


# Bird's-eye-view odometry
class FloorOdometry(object):
    """``update(frame)`` -> ``(x, y, yaw, ok)``."""

    # Store the camera and settings
    def __init__(self, camera, x_range=(0.04, 0.40), y_half=0.25, res=0.004, min_inliers=12, features=500):
        # Camera geometry
        self.camera = camera
        # Floor rectangle used (m) and BEV resolution (m / pixel)
        self.x_range, self.y_half, self.res = x_range, y_half, res
        # Matches needed for a valid step
        self.min_inliers = int(min_inliers)
        # Feature detector and matcher
        self.orb = cv2.ORB_create(nfeatures=int(features), fastThreshold=7, edgeThreshold=15, patchSize=15)
        # Brute-force Hamming matcher with cross-check
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Previous keypoints and descriptors
        self.prev = None
        # Pose in the start frame
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        # Failed steps in total and inliers of the last step
        self.lost, self.inliers = 0, 0

    # BEV pixel -> robot-frame metres
    def _to_robot(self, pts):
        # Columns grow to the right (-y), rows grow towards the robot (-x)
        x = self.x_range[1] - pts[:, 1] * self.res
        # Lateral
        y = self.y_half - pts[:, 0] * self.res
        # (N, 2) metres
        return np.stack([x, y], -1)

    # One frame
    def update(self, frame):
        """Advance the pose with a new frame; returns ``(x, y, yaw, ok)``."""
        # Grey bird's-eye view
        bev = cv2.cvtColor(self.camera.bev(frame, self.x_range, self.y_half, self.res), cv2.COLOR_BGR2GRAY)
        # Features
        kp, des = self.orb.detectAndCompute(bev, None)
        # Previous view
        prev, self.prev = self.prev, (kp, des)
        # First frame or no features: nothing to compare
        if prev is None or des is None or prev[1] is None or len(kp) < self.min_inliers:
            # Not a valid step (except the very first frame)
            ok = prev is None
            # Count failures
            self.lost += 0 if ok else 1
            # Pose unchanged
            return self.x, self.y, self.yaw, ok
        # Match previous -> current
        matches = self.matcher.match(prev[1], des)
        # Too few
        if len(matches) < self.min_inliers:
            # Failed step
            self.lost += 1
            # Pose unchanged
            return self.x, self.y, self.yaw, False
        # Matched points in robot metres: previous frame and current frame
        a = self._to_robot(np.float32([prev[0][m.queryIdx].pt for m in matches]))
        # Current
        b = self._to_robot(np.float32([kp[m.trainIdx].pt for m in matches]))
        # Rigid (+ scale) transform current -> previous: a point fixed on the floor seen now at b was at a
        M, mask = cv2.estimateAffinePartial2D(b, a, method=cv2.RANSAC, ransacReprojThreshold=0.01, maxIters=500)
        # Inlier count
        self.inliers = int(mask.sum()) if mask is not None else 0
        # Not enough agreement
        if M is None or self.inliers < self.min_inliers:
            # Failed step
            self.lost += 1
            # Pose unchanged
            return self.x, self.y, self.yaw, False
        # Rotation of the robot between the frames (the current frame is rotated by dyaw)
        dyaw = math.atan2(M[1, 0], M[0, 0])
        # Translation of the robot in the previous robot frame
        dx, dy = M[0, 2], M[1, 2]
        # Compose with the global pose
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        # Global position
        self.x, self.y = self.x + c * dx - s * dy, self.y + s * dx + c * dy
        # Global heading, wrapped to [-pi, pi]
        self.yaw = math.atan2(math.sin(self.yaw + dyaw), math.cos(self.yaw + dyaw))
        # Valid step
        return self.x, self.y, self.yaw, True
