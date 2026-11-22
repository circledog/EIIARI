"""Synthetic world for the search tests: a textured floor seen by a CameraModel.

``World.render(x, y, yaw)`` draws the camera picture of a robot at that pose
(pixels above the horizon are grey), so odometry, mapping and detection can be
checked against the exact trajectory.
"""
import math

import cv2
import numpy as np


# Floor texture plus a renderer
class World(object):
    """Square floor of ``side`` metres at ``res`` m/pixel, origin in the middle."""

    # Build the texture
    def __init__(self, camera, side=8.0, res=0.004, seed=0):
        # Camera geometry
        self.camera = camera
        # Resolution and size
        self.res, self.n = res, int(side / res)
        # Reproducible texture
        rng = np.random.RandomState(seed)
        # Multi-scale noise: a few blurred noise layers (like wood / tiles / carpet)
        tex = np.zeros((self.n, self.n), np.float32)
        # Three scales
        for k, (sigma, weight) in enumerate(((1.5, 0.5), (6, 0.8), (25, 1.0))):
            # Blurred noise layer
            tex += weight * cv2.GaussianBlur(rng.normal(0, 1, (self.n, self.n)).astype(np.float32), (0, 0), sigma)
        # Scale to grey values
        tex = cv2.normalize(tex, None, 40, 200, cv2.NORM_MINMAX)
        # Warm floor colour
        self.texture = cv2.merge([tex * 0.7, tex * 0.9, tex]).astype(np.uint8)
        # Floor point of every image pixel (robot frame), computed once
        uu, vv = np.meshgrid(np.arange(camera.size) + 0.5, np.arange(camera.size) + 0.5)
        # Robot-frame floor coordinates (NaN above the horizon)
        self.gx, self.gy = camera.ground_from_pixel(uu, vv)
        # Visible floor only up to 3 m ahead (the far floor is a blur anyway)
        far = ~(self.gx < 3.0)
        # Hide it
        self.gx[far], self.gy[far] = np.nan, np.nan

    # World metres -> texture pixel
    def to_tex(self, X, Y):
        """Column and row in the texture."""
        # Origin in the middle, rows grow towards -Y
        return X / self.res + self.n / 2.0, -Y / self.res + self.n / 2.0

    # Camera picture at a pose
    def render(self, x, y, yaw):
        """BGR picture of the robot at ``(x, y, yaw)``."""
        # Rotation of the robot
        c, s = math.cos(yaw), math.sin(yaw)
        # World floor coordinates of every pixel
        X, Y = x + c * self.gx - s * self.gy, y + s * self.gx + c * self.gy
        # Texture coordinates
        col, row = self.to_tex(X, Y)
        # Pixels without floor sample outside the texture -> border colour
        col, row = np.nan_to_num(col, nan=-10.0).astype(np.float32), np.nan_to_num(row, nan=-10.0).astype(np.float32)
        # Sample the texture
        return cv2.remap(self.texture, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(150, 150, 150))


# Poses of a square drive: straight sides, turning on the spot at the corners
def square_path(side=1.25, step=0.015, turn_deg=3.0):
    """List of ``(x, y, yaw)`` around a square, ending where it started."""
    # Poses
    poses = []
    # Start
    x, y, yaw = 0.0, 0.0, 0.0
    # Four sides
    for _ in range(4):
        # Straight
        for _ in range(int(round(side / step))):
            # Move forward
            x, y = x + step * math.cos(yaw), y + step * math.sin(yaw)
            # Record
            poses.append((x, y, yaw))
        # Turn left 90 degrees
        for _ in range(int(round(90 / turn_deg))):
            # Rotate
            yaw += math.radians(turn_deg)
            # Record
            poses.append((x, y, yaw))
    # All poses
    return poses


# Colourful synthetic poster (the object to search for)
def poster(seed, w=200, h=150):
    """Random shapes and text on a coloured background."""
    # Reproducible
    rng = np.random.RandomState(seed)
    # Background colour
    img = np.full((h, w, 3), [int(c) for c in rng.randint(60, 230, 3)], np.uint8)
    # Random rectangles, circles and lines
    for _ in range(25):
        # Colour
        col = tuple(int(c) for c in rng.randint(0, 255, 3))
        # Shape kind
        k = rng.randint(3)
        # Two random points
        p1, p2 = (int(rng.randint(w)), int(rng.randint(h))), (int(rng.randint(w)), int(rng.randint(h)))
        # Rectangle
        if k == 0:
            # Filled rectangle
            cv2.rectangle(img, p1, p2, col, -1)
        # Circle
        elif k == 1:
            # Filled circle
            cv2.circle(img, p1, int(rng.randint(5, 30)), col, -1)
        # Line
        else:
            # Thick line
            cv2.line(img, p1, p2, col, int(rng.randint(2, 6)))
    # Some text
    cv2.putText(img, "T%d" % seed, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    # Poster
    return img


# Scene with an optional poster under a random perspective
def poster_scene(target, seed, size=224):
    """Textured background with ``target`` (or None) warped into it."""
    # Reproducible
    rng = np.random.RandomState(seed)
    # Background: blurred noise in a random tint
    bg = cv2.GaussianBlur(rng.normal(120, 40, (size, size, 3)).astype(np.float32), (0, 0), 3)
    # Uint8 background
    img = np.clip(bg, 0, 255).astype(np.uint8)
    # No poster
    if target is None:
        # Background only
        return img
    # Poster size
    h, w = target.shape[:2]
    # Width in the scene: 35 % .. 70 % of the frame
    sw = size * rng.uniform(0.35, 0.7)
    # Centre
    cx, cy = rng.uniform(sw / 2, size - sw / 2), rng.uniform(sw / 3, size - sw / 3)
    # Rectangle corners, then perspective jitter (up to 12 % of the width per corner)
    sh = sw * h / w
    # Corners
    dst = np.float32([[cx - sw / 2, cy - sh / 2], [cx + sw / 2, cy - sh / 2], [cx + sw / 2, cy + sh / 2],
                      [cx - sw / 2, cy + sh / 2]]) + rng.uniform(-0.12, 0.12, (4, 2)).astype(np.float32) * sw
    # Homography poster -> scene
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
    # Brightness change
    t = np.clip(target.astype(np.float32) * rng.uniform(0.7, 1.2), 0, 255).astype(np.uint8)
    # Warped poster
    warped = cv2.warpPerspective(t, H, (size, size))
    # Its mask
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (size, size))
    # Paste
    img[mask > 0] = warped[mask > 0]
    # Slight blur (motion / focus)
    return cv2.GaussianBlur(img, (3, 3), 0)
