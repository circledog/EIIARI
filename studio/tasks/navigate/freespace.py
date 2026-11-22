"""Free-space estimate without a trained model (TASKS M3-2, zero-shot).

Idea: when the robot starts, the patch right in front of it is floor. Its
colour (HSV hue/saturation histogram) is the floor model; every pixel of the
lower part of the picture gets a floor probability by histogram
back-projection. The lower part is cut into N vertical sectors; in each
sector we walk up from the bottom row until the floor stops, which gives the
sector's free distance in [0, 1] (0 = blocked right in front, 1 = free up to
the horizon).

White tape (low saturation, bright) counts as floor when ``tape_is_floor``
(the track's lane lines are not obstacles).

Why not a depth network first: the Jetson Nano 4 GB shares its RAM with the
GPU and already runs the control model; see ``SPIKE.md``.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import cv2
import numpy as np


# Floor model and sector scores
class FreeSpace(object):
    """``fit(frame)`` once on a clear floor, then ``estimate(frame)`` every frame."""

    # Store the settings
    def __init__(self, sectors=9, horizon=0.45, floor_threshold=0.35, row_fill=0.5, tape_is_floor=True,
                 ref_box=(0.35, 0.85, 0.65, 1.0)):
        # Number of vertical sectors
        self.sectors = int(sectors)
        # Top of the analysed area as a fraction of the height (above = far away / walls)
        self.horizon = float(horizon)
        # Back-projection value (0..1) above which a pixel is floor
        self.floor_threshold = float(floor_threshold)
        # Share of floor pixels a row of a sector needs to count as free
        self.row_fill = float(row_fill)
        # Count white tape as floor
        self.tape_is_floor = bool(tape_is_floor)
        # Reference patch (x0, y0, x1, y1 as fractions): just in front of the robot
        self.ref_box = ref_box
        # Hue/saturation histogram of the floor (None until fit)
        self.hist = None

    # Learn the floor colour from the reference patch
    def fit(self, frame):
        """Build the floor histogram from ``ref_box`` of ``frame`` (BGR)."""
        # Frame size
        h, w = frame.shape[:2]
        # Reference patch in pixels
        x0, y0, x1, y1 = [int(round(f * s)) for f, s in zip(self.ref_box, (w, h, w, h))]
        # HSV of the patch
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        # 2-D histogram over hue (30 bins) and saturation (32 bins)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        # Smooth it so similar colours also count (lighting changes)
        hist = cv2.GaussianBlur(hist, (3, 3), 0)
        # Scale to 0..255 for back-projection
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        # Keep it
        self.hist = hist

    # Floor probability map (0..1) of the analysed area
    def floor_map(self, frame):
        """Float32 map of the rows from the horizon to the bottom."""
        # Fit on the first frame if nobody did
        if self.hist is None:
            # Assume the robot starts on clear floor
            self.fit(frame)
        # Frame height
        h = frame.shape[0]
        # First analysed row
        top = int(self.horizon * h)
        # HSV of the analysed area
        hsv = cv2.cvtColor(frame[top:], cv2.COLOR_BGR2HSV)
        # Back-projection of the floor histogram
        prob = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1).astype(np.float32) / 255.0
        # White tape: low saturation and bright
        if self.tape_is_floor:
            # Tape pixels count as floor
            prob[(hsv[..., 1] < 50) & (hsv[..., 2] > 170)] = 1.0
        # Remove speckles
        prob = cv2.GaussianBlur(prob, (5, 5), 0)
        # Map
        return prob

    # Sector scores
    def estimate(self, frame):
        """Return ``(scores, mask)``: free distance per sector (0..1) and the floor mask (uint8)."""
        # Floor probability
        prob = self.floor_map(frame)
        # Floor mask
        mask = (prob > self.floor_threshold).astype(np.uint8)
        # Rows and columns of the analysed area
        rows, cols = mask.shape
        # Column limits of the sectors
        edges = np.linspace(0, cols, self.sectors + 1).astype(int)
        # Scores per sector
        scores = []
        # Every sector
        for i in range(self.sectors):
            # Share of floor pixels in each row of the sector, bottom row first
            fill = mask[::-1, edges[i]:edges[i + 1]].mean(axis=1)
            # Rows that are not free enough
            blocked = np.nonzero(fill < self.row_fill)[0]
            # Free rows counted from the bottom up to the first blocked one
            free = blocked[0] if len(blocked) else rows
            # Fraction of the analysed height
            scores.append(float(free) / rows)
        # Scores and mask
        return np.asarray(scores, np.float32), mask

    # Draw the sectors on a frame (green = free far, red = blocked near)
    def draw(self, frame, scores, alpha=0.35):
        """Copy of ``frame`` with one translucent bar per sector."""
        # Copy to draw on
        out = frame.copy()
        # Frame size
        h, w = frame.shape[:2]
        # First analysed row
        top = int(self.horizon * h)
        # Column limits
        edges = np.linspace(0, w, len(scores) + 1).astype(int)
        # Overlay layer
        layer = out.copy()
        # Every sector
        for i, s in enumerate(scores):
            # Top row of the free area of this sector
            y = int(h - s * (h - top))
            # Colour from red (0) to green (1)
            color = (0, int(255 * s), int(255 * (1 - s)))
            # Filled free area
            cv2.rectangle(layer, (edges[i], y), (edges[i + 1] - 1, h - 1), color, -1)
        # Blend the layer
        cv2.addWeighted(layer, alpha, out, 1 - alpha, 0, out)
        # Horizon line
        cv2.line(out, (0, top), (w - 1, top), (255, 255, 0), 1)
        # Annotated copy
        return out
