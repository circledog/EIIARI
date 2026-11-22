"""Follow a clicked point from frame to frame (TASKS M3-3).

Template matching (normalised cross-correlation) in a window around the last
position, at three scales (the target grows as the robot approaches). The
template is updated slowly while the match is good. After ``lost_frames``
bad matches in a row the target counts as lost and the robot must stop.

Coordinates are fractions of the frame (u = x / width, v = y / height), the
same as the page sends after converting the click with the image's real
displayed size.

A match only counts when it is good (``min_score``) **and** clearly better than
the second-best place in the window (``min_margin``): on repetitive texture
an ambiguous match would otherwise drift to a wrong object. The template is
only adapted on strong matches (``update_score``).

Python 3.6 compatible (runs on the Jetson Nano).
"""
import cv2
import numpy as np


# Template tracker for one clicked point
class Tracker(object):
    """``start(frame, u, v)``, then ``update(frame)`` -> ``(u, v, score, ok)``."""

    # Store the settings
    def __init__(self, size=0.16, search=0.3, min_score=0.6, min_margin=0.1, update_score=0.85, lost_frames=5,
                 update_rate=0.1, scales=(0.92, 1.0, 1.08)):
        # Template side as a fraction of the frame width
        self.size = float(size)
        # Search window half-width as a fraction of the frame width
        self.search = float(search)
        # Correlation below this is a bad match
        self.min_score = float(min_score)
        # Best match must beat the second-best peak by this much
        self.min_margin = float(min_margin)
        # Template adapts only when the match is at least this good
        self.update_score = float(update_score)
        # Bad matches in a row that mean "lost"
        self.lost_frames = int(lost_frames)
        # Template blending rate while the match is good
        self.update_rate = float(update_rate)
        # Scales tried around the current template size
        self.scales = scales
        # Template (grey, float32), position in pixels, bad-match count
        self.template, self.pos, self.bad = None, None, 0
        # Last score and its margin over the second-best match
        self.score, self.margin = 0.0, 0.0

    # Whether a target is set
    @property
    def active(self):
        """True after ``start`` until ``clear``."""
        # A template exists
        return self.template is not None

    # Grey float image
    @staticmethod
    def _grey(frame):
        # BGR -> grey -> float32
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Cut a square patch centred on (x, y), clamped to the frame
    @staticmethod
    def _patch(img, x, y, half):
        # Image size
        h, w = img.shape[:2]
        # Centre kept inside the frame
        x, y = int(min(max(x, half), w - half - 1)), int(min(max(y, half), h - half - 1))
        # The patch and its actual centre
        return img[y - half:y + half, x - half:x + half].copy(), (x, y)

    # Start tracking the point (u, v)
    def start(self, frame, u, v):
        """Take the template around the clicked point."""
        # Grey frame
        g = self._grey(frame)
        # Frame size
        h, w = g.shape
        # Half template size in pixels (at least 6)
        self.half = max(6, int(self.size * w / 2))
        # Template and its centre (moved inside the frame if needed)
        self.template, self.pos = self._patch(g, u * w, v * h, self.half)
        # Fresh state
        self.bad, self.score = 0, 1.0

    # Forget the target
    def clear(self):
        """No target."""
        # No template
        self.template, self.pos = None, None

    # Find the target in a new frame
    def update(self, frame):
        """Return ``(u, v, score, ok)``; ``ok`` False once the target is lost."""
        # No target
        if self.template is None:
            # Nothing to track
            return None, None, 0.0, False
        # Grey frame
        g = self._grey(frame)
        # Frame size
        h, w = g.shape
        # Search window around the last position
        r = int(self.search * w)
        # Window limits
        x0, y0 = max(0, self.pos[0] - r), max(0, self.pos[1] - r)
        # Window limits (exclusive)
        x1, y1 = min(w, self.pos[0] + r), min(h, self.pos[1] + r)
        # Window image
        win = g[y0:y1, x0:x1]
        # Best match over the scales: (score, centre, template shape, correlation map, location)
        best = (-1.0, None, None, None, None)
        # Every scale
        for s in self.scales:
            # Scaled template
            t = cv2.resize(self.template, None, fx=s, fy=s) if s != 1.0 else self.template
            # Must fit into the window
            if t.shape[0] >= win.shape[0] or t.shape[1] >= win.shape[1]:
                # Skip this scale
                continue
            # Normalised cross-correlation
            res = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
            # Best position
            _, score, _, loc = cv2.minMaxLoc(res)
            # Keep the best
            if score > best[0]:
                # Score, centre in frame pixels, template size, map and location
                best = (score, (x0 + loc[0] + t.shape[1] // 2, y0 + loc[1] + t.shape[0] // 2), t.shape, res, loc)
        # Best score
        self.score = float(best[0])
        # Second-best peak away from the best one (uniqueness of the match)
        second = -1.0
        # Only when there was a match at all
        if best[3] is not None:
            # Copy of the correlation map
            res = best[3].copy()
            # Blank out the neighbourhood of the best match (one template size)
            hx, hy = best[2][1], best[2][0]
            # Neighbourhood limits
            res[max(0, best[4][1] - hy):best[4][1] + hy + 1, max(0, best[4][0] - hx):best[4][0] + hx + 1] = -1.0
            # Highest remaining peak
            second = float(res.max())
        # Margin of the best over the second-best match
        self.margin = self.score - second
        # Good match: high and unique
        if best[1] is not None and self.score >= self.min_score and self.margin >= self.min_margin:
            # New position
            self.pos = best[1]
            # Reset the bad count
            self.bad = 0
            # Slowly adapt the template (lighting, small rotations), only on strong matches
            patch, _ = self._patch(g, self.pos[0], self.pos[1], self.half)
            # Same size as the template and a strong match
            if patch.shape == self.template.shape and self.score >= self.update_score:
                # Blend
                self.template = (1 - self.update_rate) * self.template + self.update_rate * patch
        # Bad match
        else:
            # Count it
            self.bad += 1
        # Lost after too many bad matches
        ok = self.bad < self.lost_frames
        # Position as fractions
        return self.pos[0] / float(w), self.pos[1] / float(h), self.score, ok
