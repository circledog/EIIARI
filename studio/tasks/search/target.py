"""Target object: registration from photos and detection in camera frames (TASKS M4-2 / M4-3).

Registration (1-10 photos): ORB features + a hue/saturation histogram per
photo; photos with few features are flagged as hard to recognise.

Detection per frame, cheap first:

1. colour check: share of frame pixels whose hue/saturation occurs in the
   target's photos (histogram back-projection; independent of how much
   background is around the object) -- skips frames without the target's
   colours at almost no cost;
2. ORB matching with the ratio test, then a RANSAC homography; enough inliers
   and a sane (convex, not degenerate) projected outline = a hit;
3. ``Detector``: only ``confirm`` hits in a row count as "found" (no false alarms
   from a single lucky frame).

Python 3.6 compatible (runs on the Jetson Nano).
"""
import glob
import os

import cv2
import numpy as np

# Photos with fewer ORB features than this are hard to recognise
MIN_FEATURES = 60
# Longest side photos are resized to
PHOTO_SIDE = 320


# Hue/saturation histogram scaled to 0..255 (for back-projection)
def hs_hist(img):
    """30 x 32 histogram of an image (BGR), maximum 255."""
    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Histogram over hue and saturation
    h = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    # Scale so the most common colour is 255
    return cv2.normalize(h, h, 0, 255, cv2.NORM_MINMAX)


# Share of frame pixels with a colour of the target
def colour_share(frame_hsv, hist, level=8):
    """Fraction (0..1) of pixels whose back-projection reaches ``level``."""
    # Back-projection of the target histogram
    bp = cv2.calcBackProject([frame_hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    # Share of pixels with a target colour
    return float((bp >= level).mean())


# Features of the target photos
class TargetModel(object):
    """``from_images(list)`` / ``load(folder)``; ``photos`` with keypoints, descriptors, histogram."""

    # Empty model
    def __init__(self, name="target"):
        # Target name
        self.name = name
        # One dict per photo: image, kp, des, hist, size
        self.photos = []
        # Shared detector, sensitive enough for small (80-160 px) objects in 224 px frames
        self.orb = cv2.ORB_create(nfeatures=1000, fastThreshold=10, edgeThreshold=15, patchSize=15)

    # Add one photo
    def add(self, img):
        """Compute and store the photo's features; returns its info dict."""
        # Resize so the longest side is PHOTO_SIDE (small photos are enlarged: more features)
        scale = PHOTO_SIDE / float(max(img.shape[:2]))
        # Resize with the right filter
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        # Features on the grey image
        kp, des = self.orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
        # Photo record
        rec = {"image": img, "kp": kp, "des": des, "hist": hs_hist(img), "size": img.shape[1::-1],
               "features": len(kp), "hard": len(kp) < MIN_FEATURES}
        # Keep it
        self.photos.append(rec)
        # Its record
        return rec

    # Build from a list of images
    @classmethod
    def from_images(cls, images, name="target"):
        """Model of all images."""
        # New model
        m = cls(name)
        # Each image
        for img in images:
            # Add it
            m.add(img)
        # The model
        return m

    # Load every photo of a folder
    @classmethod
    def load(cls, folder):
        """Model of ``folder/*.jpg|png`` (name = folder name)."""
        # New model
        m = cls(os.path.basename(folder.rstrip(os.sep)))
        # Every photo
        for p in sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))):
            # Read it
            img = cv2.imread(p)
            # Readable only
            if img is not None:
                # Add it
                m.add(img)
        # The model
        return m

    # Summary for the page
    def describe(self):
        """Feature counts and the difficulty warning."""
        # Per photo
        return {"name": self.name, "photos": [{"features": p["features"], "hard": p["hard"]} for p in self.photos],
                "hard": bool(self.photos) and all(p["hard"] for p in self.photos)}


# Find the target in one frame
def find(model, frame, hist_threshold=0.03, ratio=0.75, min_inliers=10, frame_feats=None):
    """Return ``{'hit', 'score', 'inliers', 'box', 'photo'}`` for the best photo.

    ``box``: the photo's outline projected into the frame (4 points), when hit.
    """
    # No photos
    if not model.photos:
        # Nothing to find
        return {"hit": False, "score": 0.0, "inliers": 0, "box": None, "photo": None}
    # Colour check: share of pixels with a target colour (best photo)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Best share over the photos
    score = max(colour_share(hsv, p["hist"]) for p in model.photos)
    # Clearly not there
    if score < hist_threshold:
        # Skip the expensive part
        return {"hit": False, "score": score, "inliers": 0, "box": None, "photo": None}
    # Frame features (computed once per frame)
    kp, des = frame_feats if frame_feats is not None else model.orb.detectAndCompute(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    # No features
    if des is None or len(kp) < min_inliers:
        # Not found
        return {"hit": False, "score": score, "inliers": 0, "box": None, "photo": None}
    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # Best result over the photos
    best = {"hit": False, "score": score, "inliers": 0, "box": None, "photo": None}
    # Every photo
    for i, p in enumerate(model.photos):
        # Photo without features
        if p["des"] is None or len(p["kp"]) < 2:
            # Skip
            continue
        # Two nearest neighbours per photo feature
        pairs = bf.knnMatch(p["des"], des, k=2)
        # Ratio test
        good = [m for m, n in (q for q in pairs if len(q) == 2) if m.distance < ratio * n.distance]
        # Too few
        if len(good) < min_inliers:
            # Next photo
            continue
        # Matched points
        src = np.float32([p["kp"][m.queryIdx].pt for m in good])
        # In the frame
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        # Homography photo -> frame
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        # Inliers
        inl = int(mask.sum()) if mask is not None else 0
        # Not better than the best so far
        if H is None or inl < max(min_inliers, best["inliers"]):
            # Next photo
            continue
        # Photo outline in the frame
        w, h = p["size"]
        # Corners
        box = cv2.perspectiveTransform(np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]]), H).reshape(4, 2)
        # Sane outline: convex and not tiny / not huge
        area = cv2.contourArea(box)
        # Accept
        if cv2.isContourConvex(box.astype(np.float32)) and 64 < area < 4.0 * frame.shape[0] * frame.shape[1]:
            # New best
            best = {"hit": True, "score": score, "inliers": inl, "box": box.tolist(), "photo": i}
    # Best result
    return best


# Hits in a row -> found
class Detector(object):
    """``step(frame)`` -> result of ``find`` plus ``streak`` and ``found``."""

    # Model and confirmation count
    def __init__(self, model, confirm=3, **find_args):
        # Target model
        self.model = model
        # Hits in a row needed
        self.confirm = int(confirm)
        # Extra arguments of find()
        self.find_args = find_args
        # Current streak
        self.streak = 0

    # One frame
    def step(self, frame):
        """Detection result with ``streak`` and ``found``."""
        # Find in this frame
        r = find(self.model, frame, **self.find_args)
        # Streak of hits
        self.streak = self.streak + 1 if r["hit"] else 0
        # Confirmed after enough hits in a row
        r["streak"], r["found"] = self.streak, self.streak >= self.confirm
        # Result
        return r
