"""Labels collected while driving (TASKS M3-5), stored in project4's format.

Folder ``STUDIO_HOME/recordings/navigate/``::

    xy/xy_<x>_<y>_<id>.jpg     a click: (picture, target point), 224x224 pixels
                               -> same format as dataset_xy: train.py --data .../xy works as is
    blocked/<id>.jpg           the picture at an emergency stop ("obstacle ahead")
    free/<id>.jpg              pictures while driving without intervention (at most 1 per second)
    labels.jsonl               one line per sample: kind, file, time, target, mode, sector scores

Python 3.6 compatible (runs on the Jetson Nano).
"""
import json
import os
import threading
import time
import uuid

import cv2

# Side of the stored pictures (the networks' input size, like dataset_xy)
SIZE = 224
# Sample kinds and their folders
KINDS = ("xy", "blocked", "free")


# Writes labelled pictures with a quota check
class LabelStore(object):
    """``click(frame, u, v)``, ``blocked(frame)``, ``free(frame)``; ``counts()``."""

    # Folder and quota function
    def __init__(self, folder, quota_ok=None, free_interval=1.0):
        # Root folder of the labels
        self.folder = folder
        # quota_ok(extra_mb) -> (ok, message); None = no check
        self.quota_ok = quota_ok
        # Seconds between two 'free' pictures
        self.free_interval = float(free_interval)
        # Time of the last 'free' picture
        self._last_free = 0.0
        # One writer at a time
        self.lock = threading.Lock()
        # Create the folders
        for k in KINDS:
            # Folder of the kind
            os.makedirs(os.path.join(folder, k), exist_ok=True)
        # Message of the last refused write (quota full)
        self.refused = None

    # Resize to the stored size
    @staticmethod
    def _square(frame):
        # Already the right size
        if frame.shape[0] == SIZE and frame.shape[1] == SIZE:
            # As is
            return frame
        # Resize
        return cv2.resize(frame, (SIZE, SIZE))

    # Write one picture and its label line
    def _write(self, kind, name, frame, extra):
        # Quota check (about 15 kB per picture)
        if self.quota_ok is not None:
            # Ask
            ok, msg = self.quota_ok(0.05)
            # Full: refuse instead of filling the disk
            if not ok:
                # Remember why
                self.refused = msg
                # Not written
                return None
        # Path of the picture
        path = os.path.join(self.folder, kind, name)
        # One writer at a time
        with self.lock:
            # JPEG like the course's collection notebooks
            cv2.imwrite(path, self._square(frame))
            # Label line
            line = dict(extra, kind=kind, file=os.path.join(kind, name), t=time.time())
            # Append it
            with open(os.path.join(self.folder, "labels.jsonl"), "a", encoding="utf-8") as f:
                # ASCII JSON line
                f.write(json.dumps(line) + "\n")
        # Written
        self.refused = None
        # Path
        return path

    # A click: picture + target point
    def click(self, frame, u, v, **extra):
        """Store ``xy_<x>_<y>_<id>.jpg`` with the target in 224x224 pixels."""
        # Pixel coordinates in the stored picture
        x, y = int(round(min(max(u, 0.0), 1.0) * (SIZE - 1))), int(round(min(max(v, 0.0), 1.0) * (SIZE - 1)))
        # dataset_xy file name
        name = "xy_%03d_%03d_%s.jpg" % (x, y, uuid.uuid4())
        # Write it
        return self._write("xy", name, frame, dict(extra, x=x, y=y))

    # An emergency stop: "obstacle ahead"
    def blocked(self, frame, **extra):
        """Store the picture as a blocked sample."""
        # Write it
        return self._write("blocked", "%s.jpg" % uuid.uuid4(), frame, extra)

    # Driving without intervention: "free"
    def free(self, frame, **extra):
        """Store at most one picture per ``free_interval``; returns the path or None."""
        # Too soon after the last one
        if time.monotonic() - self._last_free < self.free_interval:
            # Skipped
            return None
        # Remember the time
        self._last_free = time.monotonic()
        # Write it
        return self._write("free", "%s.jpg" % uuid.uuid4(), frame, extra)

    # Number of samples per kind
    def counts(self):
        """``{'xy': n, 'blocked': n, 'free': n}``."""
        # Files per folder
        return dict((k, len([n for n in os.listdir(os.path.join(self.folder, k)) if n.endswith(".jpg")]))
                    for k in KINDS)
