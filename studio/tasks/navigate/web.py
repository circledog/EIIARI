"""Label API of the navigate task (web process, no torch).

    GET  /api/navigate/labels              counts per kind and the folder
    GET  /api/navigate/labels.zip          zip of all labels (train on the PC: unzip into project4/data/)
    POST /api/navigate/labels/clear {"kind"}   delete one kind ("xy", "blocked", "free") or "all"

Python 3.6 compatible (runs on the Jetson Nano).
"""
import os
import shutil
import tempfile
import zipfile

import tornado.web

# Server base handler and API decorator
from studio.server import BaseHandler, api_auth
# Studio paths
from studio import config
# Sample kinds
from studio.tasks.navigate.dataset import KINDS


# Folder of the labels
def label_dir():
    """``STUDIO_HOME/recordings/navigate``."""
    # Same folder as the runner's Context.data_dir("recordings")
    return config.home_dir("recordings", "navigate")


# Counts
class LabelsApi(BaseHandler):
    # Number of pictures per kind
    @api_auth
    def get(self):
        # Folder
        root = label_dir()
        # Count each kind
        counts = dict((k, len([n for n in os.listdir(os.path.join(root, k)) if n.endswith(".jpg")])
                       if os.path.isdir(os.path.join(root, k)) else 0) for k in KINDS)
        # Result
        self.write_json({"folder": root, "counts": counts})


# Zip download
class ZipApi(BaseHandler):
    # Every label file in one zip (folder "navigate_labels/")
    @api_auth
    def get(self):
        # Folder
        root = label_dir()
        # Temporary zip next to the system temp files (may be large)
        fd, tmp = tempfile.mkstemp(suffix=".zip")
        # Close the descriptor; zipfile opens the path
        os.close(fd)
        # Build and send, always removing the temporary file
        try:
            # Stored (JPEGs do not compress)
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
                # Every file below the folder
                for here, _, files in os.walk(root):
                    # Each file
                    for n in files:
                        # Full path
                        p = os.path.join(here, n)
                        # Inside the zip under navigate_labels/
                        z.write(p, os.path.join("navigate_labels", os.path.relpath(p, root)))
            # Download headers
            self.set_header("Content-Type", "application/zip")
            # File name
            self.set_header("Content-Disposition", 'attachment; filename="navigate_labels.zip"')
            # Stream in 1 MB pieces
            with open(tmp, "rb") as f:
                # Until the end
                while True:
                    # Next piece
                    chunk = f.read(1 << 20)
                    # Done
                    if not chunk:
                        # Leave
                        break
                    # Send
                    self.write(chunk)
            # Finish
            self.finish()
        # Clean up
        finally:
            # Remove the temporary zip
            os.remove(tmp)


# Delete labels
class ClearApi(BaseHandler):
    # One kind or all
    @api_auth
    def post(self):
        # Requested kind
        kind = self.json_body().get("kind")
        # Known kinds only
        if kind not in KINDS + ("all",):
            # Bad request
            raise tornado.web.HTTPError(400, "kind must be xy, blocked, free or all")
        # Folder
        root = label_dir()
        # Kinds to delete
        for k in (KINDS if kind == "all" else (kind,)):
            # Remove the folder of the kind
            shutil.rmtree(os.path.join(root, k), ignore_errors=True)
            # Recreate it empty
            os.makedirs(os.path.join(root, k), exist_ok=True)
        # All: also the label lines
        if kind == "all" and os.path.exists(os.path.join(root, "labels.jsonl")):
            # Remove
            os.remove(os.path.join(root, "labels.jsonl"))
        # Done
        self.write_json({"cleared": kind})


# Routes of this task
HANDLERS = [
    (r"/api/navigate/labels", LabelsApi),
    (r"/api/navigate/labels\.zip", ZipApi),
    (r"/api/navigate/labels/clear", ClearApi),
]
