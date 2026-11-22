"""Targets, sessions and maps of the search task (web process, no torch).

    GET    /api/search/targets                   registered targets with feature counts
    POST   /api/search/targets  (multipart: name, files)   register / add photos (checked)
    DELETE /api/search/targets/<name>
    GET    /api/search/sessions                  past searches
    GET    /api/search/sessions/<id>/<file>      grid.png, found.jpg, trajectory.csv, map.ply,
                                                 cloud.json, summary.json, session.json, reconstruction.json
    POST   /api/search/sessions/<id>/reconstruct  3-D map as a background job
    POST   /api/search/sessions/<id>/delete
    GET    /search/map/<id>                      map viewer page

Python 3.6 compatible (runs on the Jetson Nano).
"""
import json
import os
import re
import shutil

import tornado.web
from tornado.ioloop import IOLoop

# Server base handler and API decorator
from studio.server import BaseHandler, api_auth
# Studio paths
from studio import config, REPO_DIR
# Python interpreter for jobs
from studio.jobs import PYTHON

# Folder of this task (viewer template)
HERE = os.path.dirname(os.path.abspath(__file__))
# Safe target names and session ids
NAME = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")
# Session files the API serves, with their content types
FILES = {"grid.png": "image/png", "found.jpg": "image/jpeg", "trajectory.csv": "text/csv",
         "map.ply": "application/octet-stream", "cloud.json": "application/json", "summary.json": "application/json",
         "session.json": "application/json", "reconstruction.json": "application/json"}


# Folder of a target / session, refusing unsafe names
def _folder(kind, name):
    """``STUDIO_HOME/<kind>/<name>`` for a safe name; HTTP 400 otherwise."""
    # Safe name only
    if not NAME.match(name or ""):
        # Bad request
        raise tornado.web.HTTPError(400, "invalid name")
    # Folder
    return os.path.join(config.home_dir(kind), name)


# Describe a target folder (OpenCV loaded on first use)
def describe_target(folder):
    """Feature counts per photo, difficulty and the cross-check between photos."""
    # Target model helpers
    from studio.tasks.search.target import TargetModel, find
    # Model of all photos
    m = TargetModel.load(folder)
    # Summary
    d = m.describe()
    # Cross-check: every photo must be found by a model of the other photos (needs >= 2 photos)
    cross = []
    # Each photo
    for i, p in enumerate(m.photos):
        # Model without this photo
        others = TargetModel(m.name)
        # Its photos
        others.photos = [q for j, q in enumerate(m.photos) if j != i]
        # Found in the left-out photo?
        cross.append(bool(others.photos) and find(others, p["image"])["hit"])
    # Result per photo (None when there is only one photo)
    d["cross_check"] = cross if len(m.photos) > 1 else None
    # Summary
    return d


# Targets list / registration
class TargetsApi(BaseHandler):
    # Every target
    @api_auth
    async def get(self):
        # Root folder
        root = config.home_dir("targets")
        # Target folders
        names = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
        # Describe each in a worker thread
        out = []
        # Each target
        for n in names:
            # Features and cross-check
            out.append(await IOLoop.current().run_in_executor(None, describe_target, os.path.join(root, n)))
        # Result
        self.write_json({"targets": out})

    # Register photos
    @api_auth
    async def post(self):
        # Target name
        name = self.get_argument("name", "")
        # Its folder (checks the name)
        folder = _folder("targets", name)
        # Uploaded photos
        files = self.request.files.get("files", [])
        # 1-10 photos
        existing = len(os.listdir(folder)) if os.path.isdir(folder) else 0
        # Count check
        if not files or existing + len(files) > 10:
            # Bad request
            raise tornado.web.HTTPError(400, "1-10 photos per target")
        # Create the folder
        os.makedirs(folder, exist_ok=True)
        # Save each photo (decoded and re-encoded: only real pictures are kept)
        import cv2
        # numpy for decoding
        import numpy as np
        # Each upload
        for k, f in enumerate(files):
            # Decode
            img = cv2.imdecode(np.frombuffer(f["body"], np.uint8), cv2.IMREAD_COLOR)
            # Not a picture
            if img is None:
                # Bad request
                raise tornado.web.HTTPError(400, "not a picture: %s" % f["filename"])
            # Save as JPEG
            cv2.imwrite(os.path.join(folder, "%02d.jpg" % (existing + k)), img)
        # Describe the result
        d = await IOLoop.current().run_in_executor(None, describe_target, folder)
        # Result
        self.write_json({"target": d})


# Delete a target
class TargetApi(BaseHandler):
    # Remove its folder
    @api_auth
    def delete(self, name):
        # Folder (checks the name)
        folder = _folder("targets", name)
        # Remove it
        shutil.rmtree(folder, ignore_errors=True)
        # Done
        self.write_json({"deleted": name})


# Read a JSON file or return None
def _json(path):
    # Missing / broken files give None
    try:
        # Read it
        with open(path, encoding="utf-8") as f:
            # Parsed
            return json.load(f)
    # Unusable
    except (OSError, ValueError):
        # Nothing
        return None


# Sessions list
class SessionsApi(BaseHandler):
    # Newest first
    @api_auth
    def get(self):
        # Root folder
        root = config.home_dir("maps")
        # Each session
        out = []
        # Newest first
        for sid in sorted(os.listdir(root), reverse=True):
            # Its folder
            d = os.path.join(root, sid)
            # Folders only
            if not os.path.isdir(d):
                # Skip
                continue
            # Session settings and summary
            s, summ = _json(os.path.join(d, "session.json")) or {}, _json(os.path.join(d, "summary.json")) or {}
            # Row
            out.append({"id": sid, "target": s.get("target"), "started": s.get("started"), "state": summ.get("state"),
                        "found": summ.get("found"), "keyframes": summ.get("keyframes"), "grid": summ.get("grid"),
                        "duration_s": summ.get("duration_s"),
                        "reconstruction": _json(os.path.join(d, "reconstruction.json")),
                        "size_mb": round(config.folder_size(d) / 1e6, 1)})
        # Result
        self.write_json({"sessions": out})


# One session file
class SessionFileApi(BaseHandler):
    # Serve it
    @api_auth
    def get(self, sid, name):
        # Known files only
        if name not in FILES:
            # Not found
            raise tornado.web.HTTPError(404, "unknown file")
        # Path (checks the id)
        path = os.path.join(_folder("maps", sid), name)
        # Missing
        if not os.path.exists(path):
            # Not found
            raise tornado.web.HTTPError(404, "no such file (yet)")
        # Content type
        self.set_header("Content-Type", FILES[name])
        # Downloads keep their name
        if name in ("map.ply", "trajectory.csv"):
            # Save-as name
            self.set_header("Content-Disposition", 'attachment; filename="%s_%s"' % (sid, name))
        # Never cached (files change while a search runs)
        self.set_header("Cache-Control", "no-store")
        # Send it
        with open(path, "rb") as f:
            # Whole file (a few MB at most)
            self.finish(f.read())


# Start the 3-D reconstruction job
class ReconstructApi(BaseHandler):
    # Queue it
    @api_auth
    def post(self, sid):
        # Session folder (checks the id)
        folder = _folder("maps", sid)
        # Needs keyframes
        if not os.path.exists(os.path.join(folder, "keyframes.json")):
            # Conflict
            raise tornado.web.HTTPError(409, "this session has no keyframes")
        # Background job (CPU only; may run while a task drives)
        job = self.studio.manager.jobs.submit("3D map %s" % sid,
                                              [PYTHON, "-m", "studio.tasks.search.reconstruct", folder],
                                              cwd=REPO_DIR, kind="reconstruct", meta={"session": sid},
                                              user=self.current_user)
        # The job
        self.write_json({"job": job.to_dict()})


# Delete a session
class SessionDeleteApi(BaseHandler):
    # Remove its folder
    @api_auth
    def post(self, sid):
        # Folder (checks the id)
        folder = _folder("maps", sid)
        # Remove it
        shutil.rmtree(folder, ignore_errors=True)
        # Done
        self.write_json({"deleted": sid})


# Map viewer page
class MapPage(BaseHandler):
    # Render the viewer for a session
    @tornado.web.authenticated
    def get(self, sid):
        # Check the id
        _folder("maps", sid)
        # Template in the task folder
        self.render(os.path.join(HERE, "map.html"), sid=sid)


# Progress of reconstruction jobs ("keyframe i/n")
def _parse_reconstruct(job, text):
    # Last progress line
    for line in text.splitlines():
        # "keyframe 12/54"
        m = re.match(r"keyframe (\d+)/(\d+)", line)
        # Progress
        if m:
            # Text for the job table
            job.progress["text"] = "keyframe %s/%s" % m.groups()


# Called once by server.make_app
def setup(studio):
    """Register the reconstruction progress parser."""
    # Parser
    studio.manager.jobs.parsers["reconstruct"] = _parse_reconstruct


# Routes of this task
HANDLERS = [
    (r"/api/search/targets", TargetsApi),
    (r"/api/search/targets/([A-Za-z0-9_\-]+)", TargetApi),
    (r"/api/search/sessions", SessionsApi),
    (r"/api/search/sessions/([0-9\-]+)/reconstruct", ReconstructApi),
    (r"/api/search/sessions/([0-9\-]+)/delete", SessionDeleteApi),
    (r"/api/search/sessions/([0-9\-]+)/([a-z_.]+)", SessionFileApi),
    (r"/search/map/([0-9\-]+)", MapPage),
]
