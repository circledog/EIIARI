"""Extra pages and APIs of the lane following task (web process, no torch).

Pages: ``/lane/datasets``, ``/lane/training`` (PC), ``/lane/models``.

API::

    GET  /api/lane/datasets                          datasets with counts
    GET  /api/lane/datasets/images?ds=&split=&offset=&limit=
    GET  /api/lane/image?ds=&split=&name=&mode=raw|lanes&size=   JPEG thumbnail
    POST /api/lane/datasets/check {"ds"}             unreadable files (full decode)
    GET  /api/lane/train/schema                      training form + datasets
    POST /api/lane/train {"values"}                  start a training job (PC only)
    GET  /api/lane/models                            models/ and models_trt/
    POST /api/lane/models/delete {"kind", "name"}
    POST /api/lane/models/convert {"name", "method", "force"}   TensorRT job (car only)
    POST /api/lane/models/upload (multipart: .pth / .json / .onnx)
    GET  /api/lane/models/file?kind=&name=           download

Python 3.6 compatible (runs on the Jetson Nano).
"""
import os

import tornado.web
from tornado.ioloop import IOLoop

# Server base handler and API decorator
from studio.server import BaseHandler, api_auth
# Parameter validation
from studio import params as params_mod
# Task helpers (datasets, models, training)
from studio.tasks.lane_follow import library
# project4 folder and studio package folder
from studio import PROJECT4_DIR
# Python interpreter for jobs
from studio.jobs import PYTHON

# Folder of this task (its page templates)
HERE = os.path.dirname(os.path.abspath(__file__))
# Files a model upload may contain
UPLOAD_EXTS = (".pth", ".json", ".onnx")


# Shared helpers of the task's handlers
class LaneHandler(BaseHandler):
    """Folders, dataset lookup, role checks."""

    # Task folders for this server
    @property
    def dirs(self):
        """``library.folders(settings)``."""
        # Same folders the runner uses
        return library.folders(self.studio.settings)

    # Dataset by id or 404
    def dataset(self, ds_id):
        """The dataset dict or HTTP 404."""
        # Look it up among the known ones
        ds = library.get_dataset(self.dirs["data"], ds_id or "")
        # Unknown
        if ds is None:
            # Not found
            raise tornado.web.HTTPError(404, "unknown dataset")
        # Found
        return ds

    # Refuse on the wrong machine
    def need_role(self, role, what):
        """HTTP 409 unless this server has ``role``."""
        # Wrong machine
        if self.studio.role != role:
            # Conflict with the reason
            raise tornado.web.HTTPError(409, "%s only on the %s (this machine is '%s')" % (what, role, self.studio.role))

    # Model folder by kind
    def model_dir(self, kind):
        """``models/`` for 'torch', ``models_trt/`` for 'trt'."""
        # Known kinds only
        if kind not in ("torch", "trt"):
            # Bad request
            raise tornado.web.HTTPError(400, "kind must be torch or trt")
        # The folder
        return self.dirs["models"] if kind == "torch" else self.dirs["trt"]


# Task sub-pages
class PageHandler(LaneHandler):
    # Render one of the task's templates
    @tornado.web.authenticated
    def get(self, page):
        # Only the three pages
        if page not in ("datasets", "training", "models"):
            # Not found
            raise tornado.web.HTTPError(404)
        # Template in the task folder (extends base.html)
        self.render(os.path.join(HERE, page + ".html"), role=self.studio.role)


# Dataset list
class DatasetsApi(LaneHandler):
    # Every dataset with counts
    @api_auth
    def get(self):
        # Datasets below the data root
        self.write_json({"root": self.dirs["data"], "datasets": library.find_datasets(self.dirs["data"])})


# A page of images
class ImagesApi(LaneHandler):
    # Names and labels
    @api_auth
    def get(self):
        # The dataset
        ds = self.dataset(self.get_argument("ds", ""))
        # Requested split
        split = self.get_argument("split", "train" if ds["layout"] == "splits" else "all")
        # Page of images
        try:
            # Total and items
            total, items = library.list_images(ds, split, int(self.get_argument("offset", "0")),
                                               int(self.get_argument("limit", "48")))
        # Unknown split
        except ValueError as e:
            # Bad request
            raise tornado.web.HTTPError(400, str(e))
        # Result
        self.write_json({"total": total, "items": items, "split": split})


# One thumbnail
class ImageApi(LaneHandler):
    # JPEG with label (and lane marking)
    @api_auth
    async def get(self):
        # The dataset
        ds = self.dataset(self.get_argument("ds", ""))
        # File name (must be a label file of that split)
        name = self.get_argument("name", "")
        # Label from the name; refuses anything else
        label = library.parse_label(name)
        # Not a dataset image name (also blocks paths)
        if label is None or os.path.basename(name) != name:
            # Bad request
            raise tornado.web.HTTPError(400, "not a dataset image")
        # Full path
        path = os.path.join(library.split_folder(ds, self.get_argument("split", "all")), name)
        # Missing file
        if not os.path.exists(path):
            # Not found
            raise tornado.web.HTTPError(404, "no such image")
        # Thumbnail renderer (imports OpenCV on first use)
        from studio.tasks.lane_follow import imaging
        # Render in a worker thread (OpenCV releases the GIL)
        jpeg = await IOLoop.current().run_in_executor(
            None, imaging.render, path, self.get_argument("mode", "raw"), label,
            int(self.get_argument("size", "224")))
        # JPEG response, cacheable (images do not change)
        self.set_header("Content-Type", "image/jpeg")
        # Cache for an hour
        self.set_header("Cache-Control", "private, max-age=3600")
        # Send it
        self.finish(jpeg)


# Unreadable files of a dataset
class CheckApi(LaneHandler):
    # Full decode of every image
    @api_auth
    async def post(self):
        # The dataset
        ds = self.dataset(self.json_body().get("ds"))
        # Splits to check
        splits = list(library.SPLITS) if ds["layout"] == "splits" else ["all"]
        # Every labelled file
        paths = []
        # Each split
        for s in splits:
            # Folder of the split
            folder = library.split_folder(ds, s)
            # Labelled files
            if os.path.isdir(folder):
                # Add them
                paths += [os.path.join(folder, n) for n in sorted(os.listdir(folder)) if library.LABEL.match(n)]
        # Checker (imports PIL on first use)
        from studio.tasks.lane_follow import imaging
        # Decode in a worker thread
        bad = await IOLoop.current().run_in_executor(None, imaging.unreadable, paths)
        # Result relative to the dataset
        self.write_json({"checked": len(paths), "bad": [os.path.relpath(p, ds["path"]) for p in bad]})


# Training form
class TrainSchemaApi(LaneHandler):
    # Schema and dataset ids
    @api_auth
    def get(self):
        # Dataset ids
        ids = [d["id"] for d in library.find_datasets(self.dirs["data"])]
        # Form schema: the dataset as a drop-down of the existing datasets
        schema = [dict(p) for p in library.TRAIN_SCHEMA]
        # Replace the free-text dataset field when there are datasets
        if ids:
            # Same name, choice type (the server still validates against the dataset list)
            schema[0] = dict(schema[0], type="choice", choices=[[i, i] for i in ids], default=ids[0])
        # Schema, datasets and whether training is possible here
        self.write_json({"schema": schema, "role": self.studio.role, "datasets": ids})


# Start a training job
class TrainApi(LaneHandler):
    # Validate and queue
    @api_auth
    def post(self):
        # Training only on the PC (user's decision, 2026-11-22)
        self.need_role("pc", "training")
        # Values from the form
        values, errors = params_mod.validate(library.TRAIN_SCHEMA, self.json_body().get("values"))
        # Dataset and extra checks
        ds, more = library.check_training(values, self.dirs["data"])
        # All errors
        errors.update(more)
        # Refuse invalid values
        if errors:
            # Field errors
            self.write_json({"error": "invalid parameters", "errors": errors}, 400)
            # Done
            return
        # train.py command
        cmd = library.build_train_cmd(values, ds["path"], PYTHON, out_dir=self.dirs["models"])
        # Readable name
        name = "train %s %s e%d bs%d" % (values["arch"], ds["id"], values["epochs"], values["batch_size"])
        # Queue it (runs in project4/)
        job = self.studio.manager.jobs.submit(name, cmd, cwd=PROJECT4_DIR, kind="train",
                                              meta={"values": values, "dataset": ds["id"]}, user=self.current_user)
        # The job
        self.write_json({"job": job.to_dict()})


# Model library
class ModelsApi(LaneHandler):
    # Both folders
    @api_auth
    def get(self):
        # PyTorch models and TensorRT models
        self.write_json({"role": self.studio.role,
                         "torch": library.list_models(self.dirs["models"], "torch"),
                         "trt": library.list_models(self.dirs["trt"], "trt"),
                         "folders": {"torch": self.dirs["models"], "trt": self.dirs["trt"]}})


# Delete a model
class DeleteApi(LaneHandler):
    # Remove .pth / .json / .onnx
    @api_auth
    def post(self):
        # Kind and name
        body = self.json_body()
        # Remove the files
        try:
            # Removed file names
            removed = library.delete_model(self.model_dir(body.get("kind")), body.get("name"))
        # Bad name
        except ValueError as e:
            # Bad request
            raise tornado.web.HTTPError(400, str(e))
        # Result
        self.write_json({"removed": removed})


# Convert a model to TensorRT (background job on the car)
class ConvertApi(LaneHandler):
    # Queue convert_trt.py for one model
    @api_auth
    def post(self):
        # TensorRT exists only on the car
        self.need_role("car", "TensorRT conversion")
        # Name, method, force
        body = self.json_body()
        # Method of convert_trt.py
        method = body.get("method", "torch2trt")
        # Only its two methods
        if method not in ("torch2trt", "trtexec"):
            # Bad request
            raise tornado.web.HTTPError(400, "method must be torch2trt or trtexec")
        # Checked source path
        try:
            # Path in models/
            src = library.model_path(self.dirs["models"], body.get("name"))
        # Bad name
        except ValueError as e:
            # Bad request
            raise tornado.web.HTTPError(400, str(e))
        # Must exist
        if not os.path.exists(src):
            # Not found
            raise tornado.web.HTTPError(404, "no such model")
        # Room for the engine (about the size of the .pth)
        ok, msg = self.studio.manager.resources.quota_ok("models", os.path.getsize(src) / 1e6)
        # Full
        if not ok:
            # Conflict with the reason
            raise tornado.web.HTTPError(409, msg)
        # convert_trt.py command (its output folder is models_trt/)
        cmd = [PYTHON, "convert_trt.py", src, "--method", method, "--out", self.dirs["trt"]]
        # Redo an existing conversion
        if body.get("force"):
            # --force
            cmd.append("--force")
        # Needs the GPU and most of the RAM: waits while a task runs
        job = self.studio.manager.jobs.submit("TensorRT %s (%s)" % (os.path.basename(src), method), cmd,
                                              cwd=PROJECT4_DIR, kind="convert", needs_device_free=True,
                                              meta={"model": os.path.basename(src)}, user=self.current_user)
        # The job
        self.write_json({"job": job.to_dict()})


# Upload model files
class UploadApi(LaneHandler):
    # Multipart form with one or more files
    @api_auth
    def post(self):
        # Uploaded files (field name "files")
        files = self.request.files.get("files", [])
        # Nothing uploaded
        if not files:
            # Bad request
            raise tornado.web.HTTPError(400, "no files")
        # Total size in MB
        total = sum(len(f["body"]) for f in files) / 1e6
        # Quota of the models folder
        ok, msg = self.studio.manager.resources.quota_ok("models", total)
        # Full
        if not ok:
            # Conflict with the reason
            raise tornado.web.HTTPError(409, msg)
        # Check every name first (all or nothing)
        for f in files:
            # Plain file name with an allowed extension
            name = f["filename"]
            # Refuse paths and other types
            if not library.SAFE_NAME.match(name) or os.path.splitext(name)[1] not in UPLOAD_EXTS or \
                    name.startswith("."):
                # Bad request
                raise tornado.web.HTTPError(400, "not allowed: %s (only .pth / .json / .onnx)" % name)
        # Target folder (PyTorch models; the car converts them)
        folder = self.dirs["models"]
        # Create it if needed
        os.makedirs(folder, exist_ok=True)
        # Write each file through a temporary name
        for f in files:
            # Final path
            path = os.path.join(folder, f["filename"])
            # Temporary path in the same folder
            tmp = path + ".part"
            # Write the bytes
            with open(tmp, "wb") as out:
                # Body
                out.write(f["body"])
            # Replace in one step
            os.replace(tmp, path)
        # Saved names
        self.write_json({"saved": [f["filename"] for f in files], "folder": folder})


# Download a model file
class FileApi(LaneHandler):
    # Send a .pth / .json / .onnx
    @api_auth
    def get(self):
        # Folder of the kind
        folder = self.model_dir(self.get_argument("kind", "torch"))
        # Requested name
        name = self.get_argument("name", "")
        # Plain name with an allowed extension
        if not library.SAFE_NAME.match(name) or os.path.splitext(name)[1] not in UPLOAD_EXTS or name.startswith("."):
            # Bad request
            raise tornado.web.HTTPError(400, "invalid file name")
        # Full path
        path = os.path.join(folder, name)
        # Missing file
        if not os.path.exists(path):
            # Not found
            raise tornado.web.HTTPError(404, "no such file")
        # Binary download with the file name
        self.set_header("Content-Type", "application/octet-stream")
        # Save-as name
        self.set_header("Content-Disposition", 'attachment; filename="%s"' % name)
        # Stream in 1 MB pieces
        with open(path, "rb") as f:
            # Until the end of the file
            while True:
                # Next piece
                chunk = f.read(1 << 20)
                # Done
                if not chunk:
                    # Leave the loop
                    break
                # Send it
                self.write(chunk)
        # Finish the response
        self.finish()


# Retry a training job with half the batch size after running out of memory
def _retry_oom(queue, job):
    # New command, or None when no retry applies
    cmd = library.oom_retry_cmd(job)
    # No retry
    if cmd is None:
        # Done
        return
    # Old and new batch size
    old, batch = job.cmd[job.cmd.index("--batch-size") + 1], cmd[cmd.index("--batch-size") + 1]
    # Retries so far plus one, and a note for the page
    meta = dict(job.meta, retries=int(job.meta.get("retries", 0)) + 1, retry_of=job.id,
                note="out of memory with batch %s; retried with %s" % (old, batch))
    # Queue the retry
    queue.submit(job.name + " (batch %s)" % batch, cmd, cwd=job.cwd, kind="train", meta=meta, user=job.user)


# Called once by server.make_app
def setup(studio):
    """Register the training progress parser and the out-of-memory retry."""
    # Progress (curves, ETA) from train.py's log
    studio.manager.jobs.parsers["train"] = library.parse_train_log
    # Smaller batch after an out-of-memory failure
    studio.manager.jobs.on_finish["train"] = _retry_oom


# Routes of this task
HANDLERS = [
    (r"/lane/(datasets|training|models)", PageHandler),
    (r"/api/lane/datasets", DatasetsApi),
    (r"/api/lane/datasets/images", ImagesApi),
    (r"/api/lane/datasets/check", CheckApi),
    (r"/api/lane/image", ImageApi),
    (r"/api/lane/train/schema", TrainSchemaApi),
    (r"/api/lane/train", TrainApi),
    (r"/api/lane/models", ModelsApi),
    (r"/api/lane/models/delete", DeleteApi),
    (r"/api/lane/models/convert", ConvertApi),
    (r"/api/lane/models/upload", UploadApi),
    (r"/api/lane/models/file", FileApi),
]
