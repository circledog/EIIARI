"""Lane following task (TASKS M2): library, training command, pages / APIs, real runner."""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

from tornado import gen
from tornado.ioloop import IOLoop
from tornado.testing import gen_test

# Test helpers (temporary STUDIO_HOME, login, WebSockets)
from _base import StudioTestCase, REPO
# Message reader that never loses a message
from test_runner import Reader, parse_frame
# project4 on the import path (train.py, control.py, test helpers)
from studio import PROJECT4_DIR
# Task helpers under test
from studio.tasks.lane_follow import library
# Task declaration
from studio.tasks.lane_follow import TASK
# Job queue for the end-to-end training run
from studio import jobs

# project4 and its test helpers importable
for p in (PROJECT4_DIR, os.path.join(PROJECT4_DIR, "tests")):
    # Add once
    if p not in sys.path:
        # Put it first
        sys.path.insert(0, p)

# Synthetic labelled images (project4/tests/_helpers.py)
from _helpers import make_dataset  # noqa: E402

# Temporary folders shared by the tests in this module
TMP = tempfile.mkdtemp(prefix="studio-lane-")
# Data root with one flat and one split dataset
DATA = os.path.join(TMP, "data")
# PyTorch and TensorRT model folders
MODELS, TRT = os.path.join(TMP, "models"), os.path.join(TMP, "models_trt")
# Flat dataset (10 images)
make_dataset(os.path.join(DATA, "flat"), n=10, seed=1)
# Split dataset (train 8 / validation 3 / test 3)
for _split, _n in (("train", 8), ("validation", 3), ("test", 3)):
    # One folder per split
    make_dataset(os.path.join(DATA, "teamsplit", _split), n=_n, seed=len(_split))
# Model folders
os.makedirs(MODELS)
# TensorRT folder (empty: no TensorRT here)
os.makedirs(TRT)


# A constant-output ResNet-18 checkpoint in MODELS (no downloads)
def make_model(name="const.pth"):
    """Write ``name`` + .json whose prediction is always (0.1, -0.3)."""
    import torch
    # project4 model builder and metadata writer
    from loader import build_model, write_meta
    # Small ResNet-18 without ImageNet weights
    net = build_model("resnet18", pretrained=False)
    # Constant prediction the controller accepts
    with torch.no_grad():
        # No dependence on the input
        net.fc.weight.zero_()
        # Fixed target point
        net.fc.bias.copy_(torch.tensor([0.1, -0.3]))
    # Checkpoint path
    path = os.path.join(MODELS, name)
    # Save
    torch.save(net.state_dict(), path)
    # Metadata like train.py's (no sanity frames -> 'missing', driving allowed)
    write_meta(path, {"label": "const test model", "arch": "resnet18", "preprocessing": {"method": "lane_marking"},
                      "val_metrics": {"mean_px": 12.5}, "test_metrics": {"mean_px": 13.0, "p95_px": 30.0,
                                                                        "mae_x_px": 5, "mae_y_px": 9}})
    # The path
    return path


# Library functions without HTTP
class LibraryTest(unittest.TestCase):
    # Both dataset layouts are found with counts
    def test_find_datasets(self):
        # Datasets of the test data root
        found = dict((d["id"], d) for d in library.find_datasets(DATA))
        # Both
        self.assertEqual(set(found), {"flat", "teamsplit"})
        # Flat count
        self.assertEqual(found["flat"]["counts"], {"all": 10})
        # Split counts
        self.assertEqual(found["teamsplit"]["counts"], {"train": 8, "validation": 3, "test": 3})
        # Paging and labels
        total, items = library.list_images(found["teamsplit"], "train", 2, 3)
        # Total and page size
        self.assertEqual((total, len(items)), (8, 3))
        # Label parsed from the name
        self.assertEqual((items[0]["x"], items[0]["y"]), library.parse_label(items[0]["name"]))
        # Unknown ids never resolve (no free paths)
        self.assertIsNone(library.get_dataset(DATA, "../etc"))

    # Training form defaults equal train.py's and the networks' DEFAULTS
    def test_training_defaults_match(self):
        # project4 training entry point and networks
        import train
        # Network defaults
        import resnet18
        # MobileNetV2 defaults
        import mobilenet_v2
        # train.py defaults
        d = vars(train.build_parser().parse_args([]))
        # Form defaults
        f = dict((p["name"], p["default"]) for p in library.TRAIN_SCHEMA)
        # Same as train.py's own defaults
        for key in ("batch_size", "val_ratio", "split", "split_seed", "workers", "loss"):
            # Equal
            self.assertEqual(f[key], d[key], key)
        # Same as both networks' DEFAULTS
        for net in (resnet18, mobilenet_v2):
            # epochs, lr, finetune
            for key in ("epochs", "lr", "finetune_epochs", "finetune_lr"):
                # Equal
                self.assertEqual(f[key], net.DEFAULTS[key], (net.__name__, key))
        # Network choices are exactly loader.ARCHS
        from loader import ARCHS
        # Same set
        self.assertEqual(set(a for a, _ in library.ARCHS), set(ARCHS))

    # Every form option arrives unchanged in train.py (TASKS M2-3)
    def test_command_passes_every_option(self):
        # project4 training entry point (its parser)
        import train
        # Validation helpers
        from studio import params
        # Non-default values for every option
        values, errors = params.validate(library.TRAIN_SCHEMA, {
            "dataset": "flat", "split": "session", "val_ratio": 0.2, "split_seed": 7, "arch": "resnet18_dsnt",
            "lanes": False, "augment": True, "loss": "huber", "epochs": 33, "lr": 0.0005, "batch_size": 8,
            "early_stop": True, "seeds": "17 29 43", "workers": 2, "prune": "0.3 0.5", "finetune_epochs": 4,
            "finetune_lr": 0.00002, "onnx": True, "name": "exp_1"})
        # Valid
        self.assertEqual(errors, {})
        # Extra checks pass
        self.assertEqual(library.check_training(values, DATA)[1], {})
        # Command line
        cmd = library.build_train_cmd(values, "/data/x", "python3", out_dir="/out")
        # Parsed by train.py's own parser
        a = train.build_parser().parse_args(cmd[2:])
        # Every option as chosen
        self.assertEqual((a.data, a.out, a.arch, a.split, a.val_ratio, a.split_seed, a.no_lanes, a.augment, a.loss,
                          a.epochs, a.lr, a.batch_size, a.early_stop, a.seeds, a.workers, a.prune, a.finetune_epochs,
                          a.finetune_lr, a.onnx, a.name),
                         ("/data/x", "/out", "resnet18_dsnt", "session", 0.2, 7, True, True, "huber", 33, 0.0005, 8,
                          True, [17, 29, 43], 2, [0.3, 0.5], 4, 0.00002, True, "exp_1"))
        # Bad seeds / prune / name / dataset are refused
        _, errs = library.check_training(dict(values, seeds="x", prune="2", name="a/b", dataset="nope"), DATA)
        # All four
        self.assertEqual(set(errs), {"seeds", "prune", "name", "dataset"})

    # Control defaults of the task equal project4/control.py
    def test_control_defaults_match(self):
        # project4 controller defaults
        from control import DEFAULTS
        # Task parameter defaults
        f = dict((p["name"], p["default"]) for p in TASK["params"])
        # Every controller setting
        for key, value in DEFAULTS.items():
            # Equal
            self.assertEqual(f[key], value, key)

    # Log parsing and the out-of-memory retry
    def test_log_parser_and_oom_retry(self):
        # Fake job
        job = jobs.Job(id="x", status="running", cmd=["python3", "train.py", "--batch-size", "16"], meta={})
        # Two epoch lines in training.fit's format plus an OOM line
        library.parse_train_log(job, "[resnet18_s0] epoch   1/10  train 0.39777  val 0.22675 (72.4 px)  lr 0.001  (2.0s) *\n"
                                     "[resnet18_s0] epoch   2/10  train 0.20000  val 0.10000 (40.0 px)  lr 0.001  (4.0s) *\n"
                                     "RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB\n")
        # Progress
        p = job.progress
        # Epoch, best and ETA (8 epochs x mean 3 s)
        self.assertEqual((p["epoch"], p["epochs"], p["best_px"], p["eta_s"]), (2, 10, 40.0, 24))
        # Curve points
        self.assertEqual(len(p["curves"]["resnet18_s0"]), 2)
        # OOM noticed
        self.assertTrue(p["oom"])
        # Failed job: retry with half the batch
        job.status = "failed"
        # New command
        self.assertEqual(library.oom_retry_cmd(job)[-1], "8")
        # After 3 retries: give up
        job.meta["retries"] = 3
        # No retry
        self.assertIsNone(library.oom_retry_cmd(job))


# Training dependencies (PC only: training never runs on the car)
try:
    import torch_pruning  # noqa: F401
    # Available
    HAVE_PRUNING = True
# Missing (e.g. the robot's container)
except ImportError:
    # Skip the real training run
    HAVE_PRUNING = False


# End-to-end training job on the tiny dataset (real train.py, 1 epoch, CPU)
@unittest.skipUnless(HAVE_PRUNING, "torch-pruning not installed (training runs on the PC)")
class TrainingJobTest(unittest.TestCase):
    # Runs train.py through the job queue and parses its progress
    def test_real_training_job(self):
        # Validation helpers
        from studio import params
        # Tiny settings
        values, _ = params.validate(library.TRAIN_SCHEMA, {"dataset": "teamsplit", "epochs": 1, "batch_size": 4,
                                                          "workers": 0, "name": "tiny"})
        # Output in a temporary folder; no ImageNet download in tests
        cmd = library.build_train_cmd(values, os.path.join(DATA, "teamsplit"), sys.executable,
                                      out_dir=os.path.join(TMP, "train_out")) + ["--no-pretrained"]
        # Queue with the parser
        q = jobs.JobQueue()
        # Register the parser
        q.parsers["train"] = library.parse_train_log
        # Submit
        job = q.submit("tiny", cmd, cwd=PROJECT4_DIR, kind="train")
        # Wait (CPU training of 8 images)
        deadline = time.monotonic() + 240
        # Tick until finished
        while job.status not in jobs.FINISHED and time.monotonic() < deadline:
            # Advance
            q.tick()
            # Pause
            time.sleep(0.2)
        # Finished successfully
        self.assertEqual(job.status, "done", q.read_log(job.id)[0][-2000:])
        # Progress parsed from the real log
        self.assertEqual(job.progress.get("epoch"), 1)
        # Model written
        self.assertTrue(os.path.exists(os.path.join(TMP, "train_out", "tiny_base.pth")))


# Pages and APIs
class LaneHttpTest(StudioTestCase):
    # Test folders and the PC role
    CONFIG = {"lane_data_dir": DATA, "lane_models_dir": MODELS, "lane_trt_dir": TRT}

    # Datasets, thumbnails, check
    def test_datasets_api(self):
        # Log in
        self.login()
        # List
        status, body = self.api("GET", "/api/lane/datasets")
        # Both datasets
        self.assertEqual(sorted(d["id"] for d in body["datasets"]), ["flat", "teamsplit"])
        # A page
        status, body = self.api("GET", "/api/lane/datasets/images?ds=teamsplit&split=validation")
        # Three images
        self.assertEqual(body["total"], 3)
        # Thumbnail of the first, lane-marked
        name = body["items"][0]["name"]
        # Fetch it
        r = self.fetch("/api/lane/image?ds=teamsplit&split=validation&mode=lanes&size=160&name=" + name,
                       headers={"Cookie": self.cookie_header()})
        # JPEG
        self.assertEqual(r.body[:2], b"\xff\xd8")
        # Path tricks are refused
        r = self.fetch("/api/lane/image?ds=teamsplit&split=validation&name=../../config.json",
                       headers={"Cookie": self.cookie_header()}, raise_error=False)
        # Bad request
        self.assertEqual(r.code, 400)
        # Broken file is found by the check
        open(os.path.join(DATA, "flat", "xy_010_020_broken.jpg"), "wb").write(b"\xff\xd8garbage")
        # Check
        status, body = self.api("POST", "/api/lane/datasets/check", {"ds": "flat"})
        # Reported
        self.assertEqual(body["bad"], ["xy_010_020_broken.jpg"])
        # Clean up
        os.remove(os.path.join(DATA, "flat", "xy_010_020_broken.jpg"))
        # Sub-pages render
        for page in ("datasets", "training", "models"):
            # Page
            r = self.fetch("/lane/" + page, headers={"Cookie": self.cookie_header()})
            # OK
            self.assertEqual(r.code, 200, page)

    # Training API: invalid values refused, valid ones queued
    def test_train_api(self):
        # Log in
        self.login()
        # Schema offers the datasets as choices
        status, body = self.api("GET", "/api/lane/train/schema")
        # Choice of the two datasets
        self.assertEqual(body["schema"][0]["type"], "choice")
        # Invalid
        status, body = self.api("POST", "/api/lane/train", {"values": {"dataset": "nope", "epochs": 0}})
        # Refused with field errors
        self.assertEqual(status, 400)
        # Both fields
        self.assertTrue({"dataset", "epochs"} <= set(body["errors"]))
        # Valid
        status, body = self.api("POST", "/api/lane/train", {"values": {"dataset": "flat", "epochs": 1}})
        # Queued / started
        self.assertEqual(status, 200, body)
        # The job
        jid = body["job"]["id"]
        # Command uses the dataset folder and our models folder
        self.assertIn(os.path.join(DATA, "flat"), body["job"]["cmd"])
        # Cancel it (it would download ImageNet weights)
        self.assertTrue(self.api("POST", "/api/jobs/%s/cancel" % jid)[1]["cancelled"])

    # Model library: list, upload, download, delete; conversion refused on a PC
    def test_models_api(self):
        # Log in
        self.login()
        # A checkpoint
        make_model("lib.pth")
        # List
        status, body = self.api("GET", "/api/lane/models")
        # Listed with its metrics
        m = [x for x in body["torch"] if x["name"] == "lib.pth"][0]
        # Test error from the .json
        self.assertEqual(m["test_metrics"]["mean_px"], 13.0)
        # Download
        r = self.fetch("/api/lane/models/file?kind=torch&name=lib.json", headers={"Cookie": self.cookie_header()})
        # The .json
        self.assertIn(b"const test model", r.body)
        # Upload a new .pth / .json pair (multipart)
        boundary = "XyZ"
        # Body with two files
        parts = []
        # Each file
        for name, data in (("up.pth", b"weights"), ("up.json", b'{"label": "uploaded"}')):
            # One part
            parts.append(("--%s\r\nContent-Disposition: form-data; name=\"files\"; filename=\"%s\"\r\n"
                          "Content-Type: application/octet-stream\r\n\r\n" % (boundary, name)).encode() + data + b"\r\n")
        # Whole body
        body_bytes = b"".join(parts) + ("--%s--\r\n" % boundary).encode()
        # Send it
        r = self.fetch("/api/lane/models/upload", method="POST", body=body_bytes, raise_error=False,
                       headers={"Cookie": self.cookie_header(), "X-XSRFToken": self.cookies["_xsrf"],
                                "Content-Type": "multipart/form-data; boundary=%s" % boundary})
        # Saved
        self.assertEqual(json.loads(r.body.decode())["saved"], ["up.pth", "up.json"])
        # On disk
        self.assertTrue(os.path.exists(os.path.join(MODELS, "up.pth")))
        # Delete removes both files
        status, body = self.api("POST", "/api/lane/models/delete", {"kind": "torch", "name": "up.pth"})
        # Both
        self.assertEqual(sorted(body["removed"]), ["up.json", "up.pth"])
        # Unsafe names are refused
        self.assertEqual(self.api("POST", "/api/lane/models/delete", {"kind": "torch", "name": "../x.pth"})[0], 400)
        # TensorRT conversion only on the car
        status, body = self.api("POST", "/api/lane/models/convert", {"name": "lib.pth"})
        # Conflict with the reason
        self.assertEqual(status, 409)
        # Mentions the car
        self.assertIn("car", body["error"])


# Real runner: load a model, drive, frames with predictions
class LaneRunnerTest(StudioTestCase):
    # Test folders, PC role, dummy camera
    CONFIG = {"lane_data_dir": DATA, "lane_models_dir": MODELS, "lane_trt_dir": TRT}

    # Read control messages until one matches
    async def until(self, rd, pred, timeout):
        # Deadline
        deadline = time.monotonic() + timeout
        # Until the deadline
        while time.monotonic() < deadline:
            # Next message
            try:
                # Wait
                raw = await rd.read(max(0.01, deadline - time.monotonic()))
            # Nothing in time
            except gen.TimeoutError:
                # Give up
                return None
            # Closed
            if raw is None:
                # Give up
                return None
            # Parse
            msg = json.loads(raw)
            # Match
            if pred(msg):
                # Found
                return msg
        # Timeout
        return None

    # Send a command and wait for its reply
    async def command(self, conn, rd, cid, name, args=None):
        # Send
        conn.write_message(json.dumps({"type": "command", "id": cid, "name": name, "args": args or {}}))
        # Reply
        return await self.until(rd, lambda m: m.get("type") == "reply" and m.get("id") == cid, 30)

    # The whole drive flow with the PyTorch model on the CPU
    @gen_test(timeout=180)
    async def test_drive_flow(self):
        # A model to drive with
        make_model("drive.pth")
        # Log in
        await self.login_async()
        # Start the task with a 1-thread torch (steadier on a busy PC)
        r = await self.http_client.fetch(self.get_url("/api/tasks/lane_follow/start"), method="POST",
                                         raise_error=False, body=json.dumps({"values": {"torch_threads": 1}}),
                                         headers={"Cookie": self.cookie_header(), "X-XSRFToken": self.cookies["_xsrf"],
                                                  "Content-Type": "application/json"})
        # Started
        self.assertEqual(r.code, 200, r.body)
        # Control socket and heartbeats
        conn = await self.ws("/ws/control/lane_follow")
        # Reader
        rd = Reader(conn)
        # Heartbeat flag
        alive = [1]
        # Heartbeat loop
        async def hb():
            # While wanted
            while alive:
                # Heartbeat
                conn.write_message(json.dumps({"type": "hb"}))
                # Every 0.1 s
                await gen.sleep(0.1)
        # Run it
        IOLoop.current().spawn_callback(hb)
        # Let the socket attach and the runner see heartbeats
        await gen.sleep(0.8)
        # Driving without a model is refused
        reply = await self.command(conn, rd, 1, "start_drive")
        # no_model
        self.assertIn("no_model", reply["error"])
        # Model list
        reply = await self.command(conn, rd, 2, "list_models")
        # Our model is offered (PC: models/ folder)
        self.assertIn("drive.pth", [m["name"] for m in reply["result"]["models"]])
        # Load it
        reply = await self.command(conn, rd, 3, "load_model", {"name": "drive.pth"})
        # Accepted
        self.assertTrue(reply["ok"], reply)
        # Loaded event with the sanity result
        ev = await self.until(rd, lambda m: m.get("type") == "event" and m["event"]["kind"] == "drive.model_loaded", 60)
        # Missing sanity data (allowed to drive)
        self.assertEqual(ev["event"]["sanity"]["status"], "missing")
        # Summary from the .json
        self.assertEqual(ev["event"]["summary"]["label"], "const test model")
        # Drive
        reply = await self.command(conn, rd, 4, "start_drive")
        # Driving
        self.assertTrue(reply["ok"], reply)
        # Telemetry: driving with loop statistics
        tel = await self.until(rd, lambda m: m.get("type") == "telemetry" and m["data"].get("driving")
                               and m["data"]["loop"].get("period_p95_ms"), 10)
        # Got it
        self.assertIsNotNone(tel)
        # Print for the HANDOFF record
        print("\n[lane] loop stats while driving (CPU, ResNet-18): %s"
              % dict((k, round(v, 1)) for k, v in tel["data"]["loop"].items()))
        # A frame with the prediction
        stream = await self.ws("/ws/stream/lane_follow")
        # Reader
        srd = Reader(stream)
        # Frame header
        header, jpeg = parse_frame(await srd.read(10))
        # Constant model output
        self.assertAlmostEqual(header["meta"]["xy"][0], 0.1, places=4)
        # Motors running in the frame's readings
        self.assertGreater(header["meta"]["left"], 0.0)
        # Live parameter: max speed 0 -> motors slow to 0 (slew)
        conn.write_message(json.dumps({"type": "params", "values": {"max_speed": 0.0, "min_speed": 0.0}}))
        # Accepted
        res = await self.until(rd, lambda m: m.get("type") == "params_result", 5)
        # Applied
        self.assertEqual(res["applied"], {"max_speed": 0.0, "min_speed": 0.0})
        # Benchmark: stops the robot, times every model, resumes the loop
        reply = await self.command(conn, rd, 6, "benchmark", {"n": 3, "warmup": 1})
        # Our model timed
        self.assertTrue(reply["ok"], reply)
        # Rows with times
        self.assertTrue(all("ms" in r for r in reply["result"]["rows"]), reply)
        # Stop
        reply = await self.command(conn, rd, 5, "stop_drive")
        # Standing
        self.assertFalse(reply["result"]["driving"])
        # Stopped event (the benchmark stopped it first)
        ev = await self.until(rd, lambda m: m.get("type") == "event" and m["event"]["kind"] == "drive.drive_stopped"
                              and m["event"]["code"] in ("manual", "benchmark"), 5)
        # Got it
        self.assertIsNotNone(ev)
        # End heartbeats
        alive.pop()
        # Close sockets
        stream.close()
        # Control socket
        conn.close()
        # Stop the task
        await self.studio.manager.stop_task()


# Remove the temporary folders at the end
def tearDownModule():
    # Delete everything
    shutil.rmtree(TMP, ignore_errors=True)


# Run directly
if __name__ == "__main__":
    # unittest entry point
    unittest.main()
