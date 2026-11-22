"""Runner side of lane following: project4's DriveLoop driven from the web page.

* ``DriveLoop`` (project4/drive.py) does camera -> lane marking -> network ->
  controller -> motors -> CSV in its own thread, with its watchdog; the same
  loop as ``live_demo.ipynb``.
* Its results go into its ``LatestSlot``; the runner's frame worker renders
  the newest one only when a page asks for it (no backlog).
* Its events (model loaded, stopped, errors) are forwarded to the page.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import os

import torch

# Base class of task runtimes
from studio.runner import TaskRuntime
# Quotas for CSV recording
from studio.jobs import ResourceMonitor
# Task folders (models, ...)
from studio.tasks.lane_follow.library import folders, model_path, list_models as library_models
# project4 modules (studio.hardware put project4/ on sys.path)
import camera as cam_mod  # noqa: E402
# Steering controller and its settings
from control import SteeringController, DEFAULTS as CTRL  # noqa: E402
# Decoupled inference / control loop
from drive import DriveLoop  # noqa: E402

# Controller attributes set from the task parameters
CONTROL_KEYS = tuple(sorted(CTRL))


# Lane following runtime
class Runtime(TaskRuntime):
    """See the module docstring."""

    # Only starting to drive moves the robot
    MOTION_COMMANDS = ("start_drive",)

    # Create the controller and the loop (threads start in start())
    def __init__(self, ctx):
        # Base: context (the slot is replaced by the loop's)
        TaskRuntime.__init__(self, ctx)
        # CPU threads for torch (0 = keep the default)
        if int(ctx.params.get("torch_threads", 0)) > 0:
            # Fewer threads: steadier control period on small tensors
            torch.set_num_threads(int(ctx.params["torch_threads"]))
        # Camera sensor; the loop uses the underlying project4 source
        sensor = ctx.sensors.get("camera")
        # Source object with read() / flush() / stop(), or None
        camera = sensor.source if sensor is not None else None
        # GPU on the car, CPU elsewhere
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Controller with the defaults; parameters applied below
        self.controller = SteeringController()
        # Model folder: TensorRT on the car, PyTorch on a PC (unless chosen)
        choice = ctx.params.get("model_dir", "auto")
        # Folders of the task
        f = folders(ctx.settings)
        # Resolve "auto"
        self.model_dir = f["trt"] if choice == "trt" or (choice == "auto" and ctx.role == "car") else f["models"]
        # Kind of the models in that folder (for the library summary)
        self.model_kind = "trt" if self.model_dir == f["trt"] else "torch"
        # The project4 loop (CSV logs in STUDIO_HOME/recordings/lane_follow)
        self.loop = DriveLoop(camera, ctx.robot, self.controller, self.device,
                              log_dir=ctx.data_dir("recordings"),
                              flush_on_switch=bool(ctx.params.get("flush_on_switch", True)),
                              timeout=float(ctx.params.get("safety_timeout", 0.5)))
        # Display items come from the loop's mailbox
        self.slot = self.loop.slot
        # Apply the control parameters
        self.update_params(ctx.params)
        # Summary of the loaded model for telemetry
        self.model_summary = None

    # Start the loop threads
    def start(self):
        # Loop and watchdog
        self.loop.start()

    # Live parameter changes
    def update_params(self, values):
        # Remember them
        self.ctx.params.update(values)
        # Controller settings
        for key in CONTROL_KEYS:
            # Only given ones
            if key in values:
                # Set the attribute (the loop reads it every step)
                setattr(self.controller, key, values[key])
        # Watchdog timeout of the loop
        if "safety_timeout" in values:
            # New timeout
            self.loop.timeout = float(values["safety_timeout"])

    # Commands from the page
    def command(self, name, args):
        # Models the page can choose from
        if name == "list_models":
            # Summaries of the folder
            return {"folder": self.model_dir, "models": library_models(self.model_dir, self.model_kind)}
        # Load a model (the loop thread loads, checks and activates it)
        if name == "load_model":
            # Checked path inside the model folder
            path = model_path(self.model_dir, args.get("name", ""))
            # Must exist
            if not os.path.exists(path):
                # Refuse
                raise ValueError("no such model: %s" % args.get("name"))
            # Request it
            self.loop.request_model(path)
            # Accepted
            return {"loading": os.path.basename(path)}
        # Start driving
        if name == "start_drive":
            # Ask the loop (refuses without camera / model / passed sanity)
            ok, why = self.loop.start_drive()
            # Refused
            if not ok:
                # Reason code for the page
                raise ValueError(why)
            # Driving
            return {"driving": True}
        # Stop driving
        if name == "stop_drive":
            # Normal stop
            self.loop.stop_drive("manual")
            # Standing
            return {"driving": False}
        # CSV recording on / off
        if name == "record":
            # Switching on: check the recordings quota first
            if args.get("on"):
                # About 1 MB per minute of driving: ask for 50 MB of room
                ok, msg = ResourceMonitor().quota_ok("recordings", extra_mb=50)
                # Full
                if not ok:
                    # Refuse with the reason
                    raise ValueError(msg)
            # Open / close the file
            return {"path": self.loop.set_recording(bool(args.get("on")))}
        # Latency of every model on the same frame (live_demo's benchmark)
        if name == "benchmark":
            # Rows sorted fastest first
            return {"rows": self._benchmark(int(args.get("n", 100)), int(args.get("warmup", 20)))}
        # Anything else
        raise ValueError("unknown command %r" % name)

    # Time every model of the folder on one frame, with the robot standing still
    def _benchmark(self, n, warmup):
        import time
        import numpy as np
        # project4 model loader and lane marking
        from loader import load_model
        # Lane marking of the loop
        from preprocessing import apply_frame as mark_lanes
        # Stop the robot and pause the loop (raises if it does not answer)
        self.loop.pause()
        # Always resume
        try:
            # One frame for every model (grey without a camera)
            frame = self.loop.camera.read() if self.loop.camera is not None else None
            # Fallback frame
            if frame is None:
                # Uniform grey 224x224 BGR
                frame = np.full((224, 224, 3), 128, np.uint8)
            # Results
            rows = []
            # Every model of the folder
            for info in library_models(self.model_dir, self.model_kind):
                # Load it (errors are reported per model)
                try:
                    # Model on the loop's device
                    m = load_model(os.path.join(self.model_dir, info["name"]), self.device)
                # Report and continue
                except Exception as e:
                    # Error row
                    rows.append({"name": info["name"], "error": repr(e)})
                    # Next model
                    continue
                # Same input path as the loop
                x = cam_mod.preprocess(mark_lanes(frame), self.device, m.dtype)
                # Warm-up runs
                for _ in range(warmup):
                    # Run
                    m(x)
                # Wait for the GPU
                if self.device.type == "cuda":
                    # Synchronise
                    torch.cuda.synchronize()
                # Timer
                t0 = time.time()
                # Timed runs
                for _ in range(n):
                    # Run
                    m(x)
                # Wait for the GPU
                if self.device.type == "cuda":
                    # Synchronise
                    torch.cuda.synchronize()
                # Mean milliseconds
                ms = (time.time() - t0) * 1000.0 / n
                # Row
                rows.append({"name": info["name"], "label": info["label"], "kind": m.kind, "ms": ms, "fps": 1000.0 / ms})
                # Free it before the next one
                del m
        # Resume the loop
        finally:
            # Let it run again
            self.loop.resume()
        # Fastest first, errors last
        return sorted(rows, key=lambda r: r.get("ms", float("inf")))

    # Safety stop (e-stop, link lost, shutdown)
    def stop_motion(self, code, detail=None):
        # The loop stops the motors under its lock
        self.loop.stop_drive(code, detail=detail, fault=code not in ("shutdown", "manual", "task_stop"))

    # Moving while the loop drives
    def is_moving(self):
        # Driving flag of the loop
        return bool(self.loop.driving)

    # One loop result -> JPEG with prediction and steering bar
    def render(self, item, quality):
        # Prediction and control values (None without a model)
        xy, c = item["xy"], item["ctrl"]
        # Draw like live_demo
        img = cam_mod.draw_overlay(item["frame"], xy, c["steering"] if c else None)
        # Encode
        jpeg = cam_mod.to_jpeg(img, int(quality))
        # Readings for the page
        meta = {"driving": item["driving"], "status": item["status"], "model": item["model"],
                "left": item["left"], "right": item["right"]}
        # Prediction and control values when a model runs
        if xy is not None:
            # Normalised target point
            meta["xy"] = [float(xy[0]), float(xy[1])]
            # Angle, steering, speed and PID terms
            meta["ctrl"] = dict((k, float(c[k])) for k in ("angle", "steering", "speed", "p", "i", "d"))
            # Preprocessing + inference time
            meta["latency_ms"] = item.get("latency_ms")
        # JPEG and readings
        return jpeg, meta

    # Forward the loop's events (model objects replaced by summaries)
    def _forward_events(self):
        # Every pending event
        for ev in self.loop.drain_events():
            # Event kind
            kind = ev.pop("kind")
            # The loop's own timestamp is replaced by emit()
            ev.pop("t", None)
            # Loaded model: summary instead of the object
            if kind == "model_loaded":
                # The model object
                m = ev.pop("model")
                # Library summary of its file
                summary = library_models(os.path.dirname(m.path), self.model_kind)
                # Summary of this file
                self.model_summary = next((s for s in summary if s["name"] == os.path.basename(m.path)), None)
                # Runtime description
                ev["describe"] = m.describe()
                # Shown on the page
                ev["summary"] = self.model_summary
                # Drop the big sanity frames
                ev["sanity"] = dict((k, v) for k, v in (ev.get("sanity") or {}).items() if k != "samples")
            # Send it to the page
            self.ctx.emit("drive." + kind, **ev)

    # State for the page
    def telemetry(self):
        # Events first (they change the page state)
        self._forward_events()
        # Loop, model and recording state
        return {"driving": self.loop.driving, "loop": self.loop.stats,
                "model": os.path.basename(self.loop.model.path) if self.loop.model else None,
                "model_loading": self.loop.pending is not None,
                "sanity": dict((k, v) for k, v in (self.loop.sanity or {}).items() if k != "samples"),
                "lanes": self.loop.lanes, "recording": self.loop.csv_path if self.loop._csv else None,
                "camera": self.loop.camera is not None, "camera_error": self.ctx.sensors.errors.get("camera"),
                "robot": self.ctx.robot_info["kind"], "device": str(self.device), "model_dir": self.model_dir}

    # Stop the loop
    def close(self):
        # Motors off, threads ended, CSV closed
        self.loop.close()
