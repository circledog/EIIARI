"""Runner side of "avoid obstacles and drive to a clicked target" (TASKS M3).

One loop thread at camera rate:

    frame -> free-space sectors -> tracker (clicked target) -> planner -> target point
          -> project4 SteeringController -> motors (while driving)

* ``rule`` mode: the planner steers towards the tracked target and around
  blocked sectors. Arrived when the target is low in the picture.
* ``learned`` mode: a network trained on the collected clicks (project4
  format, trained on the PC) predicts the target point instead; the planner's
  model-free stop rule ("obstacle right ahead") still applies.
* Labels (``dataset.py``): every click, every emergency stop (blocked) and
  pictures of undisturbed driving (free).
* Safety: watchdog (no successful step for ``safety_timeout``), target lost,
  obstacle too close, plus the runner's link watchdog and e-stop.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import os
import time

import cv2

# Camera loop, watchdog and stop logic shared with the search task
from studio.camloop import CameraLoopRuntime
# Quota checks for the labels
from studio.jobs import ResourceMonitor
# Model folders of the lane task (the learned model lives there too)
from studio.tasks.lane_follow.library import folders, model_path, list_models as library_models
# Free-space estimate, tracker, planner, labels
from studio.tasks.navigate.freespace import FreeSpace
# Clicked-target tracker
from studio.tasks.navigate.tracker import Tracker
# Sector choice and target point
from studio.tasks.navigate.planner import Planner
# Label files
from studio.tasks.navigate.dataset import LabelStore
# project4 controller (studio.runner put project4/ on sys.path)
from control import SteeringController  # noqa: E402

# Controller settings taken from the task parameters
CONTROL_KEYS = ("max_speed", "min_speed", "kp", "kd", "max_output", "turn_slowdown")


# Navigate runtime
class Runtime(CameraLoopRuntime):
    """See the module docstring."""

    # Event sent when driving stops
    STOPPED_EVENT = "nav.stopped"

    # Only starting to drive moves the robot
    MOTION_COMMANDS = ("start_drive",)

    # Create the parts; the loop starts in start()
    def __init__(self, ctx):
        # Base: camera, lock, driving state, statistics
        CameraLoopRuntime.__init__(self, ctx)
        # Shorter names
        p = ctx.params
        # Free-space estimator
        self.free = FreeSpace(sectors=p["sectors"], horizon=p["horizon"], floor_threshold=p["floor_threshold"],
                              tape_is_floor=p["tape_is_floor"])
        # Clicked-target tracker
        self.tracker = Tracker()
        # Camera geometry: picture rows -> metres on the floor
        from studio.tasks.search.camera_model import CameraModel
        # Pinhole model from the parameters
        self.cam = CameraModel(size=224, hfov_deg=p["hfov_deg"], height=p["cam_height"], pitch_deg=p["cam_pitch"])
        # Planner with the stop rule, thresholds in metres
        self.planner = Planner(free_threshold=p["free_threshold"], stop_distance=p["stop_distance"],
                               horizon=p["horizon"], distance=self.cam.score_distance(p["horizon"]))
        # project4 controller with gentle defaults for indoor driving
        self.controller = SteeringController()
        # Label files with the recordings quota
        mon = ResourceMonitor()
        # Store in STUDIO_HOME/recordings/navigate
        self.labels = LabelStore(ctx.data_dir("recordings"), quota_ok=lambda mb: mon.quota_ok("recordings", mb))
        # Learned model (None until loaded) and whether to mark lanes for it
        self.model, self.model_lanes = None, False
        # Model folder: TensorRT on the car, PyTorch on a PC
        f = folders(ctx.settings)
        # Resolve "auto"
        self.model_dir = f["trt"] if (p["model_dir"] == "trt" or (p["model_dir"] == "auto" and ctx.role == "car")) \
            else f["models"]
        # Kind of the models in that folder
        self.model_kind = "trt" if self.model_dir == f["trt"] else "torch"
        # Newest planner output, controller status, target state
        self.state = {"status": "idle", "sector": None, "scores": [], "target": None}
        # Apply the parameters
        self.update_params(p)

    # ------------------------------------------------------------- params
    # Live parameter changes
    def update_params(self, values):
        # Remember them
        self.ctx.params.update(values)
        # Controller settings
        for key in CONTROL_KEYS:
            # Given ones only
            if key in values:
                # Set the attribute
                setattr(self.controller, key, values[key])
        # Free-space settings
        for key in ("horizon", "floor_threshold", "tape_is_floor"):
            # Given ones only
            if key in values:
                # Set the attribute
                setattr(self.free, key, values[key])
        # Planner settings
        for key in ("free_threshold", "stop_distance", "horizon"):
            # Given ones only
            if key in values:
                # Set the attribute
                setattr(self.planner, key, values[key])
        # A new horizon changes the row -> metres mapping
        if "horizon" in values and hasattr(self, "cam"):
            # New distance function
            self.planner.distance = self.cam.score_distance(values["horizon"])
        # Changing the mode while driving: stop first
        if "mode" in values and self.driving:
            # Normal stop
            self.stop_drive("mode_change")

    # ------------------------------------------------------------ driving
    # Start driving if possible
    def start_drive(self):
        """Raises ValueError with a reason code when refused."""
        # No pictures
        if self.camera is None:
            # Refuse
            raise ValueError("no_camera")
        # Rule mode needs a target
        if self.ctx.params["mode"] == "rule" and not self.tracker.active:
            # Refuse
            raise ValueError("no_target")
        # Learned mode needs a model
        if self.ctx.params["mode"] == "learned" and self.model is None:
            # Refuse
            raise ValueError("no_model")
        # Under the lock
        with self.lock:
            # Fresh controller (motors ramp up from 0)
            self.controller.reset()
            # Watchdog starts now
            self.last_ok = time.monotonic()
            # Driving
            self.driving = True
        # Tell the page
        self.ctx.emit("nav.started", mode=self.ctx.params["mode"])

    # ------------------------------------------------------------- loop
    # Target point from the learned model
    def _predict(self, frame):
        # project4 camera helpers and lane marking
        import camera as cam_mod
        # Lane marking (only if the model was trained with it)
        from preprocessing import apply_frame as mark_lanes
        # Network input
        x = cam_mod.preprocess(mark_lanes(frame) if self.model_lanes else frame, self.model.device, self.model.dtype)
        # Prediction as two floats
        xy = self.model(x).detach().float().cpu().numpy().flatten()
        # Normalised point
        return float(xy[0]), float(xy[1])

    # One step with a new frame (called by the base class loop)
    def step(self, frame):
        # Timings of this step
        t0 = time.monotonic()
        # Free-space sectors
        scores, _ = self.free.estimate(frame)
        # Free-space done
        t1 = time.monotonic()
        # Target state
        target = None
        # Tracking a clicked target
        if self.tracker.active:
            # Follow it
            u, v, score, ok = self.tracker.update(frame)
            # Current target
            target = {"u": u, "v": v, "score": score, "ok": ok}
            # Lost: stop and forget it
            if not ok:
                # Forget the target
                self.tracker.clear()
                # Tell the page
                self.ctx.emit("nav.target_lost", score=round(score, 3))
                # Stop when driving towards it
                if self.driving and self.ctx.params["mode"] == "rule":
                    # Fault stop
                    self.stop_drive("target_lost", fault=True)
        # Tracking done
        t2 = time.monotonic()
        # Mode
        mode = self.ctx.params["mode"]
        # Learned mode: the network chooses the point; the planner's stop rule still applies
        if mode == "learned" and self.model is not None:
            # Planner only for the safety rule
            _, pstatus, sector = self.planner.step(None, scores)
            # Network point unless the obstacle rule says stop
            xy = self._predict(frame) if pstatus != "too_close" else None
            # Status
            status = pstatus if xy is None else "ok"
        # Rule mode: planner towards the tracked target
        else:
            # Target column (None: no target)
            tu = target["u"] if target and target["ok"] else None
            # Plan
            xy, status, sector = self.planner.step(tu, scores)
            # Arrived: target low in the picture
            if target and target["ok"] and target["v"] >= float(self.ctx.params["arrive_v"]):
                # No more driving towards it
                xy, status = None, "arrived"
        # Inference / planning done
        t3 = time.monotonic()
        # Controller step (zero motors and a status when there is no point)
        if xy is not None:
            # project4 controller
            left, right, cstatus = self.controller.update(xy, t3)
        # No point to drive to
        else:
            # Zero motors
            left, right, cstatus = 0.0, 0.0, status
        # Send the motors / stop under the lock
        with self.lock:
            # Driving state now
            driving = self.driving
            # Driving and everything fine: motors
            if driving and cstatus == "ok":
                # Send
                self.ctx.robot.left_motor.value, self.ctx.robot.right_motor.value = left, right
                # Watchdog sees a good step
                self.last_ok = time.monotonic()
        # Driving but a stop condition: arrived is a normal stop, the rest are faults
        if driving and cstatus != "ok":
            # Stop with the reason
            self.stop_drive(cstatus, fault=cstatus not in ("arrived",))
            # Arrived: forget the target
            if cstatus == "arrived":
                # Clear it
                self.tracker.clear()
        # Undisturbed driving: a 'free' label now and then
        if driving and cstatus == "ok" and self.ctx.params["record_free"]:
            # Rate-limited inside
            self.labels.free(frame, mode=mode)
        # State for telemetry / display
        self.state = {"status": cstatus, "sector": sector, "scores": [round(float(s), 3) for s in scores],
                      "target": target, "xy": list(xy) if xy is not None else None, "left": left if driving else 0.0,
                      "right": right if driving else 0.0, "driving": self.driving}
        # Display item (newest only)
        self.slot.put(dict(self.state, frame=frame, scores_np=scores, steering=self.controller.last["steering"]))
        # Stage times
        return {"free": t1 - t0, "track": t2 - t1, "plan": t3 - t2, "control": time.monotonic() - t3}

    # ---------------------------------------------------------- commands
    # Commands from the page
    def command(self, name, args):
        # Click in the picture: track it and store a label
        if name == "set_target":
            # Need a frame
            if self.frame is None:
                # Refuse
                raise ValueError("no_frame")
            # Clicked position (fractions of the picture)
            u, v = float(args["u"]), float(args["v"])
            # Outside the picture
            if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                # Refuse
                raise ValueError("click outside the picture")
            # Frame of the click
            frame = self.frame
            # Track it
            self.tracker.start(frame, u, v)
            # Store the label (picture + target), in project4's format
            path = self.labels.click(frame, u, v, mode=self.ctx.params["mode"], scores=self.state.get("scores"),
                                     driving=self.driving)
            # Result (label refused when the quota is full)
            return {"u": u, "v": v, "label": os.path.basename(path) if path else None, "refused": self.labels.refused}
        # Forget the target
        if name == "clear_target":
            # Clear
            self.tracker.clear()
            # Stop when driving towards it
            if self.driving and self.ctx.params["mode"] == "rule":
                # Normal stop
                self.stop_drive("target_cleared")
            # Done
            return {}
        # Learn the floor colour again from the patch in front of the robot
        if name == "refit_floor":
            # Need a frame
            if self.frame is None:
                # Refuse
                raise ValueError("no_frame")
            # Fit
            self.free.fit(self.frame)
            # Done
            return {}
        # Start / stop driving
        if name == "start_drive":
            # Raises with a reason code when refused
            self.start_drive()
            # Driving
            return {"driving": True}
        # Stop
        if name == "stop_drive":
            # Normal stop
            self.stop_drive("manual")
            # Standing
            return {"driving": False}
        # Models for the learned mode
        if name == "list_models":
            # Summaries
            return {"folder": self.model_dir, "models": library_models(self.model_dir, self.model_kind)}
        # Load the learned model (here in the command thread; the loop keeps running)
        if name == "load_model":
            # Checked path
            path = model_path(self.model_dir, args.get("name", ""))
            # project4 loader and sanity check
            from loader import load_model
            # Sanity comparison
            import sanity
            # Stop before switching
            if self.driving:
                # Normal stop
                self.stop_drive("model_switch")
            # Load
            m = load_model(path)
            # Check
            result = sanity.check(m)
            # Refuse a model that failed its check
            if result["status"] == "failed":
                # Reason
                raise ValueError("sanity check failed: %s" % result["message"])
            # Lane marking only when trained with it
            self.model_lanes = ((m.meta or {}).get("preprocessing") or {}).get("method") == "lane_marking"
            # Active
            self.model = m
            # Summary for the page
            return {"model": os.path.basename(path), "sanity": dict((k, v) for k, v in result.items() if k != "samples"),
                    "lanes": self.model_lanes}
        # Anything else
        raise ValueError("unknown command %r" % name)

    # ----------------------------------------------------------- safety
    # Safety stop from the runner (e-stop, link lost, shutdown)
    def stop_motion(self, code, detail=None):
        # Whether the robot was driving (only then an e-stop means "obstacle ahead")
        was_driving = self.driving
        # Motors off
        self.stop_drive(code, fault=code not in ("shutdown", "task_stop"), detail=detail)
        # Emergency stop while driving = "obstacle ahead" label (TASKS M3 idea)
        if code == "estop" and was_driving and self.frame is not None:
            # Store the picture
            self.labels.blocked(self.frame, scores=self.state.get("scores"), mode=self.ctx.params["mode"])

    # Display item -> JPEG with sectors, target and steering
    def render(self, item, quality):
        # Sectors on the picture
        img = self.free.draw(item["frame"], item["scores_np"])
        # Frame size
        h, w = img.shape[:2]
        # Target marker: green while tracked
        t = item.get("target")
        # Only when a target exists
        if t and t["u"] is not None:
            # Pixel position
            px, py = int(t["u"] * w), int(t["v"] * h)
            # Circle and cross
            cv2.circle(img, (px, py), 10, (0, 255, 0) if t["ok"] else (0, 0, 255), 2)
            # Cross
            cv2.drawMarker(img, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 8, 1)
        # Planned point and line from the robot
        if item.get("xy"):
            # Pixel position of the point
            qx, qy = int((item["xy"][0] + 1) * w / 2), int((item["xy"][1] + 1) * h / 2)
            # Line from the bottom centre
            cv2.line(img, (w // 2, h - 1), (qx, qy), (255, 200, 0), 2)
        # Status text
        cv2.putText(img, item["status"], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # Encode
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        # JPEG and readings
        return (bytes(buf) if ok else b""), {"status": item["status"], "target": item.get("target"),
                                              "xy": item.get("xy"), "driving": item["driving"]}

    # State for the page
    def telemetry(self):
        # Loop, planner, labels
        return {"driving": self.driving, "mode": self.ctx.params["mode"], "loop": self.stats,
                "state": dict((k, v) for k, v in self.state.items()), "labels": self.labels.counts(),
                "labels_refused": self.labels.refused, "model": os.path.basename(self.model.path) if self.model else None,
                "camera": self.camera is not None, "camera_error": self.ctx.sensors.errors.get("camera"),
                "robot": self.ctx.robot_info["kind"]}
