"""Runner side of "object search + map" (TASKS M4).

States: ``idle`` -> ``exploring`` -> ``found`` | ``done`` (explored / time limit) | ``stopped``.

Every frame (camera rate):

1. floor odometry -> pose (metres, start frame);
2. free-space sectors (M3) on the undistorted frame -> occupancy grid;
3. target detector (colour check, ORB + homography, K hits in a row);
4. while exploring: frontier goal -> view column for M3's planner (obstacle
   avoidance + model-free stop rule) or turning on the spot; blocked goals are
   given up;
5. session files: trajectory, keyframes (for the 3-D map), grid picture,
   found photo with the object's estimated floor position.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import csv
import json
import math
import os
import threading
import time

import cv2
import numpy as np

# Camera loop, watchdog and stop logic shared with the navigate task
from studio.camloop import CameraLoopRuntime
# Studio paths and quota checks
from studio import config
# Quotas
from studio.jobs import ResourceMonitor
# M3 parts
from studio.tasks.navigate.freespace import FreeSpace
# Sector choice and stop rule
from studio.tasks.navigate.planner import Planner
# M4 parts
from studio.tasks.search.camera_model import CameraModel
# Odometry
from studio.tasks.search.odometry import FloorOdometry
# Occupancy grid
from studio.tasks.search.grid_map import GridMap
# Target detection
from studio.tasks.search.target import TargetModel, Detector
# Exploration
from studio.tasks.search.explore import choose_goal, steer_to
# project4 controller (studio.hardware put project4/ on sys.path)
from control import SteeringController  # noqa: E402


# Folder of the registered targets
def targets_dir():
    """``STUDIO_HOME/targets``."""
    # Created when missing
    return config.home_dir("targets")


# Keyframe poses written while the search runs
KEYFRAMES_JSONL = "keyframes.jsonl"


# Search runtime
class Runtime(CameraLoopRuntime):
    """See the module docstring."""

    # Event sent when driving stops
    STOPPED_EVENT = "search.stopped"

    # Starting a search moves the robot
    MOTION_COMMANDS = ("start_search",)

    # Build the parts
    def __init__(self, ctx):
        # Base: camera, lock, driving state, statistics
        CameraLoopRuntime.__init__(self, ctx)
        # Short name
        p = ctx.params
        # Camera geometry: calibration file or field of view
        if p.get("calibration"):
            # Calibrated model
            self.cam = CameraModel.from_file(p["calibration"], p["cam_height"], p["cam_pitch"])
        # Field-of-view model
        else:
            # Pinhole from the settings
            self.cam = CameraModel(size=224, hfov_deg=p["hfov_deg"], height=p["cam_height"], pitch_deg=p["cam_pitch"])
        # M3 free space and planner
        self.free = FreeSpace(sectors=9, horizon=p["horizon"], floor_threshold=p["floor_threshold"])
        # Planner with the stop rule, thresholds in metres (camera model)
        self.planner = Planner(free_threshold=p["free_threshold"], stop_distance=p["stop_distance"], horizon=p["horizon"],
                               distance=self.cam.score_distance(p["horizon"]))
        # project4 controller
        self.controller = SteeringController(max_speed=p["max_speed"], min_speed=p["min_speed"], kp=p["kp"], kd=p["kd"],
                                             max_output=p["max_output"])
        # Quota monitor
        self.mon = ResourceMonitor()
        # Lock around the session files (loop thread vs commands / close)
        self.slock = threading.RLock()
        # State machine
        self.state = "idle"
        # Session parts (created by start_search)
        self.session, self.odo, self.grid, self.detector = None, None, None, None
        # Exploration: goal, visited goals, turning direction and until when
        self.goal, self.visited, self.turn = None, [], None
        # Newest detection
        self.det = {}
        # Clock of the exploration logic (replaceable by a simulation)
        self.clock = time.monotonic
        # Exploration start and last goal choice (clock seconds)
        self._explore_t0, self._goal_t = 0.0, -1e9
        # Recent (time, x, y) while driving forward, for stuck detection
        self._moves = []

    # ----------------------------------------------------------- session
    # New session folder and fresh odometry / grid / detector
    def _new_session(self, target_name):
        # Target photos
        folder = os.path.join(targets_dir(), target_name)
        # Must exist
        if not target_name or not os.path.isdir(folder):
            # Refuse
            raise ValueError("no such target: %r (register photos on the search page first)" % target_name)
        # Model of the photos
        model = TargetModel.load(folder)
        # Needs at least one photo
        if not model.photos:
            # Refuse
            raise ValueError("target %r has no photos" % target_name)
        # Room for a session (keyframes ~15 kB each)
        ok, msg = self.mon.quota_ok("maps", 20)
        # Full
        if not ok:
            # Refuse
            raise ValueError(msg)
        # Session id and folder
        sid = time.strftime("%Y%m%d-%H%M%S")
        # Folder
        sdir = config.home_dir("maps", sid)
        # Keyframe folder
        os.makedirs(os.path.join(sdir, "keyframes"), exist_ok=True)
        # Session description (camera settings for the reconstruction)
        p = self.ctx.params
        # Settings
        info = {"id": sid, "target": target_name, "started": time.time(), "params": dict(p),
                "camera": {"size": self.cam.size, "hfov_deg": p["hfov_deg"], "height": p["cam_height"],
                           "pitch_deg": p["cam_pitch"], "calibration": p.get("calibration") or None}}
        # Write it
        with open(os.path.join(sdir, "session.json"), "w", encoding="utf-8") as f:
            # JSON
            json.dump(info, f, indent=2)
        # Trajectory file
        self._traj = open(os.path.join(sdir, "trajectory.csv"), "w", newline="", encoding="utf-8")
        # Its writer
        self._traj_w = csv.writer(self._traj)
        # Header
        self._traj_w.writerow(["t", "x", "y", "yaw", "odo_ok", "inliers", "state"])
        # Keyframe poses, one JSON line each (appending is cheap; keyframes.json is written at the end)
        self._kf_file = open(os.path.join(sdir, KEYFRAMES_JSONL), "w", encoding="utf-8")
        # Fresh parts
        self.odo, self.grid = FloorOdometry(self.cam), GridMap(side=p["map_size"], res=p["grid_res"])
        # Detector with K hits in a row
        self.detector = Detector(model, confirm=p["confirm"])
        # Keyframes so far and the pose of the last one
        self.keyframes, self._last_kf = [], None
        # Exploration memory
        self.goal, self.visited, self.turn = None, [], None
        # Exploration clock start
        self._explore_t0, self._goal_t = self.clock(), -1e9
        # Session state
        self.session = {"id": sid, "dir": sdir, "target": target_name, "started": time.time(), "found": None}

    # Keyframe when the robot moved / turned enough
    def _keyframe(self, frame, pose):
        # Last keyframe pose
        last = self._last_kf
        # Movement since then
        if last is not None and math.hypot(pose[0] - last[0], pose[1] - last[1]) < self.ctx.params["keyframe_dist"] \
                and abs(math.atan2(math.sin(pose[2] - last[2]), math.cos(pose[2] - last[2]))) < math.radians(15):
            # Not yet
            return
        # Quota check every 20 keyframes
        if len(self.keyframes) % 20 == 0:
            # Ask
            ok, _ = self.mon.quota_ok("maps", 1)
            # Full: stop adding keyframes (the run continues)
            if not ok:
                # Skip
                return
        # File name
        name = "%05d.jpg" % len(self.keyframes)
        # Save the raw frame
        cv2.imwrite(os.path.join(self.session["dir"], "keyframes", name), frame)
        # Its entry
        entry = {"file": name, "pose": [round(v, 4) for v in pose], "t": time.time()}
        # Remember it
        self.keyframes.append(entry)
        # One line on disk right away (survives a crash)
        self._kf_file.write(json.dumps(entry) + "\n")
        # Into the OS buffers
        self._kf_file.flush()
        # Last keyframe pose
        self._last_kf = pose

    # Write the session files (grid picture, summary; the full keyframes list only when ``final``)
    def _save_session(self, final=False):
        # No session
        if self.session is None:
            # Nothing
            return
        # Folder
        sdir = self.session["dir"]
        # Grid picture
        cv2.imwrite(os.path.join(sdir, "grid.png"), self.grid.image())
        # Grid geometry for the viewer
        info = {"res": self.grid.res, "n": self.grid.n, "scale": 2}
        # Full keyframes list at the end only (rewriting it every few seconds grew with the run
        # and stalled the control loop; keyframes.jsonl has every entry meanwhile)
        if final:
            # Write it
            with open(os.path.join(sdir, "keyframes.json"), "w", encoding="utf-8") as f:
                # JSON
                json.dump(self.keyframes, f)
        # Summary
        with open(os.path.join(sdir, "summary.json"), "w", encoding="utf-8") as f:
            # State, found object, grid stats, odometry failures
            json.dump({"state": self.state, "found": self.session["found"], "grid": dict(self.grid.stats(), **info),
                       "keyframes": len(self.keyframes), "odometry_lost": self.odo.lost,
                       "duration_s": round(time.time() - self.session["started"], 1)}, f, indent=2)
        # Flush the trajectory
        self._traj.flush()

    # The object was confirmed: stop, photo, position
    def _found(self, frame, det, pose):
        # Stop
        self.stop_drive("found", fault=False)
        # State
        self.state = "found"
        # Photo with the outline
        img = frame.copy()
        # Outline
        box = np.int32(det["box"])
        # Draw it
        cv2.polylines(img, [box], True, (0, 255, 0), 2)
        # Save it
        cv2.imwrite(os.path.join(self.session["dir"], "found.jpg"), img)
        # Bottom centre of the outline: where the object touches the floor
        u, v = float(box[:, 0].mean()), float(box[:, 1].max())
        # Floor point (robot frame)
        gx, gy = self.cam.ground_from_pixel(u, min(v, self.cam.size - 1))
        # Robot pose
        x, y, yaw = pose
        # World position (None when the bottom is above the horizon)
        pos = None
        # Valid floor point
        if np.isfinite(gx):
            # Rotate into the world
            c, s = math.cos(yaw), math.sin(yaw)
            # World coordinates
            pos = [round(x + c * float(gx) - s * float(gy), 3), round(y + s * float(gx) + c * float(gy), 3)]
        # Record it
        self.session["found"] = {"t": time.time(), "robot_pose": [round(v_, 3) for v_ in pose], "object_xy": pos,
                                 "inliers": det["inliers"], "photo": "found.jpg"}
        # Write the files
        self._save_session()
        # Tell the page
        self.ctx.emit("search.found", **self.session["found"])

    # ------------------------------------------------------------ driving
    # Send motor values while exploring
    def _motors(self, left, right):
        # Under the lock
        with self.lock:
            # Only while driving
            if self.driving:
                # Send
                self.ctx.robot.left_motor.value, self.ctx.robot.right_motor.value = left, right
                # Watchdog sees a good step
                self.last_ok = time.monotonic()

    # ------------------------------------------------------------ loop
    # After the base loop's own safety stops (watchdog, camera stalled, loop error)
    def on_safety_stop(self, code):
        # The search ends
        self.state = "stopped"

    # Exploration step: motors from the goal and the free space
    def _explore(self, pose, scores):
        # Clock now
        now = self.clock()
        # Time limit
        if now - self._explore_t0 > self.ctx.params["time_limit_s"]:
            # Stop
            self.stop_drive("time_limit")
            # Done
            self.state = "done"
            # Save
            self._save_session()
            # Nothing more
            return "time_limit"
        # Goal reached: remember it and pick another
        if self.goal is not None and math.hypot(self.goal[0] - pose[0], self.goal[1] - pose[1]) < 0.25:
            # Visited
            self.visited.append(self.goal)
            # New goal next
            self.goal = None
        # Pick a goal when there is none, and every 2 s (frontiers change as the map grows)
        if self.goal is None or now - self._goal_t > 2.0:
            # Remember when
            self._goal_t = now
            # Best frontier (keep the old goal if none is better)
            self.goal = choose_goal(self.grid.frontiers(), pose, self.visited) or self.goal
        # Nothing to explore any more
        if self.goal is None:
            # First frames: no map yet, just look around by turning
            if self.grid.updates < 30:
                # Turn left slowly
                p = self.ctx.params["turn_power"]
                # Motors
                self._motors(-p, p)
                # Status
                return "look_around"
            # Stop
            self.stop_drive("explored")
            # Done
            self.state = "done"
            # Save
            self._save_session()
            # Nothing more
            return "explored"
        # Turning power
        p = self.ctx.params["turn_power"]
        # A turn away from an obstacle is still running: keep turning (no oscillation)
        if self.turn is not None and now < self.turn[1]:
            # Turn on the spot
            self._motors(-p * self.turn[0], p * self.turn[0])
            # Status
            return "turn_away"
        # Turn finished
        self.turn = None
        # Direction to the goal
        kind, val = steer_to(self.goal, pose, self.ctx.params["hfov_deg"])
        # Goal beside / behind: turn on the spot
        if kind == "turn":
            # Left = right motor forward
            self._motors(-p * val, p * val)
            # Status
            return "turn"
        # Goal in view: M3 planner (obstacles, stop rule)
        xy, status, _ = self.planner.step(val, scores)
        # Moving forward without getting anywhere (wheels against an obstacle): stuck
        stuck = self._stuck(now, pose)
        # Blocked ahead / beside, or stuck: give this goal up and turn towards the freer side for a while
        if xy is None or stuck:
            # Unreachable for now
            self.visited.append(self.goal)
            # New goal next
            self.goal = None
            # Freer side: compare the left and right halves of the sectors (left = first sectors)
            left_free = sum(scores[:len(scores) // 2]) >= sum(scores[len(scores) // 2 + 1:])
            # Turn left (+1) or right (-1) for at least 1.2 s
            self.turn = (1 if left_free else -1, now + 1.2)
            # Forget the movement history
            self._moves = []
            # Turn on the spot (no forward motion)
            self._motors(-p * self.turn[0], p * self.turn[0])
            # Status
            return "stuck" if stuck else status
        # Controller
        left, right, cstatus = self.controller.update(xy, now)
        # Valid step
        if cstatus == "ok":
            # Motors
            self._motors(left, right)
        # Status
        return status if cstatus == "ok" else cstatus

    # Stuck detection: forward motion for 3 s with less than 3 cm of movement
    def _stuck(self, now, pose):
        """Record the pose; True when the robot did not move although it drove forward."""
        # Remember this pose
        self._moves.append((now, pose[0], pose[1]))
        # Keep the last 3 seconds
        self._moves = [m for m in self._moves if now - m[0] <= 3.0]
        # Need a full window
        if now - self._moves[0][0] < 2.9:
            # Not enough history
            return False
        # Movement over the window
        return math.hypot(pose[0] - self._moves[0][1], pose[1] - self._moves[0][2]) < 0.03

    # One frame (called by the base class loop; session files are not closed meanwhile)
    def step(self, frame):
        # Hold the session lock
        with self.slock:
            # Process the frame
            return self._step(frame)

    # One frame, under the session lock
    def _step(self, frame):
        # Timings
        t0 = time.monotonic()
        # Undistorted frame (pinhole) for free space and detection
        und = self.cam.undistort(frame)
        # Free-space sectors
        scores, _ = self.free.estimate(und)
        # Status of this step
        status = self.state
        # Inside a session: odometry, map, detection
        if self.session is not None and self.state in ("exploring", "idle"):
            # Pose from the floor odometry (raw frame: bev() undistorts itself)
            x, y, yaw, ok = self.odo.update(frame)
            # Pose
            pose = (x, y, yaw)
            # Odometry done
            t1 = time.monotonic()
            # Map update
            self.grid.update(pose, scores, self.cam, self.free.horizon)
            # Trajectory line
            self._traj_w.writerow(["%.3f" % time.time(), "%.4f" % x, "%.4f" % y, "%.4f" % yaw, int(ok),
                                   self.odo.inliers, self.state])
            # Keyframe
            self._keyframe(frame, pose)
            # Map done
            t2 = time.monotonic()
            # Detection
            self.det = self.detector.step(und)
            # Detection done
            t3 = time.monotonic()
            # Confirmed while exploring
            if self.det["found"] and self.state == "exploring":
                # Stop, photo, position
                self._found(und, self.det, pose)
            # Exploring: motors
            elif self.state == "exploring" and self.driving:
                # Exploration step
                status = self._explore(pose, scores)
            # Grid picture every 5 s
            if int(time.time()) % 5 == 0 and getattr(self, "_saved_t", 0) != int(time.time()):
                # Remember the second
                self._saved_t = int(time.time())
                # Save
                self._save_session()
            # Stage times
            timing = {"odometry": t1 - t0, "map": t2 - t1, "detect": t3 - t2}
        # No session: only show the picture
        else:
            # No pose
            pose, timing = None, {}
        # Display item
        self.slot.put({"frame": und, "scores": scores, "det": self.det, "status": status, "pose": pose})
        # Stage times
        return timing

    # ---------------------------------------------------------- commands
    # Commands from the page
    def command(self, name, args):
        # Registered targets
        if name == "list_targets":
            # Folder names
            return {"targets": sorted(d for d in os.listdir(targets_dir()) if os.path.isdir(os.path.join(targets_dir(), d)))}
        # Start a new search
        if name == "start_search":
            # No pictures
            if self.camera is None:
                # Refuse
                raise ValueError("no_camera")
            # Stop any running session first
            if self.session is not None:
                # Close its files
                self._close_session()
            # New session (raises with a reason); not while the loop writes
            with self.slock:
                # Create it
                self._new_session(args.get("target") or self.ctx.params.get("target", ""))
            # Under the lock
            with self.lock:
                # Fresh controller
                self.controller.reset()
                # Watchdog starts now
                self.last_ok = time.monotonic()
                # Exploring
                self.state, self.driving = "exploring", True
            # Tell the page
            self.ctx.emit("search.started", session=self.session["id"], target=self.session["target"])
            # Session id
            return {"session": self.session["id"]}
        # Stop the search (session files are kept)
        if name == "stop_search":
            # Motors off
            self.stop_drive("manual")
            # Stopped (not found)
            if self.state == "exploring":
                # State
                self.state = "stopped"
            # Save the files (not while the loop writes)
            with self.slock:
                # Write them
                self._save_session()
            # Session id
            return {"session": self.session["id"] if self.session else None}
        # Anything else
        raise ValueError("unknown command %r" % name)

    # Close the session files
    def _close_session(self):
        # Not while the loop writes
        with self.slock:
            # Save and close
            if self.session is not None:
                # Files, with the full keyframes list
                self._save_session(final=True)
                # Trajectory and keyframe files
                self._traj.close()
                # Close it
                self._kf_file.close()
            # No session
            self.session = None

    # ----------------------------------------------------------- safety
    # Safety stop from the runner
    def stop_motion(self, code, detail=None):
        # Motors off
        self.stop_drive(code, fault=code not in ("shutdown", "task_stop"))
        # A running search ends
        if self.state == "exploring":
            # Stopped
            self.state = "stopped"

    # Display item -> JPEG: sectors, detection, mini map, status
    def render(self, item, quality):
        # Sectors on the picture
        img = self.free.draw(item["frame"], item["scores"], alpha=0.25)
        # Detection outline (yellow = hit, green = confirmed)
        d = item.get("det") or {}
        # Only hits have an outline
        if d.get("hit") and d.get("box"):
            # Outline
            cv2.polylines(img, [np.int32(d["box"])], True, (0, 255, 0) if d.get("found") else (0, 255, 255), 2)
        # Mini map in the top-right corner
        if self.grid is not None:
            # Small grid picture
            mini = cv2.resize(self.grid.image(scale=1), (72, 72), interpolation=cv2.INTER_AREA)
            # Paste
            img[2:74, img.shape[1] - 74:img.shape[1] - 2] = mini
        # Status and pose
        pose = item.get("pose")
        # Text
        text = item["status"] + (" (%.2f, %.2f, %d deg)" % (pose[0], pose[1], math.degrees(pose[2])) if pose else "")
        # Draw it
        cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        # Encode
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        # JPEG and readings
        return (bytes(buf) if ok else b""), {"status": item["status"], "pose": pose, "hit": bool(d.get("hit"))}

    # State for the page
    def telemetry(self):
        # Pose and odometry
        odo = {"x": self.odo.x, "y": self.odo.y, "yaw_deg": math.degrees(self.odo.yaw), "lost": self.odo.lost,
               "inliers": self.odo.inliers} if self.odo else None
        # Detection summary
        det = dict((k, self.det.get(k)) for k in ("hit", "score", "inliers", "streak", "found")) if self.det else None
        # Everything
        return {"state": self.state, "driving": self.driving, "session": self.session["id"] if self.session else None,
                "target": self.session["target"] if self.session else None, "odometry": odo, "detection": det,
                "goal": self.goal, "visited": len(self.visited), "grid": self.grid.stats() if self.grid else None,
                "keyframes": len(self.keyframes) if self.session else 0, "loop": self.stats,
                "found": self.session["found"] if self.session else None,
                "elapsed_s": round(time.time() - self.session["started"], 1) if self.session else 0,
                "camera": self.camera is not None, "robot": self.ctx.robot_info["kind"]}

    # Save and end
    def close(self):
        # Motors off, threads end
        CameraLoopRuntime.close(self)
        # Files
        self._close_session()
