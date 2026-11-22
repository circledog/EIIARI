"""3-D map after a search run (TASKS M4-6).

Input: a session folder (``STUDIO_HOME/maps/<session>/``) with
``keyframes/<n>.jpg`` and ``keyframes.json`` (pose per keyframe) and
``session.json`` (camera settings). Output: ``map.ply`` (coloured points) and
``cloud.json`` (a subsample for the web viewer).

Two kinds of points, both in metres in the start frame (z up):

* floor: pixels of the floor in each keyframe, placed with the camera model
  (z = 0) -- the textured floor plan;
* structure: ORB features matched between neighbouring keyframes and
  triangulated with the known poses; points clearly above the floor are
  obstacles / walls / furniture. Their height comes from the multi-view
  geometry, not from an assumption.

Points are merged on a voxel grid (``voxel`` metres) and capped at
``max_points`` so a map stays below the 300 MB maps budget (one map is a few MB).

Run as a background job::

    python3 -m studio.tasks.search.reconstruct <session folder>

Python 3.6 compatible (runs on the Jetson Nano).
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

# Camera geometry
from studio.tasks.search.camera_model import CameraModel


# Camera model of a session
def session_camera(session):
    """CameraModel from ``session.json``'s camera settings."""
    # Settings
    c = session["camera"]
    # Calibration file when one was used
    if c.get("calibration"):
        # Calibrated model
        return CameraModel.from_file(c["calibration"], c["height"], c["pitch_deg"])
    # Field-of-view model
    return CameraModel(size=c["size"], hfov_deg=c["hfov_deg"], height=c["height"], pitch_deg=c["pitch_deg"])


# 3x4 projection matrix of a keyframe
def projection(cam, pose):
    """``K [R | t]`` mapping world points (start frame, metres) to pixels."""
    # Robot pose
    x, y, yaw = pose
    # World -> robot rotation (about z)
    c, s = math.cos(yaw), math.sin(yaw)
    # Rotation matrix robot -> world
    r_wr = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)
    # Camera position in the world
    cam_w = np.array([x, y, cam.height])
    # World -> camera rotation
    r = cam.R.dot(r_wr.T)
    # Translation
    t = -r.dot(cam_w)
    # Projection
    return cam.K.dot(np.hstack([r, t[:, None]]))


# Floor points of one keyframe
def floor_points(cam, img, pose, step=3, max_range=1.2):
    """``(N, 3)`` world points (z = 0) and ``(N, 3)`` RGB colours."""
    # Sample every `step` pixels
    vv, uu = np.mgrid[0:cam.size:step, 0:cam.size:step]
    # Floor coordinates in the robot frame
    gx, gy = cam.ground_from_pixel(uu + 0.5, vv + 0.5)
    # Valid floor points within range
    ok = np.isfinite(gx) & (gx > 0.02) & (np.hypot(gx, gy) < max_range)
    # Robot pose
    x, y, yaw = pose
    # Rotation
    c, s = math.cos(yaw), math.sin(yaw)
    # World coordinates
    wx, wy = x + c * gx[ok] - s * gy[ok], y + s * gx[ok] + c * gy[ok]
    # Colours (BGR -> RGB)
    col = img[vv[ok], uu[ok]][:, ::-1]
    # Points and colours
    return np.stack([wx, wy, np.zeros_like(wx)], -1), col


# Triangulated feature points between two keyframes
def structure_points(cam, img_a, pose_a, img_b, pose_b, orb, matcher, max_err=2.0, max_dist=3.0, min_baseline=0.05):
    """``(N, 3)`` world points and colours from matched ORB features.

    Pairs closer than ``min_baseline`` metres (turning on the spot) cannot be
    triangulated reliably and give nothing.
    """
    # Too little camera movement between the two keyframes
    if math.hypot(pose_b[0] - pose_a[0], pose_b[1] - pose_a[1]) < min_baseline:
        # Nothing
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    # Features
    ka, da = orb.detectAndCompute(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), None)
    # Second image
    kb, db = orb.detectAndCompute(cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY), None)
    # Need features in both
    if da is None or db is None:
        # Nothing
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    # Cross-checked matches
    matches = matcher.match(da, db)
    # Too few
    if len(matches) < 8:
        # Nothing
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    # Pixel pairs
    pa = np.float64([ka[m.queryIdx].pt for m in matches]).T
    # Second image
    pb = np.float64([kb[m.trainIdx].pt for m in matches]).T
    # Projection matrices
    Pa, Pb = projection(cam, pose_a), projection(cam, pose_b)
    # Triangulate (homogeneous)
    X = cv2.triangulatePoints(Pa, Pb, pa, pb)
    # Euclidean points
    X = (X[:3] / X[3]).T
    # Reprojection error in both views
    def reproj(P, p):
        # Project
        q = P.dot(np.hstack([X, np.ones((len(X), 1))]).T)
        # Points at infinity give inf / NaN errors and are dropped below
        with np.errstate(divide="ignore", invalid="ignore"):
            # Pixel error
            err = np.hypot(q[0] / q[2] - p[0], q[1] / q[2] - p[1])
        # Error (NaN -> inf) and depth
        return np.nan_to_num(err, nan=np.inf), q[2]
    # Errors and depths
    ea, za = reproj(Pa, pa)
    # Second view
    eb, zb = reproj(Pb, pb)
    # Distance from the first camera
    dist = np.linalg.norm(X[:, :2] - np.array(pose_a[:2]), axis=1)
    # Keep good points in front of both cameras, not below the floor, within range
    ok = (ea < max_err) & (eb < max_err) & (za > 0) & (zb > 0) & (X[:, 2] > -0.03) & (dist < max_dist)
    # Colours from the first image (BGR -> RGB)
    col = np.array([img_a[int(min(cam.size - 1, max(0, round(v)))), int(min(cam.size - 1, max(0, round(u))))][::-1]
                    for u, v in pa.T[ok]], np.uint8).reshape(-1, 3)
    # Points and colours
    return X[ok], col


# Merge points on a voxel grid
def voxelize(points, colours, voxel):
    """One point per voxel (mean position and colour)."""
    # Nothing
    if len(points) == 0:
        # Empty
        return points, colours
    # Voxel index of every point
    keys = np.floor(points / voxel).astype(np.int64)
    # Unique voxels
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    # Flat inverse index (numpy versions differ in its shape)
    inv = inv.reshape(-1)
    # Mean position per voxel
    pos = np.zeros((len(counts), 3))
    # Sum positions
    np.add.at(pos, inv, points)
    # Mean colour per voxel
    col = np.zeros((len(counts), 3))
    # Sum colours
    np.add.at(col, inv, colours.astype(np.float64))
    # Means
    return pos / counts[:, None], (col / counts[:, None]).astype(np.uint8)


# Write an ASCII PLY file
def write_ply(path, points, colours, labels=None):
    """Coloured point cloud readable by MeshLab / CloudCompare / Open3D."""
    # Open the file
    with open(path, "w", encoding="ascii") as f:
        # Header
        f.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(points))
        # One line per point
        for p, c in zip(points, colours):
            # Position and colour
            f.write("%.4f %.4f %.4f %d %d %d\n" % (p[0], p[1], p[2], c[0], c[1], c[2]))


# Build the 3-D map of a session
def reconstruct(folder, voxel=0.02, max_points=200000, floor_voxel=0.03, progress=print):
    """Write ``map.ply`` and ``cloud.json``; returns a summary dict."""
    # Start time
    t0 = time.time()
    # Session settings
    with open(os.path.join(folder, "session.json"), encoding="utf-8") as f:
        # Parsed
        session = json.load(f)
    # Keyframe poses
    with open(os.path.join(folder, "keyframes.json"), encoding="utf-8") as f:
        # List of {"file", "pose"}
        keyframes = json.load(f)
    # Camera
    cam = session_camera(session)
    # Feature detector and matcher
    orb, matcher = cv2.ORB_create(1000, fastThreshold=10, edgeThreshold=15, patchSize=15), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Collected points
    floor, floor_c, struct, struct_c = [], [], [], []
    # Previous keyframe
    prev = None
    # Every keyframe
    for i, kf in enumerate(keyframes):
        # Undistorted picture
        img = cam.undistort(cv2.imread(os.path.join(folder, "keyframes", kf["file"])))
        # Floor points
        p, c = floor_points(cam, img, kf["pose"])
        # Keep them
        floor.append(p)
        # Colours
        floor_c.append(c)
        # Structure from the previous keyframe
        if prev is not None:
            # Triangulate
            p, c = structure_points(cam, prev[0], prev[1], img, kf["pose"], orb, matcher)
            # Keep them
            struct.append(p)
            # Colours
            struct_c.append(c)
        # Remember
        prev = (img, kf["pose"])
        # Progress line (parsed by the job page)
        progress("keyframe %d/%d" % (i + 1, len(keyframes)))
    # Merge the floor on its own (coarser) grid
    fp, fc = voxelize(np.concatenate(floor) if floor else np.zeros((0, 3)),
                      np.concatenate(floor_c) if floor_c else np.zeros((0, 3), np.uint8), floor_voxel)
    # Structure points above the floor
    sp = np.concatenate(struct) if struct else np.zeros((0, 3))
    # Their colours
    sc = np.concatenate(struct_c) if struct_c else np.zeros((0, 3), np.uint8)
    # Merge them
    sp, sc = voxelize(sp, sc, voxel)
    # Everything
    pts, cols = np.concatenate([fp, sp]), np.concatenate([fc, sc])
    # Cap the size
    if len(pts) > max_points:
        # Random subset (fixed seed)
        keep = np.random.RandomState(0).choice(len(pts), max_points, replace=False)
        # Keep them
        pts, cols = pts[keep], cols[keep]
    # PLY file
    write_ply(os.path.join(folder, "map.ply"), pts, cols)
    # Subsample for the browser (at most 40 000 points)
    k = np.random.RandomState(1).choice(len(pts), min(len(pts), 40000), replace=False) if len(pts) else []
    # Viewer data
    with open(os.path.join(folder, "cloud.json"), "w", encoding="utf-8") as f:
        # Rounded positions and colours
        json.dump({"points": np.round(pts[k], 3).tolist(), "colors": cols[k].tolist(),
                   "floor_points": int(len(fp)), "structure_points": int(len(sp))}, f)
    # Summary
    summary = {"keyframes": len(keyframes), "points": int(len(pts)), "floor_points": int(len(fp)),
               "structure_points": int(len(sp)), "seconds": round(time.time() - t0, 1),
               "ply_mb": round(os.path.getsize(os.path.join(folder, "map.ply")) / 1e6, 2)}
    # Store it
    with open(os.path.join(folder, "reconstruction.json"), "w", encoding="utf-8") as f:
        # JSON
        json.dump(summary, f, indent=2)
    # Summary
    return summary


# Command-line entry point
def main(argv=None):
    """Reconstruct one session folder."""
    # Options
    ap = argparse.ArgumentParser(description="3-D map of a search session")
    # Session folder
    ap.add_argument("folder")
    # Voxel size
    ap.add_argument("--voxel", type=float, default=0.02)
    # Point cap
    ap.add_argument("--max-points", type=int, default=200000)
    # Parse
    args = ap.parse_args(argv)
    # Run, printing progress lines
    summary = reconstruct(args.folder, args.voxel, args.max_points, progress=lambda s: print(s, flush=True))
    # Final line
    print("done %s" % json.dumps(summary))
    # Success
    return 0


# Run as a module
if __name__ == "__main__":
    # Exit with the result
    sys.exit(main())
