"""Fisheye calibration of the robot camera from chessboard pictures.

Print a chessboard (inner corners 9 x 6 by default), take 15-30 pictures with
the robot camera from different angles (e.g. with the camera check task's
recording, or any 224x224 frames), then::

    python3 -m studio.tasks.search.calibrate <folder with jpgs> --out ~/.studio/camera_calibration.json

and enter that path as "Calibration file" in the search task.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np


# Calibrate from a folder of chessboard pictures
def calibrate(folder, cols=9, rows=6):
    """Return ``{'size', 'K', 'D', 'rms', 'used'}``."""
    # Chessboard corners in board units
    objp = np.zeros((1, cols * rows, 3), np.float64)
    # Grid of the inner corners
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    # Collected object / image points
    obj, img, size = [], [], None
    # Every picture
    for p in sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))):
        # Grey picture
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        # Unreadable
        if g is None:
            # Skip
            continue
        # Picture size
        size = g.shape[::-1]
        # Find the corners
        ok, corners = cv2.findChessboardCorners(g, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        # Not found
        if not ok:
            # Skip
            continue
        # Sub-pixel refinement
        corners = cv2.cornerSubPix(g, corners, (3, 3), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        # Keep them
        obj.append(objp)
        # Image points (fisheye wants (1, N, 2))
        img.append(corners.reshape(1, -1, 2).astype(np.float64))
    # Need enough views
    if len(obj) < 5:
        # Report
        raise ValueError("only %d pictures with a visible chessboard; take more (15-30)" % len(obj))
    # Outputs
    K, D = np.zeros((3, 3)), np.zeros((4, 1))
    # Fisheye calibration
    rms, K, D, _, _ = cv2.fisheye.calibrate(obj, img, size, K, D, flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
                                            cv2.fisheye.CALIB_FIX_SKEW,
                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    # Result
    return {"size": int(size[0]), "K": K.tolist(), "D": D.reshape(-1).tolist(), "rms": float(rms), "used": len(obj)}


# Command-line entry point
def main(argv=None):
    """Calibrate and write the JSON file."""
    # Options
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Picture folder
    ap.add_argument("folder")
    # Output file
    ap.add_argument("--out", default=os.path.expanduser("~/.studio/camera_calibration.json"))
    # Board size
    ap.add_argument("--cols", type=int, default=9)
    # Board size
    ap.add_argument("--rows", type=int, default=6)
    # Parse
    args = ap.parse_args(argv)
    # Calibrate
    c = calibrate(args.folder, args.cols, args.rows)
    # Folder of the output
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # Write
    with open(args.out, "w", encoding="utf-8") as f:
        # JSON
        json.dump(c, f, indent=2)
    # Report
    print("calibrated from %d pictures, RMS %.3f px -> %s" % (c["used"], c["rms"], args.out))
    # Success
    return 0


# Run as a module
if __name__ == "__main__":
    # Exit with the result
    sys.exit(main())
