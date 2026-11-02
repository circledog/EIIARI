"""Preprocessing: mark the lane lines (white tape) on the image, live.

    PC     Data Loader -> Data Augmentation -> Preprocessing -> Model training
    Robot  Camera -> (input disturbance) -> Preprocessing -> Model inference

Always on (no switch) and nothing is written to disk: ``data_loader`` calls
``mark_lanes_pil`` on every sample *after* augmentation, and ``live_demo.ipynb``
calls ``apply_frame`` on every camera frame *after* the optional Perlin /
compression disturbance. Both sides therefore run the same steps in the same
order, so noise affects the lane detection the same way in training and on
the robot.

What it does
------------
Every pixel that looks like lane tape is painted pure red, so the model gets
the lane lines handed to it while still seeing the rest of the picture (3
colour channels, networks unchanged). The labels are not touched.

The detector works per pixel instead of fitting straight lines, so curves,
hairpins and the wide tape right in front of the camera are all marked:

1. ``L`` = LAB lightness. The floor level is the 40th percentile of the lower
   middle of the image.
2. Tape = ``L > floor + delta`` (bright tape, any width), or a thin bright
   stripe: white top-hat (``tophat`` px) ``> th_top`` and ``L > floor +
   delta_top`` (far, thin, slightly dimmer tape).
3. Large bright areas high up in the image (walls, wooden floor next to the
   track) are removed: anything thicker than ``thick`` px whose bottom is in
   the upper 55 % or that touches the top edge. Thick areas lower down are
   kept -- that is tape close to the camera.
4. Small specks (< ``min_area`` px) and small blobs in the top quarter are
   dropped.

Parameters are tuned on 224x224 frames; other sizes scale the kernels. On the
198-image training set every image gets tape marked, no label point ends up on
marked pixels, and it takes ~6 ms per frame on a PC CPU.

Usage:
    image = mark_lanes_pil(image)        # PC, PIL RGB in/out (data_loader)
    frame = apply_frame(frame)           # robot, BGR uint8 in/out

Python 3.6 compatible (runs on the Jetson Nano).
"""
import glob
import os

import cv2
import numpy as np
import PIL.Image

# Public names of this module
__all__ = ["apply_frame", "mark_lanes_pil", "lane_mask", "mark_lanes",
           "LANE_COLOR_BGR", "DEFAULT_PARAMS", "CONFIG"]

# Colour painted over the tape: pure red, which does not occur on the track
LANE_COLOR_BGR = (0, 0, 255)
# Detector thresholds and kernel sizes, tuned on 224x224 frames (see module docstring)
DEFAULT_PARAMS = {"delta": 70, "th_top": 35, "delta_top": 40, "tophat": 21,
                  "thick": 31, "min_area": 40}


# Elliptical structuring element for OpenCV morphology
def _ellipse(k):
    # Round to an odd size of at least 3
    k = max(3, int(round(k)) | 1)
    # k x k ellipse
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


# Detect lane tape pixels in a frame
def lane_mask(bgr, delta=70, th_top=35, delta_top=40, tophat=21, thick=31, min_area=40):
    """uint8 mask (255 = lane tape) of a BGR uint8 image."""
    # Frame height and width
    h, w = bgr.shape[:2]
    # Scale factor for kernel sizes relative to the 224-px tuning size
    s = w / 224.0
    # LAB lightness channel (brightness independent of colour cast)
    L = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2LAB)[..., 0]
    # Floor brightness: 40th percentile of the lower middle of the image
    floor = float(np.percentile(L[h // 2:, w // 4:3 * w // 4], 40))

    # Pixels much brighter than the floor: tape of any width
    bright = (L > floor + delta).astype(np.uint8)
    # Remove single-pixel noise from the bright mask
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, _ellipse(3))
    # White top-hat: how much brighter a pixel is than its neighbourhood
    stripe = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, _ellipse(tophat * s))
    # Thin bright stripes that are also clearly brighter than the floor (far, thin tape)
    thin = ((stripe > th_top) & (L > floor + delta_top)).astype(np.uint8)
    # Candidate tape pixels
    m = bright | thin

    # thick bright areas high up = walls / floor outside the track
    thick_m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ellipse(thick * s))
    # Separate the thick areas into connected blobs
    n, lab, st, _ = cv2.connectedComponentsWithStats(thick_m, 8)
    # Check every thick blob (label 0 is the background)
    for k in range(1, n):
        # Top row and height of the blob
        y, hh = st[k, cv2.CC_STAT_TOP], st[k, cv2.CC_STAT_HEIGHT]
        # Keep it as "to remove" only if it is high up or touches the top edge
        if not (y + hh < 0.55 * h or y == 0):
            # Thick but low: wide tape near the camera, so do not remove it
            thick_m[lab == k] = 0
    # Remove the walls / background (slightly grown) from the candidates
    m[cv2.dilate(thick_m, _ellipse(5)) > 0] = 0

    # Connected blobs of the remaining candidates
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    # Pixel count of every blob
    area = st[:, cv2.CC_STAT_AREA]
    # Bottom row of every blob
    bottom = st[:, cv2.CC_STAT_TOP] + st[:, cv2.CC_STAT_HEIGHT]
    # Keep blobs that are big enough and not small blobs in the top quarter
    keep = (area >= min_area * s * s) & ~((bottom < 0.25 * h) & (area < 150 * s * s))
    # Never keep the background label
    keep[0] = False
    # Look up every pixel's blob and turn the result into a 0 / 255 mask
    return keep[lab].astype(np.uint8) * 255


# Paint the detected tape red
def mark_lanes(bgr, params=None):
    """Copy of a BGR uint8 image with the lane tape painted ``LANE_COLOR_BGR``."""
    # Work on a uint8 copy so the input stays untouched
    out = np.array(bgr, dtype=np.uint8, copy=True)
    # Paint every tape pixel in the lane colour
    out[lane_mask(out, **(params or DEFAULT_PARAMS)) > 0] = LANE_COLOR_BGR
    # Marked image
    return out


# Robot side: mark the lanes on a live camera frame
def apply_frame(frame):
    """Robot side: lane marking for a camera frame (HxWx3 BGR uint8)."""
    # Same detector and parameters as in training
    return mark_lanes(frame, DEFAULT_PARAMS)


# Training side: mark the lanes on a PIL image
def mark_lanes_pil(image):
    """Training side: lane marking for a PIL image (RGB in, RGB out)."""
    # PIL RGB -> numpy BGR (the detector works on BGR, like the camera)
    bgr = np.asarray(image.convert("RGB"))[..., ::-1]
    # Mark, convert back to RGB and wrap as a PIL image
    return PIL.Image.fromarray(np.ascontiguousarray(mark_lanes(bgr, DEFAULT_PARAMS)[..., ::-1]))


# Description of the preprocessing, stored in every model's .json
CONFIG = {"method": "lane_marking", "order": "after_augmentation",
          "color_bgr": list(LANE_COLOR_BGR), "params": DEFAULT_PARAMS}


# --------------------------------------------------------------------------- #
# CLI (PC)
# --------------------------------------------------------------------------- #
# Build a contact sheet of all lane-marked images with labels (and predictions)
def _review(args):
    """Contact sheet of the lane-marked images with labels (and predictions)."""
    # Dataset helpers: default folder and label parsing
    from data_loader import DEFAULT_DATA_DIR, parse_xy

    # Every image of the dataset
    paths = sorted(glob.glob(os.path.join(args.data or DEFAULT_DATA_DIR, "xy_*.jpg")))
    # No model unless --model is given
    model = None
    # Load the model when predictions should be shown
    if args.model:
        # Camera helpers, for the same tensor conversion as on the robot
        import camera as cam_mod
        # Checkpoint loader
        from loader import load_model
        # Load the checkpoint on the CPU
        model = load_model(args.model, "cpu")
    # (error, path, image, label point, prediction point) per image
    rows = []
    # Process every image
    for p in paths:
        # Read the image and mark its lanes
        img = apply_frame(cv2.imread(p))
        # Image height and width
        h, w = img.shape[:2]
        # Label from the file name
        x, y = parse_xy(p)
        # Defaults when there is no model
        err, pred = 0.0, None
        # Predict and measure the error when a model is loaded
        if model is not None:
            # Network output for this image as a flat numpy array
            out = model(cam_mod.preprocess(img, "cpu", model.dtype)).float().numpy().flatten()
            # Normalised prediction -> pixel coordinates
            px, py = (out[0] + 1) * w / 2.0, (out[1] + 1) * h / 2.0
            # Distance between prediction and label in pixels
            err = float(np.hypot(px - x, py - y))
            # Integer point for drawing
            pred = (int(px), int(py))
        # Store the row; clamp the label so it can be drawn
        rows.append((err, p, img, (int(min(x, w - 1)), int(min(y, h - 1))), pred))
    # With a model, show the worst predictions first
    if model is not None:
        # Sort by error, largest first
        rows.sort(key=lambda r: -r[0])

    # Thumbnails for the sheet
    tiles = []
    # Draw every row
    for i, (err, p, img, lab, pred) in enumerate(rows):
        # Draw on a copy
        t = img.copy()
        # Green dot: the label
        cv2.circle(t, lab, 5, (0, 255, 0), -1)
        # Cyan circle: the prediction, if any
        if pred is not None:
            # Draw the prediction
            cv2.circle(t, pred, 5, (255, 255, 0), 2)
        # Running number in the corner
        cv2.putText(t, "#%d" % i, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Error in pixels under the number, if a model is loaded
        if model is not None:
            # Write the error
            cv2.putText(t, "%.0fpx" % err, (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        # Shrink to thumbnail size
        tiles.append(cv2.resize(t, (args.tile, args.tile)))
        # Same order in the terminal, with file names
        print("#%-4d %6.1f  %s" % (i, err, os.path.basename(p)))
    # Thumbnails per row
    cols = args.cols
    # Pad with black tiles so the last row is full
    while len(tiles) % cols:
        # One black tile
        tiles.append(np.zeros_like(tiles[0]))
    # Stack the thumbnails into a grid
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    # Save the sheet
    cv2.imwrite(args.out, sheet)
    # Tell the user where it is and what the colours mean
    print("wrote %s  (red = marked lane, green = label, cyan circle = prediction)" % args.out)


# Command-line entry point
if __name__ == "__main__":
    import argparse
    # Top-level parser
    ap = argparse.ArgumentParser(description="Preprocessing (lane marking) tools")
    # Sub-commands
    sub = ap.add_subparsers(dest="cmd")
    # "review": contact sheet
    r = sub.add_parser("review", help="contact sheet of the lane-marked images + labels")
    # Dataset folder
    r.add_argument("--data", default=None, help="folder with xy_*.jpg")
    # Optional model whose predictions are drawn
    r.add_argument("--model", default=None, help="a .pth from train.py: sort by its error")
    # Output image file
    r.add_argument("--out", default="label_review.jpg")
    # Thumbnails per row
    r.add_argument("--cols", type=int, default=10)
    # Thumbnail size in pixels
    r.add_argument("--tile", type=int, default=160)
    # Parse the command line
    a = ap.parse_args()
    # Run the chosen sub-command
    if a.cmd == "review":
        # Build the contact sheet
        _review(a)
    # No sub-command given
    else:
        # Show the usage
        ap.print_help()
