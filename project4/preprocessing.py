"""Deterministic preprocessing, between the data loader and augmentation:
mark the lane lines (white tape) on the image.

    Data Loader -> Preprocessing -> Data Augmentation -> Model training

Always on (no switch). ``train.py`` calls ``preprocess_dataset()`` first: it
writes a lane-marked copy of every image to an output folder (kept on disk,
lossless PNG with the same ``xy_<x>_<y>_<uuid>`` name), and training reads
that folder. ``live_demo.ipynb`` calls ``apply_frame()`` on every camera frame,
so the robot sees exactly what the model was trained on.

    data/<name>/dataset_xy/xy_*.jpg  ->  data/<name>/dataset_xy_lanes/xy_*.png
                                         + preprocessing.json (parameters)

The output folder sits next to the input, so ``metadata/`` (``--split
session``) is still found. Re-running only processes new images; if the
parameters change everything is redone and stale files are removed.

What it does
------------
Every pixel that looks like lane tape is painted pure red, so the model gets
the lane lines handed to it while still seeing the rest of the picture (3
colour channels, ResNet unchanged). The labels are not touched.

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
    out_dir = preprocess_dataset("data/x/dataset_xy")   # PC, called by train.py
    frame_in = apply_frame(frame)                       # robot, BGR uint8 in/out

Python 3.6 compatible (runs on the Jetson Nano).
"""
import glob
import json
import os

import cv2
import numpy as np
import PIL.Image

__all__ = ["preprocess_dataset", "default_output_dir", "apply_frame", "lane_mask", "mark_lanes",
           "LANE_COLOR_BGR", "DEFAULT_PARAMS", "CONFIG"]

LANE_COLOR_BGR = (0, 0, 255)   # pure red: does not occur on the track
DEFAULT_PARAMS = {"delta": 70, "th_top": 35, "delta_top": 40, "tophat": 21,
                  "thick": 31, "min_area": 40}


def _ellipse(k):
    k = max(3, int(round(k)) | 1)   # odd, >= 3
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def lane_mask(bgr, delta=70, th_top=35, delta_top=40, tophat=21, thick=31, min_area=40):
    """uint8 mask (255 = lane tape) of a BGR uint8 image."""
    h, w = bgr.shape[:2]
    s = w / 224.0
    L = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2LAB)[..., 0]
    floor = float(np.percentile(L[h // 2:, w // 4:3 * w // 4], 40))

    bright = (L > floor + delta).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, _ellipse(3))
    stripe = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, _ellipse(tophat * s))
    thin = ((stripe > th_top) & (L > floor + delta_top)).astype(np.uint8)
    m = bright | thin

    # thick bright areas high up = walls / floor outside the track
    thick_m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ellipse(thick * s))
    n, lab, st, _ = cv2.connectedComponentsWithStats(thick_m, 8)
    for k in range(1, n):
        y, hh = st[k, cv2.CC_STAT_TOP], st[k, cv2.CC_STAT_HEIGHT]
        if not (y + hh < 0.55 * h or y == 0):
            thick_m[lab == k] = 0          # thick but low: wide tape near the camera
    m[cv2.dilate(thick_m, _ellipse(5)) > 0] = 0

    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    area = st[:, cv2.CC_STAT_AREA]
    bottom = st[:, cv2.CC_STAT_TOP] + st[:, cv2.CC_STAT_HEIGHT]
    keep = (area >= min_area * s * s) & ~((bottom < 0.25 * h) & (area < 150 * s * s))
    keep[0] = False
    return keep[lab].astype(np.uint8) * 255


def mark_lanes(bgr, params=None):
    """Copy of a BGR uint8 image with the lane tape painted ``LANE_COLOR_BGR``."""
    out = np.array(bgr, dtype=np.uint8, copy=True)
    out[lane_mask(out, **(params or DEFAULT_PARAMS)) > 0] = LANE_COLOR_BGR
    return out


def apply_frame(frame):
    """Robot side: lane marking for a camera frame (HxWx3 BGR uint8)."""
    return mark_lanes(frame, DEFAULT_PARAMS)


CONFIG = {"method": "lane_marking", "color_bgr": list(LANE_COLOR_BGR), "params": DEFAULT_PARAMS}


def default_output_dir(src_dir):
    """``<parent>/<name>_lanes`` next to the input folder."""
    src_dir = os.path.normpath(os.path.abspath(src_dir))
    return src_dir + "_lanes"


def preprocess_dataset(src_dir, out_dir=None, verbose=True):
    """Write a lane-marked PNG of every ``xy_*.jpg`` in ``src_dir`` to ``out_dir``.

    Returns ``out_dir``. Already processed images are skipped unless the
    parameters changed; files whose source is gone are removed.
    """
    out_dir = out_dir or default_output_dir(src_dir)
    srcs = sorted(glob.glob(os.path.join(src_dir, "xy_*.jpg")))
    if not srcs:
        raise FileNotFoundError("no xy_*.jpg found in %s" % src_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    cfg_path = os.path.join(out_dir, "preprocessing.json")
    old_cfg = None
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            old_cfg = json.load(f).get("config")
    redo = old_cfg != json.loads(json.dumps(CONFIG))

    wanted = set()
    done = 0
    for p in srcs:
        stem = os.path.splitext(os.path.basename(p))[0]
        dst = os.path.join(out_dir, stem + ".png")
        wanted.add(os.path.basename(dst))
        if not redo and os.path.exists(dst):
            continue
        img = cv2.imread(p)
        if img is None:
            raise IOError("cannot read %s" % p)
        cv2.imwrite(dst, mark_lanes(img, DEFAULT_PARAMS))
        done += 1
    removed = 0
    for q in glob.glob(os.path.join(out_dir, "xy_*.png")):
        if os.path.basename(q) not in wanted:
            os.remove(q)
            removed += 1
    with open(cfg_path, "w") as f:
        json.dump({"config": CONFIG, "source": os.path.abspath(src_dir), "n_images": len(srcs)},
                  f, indent=1)
    if verbose:
        print("preprocessing: %d images in %s (%d new, %d removed)" % (len(srcs), out_dir, done, removed))
    return out_dir


# --------------------------------------------------------------------------- #
# CLI (PC)
# --------------------------------------------------------------------------- #
def _review(args):
    """Contact sheet of the preprocessed images with labels (and predictions)."""
    from data_loader import DEFAULT_DATA_DIR, parse_xy

    out_dir = preprocess_dataset(args.data or DEFAULT_DATA_DIR, args.output)
    paths = sorted(glob.glob(os.path.join(out_dir, "xy_*.png")))
    model = None
    if args.model:
        import camera as cam_mod
        from model import load_model
        model = load_model(args.model, "cpu")
    rows = []
    for p in paths:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        x, y = parse_xy(p)
        err, pred = 0.0, None
        if model is not None:
            out = model(cam_mod.preprocess(img, "cpu", model.dtype)).float().numpy().flatten()
            px, py = (out[0] + 1) * w / 2.0, (out[1] + 1) * h / 2.0
            err = float(np.hypot(px - x, py - y))
            pred = (int(px), int(py))
        rows.append((err, p, img, (int(min(x, w - 1)), int(min(y, h - 1))), pred))
    if model is not None:
        rows.sort(key=lambda r: -r[0])

    tiles = []
    for i, (err, p, img, lab, pred) in enumerate(rows):
        t = img.copy()
        cv2.circle(t, lab, 5, (0, 255, 0), -1)
        if pred is not None:
            cv2.circle(t, pred, 5, (255, 255, 0), 2)
        cv2.putText(t, "#%d" % i, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if model is not None:
            cv2.putText(t, "%.0fpx" % err, (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(cv2.resize(t, (args.tile, args.tile)))
        print("#%-4d %6.1f  %s" % (i, err, os.path.basename(p)))
    cols = args.cols
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(args.out, sheet)
    print("wrote %s  (red = marked lane, green = label, cyan circle = prediction)" % args.out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Preprocessing (lane marking) tools")
    sub = ap.add_subparsers(dest="cmd")
    r0 = sub.add_parser("run", help="write the preprocessed folder (train.py does this itself)")
    r0.add_argument("--data", default=None, help="folder with xy_*.jpg")
    r0.add_argument("--output", default=None, help="default: <data>_lanes")
    r = sub.add_parser("review", help="contact sheet of the preprocessed images + labels")
    r.add_argument("--data", default=None, help="folder with xy_*.jpg")
    r.add_argument("--output", default=None, help="preprocessed folder (default: <data>_lanes)")
    r.add_argument("--model", default=None, help="a .pth from train.py: sort by its error")
    r.add_argument("--out", default="label_review.jpg")
    r.add_argument("--cols", type=int, default=10)
    r.add_argument("--tile", type=int, default=160)
    a = ap.parse_args()
    if a.cmd == "run":
        from data_loader import DEFAULT_DATA_DIR
        preprocess_dataset(a.data or DEFAULT_DATA_DIR, a.output)
    elif a.cmd == "review":
        _review(a)
    else:
        ap.print_help()
