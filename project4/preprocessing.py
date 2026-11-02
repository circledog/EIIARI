"""Deterministic preprocessing, between the data loader and augmentation:
mark the lane lines (white tape) on the image.

    Data Loader -> Preprocessing -> Data Augmentation -> Model training

Unlike augmentation (random, training only), preprocessing is applied the same
way every time -- in training *and* on the robot. Whether a model was trained
with it is saved in the model's ``.json``; ``live_demo.ipynb`` rebuilds the
same ``Preprocessor`` when it loads the model.

What it does (``lanes=True``; ``lanes=False`` = identity, old models unaffected)
------------------------------------------------------------------------------
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

On the robot:
    prep = Preprocessor.from_config(model.meta.get("preprocessing"))
    frame_in = prep.apply_frame(frame)          # BGR uint8 in, BGR uint8 out

Python 3.6 compatible (runs on the Jetson Nano).
"""
import os

import cv2
import numpy as np
import PIL.Image

__all__ = ["Preprocessor", "lane_mask", "mark_lanes", "LANE_COLOR_BGR", "DEFAULT_PARAMS"]

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


class Preprocessor(object):
    """Callable ``(PIL image, x_px, y_px) -> (PIL image, x_px, y_px)``."""

    def __init__(self, lanes=False, params=None):
        self.lanes = bool(lanes)
        self.params = dict(DEFAULT_PARAMS)
        self.params.update(params or {})

    # ---- config (saved in the model .json) ------------------------------- #
    def config(self):
        return {"lanes": self.lanes, "params": dict(self.params) if self.lanes else None}

    @classmethod
    def from_config(cls, cfg):
        """Rebuild from a model's ``meta['preprocessing']`` (None -> identity)."""
        cfg = cfg or {}
        if cfg.get("roi_top") or cfg.get("fixed_y") is not None:
            print("warning: this model was trained with ROI crop / fixed y, which no longer "
                  "exist; retrain it")
        return cls(lanes=cfg.get("lanes", False), params=cfg.get("params"))

    @property
    def enabled(self):
        return self.lanes

    def tag(self):
        """Short name for file names ('' when off)."""
        return "lanes" if self.lanes else ""

    def __repr__(self):
        return "Preprocessor(lanes=%s)" % self.lanes

    # ---- sample level (training) ----------------------------------------- #
    def __call__(self, image, x, y):
        if self.lanes:
            bgr = np.asarray(image.convert("RGB"))[..., ::-1]
            image = PIL.Image.fromarray(np.ascontiguousarray(mark_lanes(bgr, self.params)[..., ::-1]))
        return image, x, y

    # ---- robot side ------------------------------------------------------- #
    def apply_frame(self, frame):
        """Same transform for a camera frame (HxWx3 BGR uint8)."""
        return mark_lanes(frame, self.params) if self.lanes else frame


# --------------------------------------------------------------------------- #
# Preview / review tool (PC)
# --------------------------------------------------------------------------- #
def _review(args):
    """Contact sheet of every image with the lane marking and its label.

    With --model the model's own preprocessing is used and the sheet is sorted
    by the model's error, worst first.
    """
    import glob
    from data_loader import DEFAULT_DATA_DIR, parse_xy

    paths = sorted(glob.glob(os.path.join(args.data or DEFAULT_DATA_DIR, "xy_*.jpg")))
    model = None
    prep = Preprocessor(lanes=not args.no_lanes)
    if args.model:
        import camera as cam_mod
        from model import load_model
        model = load_model(args.model, "cpu")
        prep = Preprocessor.from_config(model.meta.get("preprocessing"))
    rows = []
    for p in paths:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        x, y = parse_xy(p)
        shown = prep.apply_frame(img)
        err, pred = 0.0, None
        if model is not None:
            out = model(cam_mod.preprocess(shown, "cpu", model.dtype)).float().numpy().flatten()
            px, py = (out[0] + 1) * w / 2.0, (out[1] + 1) * h / 2.0
            err = float(np.hypot(px - x, py - y))
            pred = (int(px), int(py))
        rows.append((err, p, shown, (int(min(x, w - 1)), int(min(y, h - 1))), pred))
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
    ap = argparse.ArgumentParser(description="Preprocessing tools")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("review", help="contact sheet: lane marking + labels (sorted by error with --model)")
    r.add_argument("--data", default=None)
    r.add_argument("--model", default=None, help="a .pth from train.py: its preprocessing, sorted by error")
    r.add_argument("--no-lanes", action="store_true", help="show the images without lane marking")
    r.add_argument("--out", default="label_review.jpg")
    r.add_argument("--cols", type=int, default=10)
    r.add_argument("--tile", type=int, default=160)
    a = ap.parse_args()
    if a.cmd == "review":
        _review(a)
    else:
        ap.print_help()
