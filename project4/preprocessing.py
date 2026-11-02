"""Deterministic preprocessing, between the data loader and augmentation.

    Data Loader -> Preprocessing -> Data Augmentation -> Model training

Unlike augmentation (random, training only), preprocessing is applied the same
way every time -- in training *and* on the robot. Its settings are saved in
each model's ``.json`` and ``live_demo.ipynb`` rebuilds the same
``Preprocessor`` when it loads the model.

Three switches, all off by default (= identity, so old models are unaffected):

* ``roi_top``  (0~0.9)  Crop away the top fraction of the image (walls, people,
                        chairs) and stretch the rest back to the original size,
                        so the model only sees the floor. Labels move with it.
* ``fixed_y``  (0~1)    Replace every label's y by a constant (fraction of the
                        height). The data's y is inconsistent (72~212 px); with
                        y fixed the model only has to learn x. 0.65 matches the
                        auto-labeller's look-ahead (median of the data: 0.63).
* ``exclude``  (file)   Text file with one image file name per line (``#`` for
                        comments); those images are dropped from training.
                        Use ``python preprocessing.py review`` to find them.

On the robot:
    prep = Preprocessor.from_config(model.meta.get("preprocessing"))
    frame_in = prep.apply_frame(frame)          # BGR uint8 in, BGR uint8 out
    xy_full = prep.to_full(xy_model)            # model output -> full-frame coords

Python 3.6 compatible (``apply_frame`` / ``to_full`` run on the Jetson Nano).
"""
import os

import numpy as np
import PIL.Image

__all__ = ["Preprocessor", "read_exclude"]


def read_exclude(path):
    """Set of file names listed in an exclude file (empty set if None)."""
    if not path:
        return set()
    names = set()
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                names.add(os.path.basename(line))
    return names


class Preprocessor(object):
    """Callable ``(PIL image, x_px, y_px) -> (PIL image, x_px, y_px)``."""

    def __init__(self, roi_top=0.0, fixed_y=None, exclude=None):
        if not 0.0 <= roi_top < 0.9:
            raise ValueError("roi_top must be in [0, 0.9)")
        if fixed_y is not None and not 0.0 <= fixed_y <= 1.0:
            raise ValueError("fixed_y must be in [0, 1]")
        self.roi_top = float(roi_top)
        self.fixed_y = None if fixed_y is None else float(fixed_y)
        self.exclude = exclude
        self.excluded = read_exclude(exclude)

    # ---- config (saved in the model .json) ------------------------------- #
    def config(self):
        return {"roi_top": self.roi_top, "fixed_y": self.fixed_y,
                "exclude": os.path.basename(self.exclude) if self.exclude else None,
                "exclude_path": os.path.abspath(self.exclude) if self.exclude else None,
                "n_excluded": len(self.excluded)}

    @classmethod
    def from_config(cls, cfg):
        """Rebuild from a model's ``meta['preprocessing']`` (None -> identity).

        The exclude list only matters for training, so it is not reloaded.
        """
        cfg = cfg or {}
        return cls(roi_top=cfg.get("roi_top") or 0.0, fixed_y=cfg.get("fixed_y"))

    @property
    def enabled(self):
        return self.roi_top > 0 or self.fixed_y is not None or bool(self.excluded)

    def tag(self):
        """Short name for file names, e.g. 'roi30+fy65+ex12' ('' when off)."""
        parts = []
        if self.roi_top > 0:
            parts.append("roi%02d" % round(self.roi_top * 100))
        if self.fixed_y is not None:
            parts.append("fy%02d" % round(self.fixed_y * 100))
        if self.excluded:
            parts.append("ex%d" % len(self.excluded))
        return "+".join(parts)

    def __repr__(self):
        return "Preprocessor(roi_top=%s, fixed_y=%s, excluded=%d)" % (
            self.roi_top, self.fixed_y, len(self.excluded))

    # ---- dataset level ---------------------------------------------------- #
    def keep(self, path):
        return os.path.basename(path) not in self.excluded

    # ---- sample level (training) ----------------------------------------- #
    def __call__(self, image, x, y):
        w, h = image.size
        if self.fixed_y is not None:
            y = self.fixed_y * h
        if self.roi_top > 0:
            image = self._crop(image)
            top = self.roi_top * h
            y = max(0.0, (y - top) * h / (h - top))
        return image, x, y

    def _crop(self, image):
        w, h = image.size
        top = self.roi_top * h
        return image.resize((w, h), PIL.Image.BILINEAR, box=(0, top, w, h))

    # ---- robot side ------------------------------------------------------- #
    def apply_frame(self, frame):
        """Same image transform for a camera frame (HxWx3 uint8, any channel order).

        Uses the exact PIL code path as training so both sides match bit for bit.
        """
        if self.roi_top <= 0:
            return frame
        return np.asarray(self._crop(PIL.Image.fromarray(frame)))

    def to_full(self, xy):
        """Model output (normalised, in the cropped image) -> normalised full frame."""
        x, y = float(xy[0]), float(xy[1])
        if self.roi_top > 0:
            # normalised -> fraction of crop height -> fraction of full height -> normalised
            frac = (y + 1.0) / 2.0
            y = (self.roi_top + frac * (1.0 - self.roi_top)) * 2.0 - 1.0
        return np.array([x, y], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Label review tool (PC)
# --------------------------------------------------------------------------- #
def _review(args):
    """Contact sheet of every image with its label (and the model's prediction).

    With --model the sheet is sorted by error, worst first -- mislabelled
    images usually show up at the top. Copy their file names into an exclude
    file and retrain with ``--exclude``.
    """
    import glob
    import cv2
    from data_loader import DEFAULT_DATA_DIR, parse_xy

    paths = sorted(glob.glob(os.path.join(args.data or DEFAULT_DATA_DIR, "xy_*.jpg")))
    excluded = read_exclude(args.exclude)
    rows = []
    model = prep = None
    if args.model:
        import torch
        import camera as cam_mod
        from model import load_model
        model = load_model(args.model, "cpu")
        prep = Preprocessor.from_config(model.meta.get("preprocessing"))
    for p in paths:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        x, y = parse_xy(p)
        err, pred = 0.0, None
        if model is not None:
            out = model(cam_mod.preprocess(prep.apply_frame(img), "cpu", model.dtype))
            pred = prep.to_full(out.float().numpy().flatten())
            px, py = (pred[0] + 1) * w / 2.0, (pred[1] + 1) * h / 2.0
            err = abs(px - x) if prep.fixed_y is not None else float(np.hypot(px - x, py - y))
            pred = (int(px), int(py))
        rows.append((err, p, img, (int(x), int(y)), pred))
    if model is not None:
        rows.sort(key=lambda r: -r[0])

    tiles = []
    for i, (err, p, img, lab, pred) in enumerate(rows):
        t = img.copy()
        cv2.line(t, (t.shape[1] // 2, t.shape[0] - 1), lab, (0, 255, 0), 2)
        cv2.circle(t, lab, 5, (0, 255, 0), -1)
        if pred is not None:
            cv2.circle(t, pred, 5, (0, 0, 255), 2)
        color = (0, 0, 255) if os.path.basename(p) in excluded else (255, 255, 255)
        cv2.putText(t, "#%d" % i, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if model is not None:
            cv2.putText(t, "%.0fpx" % err, (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        tiles.append(cv2.resize(t, (args.tile, args.tile)))
        print("#%-4d %6.1f  %s%s" % (i, err, os.path.basename(p),
                                       "  (excluded)" if os.path.basename(p) in excluded else ""))
    cols = args.cols
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(args.out, sheet)
    print("wrote %s  (green = label, red circle = prediction, red number = excluded)" % args.out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Preprocessing tools")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("review", help="contact sheet of labels (sorted by error with --model)")
    r.add_argument("--data", default=None)
    r.add_argument("--model", default=None, help="a .pth from train.py: sort by its error")
    r.add_argument("--exclude", default=None, help="mark images already in this exclude file")
    r.add_argument("--out", default="label_review.jpg")
    r.add_argument("--cols", type=int, default=10)
    r.add_argument("--tile", type=int, default=160)
    a = ap.parse_args()
    if a.cmd == "review":
        _review(a)
    else:
        ap.print_help()
