"""Camera input and pre-processing.

``open_camera(source)`` returns an object with ``read()`` (latest BGR uint8
frame, HxWx3) and ``stop()``:

* ``'jetbot'``  -- ``jetbot.Camera`` (CSI camera on the robot, 224x224).
* ``'opencv'``  -- ``cv2.VideoCapture(index)`` (USB webcam / PC testing).
* ``'folder'``  -- loops over jpgs in a folder (offline testing with the dataset).
* ``'dummy'``   -- synthetic frames, no hardware needed.

``preprocess(frame)`` reproduces the slides' P.59 ``preprocess`` (BGR frame in,
normalised NCHW tensor out), with the dtype chosen per model instead of a
hard-coded ``.half()`` -- trtexec engines expect FP32 (handoff 4.2).

Python 3.6 compatible (runs on the Jetson Nano).
"""
import glob
import os
import time

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF

__all__ = ["open_camera", "preprocess", "draw_overlay", "to_jpeg", "degrade"]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
class _JetbotSource(object):
    def __init__(self, width, height, fps):
        from jetbot import Camera
        try:
            self.camera = Camera.instance(width=width, height=height, fps=fps)
        except AttributeError:  # newer jetbot: plain constructor
            self.camera = Camera(width=width, height=height, fps=fps)

    def read(self):
        return self.camera.value

    def stop(self):
        self.camera.stop()


class _OpenCVSource(object):
    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index)
        self.size = (width, height)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv2.resize(frame, self.size)

    def stop(self):
        self.cap.release()


class _FolderSource(object):
    def __init__(self, folder, width, height, fps):
        self.paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        if not self.paths:
            raise FileNotFoundError("no jpg in %s" % folder)
        self.size = (width, height)
        self.period = 1.0 / fps
        self.t0 = time.time()

    def read(self):
        i = int((time.time() - self.t0) / self.period) % len(self.paths)
        return cv2.resize(cv2.imread(self.paths[i]), self.size)

    def stop(self):
        pass


class _DummySource(object):
    """A moving synthetic 'road' so the whole loop can run without hardware."""

    def __init__(self, width, height):
        self.size = (width, height)
        self.t0 = time.time()

    def read(self):
        w, h = self.size
        img = np.full((h, w, 3), 60, np.uint8)
        off = int(0.25 * w * np.sin(time.time() - self.t0))
        cx = w // 2 + off
        cv2.line(img, (cx - w // 3, h - 1), (cx - w // 12, h // 3), (255, 255, 255), 3)
        cv2.line(img, (cx + w // 3, h - 1), (cx + w // 12, h // 3), (255, 255, 255), 3)
        return img

    def stop(self):
        pass


def open_camera(source="jetbot", width=224, height=224, fps=21, index=0, folder=None):
    """Open a frame source; see module docstring."""
    if source == "jetbot":
        return _JetbotSource(width, height, fps)
    if source == "opencv":
        return _OpenCVSource(index, width, height)
    if source == "folder":
        return _FolderSource(folder, width, height, fps=2)
    if source == "dummy":
        return _DummySource(width, height)
    raise ValueError("unknown camera source %r" % source)


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #
_norm_cache = {}


def _mean_std(device, dtype):
    key = (str(device), dtype)
    if key not in _norm_cache:
        _norm_cache[key] = (torch.tensor(MEAN, device=device, dtype=dtype)[:, None, None],
                            torch.tensor(STD, device=device, dtype=dtype)[:, None, None])
    return _norm_cache[key]


def preprocess(frame, device="cpu", dtype=torch.float32):
    """BGR uint8 HxWx3 -> normalised (1, 3, 224, 224) tensor (channels stay BGR).

    Matches training (``data_loader.image_to_tensor``): the course flips RGB to
    BGR when training, and the JetBot camera already delivers BGR.
    """
    if frame.shape[0] != INPUT_SIZE or frame.shape[1] != INPUT_SIZE:
        frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    image = PIL.Image.fromarray(frame)
    t = TF.to_tensor(image).to(device=device, dtype=dtype)
    mean, std = _mean_std(t.device, dtype)
    t.sub_(mean).div_(std)
    return t[None, ...]


def degrade(frame, perlin=False, compression=False):
    """Optionally apply augmentation-style disturbances to a live BGR frame.

    Lets the UI test how robust each checkpoint is to lighting noise /
    compression artefacts at inference time. Returns a BGR frame.
    """
    if not (perlin or compression):
        return frame
    from augmentation import Augmenter
    aug = Augmenter(perlin=perlin, compression=compression, perlin_p=1.0, compression_p=1.0)
    rgb = PIL.Image.fromarray(frame[:, :, ::-1])
    rgb, _, _ = aug(rgb, 0.0, 0.0)
    return np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
def draw_overlay(frame, xy=None, steering=None):
    """Copy of ``frame`` with the predicted target point and a steering bar."""
    out = frame.copy()
    h, w = out.shape[:2]
    if xy is not None:
        px = int(round((float(xy[0]) + 1) * w / 2.0))
        py = int(round((float(xy[1]) + 1) * h / 2.0))
        cv2.line(out, (w // 2, h - 1), (px, py), (0, 255, 0), 2)
        cv2.circle(out, (px, py), 5, (0, 255, 0), -1)
    if steering is not None:
        cx = w // 2
        bar = int(max(-1.0, min(1.0, steering)) * (w // 2 - 4))
        cv2.rectangle(out, (cx, 4), (cx + bar, 10), (0, 165, 255), -1)
        cv2.line(out, (cx, 2), (cx, 12), (255, 255, 255), 1)
    return out


def to_jpeg(frame, quality=80):
    """BGR frame -> JPEG bytes for ``ipywidgets.Image`` (like jetbot.bgr8_to_jpeg)."""
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return bytes(buf) if ok else b""
