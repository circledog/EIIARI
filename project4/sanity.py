"""Deployment checks: weights fingerprint and expected-output comparison.

When ``train.py`` saves a model it stores, in the model's ``.json``:

* ``weights_sha256`` -- fingerprint of the ``.pth``; ``loader.load_model``
  refuses a file whose fingerprint does not match (broken / partial copy, or
  a ``.json`` that belongs to another file).
* ``sanity`` -- a few lane-marked validation frames (PNG, lossless, so every
  machine decodes exactly the same pixels) and the model's FP32 output for
  each of them, computed on the PC.

On the robot ``check(model)`` runs those frames through the loaded model
(PyTorch FP16, pruned rebuild or TensorRT engine) and compares the outputs.
If the largest difference exceeds the tolerance (1 px for FP32, 2 px for
FP16 / TensorRT) something went wrong in the conversion and ``live_demo``
does not allow driving.

The frames are stored after lane marking, so the check covers the network
and its runtime only; lane marking itself is the same code on both sides.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import base64
import copy
import hashlib

import cv2
import numpy as np
import torch

# Same tensor conversion as on the robot
import camera as cam_mod
# Same lane marking as on the robot
from preprocessing import apply_frame

# Public names of this module
__all__ = ["file_sha256", "make_sanity", "check", "N_SAMPLES"]

# Number of frames stored per model
N_SAMPLES = 8
# Pixels per normalised unit (224-px input)
PX_PER_UNIT = 112.0


# Fingerprint of a file
def file_sha256(path):
    """Hex SHA-256 of a file, read in chunks."""
    # Hash object
    h = hashlib.sha256()
    # Read the file in binary mode
    with open(path, "rb") as f:
        # 1 MiB at a time until the end of the file
        for chunk in iter(lambda: f.read(1 << 20), b""):
            # Add the chunk to the hash
            h.update(chunk)
    # Hex digest
    return h.hexdigest()


# Build the sanity data for a freshly trained model (PC side)
def make_sanity(model, image_paths, n=N_SAMPLES):
    """``{"samples": [{"file", "png_b64", "expected"}]}`` for the first ``n`` paths."""
    # Evaluate a CPU FP32 copy so the reference is exact
    ref = _cpu_fp32_copy(model)
    # One entry per frame
    samples = []
    # The first n images (validation images, chosen by the caller)
    for path in list(image_paths)[:n]:
        # Raw camera-like BGR frame, resized like the robot's 224x224 camera
        frame = cv2.resize(cv2.imread(path), (224, 224))
        # Lane marking, exactly as on the robot
        marked = apply_frame(frame)
        # Lossless encoding so the robot decodes identical pixels
        ok, png = cv2.imencode(".png", marked)
        # Reference output of the network
        with torch.no_grad():
            # Robot tensor conversion, FP32 on the CPU
            out = ref(cam_mod.preprocess(marked, "cpu", torch.float32)).flatten().tolist()
        # Store file name, image and expected output
        samples.append({"file": path.replace("\\", "/").split("/")[-1],
                        "png_b64": base64.b64encode(png.tobytes()).decode("ascii"),
                        "expected": [float(v) for v in out]})
    # Data for the .json
    return {"samples": samples}


# Compare a loaded model with the stored expectations (robot side)
def check(model, meta=None):
    """Run the stored frames through ``model`` (a ``loader.LoadedModel``).

    Returns a dict: ``status`` ('passed' | 'failed' | 'missing'),
    ``max_error_px``, ``tolerance_px``, ``samples``, ``message``.
    """
    # Metadata from the model unless given
    meta = meta if meta is not None else (model.meta or {})
    # Stored frames, if any
    samples = (meta.get("sanity") or {}).get("samples") or []
    # FP32 PyTorch must match tightly; FP16 and TensorRT get a looser bound
    tol = 1.0 if (model.kind == "torch" and model.dtype == torch.float32) else 2.0
    # Older models have no stored frames
    if not samples:
        # Nothing to compare
        return {"status": "missing", "max_error_px": None, "tolerance_px": tol, "samples": 0,
                "message": "這個模型的 .json 沒有 sanity 資料（舊版 train.py 產生），無法比對"}
    # Largest difference so far
    worst = 0.0
    # Every stored frame
    for s in samples:
        # Decode the lossless frame
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(s["png_b64"]), np.uint8), cv2.IMREAD_COLOR)
        # Model output for it, as floats
        out = model(cam_mod.preprocess(frame, model.device, model.dtype)).float().cpu().numpy().flatten()
        # NaN / inf output is a failure on its own
        if not np.all(np.isfinite(out)):
            # Report it right away
            return {"status": "failed", "max_error_px": float("inf"), "tolerance_px": tol,
                    "samples": len(samples), "message": "模型輸出 NaN / inf"}
        # Largest coordinate difference in pixels
        worst = max(worst, float(np.max(np.abs(out - np.asarray(s["expected"])))) * PX_PER_UNIT)
    # Passed if within the tolerance
    status = "passed" if worst <= tol else "failed"
    # Result for the UI
    return {"status": status, "max_error_px": worst, "tolerance_px": tol, "samples": len(samples),
            "message": "%d 張比對，最大差 %.3f px（容許 %.1f px）" % (len(samples), worst, tol)}


# CPU FP32 copy of a network, leaving the original untouched
def _cpu_fp32_copy(model):
    # Deep copy (the training model keeps its device and dtype), on the CPU, FP32, eval mode
    return copy.deepcopy(model).cpu().float().eval()
