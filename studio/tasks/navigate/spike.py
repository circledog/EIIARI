"""TASKS M3-1 spike: how fast are the free-space options on this machine?

Run on the Jetson Nano (inside the Jupyter container) and paste the output
into SPIKE.md::

    cd /workspace/EIIARI && python3 -m studio.tasks.navigate.spike
    python3 -m studio.tasks.navigate.spike --depth-onnx depth_anything_v2_vits_252.onnx

Measures, per input size:

* the floor-appearance method (``freespace.py``) and the tracker (``tracker.py``),
  pure OpenCV on the CPU;
* optionally a monocular depth network exported to ONNX, built with ``trtexec``
  in FP16 (needs TensorRT; the ONNX file is not part of the repository).

Python 3.6 compatible (runs on the Jetson Nano).
"""
import argparse
import os
import re
import subprocess
import time

import numpy as np

# Methods under test
from studio.tasks.navigate.freespace import FreeSpace
# Tracker
from studio.tasks.navigate.tracker import Tracker

# trtexec of JetPack
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


# Synthetic floor picture of a given size
def picture(size, seed=0):
    """Noisy floor with a box, like the tests."""
    import cv2
    # Reproducible noise
    rng = np.random.RandomState(seed)
    # Floor
    img = np.clip(rng.normal((60, 90, 130), 12, (size, size, 3)), 0, 255).astype(np.uint8)
    # Box
    cv2.rectangle(img, (size // 3, size // 2), (2 * size // 3, 3 * size // 4), (170, 80, 40), -1)
    # Picture
    return img


# Median milliseconds of fn() over n runs
def median_ms(fn, n=200, warmup=10):
    """Run ``fn`` and return the median time in ms."""
    # Warm-up
    for _ in range(warmup):
        # Run
        fn()
    # Times
    times = []
    # Timed runs
    for _ in range(n):
        # Start
        t0 = time.perf_counter()
        # Run
        fn()
        # Duration
        times.append((time.perf_counter() - t0) * 1000.0)
    # Median
    return float(np.median(times))


# Build and time an ONNX model with trtexec (FP16)
def time_onnx(path):
    """Median GPU compute time in ms reported by trtexec, or the error text."""
    # trtexec must exist
    if not os.path.exists(TRTEXEC):
        # Not on this machine
        return "trtexec not found (%s)" % TRTEXEC
    # Build in FP16 and run the benchmark
    out = subprocess.run([TRTEXEC, "--onnx=%s" % path, "--fp16", "--workspace=1024", "--avgRuns=50"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=3600)
    # Text output
    text = out.stdout.decode("utf-8", "replace")
    # "GPU Compute ... median = X ms" (TensorRT 7 / 8 formats)
    m = re.search(r"GPU Compute.*?median\s*=\s*([\d.]+)\s*ms", text, re.S) or re.search(r"median:\s*([\d.]+)\s*ms", text)
    # Number or the tail of the log
    return float(m.group(1)) if m else text[-800:]


# Command-line entry point
def main(argv=None):
    """Print a table of timings."""
    # Options
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Input sizes
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 256, 320])
    # Optional depth network
    ap.add_argument("--depth-onnx", default=None, help="ONNX depth model to time with trtexec (FP16)")
    # Parse
    args = ap.parse_args(argv)
    # Header
    print("%-6s %-14s %-14s" % ("size", "freespace ms", "tracker ms"))
    # Every size
    for size in args.sizes:
        # Test picture
        img = picture(size)
        # Estimator fitted on it
        fs = FreeSpace()
        # Fit
        fs.fit(img)
        # Tracker started on it
        tr = Tracker()
        # Start
        tr.start(img, 0.5, 0.3)
        # Timings
        print("%-6d %-14.2f %-14.2f" % (size, median_ms(lambda: fs.estimate(img)), median_ms(lambda: tr.update(img))))
    # Depth network
    if args.depth_onnx:
        # Time it
        print("depth %s: %s ms (trtexec FP16, GPU compute median)" % (args.depth_onnx, time_onnx(args.depth_onnx)))


# Run as a module
if __name__ == "__main__":
    # Entry point
    main()
