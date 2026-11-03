"""Convert uploaded PyTorch models to TensorRT on the JetBot (run on the Nano).

Folders (both ignored by git):
    models/       PyTorch models uploaded from the PC (train.py output: .pth + .json [+ .onnx])
    models_trt/   TensorRT models made by this script; live_demo only lists these

Usage (on the Nano, inside project4/):
    python3 convert_trt.py                     # every models/*.pth without a TRT version yet
    python3 convert_trt.py models/X.pth        # just this one
    python3 convert_trt.py --force             # convert again even if models_trt/X_trt.pth exists
    python3 convert_trt.py --method trtexec    # ONNX -> trtexec instead of torch2trt (needs X.onnx)

Methods:
    torch2trt (default)  rebuilds the network from the .pth (pruned or not) and
                         converts it in-process with the course's torch2trt, FP16 inside,
                         FP32 input. No ONNX file needed.
    trtexec              /usr/src/tensorrt/bin/trtexec --onnx=X.onnx --fp16, then packs
                         the engine into a TRTModule .pth (compression.wrap_trt_engine).

After converting, each model is checked before it is used:
    * reloaded from disk with loader.load_model (as live_demo does),
    * compared with the outputs stored by train.py (sanity frames, <= 2 px),
    * timed (model only, batch 1).
The result goes into models_trt/X_trt.json ("trt_validation"); live_demo
repeats the comparison when loading and refuses to drive if it fails.

Python 3.6 compatible.
"""
import argparse
import glob
import os
import subprocess
import time

import numpy as np
import torch

# Checkpoint loading and .json handling
from loader import load_model, read_meta, write_meta
# Fingerprint and expected-output comparison
import sanity

# Folder with this file (project4/)
_HERE = os.path.dirname(os.path.abspath(__file__))
# Uploaded PyTorch models
SRC_DIR = os.path.join(_HERE, "models")
# Converted TensorRT models
DST_DIR = os.path.join(_HERE, "models_trt")
# trtexec location in JetPack
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
# GPU used for conversion and checks
DEVICE = "cuda"


# Output path of the TRT version of a model
def trt_path(src, dst_dir=DST_DIR):
    """``models/X.pth`` -> ``models_trt/X_trt.pth``."""
    # File name without extension
    stem = os.path.splitext(os.path.basename(src))[0]
    # Same name with the _trt suffix in the TRT folder
    return os.path.join(dst_dir, stem + "_trt.pth")


# Convert with torch2trt inside this process
def convert_torch2trt(src, out, workspace=1 << 25):
    """Rebuild the network from ``src`` and save a TRTModule state_dict to ``out``."""
    # The course's TensorRT converter (only on the JetBot)
    from torch2trt import torch2trt
    # FP32 PyTorch model on the GPU (pruned networks are rebuilt from their shapes)
    model = load_model(src, DEVICE, half=False)
    # Example input with the network's input shape
    sample = torch.zeros((1, 3, 224, 224), device=DEVICE)
    # Build the engine: FP16 inside, FP32 input / output
    trt_model = torch2trt(model.module, [sample], fp16_mode=True, max_workspace_size=workspace)
    # Save it in the format loader.load_model recognises (dict with an "engine" key)
    torch.save(trt_model.state_dict(), out)


# Convert with trtexec from the ONNX file
def convert_trtexec(src, out):
    """``X.onnx`` next to ``src`` -> trtexec engine -> TRTModule .pth at ``out``."""
    # Engine packing helper
    from compression import wrap_trt_engine
    # ONNX file exported by train.py --onnx
    onnx = os.path.splitext(src)[0] + ".onnx"
    # Without it trtexec has nothing to read
    if not os.path.exists(onnx):
        # Explain how to get it
        raise FileNotFoundError("%s not found: train with --onnx, or use --method torch2trt" % onnx)
    # Raw engine file next to the output
    engine = os.path.splitext(out)[0] + ".engine"
    # Build the engine (FP16 inside, FP32 input)
    subprocess.check_call([TRTEXEC, "--onnx=" + onnx, "--saveEngine=" + engine, "--fp16"])
    # Pack it into a TRTModule .pth; the .json is written by convert_one
    wrap_trt_engine(engine, out)


# Wait for queued GPU work (no-op without CUDA)
def _sync():
    # Only when a GPU is present
    if torch.cuda.is_available():
        # Block until the GPU is idle
        torch.cuda.synchronize()


# Model-only latency of a loaded model
def time_model(model, n=100, warmup=20):
    """(median ms, p95 ms) for batch-1 inference, excluding camera / preprocessing."""
    # Zero input in the dtype the model expects
    x = torch.zeros((1, 3, 224, 224), device=model.device, dtype=model.dtype)
    # Warm-up runs, not timed
    for _ in range(warmup):
        # Run the network
        model(x)
    # Duration of every timed run
    times = []
    # Timed runs
    for _ in range(n):
        # Make sure earlier GPU work is done
        _sync()
        # Start time
        t0 = time.time()
        # Run the network
        model(x)
        # Wait for the GPU
        _sync()
        # Duration in milliseconds
        times.append((time.time() - t0) * 1000.0)
    # Median and 95th percentile
    return float(np.median(times)), float(np.percentile(times, 95))


# Convert one model, verify it and write its .json
def convert_one(src, method, dst_dir=DST_DIR):
    """Convert ``src`` and return the validation dict (``status`` 'passed' / 'failed' / 'missing')."""
    # Output file
    out = trt_path(src, dst_dir)
    # Show what is happening
    print("== %s -> %s (%s)" % (os.path.basename(src), os.path.relpath(out, _HERE), method))
    # The source must be intact before spending minutes on it (checks its SHA256)
    load_model(src, "cpu")
    # Start time of the conversion
    t0 = time.time()
    # Chosen method
    if method == "torch2trt":
        # In-process conversion
        convert_torch2trt(src, out)
    # trtexec from ONNX
    else:
        # External conversion
        convert_trtexec(src, out)
    # Conversion time
    seconds = time.time() - t0
    # Metadata of the PyTorch model (sanity frames, metrics, label, ...)
    meta = read_meta(src)
    # Same metadata, marked as the TensorRT version
    meta = dict(meta, label=(meta.get("label") or os.path.basename(src)) + " | TensorRT",
                runtime="tensorrt", trt_method=method, source=os.path.basename(src),
                source_sha256=meta.get("weights_sha256"), weights_sha256=sanity.file_sha256(out))
    # Write it so load_model can verify the new file
    write_meta(out, meta)
    # Reload from disk exactly as live_demo does
    model = load_model(out, DEVICE)
    # Compare with the outputs stored at training time
    check = sanity.check(model)
    # Model-only latency
    median_ms, p95_ms = time_model(model)
    # Everything about the conversion for the report
    meta["trt_validation"] = dict(check, convert_seconds=seconds, median_model_ms=median_ms, p95_model_ms=p95_ms,
                                  input_dtype=str(model.dtype).replace("torch.", ""),
                                  device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__,
                                  timing_scope="batch 1, model only (no camera / preprocessing / control)")
    # Store it next to the model
    write_meta(out, meta)
    # One-line result
    print("   %s: %s | %.1f ms median, %.1f ms p95 | %.0f s"
          % (check["status"].upper(), check["message"], median_ms, p95_ms, seconds))
    # Result for the summary
    return check


# Command-line entry point
def main():
    # Parser that shows the module docstring as help
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Models to convert (default: all in models/)
    ap.add_argument("models", nargs="*", help="PyTorch .pth files (default: every models/*.pth)")
    # Conversion method
    ap.add_argument("--method", choices=["torch2trt", "trtexec"], default="torch2trt")
    # Convert again even if a TRT version exists
    ap.add_argument("--force", action="store_true", help="also convert models that already have a TRT version")
    # Output folder
    ap.add_argument("--out", default=DST_DIR, help="output folder (default models_trt/)")
    # Parse the command line
    args = ap.parse_args()
    # TensorRT needs the GPU
    if not torch.cuda.is_available():
        # Explain where to run it
        raise SystemExit("CUDA not available: run this on the JetBot")
    # Create the output folder if needed
    if not os.path.isdir(args.out):
        # Make it
        os.makedirs(args.out)
    # Given files, or every PyTorch model in models/
    sources = args.models or sorted(glob.glob(os.path.join(SRC_DIR, "*.pth")))
    # Skip models that already have a TRT version unless --force
    todo = [s for s in sources if args.force or not os.path.exists(trt_path(s, args.out))]
    # Mention the skipped ones
    for s in sorted(set(sources) - set(todo)):
        # One line each
        print("skip %s (already converted; --force to redo)" % os.path.basename(s))
    # Nothing to do
    if not todo:
        # Explain
        print("nothing to convert (models in %s: %d)" % (SRC_DIR, len(sources)))
        # Done
        return
    # (file, status) per model
    results = []
    # Convert one after the other
    for src in todo:
        # One model failing must not stop the others
        try:
            # Convert and verify
            status = convert_one(src, args.method, args.out)["status"]
        # Show the error and continue
        except Exception as e:
            # Error text
            print("   ERROR: %r" % e)
            # Record the failure
            status = "error"
        # Remember the result
        results.append((os.path.basename(src), status))
        # Free GPU memory before the next model
        torch.cuda.empty_cache()
    # Summary of all models
    print("\nsummary:")
    # One line per model
    for name, status in results:
        # Status and file name
        print("  %-8s %s" % (status, name))
    # Remind what failed means
    if any(s != "passed" for _, s in results):
        # live_demo blocks driving with these models
        print("models that did not pass are refused for driving by live_demo")


# Run when executed as a script
if __name__ == "__main__":
    # Start
    main()
