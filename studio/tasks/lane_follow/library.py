"""Lane following: datasets, models and training commands (web process, no torch).

* Datasets: folders under the data root with ``xy_<x>_<y>_<id>.jpg`` files,
  either flat or split into ``train/ validation/ [test/]`` (``train.py``'s two layouts).
* Models: ``.pth`` + ``.json`` in ``models/`` (PyTorch, from the PC) and
  ``models_trt/`` (TensorRT, converted on the car).
* Training: ``TRAIN_SCHEMA`` (the form) -> ``build_train_cmd`` -> ``train.py``
  command line, run as a background job; ``parse_train_log`` turns the log
  into progress (loss curves, ETA).

Python 3.6 compatible (runs on the Jetson Nano).
"""
import glob
import json
import os
import re
import time

# project4 folder
from studio import PROJECT4_DIR

# Network names of project4/loader.py ARCHS (choices of --arch)
ARCHS = [["resnet18", "ResNet-18"], ["mobilenet_v2", "MobileNetV2"], ["resnet18_coord", "ResNet-18 + CoordConv"],
         ["resnet18_dsnt", "ResNet-18 + DSNT"], ["resnet18_coord_dsnt", "ResNet-18 + CoordConv + DSNT"]]
# Split folder names of the team's layout
SPLITS = ("train", "validation", "test")
# Label file names: xy_<x>_<y>_<id>.jpg
LABEL = re.compile(r"^xy_(\d+)_(\d+)_.+\.jpg$")
# Safe model / dataset names (no path separators)
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.+\-]{1,200}$")


# Folders used by the task (settings may override them)
def folders(settings):
    """``{'data', 'models', 'trt', 'runs'}`` absolute folders."""
    # Overrides from the server settings, project4 defaults otherwise
    return {"data": settings.get("lane_data_dir") or os.path.join(PROJECT4_DIR, "data"),
            "models": settings.get("lane_models_dir") or os.path.join(PROJECT4_DIR, "models"),
            "trt": settings.get("lane_trt_dir") or os.path.join(PROJECT4_DIR, "models_trt")}


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
# Number of labelled jpgs in a folder
def _count(folder):
    # Files matching the label pattern
    return sum(1 for n in os.listdir(folder) if LABEL.match(n)) if os.path.isdir(folder) else 0


# Every dataset below the data root
def find_datasets(root, max_depth=3):
    """List of ``{'id', 'path', 'layout', 'counts'}``; ``id`` is the path relative to ``root``."""
    # Found datasets
    out = []
    # No data folder yet
    if not os.path.isdir(root):
        # Nothing
        return out
    # Walk the tree (followlinks: project4/data may be a symlink)
    for here, dirs, files in os.walk(root, followlinks=True):
        # Depth below the root
        depth = os.path.relpath(here, root).count(os.sep) + (0 if here == root else 1)
        # Too deep: do not descend further
        if depth >= max_depth:
            # Stop descending
            dirs[:] = []
        # Split layout: train/ and validation/ with labelled images
        if "train" in dirs and "validation" in dirs:
            # Counts per split
            counts = dict((s, _count(os.path.join(here, s))) for s in SPLITS)
            # A dataset
            out.append({"id": os.path.relpath(here, root), "path": here, "layout": "splits", "counts": counts})
            # Its split folders are not datasets of their own
            dirs[:] = []
            # Next folder
            continue
        # Flat layout: labelled images directly in the folder
        n = sum(1 for f in files if LABEL.match(f))
        # A dataset when it has labelled images
        if n:
            # Flat dataset
            out.append({"id": os.path.relpath(here, root), "path": here, "layout": "flat", "counts": {"all": n}})
    # Stable order
    out.sort(key=lambda d: d["id"])
    # Datasets
    return out


# Dataset by id (refuses ids that leave the data root)
def get_dataset(root, ds_id):
    """The dataset dict or None."""
    # Known datasets only (never a free path from the browser)
    for d in find_datasets(root):
        # Same id
        if d["id"] == ds_id:
            # Found
            return d
    # Unknown
    return None


# Image folder of a dataset split
def split_folder(ds, split):
    """Folder holding the images of ``split`` ('all' for flat datasets)."""
    # Split layout: the sub-folder
    if ds["layout"] == "splits":
        # Only known splits
        if split not in SPLITS:
            # Refuse
            raise ValueError("unknown split %r" % split)
        # Sub-folder
        return os.path.join(ds["path"], split)
    # Flat layout: the folder itself
    return ds["path"]


# Pixel label from a file name
def parse_label(name):
    """``(x, y)`` pixels from ``xy_<x>_<y>_<id>.jpg``, or None."""
    # Match the pattern
    m = LABEL.match(os.path.basename(name))
    # Two integers
    return (int(m.group(1)), int(m.group(2))) if m else None


# A page of images of a split
def list_images(ds, split, offset=0, limit=48):
    """``(total, [{'name', 'x', 'y'}])``, sorted by name."""
    # Folder of the split
    folder = split_folder(ds, split)
    # Labelled files, sorted
    names = sorted(n for n in os.listdir(folder) if LABEL.match(n)) if os.path.isdir(folder) else []
    # Requested page
    page = names[max(0, offset):max(0, offset) + max(1, min(500, limit))]
    # Names with labels
    return len(names), [dict(zip(("name", "x", "y"), (n,) + parse_label(n))) for n in page]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
# Read a model's .json (empty dict when missing / broken)
def read_meta(pth):
    """Metadata next to ``pth``."""
    # Same name with .json
    p = os.path.splitext(pth)[0] + ".json"
    # Missing or broken files give an empty dict
    try:
        # Read it
        with open(p, encoding="utf-8") as f:
            # Parsed
            return json.load(f)
    # Unusable
    except (OSError, ValueError):
        # Empty
        return {}


# One model's summary for the library page
def model_info(pth, kind):
    """Short JSON-able summary of a checkpoint (``kind``: 'torch' or 'trt')."""
    # Metadata
    meta = read_meta(pth)
    # File statistics
    st = os.stat(pth)
    # Sanity data (frames) is large: only report whether it exists
    sanity = meta.get("sanity") or {}
    # Fields shown in the table / comparison
    return {
        "name": os.path.basename(pth), "kind": kind, "size_mb": round(st.st_size / 1e6, 1),
        "mtime": st.st_mtime, "label": meta.get("label"), "arch": meta.get("arch"),
        "has_json": bool(meta), "has_onnx": os.path.exists(os.path.splitext(pth)[0] + ".onnx"),
        "val_metrics": meta.get("val_metrics"), "test_metrics": meta.get("test_metrics"),
        "params": meta.get("params"), "macs": meta.get("macs"), "compression": meta.get("compression"),
        "augmentation": (meta.get("augmentation") or {}).get("tag"),
        "preprocessing": (meta.get("preprocessing") or {}).get("method"),
        "training": meta.get("training"), "epochs": meta.get("epochs"), "lr": meta.get("lr"),
        "batch_size": meta.get("batch_size"), "seed": meta.get("seed"), "data": meta.get("data"),
        "sanity_frames": len(sanity.get("samples") or []), "weights_sha256": meta.get("weights_sha256"),
        "trt_validation": meta.get("trt_validation"), "code_version": meta.get("code_version"),
    }


# Every model of a folder
def list_models(folder, kind):
    """Summaries of every ``.pth`` in ``folder``, newest first."""
    # No folder yet
    if not os.path.isdir(folder):
        # Nothing
        return []
    # Every checkpoint
    out = [model_info(p, kind) for p in glob.glob(os.path.join(folder, "*.pth"))]
    # Newest first
    out.sort(key=lambda m: m["mtime"], reverse=True)
    # Summaries
    return out


# Path of a model file, refusing anything that is not a plain file name
def model_path(folder, name):
    """``folder/name`` for a safe ``.pth`` name; raises ``ValueError`` otherwise."""
    # Plain name ending in .pth
    if not SAFE_NAME.match(name or "") or not name.endswith(".pth") or name.startswith("."):
        # Refuse
        raise ValueError("invalid model name")
    # Inside the folder
    return os.path.join(folder, name)


# Delete a model and its side files
def delete_model(folder, name):
    """Remove ``name`` (.pth) plus its .json / .onnx; returns the removed file names."""
    # Checked path
    pth = model_path(folder, name)
    # Base path without extension
    base = os.path.splitext(pth)[0]
    # Removed files
    removed = []
    # The three file types
    for ext in (".pth", ".json", ".onnx"):
        # Existing ones only
        if os.path.exists(base + ext):
            # Delete
            os.remove(base + ext)
            # Remember
            removed.append(os.path.basename(base + ext))
    # Removed file names
    return removed


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
# Training form; defaults equal train.py's / resnet18.DEFAULTS (checked by a test)
TRAIN_SCHEMA = [
    {"name": "dataset", "type": "string", "default": "", "label": "Dataset", "group": "Data",
     "help": "Folder below project4/data (flat, or with train/ validation/ test/)."},
    {"name": "split", "type": "choice", "default": "random", "label": "Split", "group": "Data",
     "choices": [["random", "Random"], ["session", "By recording session"]],
     "help": "Only for flat datasets; split datasets use their folders as they are."},
    {"name": "val_ratio", "type": "float", "min": 0.05, "max": 0.5, "step": 0.05, "default": 0.1,
     "label": "Validation share", "group": "Data"},
    {"name": "split_seed", "type": "int", "min": 0, "max": 100000, "default": 0, "label": "Split seed",
     "group": "Data"},
    {"name": "arch", "type": "choice", "default": "resnet18", "label": "Network", "group": "Network", "choices": ARCHS},
    {"name": "lanes", "type": "bool", "default": True, "label": "Mark lane lines", "group": "Network",
     "help": "Off = train on plain images (train.py --no-lanes, control group)."},
    {"name": "augment", "type": "bool", "default": False, "label": "Augmentation", "group": "Network",
     "help": "All combinations of at most 2 of 7 methods (29 per image)."},
    {"name": "loss", "type": "choice", "default": "mse", "label": "Loss", "group": "Training",
     "choices": [["mse", "MSE"], ["huber", "Huber"]]},
    {"name": "epochs", "type": "int", "min": 1, "max": 1000, "default": 10, "label": "Epochs", "group": "Training"},
    {"name": "lr", "type": "float", "min": 0.00001, "max": 0.1, "step": 0.0001, "default": 0.001,
     "label": "Learning rate", "group": "Training"},
    {"name": "batch_size", "type": "int", "min": 1, "max": 256, "default": 16, "label": "Batch size",
     "group": "Training", "help": "Halved automatically and retried when the GPU runs out of memory."},
    {"name": "early_stop", "type": "bool", "default": False, "label": "Early stop", "group": "Training"},
    {"name": "seeds", "type": "string", "default": "0", "max_length": 60, "label": "Seeds", "group": "Training",
     "help": "Space-separated, e.g. '17 29 43' (mean +- std over seeds)."},
    {"name": "workers", "type": "int", "min": 0, "max": 16, "default": 4, "label": "Data loader workers",
     "group": "Training"},
    {"name": "prune", "type": "string", "default": "", "max_length": 60, "label": "Pruning ratios",
     "group": "Compression", "help": "Space-separated, e.g. '0.3 0.5'; empty = no pruning."},
    {"name": "finetune_epochs", "type": "int", "min": 1, "max": 500, "default": 15, "label": "Fine-tune epochs",
     "group": "Compression"},
    {"name": "finetune_lr", "type": "float", "min": 0.000001, "max": 0.1, "step": 0.00001, "default": 0.0001,
     "label": "Fine-tune learning rate", "group": "Compression"},
    {"name": "onnx", "type": "bool", "default": False, "label": "Also export ONNX", "group": "Compression",
     "help": "Needed for TensorRT conversion with trtexec."},
    {"name": "name", "type": "string", "default": "", "max_length": 80, "label": "Run name", "group": "Output",
     "help": "Prefix of the output files; empty = automatic."},
]


# Parse "a b c" into numbers
def _numbers(text, kind, low, high, max_count):
    # Words of the text
    words = (text or "").split()
    # Too many values
    if len(words) > max_count:
        # Refuse
        raise ValueError("at most %d values" % max_count)
    # Converted values
    out = []
    # Every word
    for w in words:
        # Convert
        v = kind(w)
        # Range check
        if not low <= v <= high:
            # Refuse
            raise ValueError("%s is outside %s..%s" % (w, low, high))
        # Keep
        out.append(v)
    # Numbers
    return out


# Extra checks of a validated training form
def check_training(values, data_root):
    """Return ``(dataset, errors)`` for values already checked against ``TRAIN_SCHEMA``."""
    # name -> message
    errors = {}
    # Dataset must be one of the known ones
    ds = get_dataset(data_root, values.get("dataset", ""))
    # Unknown dataset
    if ds is None:
        # Report
        errors["dataset"] = "choose a dataset"
    # Seeds: 1..10 integers
    try:
        # Parse
        seeds = _numbers(values.get("seeds"), int, 0, 100000, 10)
        # At least one
        if not seeds:
            # Report
            errors["seeds"] = "at least one seed"
    # Bad numbers
    except ValueError as e:
        # Report
        errors["seeds"] = str(e)
    # Pruning ratios: up to 5 floats in (0, 0.95]
    try:
        # Parse
        _numbers(values.get("prune"), float, 0.01, 0.95, 5)
    # Bad numbers
    except ValueError as e:
        # Report
        errors["prune"] = str(e)
    # Run name: safe characters only
    if values.get("name") and not SAFE_NAME.match(values["name"]):
        # Report
        errors["name"] = "use letters, digits, '.', '_', '+', '-'"
    # Dataset and errors
    return ds, errors


# train.py command line for a checked form
def build_train_cmd(values, data_path, python, out_dir=None):
    """List of arguments: ``[python, 'train.py', ...]`` (run in project4/)."""
    # Interpreter and script
    cmd = [python, "train.py", "--data", data_path, "--arch", values["arch"]]
    # Output folder override (tests, other locations)
    if out_dir:
        # --out
        cmd += ["--out", out_dir]
    # Options with a value
    cmd += ["--epochs", str(values["epochs"]), "--lr", repr(float(values["lr"])),
            "--batch-size", str(values["batch_size"]), "--loss", values["loss"],
            "--split", values["split"], "--val-ratio", repr(float(values["val_ratio"])),
            "--split-seed", str(values["split_seed"]), "--workers", str(values["workers"]),
            "--finetune-epochs", str(values["finetune_epochs"]),
            "--finetune-lr", repr(float(values["finetune_lr"]))]
    # Seeds (one or more)
    cmd += ["--seeds"] + values["seeds"].split()
    # Pruning ratios when given
    if values["prune"].split():
        # --prune r1 r2 ...
        cmd += ["--prune"] + values["prune"].split()
    # Flags
    for key, flag in (("augment", "--augment"), ("early_stop", "--early-stop"), ("onnx", "--onnx")):
        # Only when on
        if values[key]:
            # Add the flag
            cmd.append(flag)
    # Lane marking off = --no-lanes
    if not values["lanes"]:
        # Control group
        cmd.append("--no-lanes")
    # Optional run name
    if values["name"]:
        # --name
        cmd += ["--name", values["name"]]
    # Full command
    return cmd


# One epoch line of training.fit's log
EPOCH_LINE = re.compile(r"^\[(?P<tag>[^\]]+)\] epoch\s+(?P<e>\d+)/(?P<n>\d+)\s+train (?P<train>[\d.eE+-]+)\s+"
                        r"val (?P<val>[\d.eE+-]+) \((?P<px>[\d.]+) px\)\s+lr \S+\s+\((?P<sec>[\d.]+)s\)(?P<best> \*)?")


# Progress parser for train.py logs (JobQueue.parsers['train'])
def parse_train_log(job, text):
    """Update ``job.progress`` with epochs, loss curves, ETA and a short text."""
    # Short name
    p = job.progress
    # Curves per phase tag: [[epoch, train, val, px], ...]
    curves = p.setdefault("curves", {})
    # Every new line
    for line in text.splitlines():
        # Epoch line
        m = EPOCH_LINE.match(line.strip())
        # Other lines: remember errors that need special handling
        if not m:
            # GPU / CPU out of memory (CUDA or the CPU allocator)
            if "out of memory" in line.lower() or "can't allocate memory" in line.lower():
                # Mark for the retry hook
                p["oom"] = True
            # Next line
            continue
        # Phase (seed / pruning tag)
        tag = m.group("tag")
        # Point of the curve
        curves.setdefault(tag, []).append([int(m.group("e")), float(m.group("train")), float(m.group("val")),
                                           float(m.group("px"))])
        # Epoch seconds (running mean over the last 5)
        secs = p.setdefault("epoch_seconds", [])
        # Add this epoch's duration
        secs.append(float(m.group("sec")))
        # Keep the last 5
        del secs[:-5]
        # Current phase and position
        p.update({"phase": tag, "epoch": int(m.group("e")), "epochs": int(m.group("n")),
                  "eta_s": round((int(m.group("n")) - int(m.group("e"))) * sum(secs) / len(secs)),
                  "best_px": min(pt[3] for pt in curves[tag])})
        # One-line summary for the job table
        p["text"] = "%s: epoch %d/%d, val %.1f px, phase ends in ~%ds" % (
            tag, p["epoch"], p["epochs"], float(m.group("px")), p["eta_s"])
    # Remember when it was last updated
    p["updated"] = time.time()


# Arguments of a retry with half the batch size after running out of memory
def oom_retry_cmd(job):
    """New command with ``--batch-size`` halved, or None (no OOM, batch already 1, already retried 3 times)."""
    # Only failed jobs that ran out of memory
    if job.status != "failed" or not job.progress.get("oom"):
        # No retry
        return None
    # Retries so far
    if int(job.meta.get("retries", 0)) >= 3:
        # Give up
        return None
    # Copy of the command
    cmd = list(job.cmd)
    # Position of the batch size value
    i = cmd.index("--batch-size") + 1
    # Current value
    batch = int(cmd[i])
    # Cannot go lower
    if batch <= 1:
        # No retry
        return None
    # Half, at least 1
    cmd[i] = str(max(1, batch // 2))
    # New command
    return cmd
