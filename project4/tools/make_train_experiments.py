"""Generate project4/train_experiments.ipynb.

Usage (inside project4/):
    python3 tools/make_train_experiments.py            # writes train_experiments.ipynb
    python3 tools/make_train_experiments.py OUT.ipynb  # somewhere else

Edit the experiments here, not in the notebook, then regenerate: the
notebook is overwritten (outputs included).
"""
import json
import os
import sys

# Output file: first argument, or train_experiments.ipynb next to this folder
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                         "train_experiments.ipynb")

# Notebook cells in order
cells = []


# Append a markdown cell
def md(text):
    # Source split into lines, keeping the newlines
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


# Append an empty-output code cell
def code(text):
    # Source split into lines, keeping the newlines
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md('''
# 訓練實驗（PC / Colab，有 GPU）

這個 notebook 由 `tools/make_train_experiments.py` 產生；要改實驗請改那個檔案再重新產生（會覆蓋這個 notebook）。

先跑第 0 節安裝環境。之後每一格程式碼 = 一個訓練實驗：上面寫設定，最後一行 `run(...)` 會呼叫 `train.py` 訓練，訓練完印出 **test** 的結果。

- **不會重複訓練**：檔名由所有設定自動產生（例如 `resnet18_aug2_mse_e5_lr0.001_bs16`，3 個 seed 的檔案再加 `_s17` 等）。`models/` 裡已經有同名的模型時就不訓練，直接讀出上次的 test 結果。要重新訓練就把那個模型的 `.pth` / `.json` 刪掉。
- 訓練方式是隊友流程 ①（部分凍結、凍結 BN、AdamW、cosine、gradient clipping、用 validation px 挑模型、提早停止，見 `training.py`）；`loss` 可選 `"mse"` 或 `"huber"`。檔名多了 loss 一段，和舊訓練方式的檔案不會撞名。
- 設定沒寫的（`epochs = None`、`lr = None`）用該網路檔的 `DEFAULTS`（目前是 10 epochs、lr 1e-3）；batch size 預設 16。
- `augment = True`：每張訓練圖用 7 種方法（翻轉、旋轉、裁切、顏色、噪聲、模糊、壓縮）裡最多 2 種的所有組合（29 種，含原圖）展開，一個 epoch 看 277 × 29 張；`False` 只用原圖。
- **公平比較**：augmentation 開 5 epochs = 277 × 29 × 5 = 40,165 個樣本；關 145 epochs = 277 × 145 = 40,165 個樣本，兩邊練一樣多。提早停止預設關閉（`early_stop = False`），兩邊都跑完全程、挑 validation 最好的 epoch；開了檔名多 `_es`。
- `lanes = False`：不標車道線，用原圖訓練（對照組），檔名多 `_nolane`；live_demo 會自動對這種模型不標車道線。
- `seeds = [17, 29, 43]`：每個設定跑 3 次，看平均和標準差（和隊友相同），檔案是 `..._s17_base.pth` 等，另外寫一份 `..._seeds.json`。改成 `[0]` 只跑一次。
- 結果多了**轉向角誤差**（`angle`，度）：用車上預設的轉向公式把預測點和標籤點都換成轉向角再相減，比像素誤差更接近開車的好壞。
- 最後一格把這次執行過的實驗（含跳過的）整理成一張表。
- 想加實驗：複製一格、改設定即可。
''')

md('''
## 0. 安裝環境

在 `project4/` 資料夾裡開這個 notebook。下面這格會用目前 kernel 的 Python 安裝 `requirements-pc.txt`（torch、torchvision、torch-pruning、onnx、onnxruntime、numpy、pillow、opencv-python-headless）；已經裝好的套件會直接略過，所以每次 Run All 都可以留著。

- Colab 已經內建有 CUDA 的 torch，不會重裝。
- 自己的電腦要用 GPU 的話，torch 請先照 https://pytorch.org 選對 CUDA 版本安裝，再跑這格。
- 資料集放在 `data/splits/{train,validation,test}/`（隊友切好的資料夾）。
''')

code('''
import subprocess
import sys

# Some virtual environments (e.g. made by uv) come without pip: add it first
if subprocess.call([sys.executable, "-m", "pip", "--version"], stdout=subprocess.DEVNULL) != 0:
    # Bundled pip installer of the standard library
    subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
# Install / check the PC packages with this kernel's Python (already installed ones are skipped)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-pc.txt"])
import torch
# Show what will be used for training
print("torch %s | CUDA available: %s%s" % (torch.__version__, torch.cuda.is_available(),
      " (%s)" % torch.cuda.get_device_name(0) if torch.cuda.is_available() else " -> training on CPU is very slow"))
''')

code('''
# --- Shared settings (change here once for every experiment) -----------------
import json
import os
import subprocess
import sys

# Network registry (for the default epochs / lr and the file-name prefix)
from loader import ARCHS
# Augmentation switch -> file-name tag ("aug2" / "none")
from augmentation import Augmenter

# Dataset with the team's train/ validation/ test/ folders (test is needed for the results)
DATA_DIR = "data/splits"
# Where train.py writes the models (.pth + .json); live_demo / convert_trt read from here
OUT_DIR = "models"
# DataLoader worker processes (0 if the machine complains about shared memory)
WORKERS = 4
# Results of every experiment run in this session, for the summary table
RESULTS = {}


# File-name stem built from every setting, so different settings never share a name
def run_name(arch, epochs, lr, augment, batch_size, seeds, loss, lanes=True, early_stop=False):
    """e.g. 'resnet18_aug2_mse_e5_lr0.001_bs16' (several seeds: no seed part, train.py adds _s<seed>)."""
    # Augmentation part of the name, same tag train.py uses
    aug = Augmenter(enabled=augment).tag()
    # Network, augmentation, lane marking, loss, epochs, learning rate and batch size
    name = "%s_%s%s_%s_e%d_lr%g_bs%d" % (ARCHS[arch].FILE_TAG, aug, "" if lanes else "_nolane", loss, epochs, lr,
                                       batch_size)
    # Early stopping changes the result, so it is part of the name
    name += "_es" if early_stop else ""
    # A single seed is part of the name; several seeds get their own suffixes from train.py
    return name + ("_seed%d" % seeds[0] if len(seeds) == 1 else "")


# Test metrics of a finished experiment, read from the files train.py wrote
def read_results(name, seeds):
    """{"test": {...}, "val": {...}, "macs": ...}; several seeds give mean / std per metric."""
    # One seed: the baseline's .json
    if len(seeds) == 1:
        # Metadata written next to the .pth
        with open(os.path.join(OUT_DIR, name + "_base.json")) as f:
            # Parsed metadata
            meta = json.load(f)
        # Test (None without a test split), validation and size
        return {"test": meta.get("test_metrics"), "val": meta.get("val_metrics"), "macs": meta.get("macs")}
    # Several seeds: the summary train.py wrote
    with open(os.path.join(OUT_DIR, name + "_seeds.json")) as f:
        # Baseline statistics
        base = json.load(f)["summary"]["base"]
    # Mean / std per metric for test and validation
    return {"test": base.get("test"), "val": base.get("val"), "macs": None}


# Has this exact experiment been trained already?
def finished(name, seeds):
    # One seed: the baseline weights and their .json; several seeds: the summary file
    files = [name + "_base.pth", name + "_base.json"] if len(seeds) == 1 else [name + "_seeds.json"]
    # All of them must exist
    return all(os.path.exists(os.path.join(OUT_DIR, f)) for f in files)


# Print the test (or, without a test split, validation) metrics
def show(name, res, seeds):
    # Prefer test; fall back to validation and say so
    split, m = ("test", res["test"]) if res["test"] else ("val (no test split!)", res["val"])
    # One seed: plain numbers
    if len(seeds) == 1:
        # MSE and pixel errors
        print("%s  [%s]  mse %.5f | mean %.1f px | x %.1f px | y %.1f px | p95 %.1f px | angle %s deg"
              % (name, split, m["mse"], m["mean_px"], m["mae_x_px"], m["mae_y_px"], m["p95_px"],
                 "%.1f" % m["angle_deg"] if "angle_deg" in m else "-"))
    # Several seeds: mean +- std
    else:
        # MSE and pixel errors with their spread over the seeds
        print("%s  [%s, seeds %s]  mse %.5f +- %.5f | mean %.1f +- %.1f px | p95 %.1f +- %.1f px | angle %s deg"
              % (name, split, seeds, m["mse"]["mean"], m["mse"]["std"], m["mean_px"]["mean"],
                 m["mean_px"]["std"], m["p95_px"]["mean"], m["p95_px"]["std"],
                 "%.1f +- %.1f" % (m["angle_deg"]["mean"], m["angle_deg"]["std"]) if "angle_deg" in m else "-"))


# Train one experiment unless it already exists, then print its test results
def run(arch, epochs=None, lr=None, augment=False, batch_size=16, seeds=(17, 29, 43), loss="mse", lanes=True,
        early_stop=False):
    # Defaults of the chosen network for anything left as None
    epochs = epochs if epochs is not None else ARCHS[arch].DEFAULTS["epochs"]
    # Learning rate default
    lr = lr if lr is not None else ARCHS[arch].DEFAULTS["lr"]
    # Seeds as a list
    seeds = list(seeds)
    # File-name stem of this experiment
    name = run_name(arch, epochs, lr, augment, batch_size, seeds, loss, lanes, early_stop)
    # Same settings trained before: reuse the result
    if finished(name, seeds):
        # Tell the user why nothing is trained
        print("already trained, skipping: %s (delete its files in %s/ to train again)" % (name, OUT_DIR))
    # New experiment: run train.py
    else:
        # Command line, same as typing it in a terminal
        cmd = [sys.executable, "train.py", "--arch", arch, "--data", DATA_DIR, "--out", OUT_DIR,
               "--name", name, "--epochs", str(epochs), "--lr", str(lr), "--batch-size", str(batch_size),
               "--workers", str(WORKERS), "--seeds"] + [str(s) for s in seeds]
        # Coordinate loss
        cmd += ["--loss", loss]
        # Augmentation switch
        cmd += ["--augment"] if augment else []
        # Lane-marking switch
        cmd += [] if lanes else ["--no-lanes"]
        # Early-stopping switch
        cmd += ["--early-stop"] if early_stop else []
        # Show the command
        print("$ " + " ".join(cmd), flush=True)
        # Run it and stream its log into the notebook
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                                env=dict(os.environ, PYTHONUNBUFFERED="1"))
        # Every log line as it comes
        for line in proc.stdout:
            # Pass it through
            print(line, end="", flush=True)
        # A failed run must not look finished
        if proc.wait() != 0:
            # Stop Run All here so the error is visible
            raise RuntimeError("train.py failed for %s (exit code %d)" % (name, proc.returncode))
    # Results from the files
    res = read_results(name, seeds)
    # Remember them for the summary
    RESULTS[name] = dict(res, arch=arch, seeds=seeds)
    # Print the test results
    show(name, res, seeds)
''')

# One experiment cell: network, epochs, augmentation, lane marking
CELL = '''
# Network
model = "%s"
# Training length (augmentation on: 5 epochs = 40,165 samples; off: 145 epochs = the same)
epochs = %d
# Learning rate (None = the network's DEFAULTS)
lr = None
# Augmentation on / off (subsets of at most 2 of the 7 methods)
augment = %s
# Lane marking on / off (False = plain images, the comparison)
lanes = %s
# Early stopping (off: train all epochs, keep the best validation epoch)
early_stop = False
# Batch size
batch_size = 16
# Coordinate loss: "mse" or "huber"
loss = "mse"
# Training seeds (mean +- std over them)
seeds = [17, 29, 43]
# Train (or reuse) and print the test results
run(model, epochs, lr, augment, batch_size, seeds, loss, lanes, early_stop)
'''

# The five networks
NETS = ["resnet18", "mobilenet_v2", "resnet18_coord", "resnet18_dsnt", "resnet18_coord_dsnt"]

# Section 1: augmentation on, same sample budget as section 2
md('''
## 1. 五種網路，augmentation 開，5 epochs（40,165 個樣本）
''')

# One cell per network
for arch in NETS:
    # 5 epochs with augmentation, lane marking on
    code(CELL % (arch, 5, True, True))

# Section 2: augmentation off, same sample budget
md('''
## 2. 五種網路，augmentation 關，145 epochs（40,165 個樣本，和第 1 節一樣多）
''')

# One cell per network
for arch in NETS:
    # 145 epochs without augmentation, lane marking on
    code(CELL % (arch, 145, False, True))

# Section 3: the lane-marking comparison
md('''
## 3. 不標車道線的對照組：五種網路，augmentation 開，5 epochs

和第 1 節只差在 `lanes = False`。比較第 1 節和這一節，就知道車道線前處理有沒有幫助。
''')

# One cell per network
for arch in NETS:
    # Same as section 1 without lane marking
    code(CELL % (arch, 5, True, False))

# Summary section
md('''
## 4. 總表（這次執行過的所有實驗，test 結果）
''')

code('''
# One row per experiment: test metrics (mean over seeds when there are several)
def value(m, key):
    # Several seeds store {"mean", "std", "values"}; one seed stores the number
    return m[key]["mean"] if isinstance(m[key], dict) else m[key]


# Header of the table
print("%-54s %-6s %9s %8s %7s %7s %7s %9s %6s" % ("experiment", "split", "mse", "mean px", "x px", "y px",
                                                "p95 px", "angle deg", "GMACs"))
# Experiments sorted by mean pixel error (best first)
rows = sorted(RESULTS.items(), key=lambda kv: value(kv[1]["test"] or kv[1]["val"], "mean_px"))
# One line each
for name, res in rows:
    # Test metrics, or validation when there is no test split
    split, m = ("test", res["test"]) if res["test"] else ("val", res["val"])
    # Size in GMACs when known
    macs = "%.2f" % (res["macs"] / 1e9) if res["macs"] else "-"
    # Seed count marker
    tag = name + (" (%d seeds)" % len(res["seeds"]) if len(res["seeds"]) > 1 else "")
    # Steering-angle error (older .json files have none)
    angle = "%.1f" % value(m, "angle_deg") if "angle_deg" in m else "-"
    # The row
    print("%-54s %-6s %9.5f %8.1f %7.1f %7.1f %7.1f %9s %6s" % (tag, split, value(m, "mse"), value(m, "mean_px"),
                                                             value(m, "mae_x_px"), value(m, "mae_y_px"),
                                                             value(m, "p95_px"), angle, macs))
''')

# Notebook structure
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
# Write it (UTF-8: the markdown is Chinese)
with open(OUT, "w", encoding="utf-8") as f:
    # Same layout Jupyter uses
    json.dump(nb, f, indent=1, ensure_ascii=False)
# Report
print("wrote %s (%d cells)" % (os.path.normpath(OUT), len(cells)))
