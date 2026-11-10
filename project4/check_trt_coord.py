"""JetBot test: do the CoordConv / DSNT networks convert to TensorRT correctly?

Run on the JetBot (inside project4/, no motors or camera used):
    python3 check_trt_coord.py

Tests, with random weights (no download needed):
    resnet18, resnet18_coord, resnet18_dsnt, resnet18_coord_dsnt  -- the
        pipeline's own networks from loader.ARCHS (CoordConv adds the two
        coordinate channels inside the network; DSNT uses a heatmap head)
    cat_const / cat_expand -- the two simplest ways of adding a stored
        constant, for diagnosis

Each is converted with torch2trt (FP16, as convert_trt.py does) and, if
trtexec exists, through ONNX -> trtexec. Conversion uses input A; the
TensorRT result for a *different* input B is compared with PyTorch FP32 on B
(differences below 2 px count as OK), so values frozen into constants show
up as a failure.

Expected (checked offline with check_trt_offline.py on torch2trt
2019-10 .. 2022-07): every pipeline network OK; cat_const fails on torch2trt
before 2020-09; cat_expand fails from 2021-01 on.

Python 3.6 compatible.
"""
import os
import platform
import subprocess
import tempfile
import time
import traceback

import torch
import torch.nn as nn
import torchvision

# Pipeline networks
from loader import ARCHS

# Input size of every model in this project
SIZE = 224
# Allowed difference in normalised units (2 px on a 224-px image)
TOLERANCE = 2.0 / 112.0
# trtexec location in JetPack
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


# Fixed coordinate maps: channel 0 = x (left -1 .. right +1), channel 1 = y (top -1 .. bottom +1)
def coordinate_maps(size=SIZE):
    """(1, 2, size, size) float tensor of x / y coordinates."""
    # Evenly spaced positions from -1 to 1
    line = torch.linspace(-1.0, 1.0, size)
    # x varies along the width
    xs = line.view(1, 1, 1, size).expand(1, 1, size, size)
    # y varies along the height
    ys = line.view(1, 1, size, 1).expand(1, 1, size, size)
    # Stack into one contiguous tensor
    return torch.cat([xs, ys], dim=1).contiguous()


# Variant A: concatenate the stored maps directly (batch 1 only)
class CatConst(nn.Module):
    # Small conv on the 5-channel result so there is something to compute
    def __init__(self):
        # Initialise nn.Module
        super(CatConst, self).__init__()
        # Stored coordinate maps
        self.register_buffer("coords", coordinate_maps())
        # 5 -> 8 channel convolution
        self.conv = nn.Conv2d(5, 8, 3, padding=1)

    # Append the maps and run the conv
    def forward(self, x):
        # (1, 3, H, W) + (1, 2, H, W) -> (1, 5, H, W)
        return self.conv(torch.cat([x, self.coords], dim=1))


# Variant B: expand the maps to the batch size first
class CatExpand(CatConst):
    # Expand, append and run the conv
    def forward(self, x):
        # Coordinate maps repeated for every image of the batch
        grid = self.coords.expand(x.shape[0], -1, -1, -1)
        # (B, 3, H, W) + (B, 2, H, W) -> (B, 5, H, W)
        return self.conv(torch.cat([x, grid], dim=1))


# Largest difference between two outputs
def max_diff(a, b):
    # Compare as FP32 on the CPU
    return float((a.float().cpu() - b.float().cpu()).abs().max())


# Convert with torch2trt and compare
def try_torch2trt(name, model, x, ref, x2):
    """'OK' / 'FAIL' line for one model converted with torch2trt."""
    # Converter from the course image
    from torch2trt import torch2trt
    # Conversion can raise for unsupported layers
    try:
        # Start time
        t0 = time.time()
        # FP16 inside, FP32 input, like convert_trt.py
        trt_model = torch2trt(model, [x], fp16_mode=True, max_workspace_size=1 << 25)
        # TensorRT output for the second input
        out = trt_model(x2)
        # Difference to PyTorch FP32
        d = max_diff(out, ref)
        # Verdict
        status = "OK  " if d <= TOLERANCE else "FAIL"
        # Report line
        return "%s torch2trt  %-19s max diff %.5f (%.2f px)  %.0f s" % (status, name, d, d * 112, time.time() - t0)
    # Unsupported operation or other conversion error
    except Exception as e:
        # Report the error, with the last frames of the traceback for debugging
        return "FAIL torch2trt  %-19s %r\n%s" % (name, e, "".join(traceback.format_exc().splitlines(True)[-6:]))


# Convert through ONNX and trtexec and compare
def try_trtexec(name, model, x, ref, x2, workdir):
    """'OK' / 'FAIL' / 'SKIP' line for one model built with trtexec."""
    # trtexec is part of JetPack; skip elsewhere
    if not os.path.exists(TRTEXEC):
        # Nothing to test
        return "SKIP trtexec    %-19s (no %s)" % (name, TRTEXEC)
    # Engine packing and loading helpers of this project
    from compression import wrap_trt_engine
    # Loader that detects the engine's input dtype
    from loader import load_model
    # Files in the temporary folder
    onnx, engine, wrapped = [os.path.join(workdir, name + ext) for ext in (".onnx", ".engine", "_trt.pth")]
    # Any step can fail
    try:
        # Export with the same settings as compression.export_onnx
        torch.onnx.export(model, x, onnx, opset_version=11, input_names=["input_0"], output_names=["output_0"])
        # Build the engine (FP16 inside)
        subprocess.check_call([TRTEXEC, "--onnx=" + onnx, "--saveEngine=" + engine, "--fp16"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # Pack it for TRTModule
        wrap_trt_engine(engine, wrapped)
        # Load it like live_demo does
        m = load_model(wrapped, "cuda")
        # TensorRT output for the second input
        out = m(x2.to(m.dtype))
        # Difference to PyTorch FP32
        d = max_diff(out, ref)
        # Verdict
        status = "OK  " if d <= TOLERANCE else "FAIL"
        # Report line
        return "%s trtexec    %-19s max diff %.5f (%.2f px)" % (status, name, d, d * 112)
    # Report the error
    except Exception as e:
        # Short error line
        return "FAIL trtexec    %-19s %r" % (name, e)


# Run every test and print a summary
def main():
    # Versions matter for interpreting the result
    print("python %s | torch %s | torchvision %s" % (platform.python_version(), torch.__version__,
                                                    torchvision.__version__))
    # TensorRT version, if installed
    try:
        import tensorrt
        # Show it
        print("tensorrt %s" % tensorrt.__version__)
    # Not installed
    except ImportError:
        # Show that it is missing
        print("tensorrt: not installed")
    # torch2trt location and how its cat converter handles constants
    try:
        import torch2trt
        # Folder of the installed package
        folder = os.path.dirname(torch2trt.__file__)
        # Show it
        print("torch2trt %s" % folder)
        # Source of the cat converter
        cat_src = open(os.path.join(folder, "converters", "cat.py"), encoding="utf-8").read()
        # Era of the converter: add_missing_trt_tensors (2020-09 on) or trt_ (2019-09 .. 2020-08)
        era = ("2020-09 or newer" if "add_missing_trt_tensors" in cat_src else
               "2019-09 .. 2020-08 (cat_const expected to fail)" if "trt_(" in cat_src else "before 2019-09")
        # Show it
        print("torch2trt cat converter: %s" % era)
    # Not installed
    except ImportError:
        # Show that it is missing
        print("torch2trt: not installed")
    # TensorRT needs the GPU
    if not torch.cuda.is_available():
        # Explain where to run it
        print("CUDA not available: run this on the JetBot. Only the PyTorch / ONNX side was checked.")
    # Run on the GPU when there is one
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fixed random inputs so results are reproducible
    torch.manual_seed(0)
    # Conversion input and comparison input (batch 1, camera size)
    x, x2 = torch.randn(1, 3, SIZE, SIZE, device=device), torch.randn(1, 3, SIZE, SIZE, device=device)
    # Pipeline networks with random weights, then the two probes
    models = [(n, ARCHS[n].build(pretrained=False)) for n in
              ("resnet18", "resnet18_coord", "resnet18_dsnt", "resnet18_coord_dsnt")]
    # Probes
    models += [("cat_const", CatConst()), ("cat_expand", CatExpand())]
    # Result lines
    lines = []
    # Temporary folder for ONNX / engine files
    workdir = tempfile.mkdtemp(prefix="coordcheck_")
    # Test each model
    for name, model in models:
        # Give zero-initialised weights (CoordConv's extra channels) values so they matter,
        # and sharpen DSNT heatmaps so the output depends on the image
        with torch.no_grad():
            # Every parameter
            for p in model.parameters():
                # All-zero weight tensors only
                if p.dim() > 1 and not p.any():
                    # Small random values
                    p.normal_(0, 0.05)
            # DSNT head
            if hasattr(model, "head"):
                # Peaked heatmap
                model.head[-1].weight.mul_(40)
        # Evaluation mode on the chosen device
        model = model.to(device).eval()
        # PyTorch FP32 reference output for the comparison input
        with torch.no_grad():
            # Reference
            ref = model(x2)
        # The CoordConv networks must also train with batches > 1
        if name.startswith("resnet18_coord"):
            # A batch of 4 must run in PyTorch (expand path)
            with torch.no_grad():
                # Training-sized batch
                model(torch.randn(4, 3, SIZE, SIZE, device=device))
            # Record that the training path works
            lines.append("OK   pytorch    %-19s batch 4 (training path)" % name)
        # TensorRT tests need CUDA
        if device.type == "cuda":
            # torch2trt, if installed
            try:
                import torch2trt
                # Convert and compare
                lines.append(try_torch2trt(name, model, x, ref, x2))
            # Not installed
            except ImportError:
                # Report the skip
                lines.append("SKIP torch2trt  %-19s (not installed)" % name)
            # ONNX -> trtexec
            lines.append(try_trtexec(name, model, x, ref, x2, workdir))
    # Summary
    print("\n".join(lines))
    # Interpretation
    print("\nA network can run on TensorRT if it is OK on at least one path (torch2trt or trtexec); "
          "cat_const / cat_expand are only diagnostics.")


# Run when executed as a script
if __name__ == "__main__":
    # Start
    main()
