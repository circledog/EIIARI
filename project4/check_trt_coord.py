"""JetBot test: can TensorRT convert coordinate channels added *inside* a model?

Run on the JetBot (inside project4/, no motors or camera used):
    python3 check_trt_coord.py

A CoordConv-style model appends two fixed coordinate maps (x, y in [-1, 1])
to the 3 colour channels inside ``forward``, so preprocessing and the camera
input stay 3-channel. The maps are a stored buffer (a constant), and old
torch2trt versions differ in how they handle constants, so this script tries
the candidate implementations and reports which ones convert correctly:

    A  cat_const      torch.cat([x, coords])                 coords is a (1, 2, H, W) buffer
    B  cat_expand     torch.cat([x, coords.expand(B, ...)])  expand on a constant
    C  resnet18_coord ResNet-18 whose forward uses A for batch 1, B for training batches

Each is converted with torch2trt (FP16, as convert_trt.py does) and, if
trtexec exists, through ONNX -> trtexec. The TensorRT output is compared with
PyTorch FP32; differences below 2 px (0.018 in normalised units) count as OK.
Random weights are used (no download), which is enough to test conversion.

Expected from reading the torch2trt source: A converts on torch2trt versions
from 2019-09-30 on (constants are added automatically in torch.cat); B fails
on every version (the expand converter needs a non-constant input). So the
real model should use A at batch 1 -- which is what C does.

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


# Variant C: ResNet-18 with coordinate channels inside, as a real model would be built
class ResNet18Coord(nn.Module):
    # ResNet-18 whose first conv takes 5 channels and whose head outputs (x, y)
    def __init__(self):
        # Initialise nn.Module
        super(ResNet18Coord, self).__init__()
        # Stored coordinate maps
        self.register_buffer("coords", coordinate_maps())
        # Stock ResNet-18 without pretrained weights (no download needed here)
        try:
            # New torchvision API
            self.net = torchvision.models.resnet18(weights=None)
        # Old torchvision on the JetBot
        except TypeError:
            # Old API
            self.net = torchvision.models.resnet18(pretrained=False)
        # First conv with 5 input channels
        self.net.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Regression head for (x, y)
        self.net.fc = nn.Linear(512, 2)

    # Append the maps and run the network
    def forward(self, x):
        # Batch 1 (inference, TensorRT tracing): use the constant directly (variant A)
        if x.shape[0] == 1:
            # Stored maps as they are
            grid = self.coords
        # Training batches: repeat the maps for every image (PyTorch only)
        else:
            # Expanded maps
            grid = self.coords.expand(x.shape[0], -1, -1, -1)
        # 3 colour channels + 2 coordinate channels
        return self.net(torch.cat([x, grid], dim=1))


# Largest difference between two outputs
def max_diff(a, b):
    # Compare as FP32 on the CPU
    return float((a.float().cpu() - b.float().cpu()).abs().max())


# Convert with torch2trt and compare
def try_torch2trt(name, model, x, ref):
    """'OK' / 'FAIL' line for one model converted with torch2trt."""
    # Converter from the course image
    from torch2trt import torch2trt
    # Conversion can raise for unsupported layers
    try:
        # Start time
        t0 = time.time()
        # FP16 inside, FP32 input, like convert_trt.py
        trt_model = torch2trt(model, [x], fp16_mode=True, max_workspace_size=1 << 25)
        # TensorRT output
        out = trt_model(x)
        # Difference to PyTorch FP32
        d = max_diff(out, ref)
        # Verdict
        status = "OK  " if d <= TOLERANCE else "FAIL"
        # Report line
        return "%s torch2trt  %-15s max diff %.5f (%.2f px)  %.0f s" % (status, name, d, d * 112, time.time() - t0)
    # Unsupported operation or other conversion error
    except Exception as e:
        # Report the error, with the last frames of the traceback for debugging
        return "FAIL torch2trt  %-15s %r\n%s" % (name, e, "".join(traceback.format_exc().splitlines(True)[-6:]))


# Convert through ONNX and trtexec and compare
def try_trtexec(name, model, x, ref, workdir):
    """'OK' / 'FAIL' / 'SKIP' line for one model built with trtexec."""
    # trtexec is part of JetPack; skip elsewhere
    if not os.path.exists(TRTEXEC):
        # Nothing to test
        return "SKIP trtexec    %-15s (no %s)" % (name, TRTEXEC)
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
        # TensorRT output
        out = m(x.to(m.dtype))
        # Difference to PyTorch FP32
        d = max_diff(out, ref)
        # Verdict
        status = "OK  " if d <= TOLERANCE else "FAIL"
        # Report line
        return "%s trtexec    %-15s max diff %.5f (%.2f px)" % (status, name, d, d * 112)
    # Report the error
    except Exception as e:
        # Short error line
        return "FAIL trtexec    %-15s %r" % (name, e)


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
        cat_src = open(os.path.join(folder, "converters", "cat.py")).read()
        # Constants are handled by trt_ (2019-09 .. 2020-08) or add_missing_trt_tensors (2020-09 on)
        handles = "add_missing_trt_tensors" in cat_src or "trt_(" in cat_src
        # Static verdict
        print("torch2trt cat converter adds constants: %s" % ("yes" if handles else "NO (old version)"))
    # Not installed
    except ImportError:
        # Show that it is missing
        print("torch2trt: not installed")
    # TensorRT needs the GPU
    if not torch.cuda.is_available():
        # Explain where to run it
        print("CUDA not available: run this on the JetBot. Only the PyTorch / ONNX side was checked.")
    # The three candidate models, on the GPU when there is one
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fixed random input so results are reproducible
    torch.manual_seed(0)
    # One camera-sized input (batch 1)
    x = torch.randn(1, 3, SIZE, SIZE, device=device)
    # Name and model of every variant
    models = [("cat_const", CatConst()), ("cat_expand", CatExpand()), ("resnet18_coord", ResNet18Coord())]
    # Result lines
    lines = []
    # Temporary folder for ONNX / engine files
    workdir = tempfile.mkdtemp(prefix="coordcheck_")
    # Test each variant
    for name, model in models:
        # Evaluation mode on the chosen device
        model = model.to(device).eval()
        # PyTorch FP32 reference output
        with torch.no_grad():
            # Reference
            ref = model(x)
        # The real model must also train with batches > 1
        if name == "resnet18_coord":
            # A batch of 4 must run in PyTorch (expand path)
            with torch.no_grad():
                # Training-sized batch
                model(torch.randn(4, 3, SIZE, SIZE, device=device))
            # Record that the training path works
            lines.append("OK   pytorch    resnet18_coord batch 4 (training path)")
        # TensorRT tests need CUDA
        if device.type == "cuda":
            # torch2trt, if installed
            try:
                import torch2trt
                # Convert and compare
                lines.append(try_torch2trt(name, model, x, ref))
            # Not installed
            except ImportError:
                # Report the skip
                lines.append("SKIP torch2trt  %-15s (not installed)" % name)
            # ONNX -> trtexec
            lines.append(try_trtexec(name, model, x, ref, workdir))
    # Summary
    print("\n".join(lines))
    # Interpretation
    print("\nThe real model uses the resnet18_coord pattern: it can run on TensorRT if resnet18_coord is OK "
          "on at least one path (torch2trt or trtexec).")


# Run when executed as a script
if __name__ == "__main__":
    # Start
    main()
