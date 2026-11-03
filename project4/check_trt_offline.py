"""PC test without a GPU: does torch2trt convert our networks correctly?

The real torch2trt source (a git checkout of the version on the robot) runs
against ``fake_tensorrt/``, an offline stand-in for the TensorRT Python API.
torch2trt builds its TensorRT network as usual; the stand-in then evaluates
that network with PyTorch operations (TensorRT semantics) and the result is
compared with the original model:

* converted with input A, evaluated with a *different* input B, compared
  with ``model(B)`` -- so values torch2trt silently froze into constants
  (an unconverted operation) show up as a mismatch;
* missing converters, crashing converters and invalid shapes / axes are
  reported as FAIL with the reason.

Not covered (only the robot can tell): TensorRT engine-builder limits on the
Nano and FP16 rounding (the evaluation is FP32). ``check_trt_coord.py`` on
the robot stays the final confirmation. torch2trt >= 0.5 (2024, dynamic
shapes, not for JetPack 4) uses layers the stand-in does not model.

Controls that show the check itself works: ``probe_cat_expand`` must FAIL
from 2021-01 on (expand converter), ``probe_frozen`` must FAIL everywhere
(torch.softmax is frozen into a constant), ``teammate_dsnt`` (interpolate)
reports the unmodelled resize layer.

Usage (PC, project4/):
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt /tmp/torch2trt
    git -C /tmp/torch2trt worktree add /tmp/t2t_v020 v0.2.0
    python check_trt_offline.py --t2t /tmp/t2t_v020 [/tmp/t2t_other ...] \\
        [--trt-version 7.1.3 8.0.1] [--models resnet18 resnet18_coord ...]

Each (torch2trt checkout, TensorRT version) pair runs in its own process.
"""
import argparse
import contextlib
import io
import os
import subprocess
import sys
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

# Folder with this file (project4/)
_HERE = os.path.dirname(os.path.abspath(__file__))
# Folder that holds the offline "tensorrt" package
FAKE_TRT_DIR = os.path.join(_HERE, "fake_tensorrt")
# Allowed difference in normalised units (0.1 px on a 224-px image; FP32 evaluation)
TOLERANCE = 0.1 / 112.0
# A model whose output barely depends on the input cannot show frozen values: require
# the two test inputs to change the output by at least this much (1 px)
MIN_SENSITIVITY = 1.0 / 112.0


# Negative control: torch.softmax had no torch2trt converter before 2022, so its value is
# frozen into a constant -- this probe must FAIL on those versions (shows the check works)
class FrozenProbe(nn.Module):
    # Strided conv to a 28x28 map, softmax, linear head
    def __init__(self):
        # Initialise nn.Module
        super(FrozenProbe, self).__init__()
        # 224 -> 28 with one output channel
        self.conv = nn.Conv2d(3, 1, 8, stride=8)
        # 784 cells -> (x, y)
        self.fc = nn.Linear(784, 2)
        # Large weights so the softmax is peaked and the output depends on the input
        with torch.no_grad():
            # Sharpen
            self.conv.weight.mul_(20)

    # torch.softmax instead of F.softmax
    def forward(self, x):
        # Probabilities over the cells, then the head
        return self.fc(torch.softmax(self.conv(x).view(-1, 784), dim=1))


# The teammate's original DSNT operations (F.interpolate, flatten, torch.softmax), for comparison
def teammate_dsnt():
    """resnet18_dsnt weights with the teammate's forward ops instead of the fixed Linear layers."""
    # Our DSNT network
    import resnet18_dsnt as d

    # Same network, original heatmap computation
    class TeammateDSNT(d.ResNet18DSNT):
        # Teammate's heatmaps(): interpolate + flatten + torch.softmax
        def heatmaps(self, images):
            # Backbone
            b = self.backbone
            # Stem
            x = b.maxpool(b.relu(b.bn1(b.conv1(images))))
            # Residual stages
            x = b.layer4(b.layer3(b.layer2(b.layer1(x))))
            # 14x14 -> 28x28 logits
            logits = F.interpolate(self.head(x), size=(28, 28), mode="bilinear", align_corners=False)
            # Softmax over the flattened cells
            return torch.softmax(logits.flatten(2), dim=2).view(-1, 784)
    # Build it with random weights
    return TeammateDSNT(d.backbone(False))


# Networks to test: the project's registered networks plus the probe variants
def build_models(names):
    """[(name, model)] in eval mode with non-trivial weights (pretrained backbone, random heads)."""
    import torch
    # Project network registry
    from loader import ARCHS
    # Probe variants used on the robot
    import check_trt_coord as probes
    # Probe models by name
    extra = {"probe_cat_const": probes.CatConst, "probe_cat_expand": probes.CatExpand,
             "probe_frozen": FrozenProbe, "teammate_dsnt": teammate_dsnt}
    # Built models
    out = []
    # Every requested name
    for name in names:
        # Same weights for every run of the same name
        torch.manual_seed(0)
        # Registered project network: ImageNet weights when available (random deep networks
        # can give the same output for every input), random weights otherwise
        if name in ARCHS:
            # Download / cache may be unavailable
            try:
                # Pretrained backbone, random head
                model = ARCHS[name].build(pretrained=True)
            # Offline without cached weights
            except Exception:
                # Random weights
                model = ARCHS[name].build(pretrained=False)
        # Probe variant
        elif name in extra:
            # Build it
            model = extra[name]()
        # Unknown name
        else:
            # Tell the user
            raise SystemExit("unknown model %r (choose from %s)" % (name, ", ".join(list(ARCHS) + list(extra))))
        # Give zero-initialised weights (e.g. CoordConv's extra channels) real values, so
        # every input channel influences the output and a lost channel would be noticed
        with torch.no_grad():
            # Every parameter
            for p in model.parameters():
                # All-zero tensors only
                if p.dim() > 1 and not p.any():
                    # Small random values
                    p.normal_(0, 0.05)
            # Random heatmap heads give an almost flat heatmap (output ~0 for every input);
            # sharpen them so the output really depends on the image
            if hasattr(model, "head"):
                # Larger last-layer weights -> peaked heatmap
                model.head[-1].weight.mul_(40)
            # Random deep networks (MobileNetV2) can give almost the same output for every
            # input; scale the trainable output layer until two inputs differ by >= 4 px
            sensitive_output(model.eval())
        # Evaluation mode (BatchNorm uses its running statistics)
        out.append((name, model.eval()))
    # All models
    return out


# Make a model's output depend visibly on its input
def sensitive_output(model):
    """Scale the last trainable Linear layer so two random inputs move the output by >= 4 px."""
    # Two fixed random inputs
    gen = torch.Generator().manual_seed(2)
    # Input pair
    xa, xb = torch.randn(1, 3, 224, 224, generator=gen), torch.randn(1, 3, 224, 224, generator=gen)
    # Output change between them
    with torch.no_grad():
        # Max absolute difference
        s = float((model(xa) - model(xb)).abs().max())
    # Trainable Linear layers (DSNT's fixed ones are excluded)
    linears = [m for m in model.modules() if isinstance(m, nn.Linear) and m.weight.requires_grad]
    # Already sensitive enough, or nothing to scale
    if s >= 4 * MIN_SENSITIVITY or s == 0 or not linears:
        # Leave the model as it is
        return
    # Scale the last one (the output head)
    with torch.no_grad():
        # Output changes scale linearly with this layer's weights
        linears[-1].weight.mul_(4 * MIN_SENSITIVITY / s)


# Convert one model with torch2trt and evaluate the captured network
def check_one(name, model, torch2trt_fn, fake):
    """'OK' / 'FAIL' / 'WEAK' line for one model (WEAK: output hardly depends on the input)."""
    # Two different camera-sized inputs
    torch.manual_seed(1)
    # Input used for conversion, input used for evaluation
    xa, xb = torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)
    # Reference output for the evaluation input
    with torch.no_grad():
        # PyTorch FP32
        ref = model(xb)
        # How much the output changes between the two inputs
        sensitivity = float((model(xa) - ref).abs().max())
    # torch2trt prints warnings for unsupported methods
    log = io.StringIO()
    # Conversion or evaluation can fail
    try:
        # Forget networks of earlier models
        del fake.CAPTURED[:]
        # Fake inputs for the conversion run
        fake.FEED[:] = [xa]
        # Capture torch2trt's printed warnings
        with contextlib.redirect_stdout(log):
            # Same call as convert_trt.py
            torch2trt_fn(model, [xa], fp16_mode=True, max_workspace_size=1 << 25)
        # The network torch2trt asked TensorRT to build
        network = fake.CAPTURED[-1]
        # Its output for the other input
        out = fake.evaluate(network, [xb])[0]
        # Shape must match the model's output
        if tuple(out.shape) != tuple(ref.shape):
            # Report the mismatch
            return "FAIL %-19s output shape %s, model %s" % (name, tuple(out.shape), tuple(ref.shape))
        # Largest difference
        d = float((out - ref).abs().max())
        # Warnings torch2trt printed, if any
        warn = " | " + log.getvalue().strip().replace("\n", " / ")[:200] if log.getvalue().strip() else ""
        # Verdict
        status = "FAIL" if d > TOLERANCE else ("OK  " if sensitivity >= MIN_SENSITIVITY else "WEAK")
        # torch2trt names methods it cannot convert: their values end up frozen
        if "unsupported method" in log.getvalue():
            # Never acceptable, even if this input pair happens to match
            status = "FAIL"
        # Report line
        return "%s %-19s max diff %.3f px (output moves %.1f px between inputs), %d layers%s" % (
            status, name, d * 112, sensitivity * 112, network.num_layers, warn)
    # Any failure
    except Exception as e:
        # Innermost frame where it happened
        where = traceback.extract_tb(sys.exc_info()[2])[-1]
        # Report the error and its location
        return "FAIL %-19s %s: %s  (%s:%d)" % (name, type(e).__name__, str(e)[:200],
                                              os.path.basename(where.filename), where.lineno)


# Worker: one torch2trt checkout with one fake TensorRT version
def worker(t2t_dir, trt_version, names):
    # Version the fake TensorRT reports
    os.environ["FAKE_TRT_VERSION"] = trt_version
    # The fake tensorrt package first, then the torch2trt checkout
    sys.path[:0] = [FAKE_TRT_DIR, os.path.abspath(t2t_dir)]
    # Project modules
    sys.path.append(_HERE)
    # The offline TensorRT
    import tensorrt as fake
    # torch2trt prints its warnings on import too
    with contextlib.redirect_stdout(io.StringIO()):
        # The torch2trt version under test
        import torch2trt
        # Its main module
        from torch2trt import torch2trt as t2t_module
    # Engines are placeholders: replace the TRTModule wrapper, it would try to use them
    sys.modules["torch2trt.torch2trt"].TRTModule = lambda *a, **k: fake._Auto()
    # Test every model
    for name, model in build_models(names):
        # One line per model
        print(check_one(name, model, t2t_module, fake), flush=True)


# Command-line entry point
def main():
    # Parser that shows the module docstring as help
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # torch2trt checkouts to test
    ap.add_argument("--t2t", nargs="+", required=True, help="torch2trt source checkouts")
    # TensorRT versions to pretend (converters are enabled by version)
    ap.add_argument("--trt-version", nargs="+", default=["7.1.3", "8.0.1"])
    # Models to test
    ap.add_argument("--models", nargs="+", default=None, help="default: every network in loader.ARCHS + probes")
    # Internal: run one combination in this process
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    # Parse the command line
    args = ap.parse_args()
    # Default model list
    if args.models is None:
        # Project registry
        sys.path.append(_HERE)
        # Registered networks
        from loader import ARCHS
        # All of them plus the probes
        args.models = list(ARCHS) + ["probe_cat_const", "probe_cat_expand", "probe_frozen", "teammate_dsnt"]
    # Worker mode: exactly one checkout and version
    if args.worker:
        # Run the tests here
        worker(args.t2t[0], args.trt_version[0], args.models)
        # Done
        return
    # Every combination in a fresh process (torch2trt patches torch globally)
    for t2t_dir in args.t2t:
        # Commit date of the checkout, if it is a git checkout
        try:
            # e.g. "2021-03-02 15da623"
            info = subprocess.check_output(["git", "-C", t2t_dir, "log", "-1", "--format=%ad %h", "--date=short"],
                                           stderr=subprocess.DEVNULL).decode().strip()
        # Not a git checkout
        except Exception:
            # No date
            info = "?"
        # Every TensorRT version
        for version in args.trt_version:
            # Header
            print("\n== torch2trt %s (%s), TensorRT %s" % (os.path.basename(os.path.normpath(t2t_dir)), info, version),
                  flush=True)
            # Run the worker
            subprocess.call([sys.executable, os.path.abspath(__file__), "--worker", "--t2t", t2t_dir,
                             "--trt-version", version, "--models"] + args.models)


# Run when executed as a script
if __name__ == "__main__":
    # Start
    main()
