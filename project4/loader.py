"""Shared model loading for every network (``resnet18.py``, ``mobilenet_v2.py``).

* ``build_model(arch)``         -- network for training (``ARCHS`` lists them).
* ``detect_arch(state_dict)``   -- which network a state_dict belongs to.
* ``rebuild_from_state_dict``   -- recreate a (DepGraph-pruned) network whose
  channel counts differ from the stock one, using the tensor shapes in the
  state_dict (slides P.55's plain ``load_state_dict`` fails after pruning).
* ``load_model`` -- load either a PyTorch state_dict ``.pth`` or a TensorRT
  ``TRTModule`` ``.pth`` (dict with an ``engine`` key) for inference.
* ``list_models`` / ``read_meta`` / ``write_meta`` -- checkpoint + ``.json``.

To add a network: write ``<name>.py`` with ``NAME``, ``FILE_TAG``, ``LABEL``,
``HEAD_KEY`` and ``build(pretrained, num_outputs)``, then add it to ``ARCHS``.

Kept Python 3.6 / old-torchvision compatible because it runs on the Jetson Nano.
"""
import glob
import json
import os

import torch
import torch.nn as nn

# Network definition: MobileNetV2
import mobilenet_v2
# Network definition: ResNet-18
import resnet18

# Public names of this module
__all__ = ["ARCHS", "build_model", "detect_arch", "rebuild_from_state_dict", "load_model",
           "LoadedModel", "list_models", "read_meta", "write_meta"]

# Registry of the supported networks, keyed by their --arch name
ARCHS = {resnet18.NAME: resnet18, mobilenet_v2.NAME: mobilenet_v2}


# Create a fresh network for training
def build_model(arch="resnet18", pretrained=True, num_outputs=2):
    """Network used for training."""
    # Reject names that are not in the registry
    if arch not in ARCHS:
        # Tell the user which names are valid
        raise ValueError("unknown arch %r (choose from %s)" % (arch, ", ".join(sorted(ARCHS))))
    # Let the network's own module build it
    return ARCHS[arch].build(pretrained=pretrained, num_outputs=num_outputs)


# Work out which network a saved state_dict belongs to
def detect_arch(state_dict):
    """Name of the network a state_dict belongs to (from its head layer key)."""
    # Try every registered network
    for name, mod in ARCHS.items():
        # Its output layer key is unique, so finding it identifies the network
        if mod.HEAD_KEY in state_dict:
            # Found the matching network
            return name
    # No known output layer key: not one of our networks
    raise ValueError("unknown network: none of %s in the state_dict"
                     % ", ".join(m.HEAD_KEY for m in ARCHS.values()))


# Recreate a (possibly pruned) network from nothing but its weights
def rebuild_from_state_dict(state_dict, arch=None, num_outputs=2):
    """Network whose conv / bn / linear sizes match ``state_dict``, weights loaded.

    Works for unpruned and structurally pruned networks: every Conv2d /
    BatchNorm2d / Linear whose shape differs is replaced by a module of the
    stored shape (stride / padding / kernel are kept from the stock layer;
    depthwise convolutions stay depthwise).
    """
    # Detect the network unless the caller says which one it is
    arch = arch or detect_arch(state_dict)
    # Start from the stock, unpruned network (no download needed)
    model = ARCHS[arch].build(pretrained=False, num_outputs=num_outputs)
    # Map from dotted module name to module, for looking up parents
    modules = dict(model.named_modules())
    # Visit every layer of the stock network
    for name, mod in list(modules.items()):
        # Key of this layer's weight in the state_dict
        wkey = name + ".weight"
        # Layers without weights (ReLU, pooling, ...) need no change
        if wkey not in state_dict:
            # Go to the next layer
            continue
        # Shape the saved (maybe pruned) weight has
        shape = tuple(state_dict[wkey].shape)
        # Convolution whose channel counts were changed by pruning
        if isinstance(mod, nn.Conv2d) and tuple(mod.weight.shape) != shape:
            # A depthwise conv has one group per input channel
            depthwise = mod.groups > 1 and mod.groups == mod.in_channels
            # Depthwise stays depthwise: groups follow the new channel count
            groups = shape[0] if depthwise else mod.groups
            # Same kernel / stride / padding, new channel counts
            new = nn.Conv2d(shape[1] * groups, shape[0], kernel_size=mod.kernel_size,
                            stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                            groups=groups, bias=(name + ".bias") in state_dict)
        # BatchNorm whose number of channels changed
        elif isinstance(mod, nn.BatchNorm2d) and mod.num_features != shape[0]:
            # New BatchNorm with the pruned channel count and the same settings
            new = nn.BatchNorm2d(shape[0], eps=mod.eps, momentum=mod.momentum)
        # Linear layer whose input size changed
        elif isinstance(mod, nn.Linear) and tuple(mod.weight.shape) != shape:
            # New Linear with the saved input / output sizes
            new = nn.Linear(shape[1], shape[0], bias=(name + ".bias") in state_dict)
        # Shape already matches: keep the stock layer
        else:
            # Go to the next layer
            continue
        # Split "layer1.0.conv1" into parent "layer1.0" and attribute "conv1"
        parent_name, _, child = name.rpartition(".")
        # Top-level layers hang directly off the model
        parent = modules[parent_name] if parent_name else model
        # Swap in the resized layer
        setattr(parent, child, new)
    # All shapes match now, so the weights load without size mismatch
    model.load_state_dict(state_dict)
    # Ready-to-use network with the saved weights
    return model


# --------------------------------------------------------------------------- #
# Metadata sidecar: <model>.json next to <model>.pth
# --------------------------------------------------------------------------- #
# Path of the .json that belongs to a checkpoint
def _meta_path(path):
    # Same name, .json extension
    return os.path.splitext(path)[0] + ".json"


# Save a checkpoint's metadata (label, history, MACs, ...)
def write_meta(path, meta):
    # Open the sidecar .json for writing
    with open(_meta_path(path), "w") as f:
        # Pretty-printed, keeping non-ASCII text readable
        json.dump(meta, f, indent=2, ensure_ascii=False)


# Load a checkpoint's metadata, or an empty dict if there is none
def read_meta(path):
    # Path of the sidecar .json
    p = _meta_path(path)
    # Only read it if it exists
    if os.path.exists(p):
        # Open the .json
        with open(p) as f:
            # Parsed metadata
            return json.load(f)
    # No metadata for this checkpoint
    return {}


# List every checkpoint in a folder for the live demo dropdown
def list_models(directory):
    """``[(label, path), ...]`` for every .pth in ``directory`` (for the UI dropdown)."""
    # Collected (label, path) pairs
    out = []
    # Every .pth in the folder, in a stable order
    for path in sorted(glob.glob(os.path.join(directory, "*.pth"))):
        # Its metadata, if any
        meta = read_meta(path)
        # Show the label from the .json, or the file name when there is none
        label = meta.get("label") or os.path.basename(path)
        # Add the dropdown entry
        out.append((label, path))
    # All entries
    return out


# --------------------------------------------------------------------------- #
# Inference loading
# --------------------------------------------------------------------------- #
# Uniform wrapper around a PyTorch or TensorRT model for inference
class LoadedModel(object):
    """Callable wrapper: ``model(tensor) -> tensor``, plus info for the UI.

    Attributes: ``kind`` ('torch' or 'trt'), ``dtype`` (input dtype the model
    expects -- feed ``camera.preprocess(..., dtype=model.dtype)``), ``meta``.
    """

    # Keep the module and everything the UI needs to know about it
    def __init__(self, module, kind, dtype, device, path, meta):
        # The actual network (nn.Module or TRTModule)
        self.module = module
        # 'torch' or 'trt'
        self.kind = kind
        # Input dtype the network expects (float16 / float32)
        self.dtype = dtype
        # Device the network runs on
        self.device = device
        # Checkpoint file it was loaded from
        self.path = path
        # Contents of the checkpoint's .json
        self.meta = meta

    # Run the network on one input batch
    def __call__(self, x):
        # No gradients are needed for inference
        with torch.no_grad():
            # Move the input to the right device / dtype and run the network
            return self.module(x.to(self.device, self.dtype))

    # One-line description for the UI, e.g. "x.pth [torch, float16]"
    def describe(self):
        # File name, runtime and input dtype
        return "%s [%s, %s]" % (os.path.basename(self.path), self.kind,
                                str(self.dtype).replace("torch.", ""))


# Find out whether a TensorRT engine wants FP16 or FP32 input
def _trt_input_dtype(module):
    """Input dtype of a TRTModule's engine; FP32 if it cannot be determined."""
    # Any failure below means we cannot tell, so fall back to FP32
    try:
        import tensorrt as trt
        # The deserialized engine inside the TRTModule
        engine = module.engine
        # TensorRT >= 8.5 API
        try:
            # Name of the first input tensor
            name = module.input_names[0]
            # Ask the engine for that tensor's dtype
            trt_dtype = engine.get_tensor_dtype(name)
        # Older TensorRT (JetPack 4.x) uses binding indices instead
        except AttributeError:
            # Look up the binding index of the input and ask for its dtype
            trt_dtype = engine.get_binding_dtype(engine.get_binding_index(module.input_names[0]))
        # Map the TensorRT dtype to the torch dtype
        return torch.float16 if trt_dtype == trt.float16 else torch.float32
    # TensorRT missing or API mismatch
    except Exception:
        # Safe default: trtexec engines take FP32 input
        return torch.float32


# Load a checkpoint (PyTorch weights or wrapped TensorRT engine) for inference
def load_model(path, device=None, half=None):
    """Load a ``.pth`` for inference.

    * dict with ``'engine'`` -> TensorRT via ``torch2trt.TRTModule``; input dtype
      is read from the engine (trtexec engines take FP32, torch2trt fp16 ones FP16).
    * otherwise a state_dict -> network detected from its keys and rebuilt
      from the tensor shapes (pruned or not).
      ``half`` defaults to True on CUDA (like slides P.55), False on CPU.
    """
    # Default device: GPU if there is one
    if device is None:
        # CUDA on the Nano, CPU on a machine without a GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Accept strings like "cpu" as well as torch.device objects
    device = torch.device(device)
    # Metadata shown in the UI
    meta = read_meta(path)
    # torch >= 2.6 defaults to weights_only=True, which rejects the engine bytearray
    try:
        # Load the file with full unpickling allowed
        obj = torch.load(path, map_location="cpu", weights_only=False)
    # Old torch on the Nano has no weights_only argument
    except TypeError:
        # Load the file the old way
        obj = torch.load(path, map_location="cpu")

    # A dict with an 'engine' key is a wrapped TensorRT engine
    if isinstance(obj, dict) and "engine" in obj:
        # Only available on the robot
        from torch2trt import TRTModule
        # Empty TensorRT module
        module = TRTModule()
        # Deserialize the engine into it
        module.load_state_dict(obj)
        # Wrap it, with the input dtype read from the engine
        return LoadedModel(module, "trt", _trt_input_dtype(module), device, path, meta)

    # Some checkpoints nest the weights under a 'state_dict' key
    if isinstance(obj, dict) and "state_dict" in obj:
        # Unwrap the actual weights
        obj = obj["state_dict"]
    # Rebuild the right network (pruned or not) from the weights
    module = rebuild_from_state_dict(obj)
    # Default precision: FP16 on CUDA (like slides P.55), FP32 on CPU
    if half is None:
        # Half precision only makes sense on the GPU
        half = device.type == "cuda"
    # Input dtype that matches the chosen precision
    dtype = torch.float16 if half else torch.float32
    # Move to the device and switch to evaluation mode
    module = module.to(device).eval()
    # Convert the weights to FP16 if requested
    if half:
        # Half-precision weights
        module = module.half()
    # Wrap it for the UI
    return LoadedModel(module, "torch", dtype, device, path, meta)
