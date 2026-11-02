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

import mobilenet_v2
import resnet18

__all__ = ["ARCHS", "build_model", "detect_arch", "rebuild_from_state_dict", "load_model",
           "LoadedModel", "list_models", "read_meta", "write_meta"]

ARCHS = {resnet18.NAME: resnet18, mobilenet_v2.NAME: mobilenet_v2}


def build_model(arch="resnet18", pretrained=True, num_outputs=2):
    """Network used for training."""
    if arch not in ARCHS:
        raise ValueError("unknown arch %r (choose from %s)" % (arch, ", ".join(sorted(ARCHS))))
    return ARCHS[arch].build(pretrained=pretrained, num_outputs=num_outputs)


def detect_arch(state_dict):
    """Name of the network a state_dict belongs to (from its head layer key)."""
    for name, mod in ARCHS.items():
        if mod.HEAD_KEY in state_dict:
            return name
    raise ValueError("unknown network: none of %s in the state_dict"
                     % ", ".join(m.HEAD_KEY for m in ARCHS.values()))


def rebuild_from_state_dict(state_dict, arch=None, num_outputs=2):
    """Network whose conv / bn / linear sizes match ``state_dict``, weights loaded.

    Works for unpruned and structurally pruned networks: every Conv2d /
    BatchNorm2d / Linear whose shape differs is replaced by a module of the
    stored shape (stride / padding / kernel are kept from the stock layer;
    depthwise convolutions stay depthwise).
    """
    arch = arch or detect_arch(state_dict)
    model = ARCHS[arch].build(pretrained=False, num_outputs=num_outputs)
    modules = dict(model.named_modules())
    for name, mod in list(modules.items()):
        wkey = name + ".weight"
        if wkey not in state_dict:
            continue
        shape = tuple(state_dict[wkey].shape)
        if isinstance(mod, nn.Conv2d) and tuple(mod.weight.shape) != shape:
            depthwise = mod.groups > 1 and mod.groups == mod.in_channels
            groups = shape[0] if depthwise else mod.groups
            new = nn.Conv2d(shape[1] * groups, shape[0], kernel_size=mod.kernel_size,
                            stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                            groups=groups, bias=(name + ".bias") in state_dict)
        elif isinstance(mod, nn.BatchNorm2d) and mod.num_features != shape[0]:
            new = nn.BatchNorm2d(shape[0], eps=mod.eps, momentum=mod.momentum)
        elif isinstance(mod, nn.Linear) and tuple(mod.weight.shape) != shape:
            new = nn.Linear(shape[1], shape[0], bias=(name + ".bias") in state_dict)
        else:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        setattr(parent, child, new)
    model.load_state_dict(state_dict)
    return model


# --------------------------------------------------------------------------- #
# Metadata sidecar: <model>.json next to <model>.pth
# --------------------------------------------------------------------------- #
def _meta_path(path):
    return os.path.splitext(path)[0] + ".json"


def write_meta(path, meta):
    with open(_meta_path(path), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def read_meta(path):
    p = _meta_path(path)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def list_models(directory):
    """``[(label, path), ...]`` for every .pth in ``directory`` (for the UI dropdown)."""
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "*.pth"))):
        meta = read_meta(path)
        label = meta.get("label") or os.path.basename(path)
        out.append((label, path))
    return out


# --------------------------------------------------------------------------- #
# Inference loading
# --------------------------------------------------------------------------- #
class LoadedModel(object):
    """Callable wrapper: ``model(tensor) -> tensor``, plus info for the UI.

    Attributes: ``kind`` ('torch' or 'trt'), ``dtype`` (input dtype the model
    expects -- feed ``camera.preprocess(..., dtype=model.dtype)``), ``meta``.
    """

    def __init__(self, module, kind, dtype, device, path, meta):
        self.module = module
        self.kind = kind
        self.dtype = dtype
        self.device = device
        self.path = path
        self.meta = meta

    def __call__(self, x):
        with torch.no_grad():
            return self.module(x.to(self.device, self.dtype))

    def describe(self):
        return "%s [%s, %s]" % (os.path.basename(self.path), self.kind,
                                str(self.dtype).replace("torch.", ""))


def _trt_input_dtype(module):
    """Input dtype of a TRTModule's engine; FP32 if it cannot be determined."""
    try:
        import tensorrt as trt
        engine = module.engine
        try:
            name = module.input_names[0]
            trt_dtype = engine.get_tensor_dtype(name)  # TensorRT >= 8.5
        except AttributeError:
            trt_dtype = engine.get_binding_dtype(engine.get_binding_index(module.input_names[0]))
        return torch.float16 if trt_dtype == trt.float16 else torch.float32
    except Exception:
        return torch.float32


def load_model(path, device=None, half=None):
    """Load a ``.pth`` for inference.

    * dict with ``'engine'`` -> TensorRT via ``torch2trt.TRTModule``; input dtype
      is read from the engine (trtexec engines take FP32, torch2trt fp16 ones FP16).
    * otherwise a state_dict -> network detected from its keys and rebuilt
      from the tensor shapes (pruned or not).
      ``half`` defaults to True on CUDA (like slides P.55), False on CPU.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    meta = read_meta(path)
    try:  # torch >= 2.6 defaults to weights_only=True, which rejects the engine bytearray
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # old torch on the Nano has no weights_only argument
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and "engine" in obj:
        from torch2trt import TRTModule  # only available on the robot
        module = TRTModule()
        module.load_state_dict(obj)
        return LoadedModel(module, "trt", _trt_input_dtype(module), device, path, meta)

    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    module = rebuild_from_state_dict(obj)
    if half is None:
        half = device.type == "cuda"
    dtype = torch.float16 if half else torch.float32
    module = module.to(device).eval()
    if half:
        module = module.half()
    return LoadedModel(module, "torch", dtype, device, path, meta)
