"""Model compression: DepGraph structured channel pruning (torch-pruning).

PC side:
    prune_model()      -- MetaPruner + GroupMagnitudeImportance(p=2), output Linear ignored,
                          round_to=8. Returns the physically smaller model.
    count_macs_params()
    export_onnx()      -- opset 11, input_0 / output_0, legacy exporter.

Robot side (no torch-pruning needed):
    wrap_trt_engine()  -- turn a ``trtexec`` .engine into a TRTModule .pth.

``torch_pruning`` and ``onnx`` are imported lazily so that this module can be
imported on the Nano just for ``wrap_trt_engine``.
"""
import copy
import os

import torch
import torch.nn as nn

__all__ = ["prune_model", "count_macs_params", "export_onnx", "wrap_trt_engine"]


def _example_input(device=None):
    return torch.randn(1, 3, 224, 224, device=device)


def count_macs_params(model, device=None):
    """(MACs, params) for a 1x3x224x224 input, via torch-pruning's counter."""
    import torch_pruning as tp
    model = model.eval()
    if device is None:
        device = next(model.parameters()).device
    macs, params = tp.utils.count_ops_and_params(model, _example_input(device))
    return int(macs), int(params)


def prune_model(model, ratio, round_to=8, inplace=False):
    """DepGraph structured pruning of ``model`` to ``ratio`` channel sparsity.

    Coupled layers (residual adds, BN after conv, ...) are pruned together
    through the dependency graph, so the result is a valid, smaller network.
    The output Linear layer (``fc`` / ``classifier[1]``) is never pruned. Fine-tune afterwards (required).

    Returns (pruned_model, info) where info has MACs / params before & after.
    """
    import torch_pruning as tp
    if not inplace:
        model = copy.deepcopy(model)
    model = model.eval()
    device = next(model.parameters()).device
    example = _example_input(device)

    ignored = [m for m in model.modules() if isinstance(m, nn.Linear)]
    macs0, params0 = tp.utils.count_ops_and_params(model, example)
    pruner = tp.pruner.MetaPruner(
        model,
        example,
        importance=tp.importance.GroupMagnitudeImportance(p=2),
        pruning_ratio=ratio,
        ignored_layers=ignored,
        round_to=round_to,
    )
    pruner.step()
    macs1, params1 = tp.utils.count_ops_and_params(model, example)
    info = {
        "ratio": ratio,
        "round_to": round_to,
        "macs_before": int(macs0), "params_before": int(params0),
        "macs_after": int(macs1), "params_after": int(params1),
    }
    return model, info


def export_onnx(model, path, opset=11):
    """Export for ``trtexec`` (TensorRT on JetPack 4.x handles opset 11)."""
    model = copy.deepcopy(model).float().cpu().eval()
    kwargs = dict(opset_version=opset, input_names=["input_0"], output_names=["output_0"],
                  do_constant_folding=True)
    try:
        torch.onnx.export(model, _example_input(), path, dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(model, _example_input(), path, **kwargs)
    return path


def wrap_trt_engine(engine_path, out_path, input_names=("input_0",), output_names=("output_0",)):
    """Pack a serialized TensorRT engine into the dict format ``TRTModule`` loads.

    Run on the Nano after:
        /usr/src/tensorrt/bin/trtexec --onnx=X.onnx --saveEngine=X.engine --fp16
    The engine input stays FP32, so feed FP32 tensors (``model.load_model``
    detects this automatically).
    """
    with open(engine_path, "rb") as f:
        engine_bytes = f.read()
    torch.save({"engine": bytearray(engine_bytes),
                "input_names": list(input_names),
                "output_names": list(output_names)}, out_path)

    # Carry over the PyTorch model's metadata (X.json next to X.engine) so the
    # live-demo dropdown shows e.g. "ResNet18 | aug=none | DepGraph 30% | TensorRT".
    from loader import read_meta, write_meta
    meta = read_meta(engine_path)
    if meta:
        meta = dict(meta, label=meta.get("label", "") + " | TensorRT", runtime="tensorrt",
                    engine=os.path.basename(engine_path))
        write_meta(out_path, meta)
    return out_path
