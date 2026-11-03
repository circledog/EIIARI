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

# Public names of this module
__all__ = ["prune_model", "count_macs_params", "export_onnx", "wrap_trt_engine"]


# Dummy input with the network's input shape (one 224x224 RGB image)
def _example_input(device=None):
    # Random tensor on the requested device; only the shape matters
    return torch.randn(1, 3, 224, 224, device=device)


# Count multiply-accumulates and parameters of a network
def count_macs_params(model, device=None):
    """(MACs, params) for a 1x3x224x224 input, via torch-pruning's counter."""
    import torch_pruning as tp
    # Evaluation mode so BatchNorm / Dropout behave deterministically
    model = model.eval()
    # Default to the device the model already lives on
    if device is None:
        # Read the device from the first parameter
        device = next(model.parameters()).device
    # Trace the network once and count operations and parameters
    macs, params = tp.utils.count_ops_and_params(model, _example_input(device))
    # Return plain ints so they can be written to JSON
    return int(macs), int(params)


# Remove a fraction of the channels with DepGraph structured pruning
def prune_model(model, ratio, round_to=8, inplace=False):
    """DepGraph structured pruning of ``model`` to ``ratio`` channel sparsity.

    Coupled layers (residual adds, BN after conv, ...) are pruned together
    through the dependency graph, so the result is a valid, smaller network.
    The output Linear layer (``fc`` / ``classifier[1]``) is never pruned. Fine-tune afterwards (required).

    Returns (pruned_model, info) where info has MACs / params before & after.
    """
    import torch_pruning as tp
    # Work on a copy unless the caller allows changing the original
    if not inplace:
        # Deep copy keeps the unpruned baseline intact
        model = copy.deepcopy(model)
    # Evaluation mode for tracing the dependency graph
    model = model.eval()
    # Device of the model, so the example input lives there too
    device = next(model.parameters()).device
    # Example input used to trace the network
    example = _example_input(device)

    # Never prune the output Linear layer: it must keep its 2 outputs
    ignored = [m for m in model.modules() if isinstance(m, nn.Linear)]
    # Size of the network before pruning
    macs0, params0 = tp.utils.count_ops_and_params(model, example)
    # DepGraph pruner: L2 magnitude of each coupled channel group decides what goes
    pruner = tp.pruner.MetaPruner(
        model,
        example,
        importance=tp.importance.GroupMagnitudeImportance(p=2),
        pruning_ratio=ratio,
        ignored_layers=ignored,
        round_to=round_to,
    )
    # Physically remove the selected channels
    pruner.step()
    # Size of the network after pruning
    macs1, params1 = tp.utils.count_ops_and_params(model, example)
    # Summary that train.py stores in the model's .json
    info = {
        "ratio": ratio,
        "round_to": round_to,
        "macs_before": int(macs0), "params_before": int(params0),
        "macs_after": int(macs1), "params_after": int(params1),
    }
    # Pruned network (needs fine-tuning) and its summary
    return model, info


# Write the network to an ONNX file for trtexec on the Nano
def export_onnx(model, path, opset=11):
    """Export for ``trtexec`` (TensorRT on JetPack 4.x handles opset 11)."""
    # FP32 CPU copy in eval mode, so the original model is left untouched
    model = copy.deepcopy(model).float().cpu().eval()
    # Export options shared by both torch versions: fixed tensor names, folded constants
    kwargs = dict(opset_version=opset, input_names=["input_0"], output_names=["output_0"],
                  do_constant_folding=True)
    # New torch: force the legacy (TorchScript) exporter, which supports opset 11
    try:
        # Export with dynamo disabled
        torch.onnx.export(model, _example_input(), path, dynamo=False, **kwargs)
    # Older torch without the dynamo flag
    except TypeError:
        # Export with the only exporter old torch has
        torch.onnx.export(model, _example_input(), path, **kwargs)
    # Path of the written file
    return path


# Package a trtexec engine so torch2trt's TRTModule can load it (runs on the Nano)
def wrap_trt_engine(engine_path, out_path, input_names=("input_0",), output_names=("output_0",)):
    """Pack a serialized TensorRT engine into the dict format ``TRTModule`` loads.

    Run on the Nano after:
        /usr/src/tensorrt/bin/trtexec --onnx=X.onnx --saveEngine=X.engine --fp16
    (``convert_trt.py --method trtexec`` does both steps and the checks.)
    (``convert_trt.py --method trtexec`` does both steps and the checks.)
    The engine input stays FP32, so feed FP32 tensors (``loader.load_model``
    detects this automatically).
    """
    # Open the serialized engine produced by trtexec
    with open(engine_path, "rb") as f:
        # Read the whole engine into memory
        engine_bytes = f.read()
    # Save it in the dict layout TRTModule.load_state_dict expects
    torch.save({"engine": bytearray(engine_bytes),
                "input_names": list(input_names),
                "output_names": list(output_names)}, out_path)

    # Carry over the PyTorch model's metadata (X.json next to X.engine) so the
    # live-demo dropdown shows e.g. "ResNet18 | aug=none | DepGraph 30% | TensorRT".
    from loader import read_meta, write_meta
    # Metadata of the matching PyTorch model, if its .json sits next to the engine
    meta = read_meta(engine_path)
    # Only write a .json when there was one to copy
    if meta:
        # Same metadata, labelled as the TensorRT version
        meta = dict(meta, label=meta.get("label", "") + " | TensorRT", runtime="tensorrt",
                    engine=os.path.basename(engine_path))
        # Fingerprint of the wrapped file (the PyTorch one no longer applies)
        from sanity import file_sha256
        # Record it so load_model can verify the copy
        meta["weights_sha256"] = file_sha256(out_path)
        # Write X_trt.json next to the wrapped .pth
        write_meta(out_path, meta)
    # Path of the wrapped .pth
    return out_path
