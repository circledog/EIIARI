"""PC training entry point: baseline -> DepGraph pruning -> finetune -> export.

Preprocessing (lane marking) always runs, live inside the DataLoader after
augmentation; nothing is written to disk. ``--arch`` picks the network
(``resnet18.py`` / ``mobilenet_v2.py``).

Examples
--------
    # baseline only, no augmentation (course-equivalent, minus the hflip bug)
    python train.py

    # general + Perlin augmentation, then prune 30 % and 50 %, export ONNX
    python train.py --general --perlin --prune 0.3 0.5 --onnx

    # the same with MobileNetV2
    python train.py --arch mobilenet_v2 --general --perlin --prune 0.3 0.5 --onnx

    # use the team's fixed split: data/splits/{train,validation,test}/
    # (validation picks the best epoch, test is scored once per saved model)
    python train.py --data data/splits --general --perlin --prune 0.3

    # prune an existing baseline without retraining it
    # (augmentation switches and run name are taken from its .json)
    python train.py --base models/resnet18_none_base.pth --prune 0.3 --onnx
    # (the architecture is detected from the file)

Outputs in ``--out`` (default ``models/``), per run name ``<arch>_<aug>`` (``resnet18_gen``, ``mobilenetv2_gen``, ...):
    <name>_base.pth / .json          baseline state_dict + metadata
    <name>_pruned30.pth / .json      pruned + finetuned state_dict + metadata
    <name>_*.onnx                    with --onnx (for trtexec on the Nano)
The .json holds the loss history, MACs/params and data split for the report.
"""
import argparse
import copy
import json
import os
import time

import torch
import torch.nn.functional as F

# Pruning, size counting and ONNX export
from compression import count_macs_params, export_onnx, prune_model
# Dataset folder default and DataLoader construction (incl. lane marking)
from data_loader import DEFAULT_DATA_DIR, has_split_folders, make_loaders
# Network registry, building, rebuilding from weights and .json handling
from loader import ARCHS, build_model, detect_arch, read_meta, rebuild_from_state_dict, write_meta
# Description of the lane-marking preprocessing, saved in every .json
from preprocessing import CONFIG as PREPROCESSING


# Average validation loss of a model
def evaluate(model, loader, device):
    # Evaluation mode (BatchNorm uses running statistics, no dropout)
    model.eval()
    # Summed loss and number of images
    total, n = 0.0, 0
    # No gradients needed for evaluation
    with torch.no_grad():
        # Every validation batch
        for images, labels in loader:
            # Move the batch to the training device
            images, labels = images.to(device), labels.to(device)
            # Summed squared error, halved so it equals the per-image mean over (x, y)
            total += F.mse_loss(model(images), labels, reduction="sum").item() / 2.0
            # Count the images
            n += images.size(0)
    # Mean loss per image (guard against an empty loader)
    return total / max(n, 1)


# Train a model and keep its best weights
def fit(model, train_loader, val_loader, device, epochs, lr, tag):
    """Adam + MSE, keep the weights with the lowest validation loss."""
    # Adam optimiser, as in the course
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Best validation loss so far, its weights, and the per-epoch log
    best_loss, best_state, history = float("inf"), None, []
    # One iteration per epoch
    for epoch in range(1, epochs + 1):
        # Start time of the epoch
        t0 = time.time()
        # Training mode
        model.train()
        # Summed training loss and number of images
        total, n = 0.0, 0
        # Every training batch
        for images, labels in train_loader:
            # Move the batch to the training device
            images, labels = images.to(device), labels.to(device)
            # Clear the gradients of the previous step
            optimizer.zero_grad()
            # Mean squared error between prediction and label
            loss = F.mse_loss(model(images), labels)
            # Back-propagate
            loss.backward()
            # Update the weights
            optimizer.step()
            # Accumulate the loss weighted by batch size
            total += loss.item() * images.size(0)
            # Count the images
            n += images.size(0)
        # Mean training loss of the epoch
        train_loss = total / max(n, 1)
        # Validation loss after the epoch
        val_loss = evaluate(model, val_loader, device)
        # Log both losses for the report
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        # Marker printed next to a new best epoch
        mark = ""
        # New best validation loss
        if val_loss < best_loss:
            # Remember the loss and a copy of the weights
            best_loss, best_state, mark = val_loss, copy.deepcopy(model.state_dict()), " *"
        # Progress line
        print("[%s] epoch %3d/%d  train %.5f  val %.5f  (%.1fs)%s"
              % (tag, epoch, epochs, train_loss, val_loss, time.time() - t0, mark))
    # Restore the best weights
    model.load_state_dict(best_state)
    # Best validation loss and the full log
    return best_loss, history


# Text describing the augmentation, used in the dropdown label
def _variant_label(aug):
    # e.g. "aug=gen+perlin"
    return "aug=%s" % aug.tag()


# Write a checkpoint, optionally its ONNX file, and its .json
def save(model, path, meta, onnx):
    # Save the weights (state_dict)
    torch.save(model.state_dict(), path)
    # Also export ONNX if requested
    if onnx:
        # Export next to the .pth and record the file name in the metadata
        meta["onnx"] = os.path.basename(export_onnx(model, os.path.splitext(path)[0] + ".onnx"))
    # Write the .json sidecar
    write_meta(path, meta)
    # Report the file
    print("saved %s" % path)


# Loss on the held-out test split, evaluated once per finished model
def report_test(model, test_loader, device, tag):
    """Test loss of the final (best-validation) weights, or None without a test split."""
    # Nothing to evaluate without a test/ folder
    if test_loader is None:
        # Stored as null in the .json
        return None
    # Average loss over the test images
    loss = evaluate(model, test_loader, device)
    # Show it next to the training log
    print("[%s] test loss %.5f  (%d images, never used for training or model selection)"
          % (tag, loss, len(test_loader.dataset)))
    # Value for the .json
    return loss


# Command-line training pipeline
def main():
    # Parser that shows the module docstring as help text
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Dataset folder
    ap.add_argument("--data", default=DEFAULT_DATA_DIR,
                    help="folder with xy_*.jpg, or with train/ validation/ [test/] sub-folders (used as is)")
    # Network to train
    ap.add_argument("--arch", choices=sorted(ARCHS), default="resnet18",
                    help="network (ignored with --base: taken from the baseline)")
    # Output folder for checkpoints (default project4/models)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
    # Optional run name (prefix of the output files)
    ap.add_argument("--name", default=None, help="run name (default <arch>_<aug>)")
    # Group of augmentation switches in the help text
    g = ap.add_argument_group("augmentation switches (all off = none)")
    # Switch: general augmentation
    g.add_argument("--general", action="store_true", help="flip / rotate / crop / colour jitter")
    # Switch: Perlin noise
    g.add_argument("--perlin", action="store_true", help="Perlin-noise lighting")
    # Switch: compression artefacts
    g.add_argument("--compression", action="store_true", help="JPEG / 144p artefacts")
    # Baseline training epochs
    ap.add_argument("--epochs", type=int, default=70)
    # Baseline learning rate
    ap.add_argument("--lr", type=float, default=1e-3)
    # Batch size
    ap.add_argument("--batch-size", type=int, default=8)
    # Share of images used for validation
    ap.add_argument("--val-ratio", type=float, default=0.1)
    # How to split train / validation
    ap.add_argument("--split", choices=["random", "session"], default="random",
                    help="session: keep shooting segments together (needs the metadata folder)")
    # Metadata folder for the session split
    ap.add_argument("--metadata", default=None, help="metadata folder (default: next to --data)")
    # Pause that starts a new shooting segment
    ap.add_argument("--segment-gap", type=float, default=60.0,
                    help="seconds between captures that start a new segment (--split session)")
    # DataLoader worker processes
    ap.add_argument("--workers", type=int, default=4)
    # Random seed for split, shuffling and initialisation
    ap.add_argument("--seed", type=int, default=0)
    # Existing baseline to prune instead of training a new one
    ap.add_argument("--base", default=None, help="existing baseline .pth: skip baseline training")
    # Pruning ratios (one pruned model per ratio)
    ap.add_argument("--prune", type=float, nargs="*", default=[], help="pruning ratios, e.g. 0.3 0.5")
    # Channel counts after pruning are rounded to a multiple of this
    ap.add_argument("--round-to", type=int, default=8)
    # Fine-tuning epochs after pruning
    ap.add_argument("--finetune-epochs", type=int, default=15)
    # Fine-tuning learning rate
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    # Also export ONNX files
    ap.add_argument("--onnx", action="store_true", help="also export ONNX (opset 11)")
    # Testing only: random initialisation instead of ImageNet weights
    ap.add_argument("--no-pretrained", action="store_true", help="(testing) skip ImageNet weights")
    # Training device
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Parse the command line
    args = ap.parse_args()

    # Seed torch for reproducible initialisation
    torch.manual_seed(args.seed)
    # Training device
    device = torch.device(args.device)
    # Create the output folder if needed
    os.makedirs(args.out, exist_ok=True)

    # Settings inherited from an existing baseline
    if args.base:
        # The network type comes from the baseline's weights
        args.arch = detect_arch(torch.load(args.base, map_location="cpu"))
        # Reuse the baseline's augmentation switches and run name
        # unless overridden, so pruned files stay grouped with their baseline.
        base_aug = read_meta(args.base).get("augmentation", {})
        # Only when no augmentation switch was given on the command line
        if not (args.general or args.perlin or args.compression):
            # Baseline's general switch
            args.general = bool(base_aug.get("general"))
            # Baseline's Perlin switch
            args.perlin = bool(base_aug.get("perlin"))
            # Baseline's compression switch
            args.compression = bool(base_aug.get("compression"))
        # Baseline file name without extension
        stem = os.path.splitext(os.path.basename(args.base))[0]
        # Derive the run name from "<name>_base"
        if args.name is None and stem.endswith("_base"):
            # Strip the "_base" suffix
            args.name = stem[:-len("_base")]

    # The team's sub-folders decide the split; --split / --val-ratio do not apply
    if has_split_folders(args.data) and (args.split != "random" or args.val_ratio != 0.1):
        # Tell the user their split options are ignored
        print("note: %s has train/validation sub-folders: --split / --val-ratio are ignored" % args.data)
    # Training / validation / test loaders (lane marking happens inside)
    train_loader, val_loader, test_loader, aug = make_loaders(
        args.data, general=args.general, perlin=args.perlin, compression=args.compression,
        val_ratio=args.val_ratio, batch_size=args.batch_size, num_workers=args.workers, seed=args.seed,
        split=args.split, metadata_dir=args.metadata, segment_gap=args.segment_gap)
    # Module of the chosen network (names and labels)
    arch = ARCHS[args.arch]
    # Run name, e.g. "mobilenetv2_gen"
    name = args.name or "%s_%s" % (arch.FILE_TAG, aug.tag())
    # Metadata shared by the baseline and the pruned models
    common = {
        "arch": args.arch,
        "preprocessing": PREPROCESSING,
        "augmentation": {"general": aug.general, "perlin": aug.perlin,
                         "compression": aug.compression, "tag": aug.tag()},
        "data": os.path.abspath(args.data),
        "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset),
        "n_test": len(test_loader.dataset) if test_loader is not None else 0,
        "val_ratio": args.val_ratio, "seed": args.seed, "batch_size": args.batch_size,
        "split": train_loader.split_info,
    }
    # Summary of the setup
    print("data: %d train / %d val / %d test, %s, %r, lane marking on, device %s"
          % (common["n_train"], common["n_val"], common["n_test"], arch.LABEL, aug, device))
    # How the data was split
    print("split: %s" % train_loader.split_info)

    # ---- baseline --------------------------------------------------------- #
    # Prune an existing baseline
    if args.base:
        # Rebuild the baseline network from its weights
        model = rebuild_from_state_dict(torch.load(args.base, map_location="cpu")).to(device)
        # Its metadata
        base_meta = read_meta(args.base)
        # Its file
        base_path = args.base
        # Show its validation loss on the current split
        print("loaded baseline %s (val %.5f)" % (args.base, evaluate(model, val_loader, device)))
        # Also show its test loss when there is a test split
        report_test(model, test_loader, device, "base")
    # Train a new baseline
    else:
        # Fresh network (ImageNet weights unless --no-pretrained)
        model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
        # Train it
        best, history = fit(model, train_loader, val_loader, device, args.epochs, args.lr, "base")
        # Its size
        macs, params = count_macs_params(model)
        # Metadata of the baseline
        base_meta = dict(common, label="%s | %s | no compression" % (arch.LABEL, _variant_label(aug)),
                         compression=None, epochs=args.epochs, lr=args.lr,
                         best_val_loss=best, test_loss=report_test(model, test_loader, device, "base"),
                         macs=macs, params=params, history=history)
        # File of the baseline
        base_path = os.path.join(args.out, name + "_base.pth")
        # Save it
        save(model, base_path, base_meta, args.onnx)

    # ---- DepGraph pruning + finetune -------------------------------------- #
    # One pruned model per ratio
    for ratio in args.prune:
        # Prune a copy of the baseline
        pruned, info = prune_model(model, ratio, round_to=args.round_to)
        # Make sure it is on the training device
        pruned = pruned.to(device)
        # Validation loss right after pruning, before fine-tuning
        before = evaluate(pruned, val_loader, device)
        # Report the size reduction
        print("pruned %.0f%%: MACs %.2fG -> %.2fG, params %.2fM -> %.2fM, val before finetune %.5f"
              % (ratio * 100, info["macs_before"] / 1e9, info["macs_after"] / 1e9,
                 info["params_before"] / 1e6, info["params_after"] / 1e6, before))
        # File suffix, e.g. "pruned30"
        tag = "pruned%02d" % round(ratio * 100)
        # Fine-tune the pruned network
        best, history = fit(pruned, train_loader, val_loader, device,
                            args.finetune_epochs, args.finetune_lr, tag)
        # Metadata of the pruned model
        meta = dict(common, label="%s | %s | DepGraph %d%%" % (arch.LABEL, _variant_label(aug), round(ratio * 100)),
                    compression=dict(info, method="DepGraph", importance="GroupMagnitude(p=2)"),
                    base=os.path.basename(base_path), base_val_loss=base_meta.get("best_val_loss"),
                    val_loss_before_finetune=before, epochs=args.finetune_epochs, lr=args.finetune_lr,
                    best_val_loss=best, test_loss=report_test(pruned, test_loader, device, tag),
                    macs=info["macs_after"], params=info["params_after"],
                    history=history)
        # Save it
        save(pruned, os.path.join(args.out, "%s_%s.pth" % (name, tag)), meta, args.onnx)

    # Machine-readable summary of where the files went
    print(json.dumps({"out": args.out, "name": name}, indent=None))


# Run the pipeline when executed as a script
if __name__ == "__main__":
    # Start training
    main()
