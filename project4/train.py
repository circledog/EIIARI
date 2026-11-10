"""PC training entry point: baseline -> DepGraph pruning -> finetune -> export.

Preprocessing (lane marking) always runs, live inside the DataLoader after
augmentation; nothing is written to disk. ``--arch`` picks the network; the
training routine itself lives in that network's file (``resnet18.train`` /
``mobilenet_v2.train``), together with its default epochs and learning rates.
This script only decides *what* to run (task) and calls the network's code.

Examples
--------
    # baseline only, no augmentation (course-equivalent, minus the hflip bug)
    python train.py

    # augmentation: subsets of at most 2 methods (each image x 29 per epoch), then prune 30 % and 50 %, export ONNX
    python train.py --augment --prune 0.3 0.5 --onnx

    # the same with MobileNetV2
    python train.py --arch mobilenet_v2 --augment --prune 0.3 0.5 --onnx

    # use the team's fixed split: data/splits/{train,validation,test}/
    # (validation picks the best epoch, test is scored once per saved model)
    python train.py --data data/splits --augment --prune 0.3

    # the teammate's variants (CoordConv / DSNT / both), same options as any network
    python train.py --arch resnet18_coord --data data/splits --augment --seeds 17 29 43
    python train.py --arch resnet18_dsnt --data data/splits --augment --prune 0.3

    # three training seeds on the same split: mean +- std for the report
    python train.py --data data/splits --augment --seeds 0 1 2 --prune 0.3

    # prune an existing baseline without retraining it
    # (augmentation switch and run name are taken from its .json)
    python train.py --base models/resnet18_none_base.pth --prune 0.3 --onnx
    # (the architecture is detected from the file)

Outputs in ``--out`` (default ``models/``), per run name ``<arch>_<aug>`` (``resnet18_aug2``, ``mobilenetv2_none``, ...):
    <name>_base.pth / .json          baseline state_dict + metadata
    <name>_pruned30.pth / .json      pruned + finetuned state_dict + metadata
    <name>_*.onnx                    with --onnx (for trtexec on the Nano)
    <name>_seeds.json                with several --seeds: mean / std per model
With several seeds every file name gets ``_s<seed>`` (``resnet18_aug2_s1_base.pth``).
The .json holds the loss history, MSE and pixel errors (validation and test),
MACs/params, data split, the weights' SHA256 and the sanity frames that
``live_demo`` uses to check the model on the robot.
"""
import argparse
import json
import os

import numpy as np
import torch

# Names of the augmentation methods, saved in every .json
from augmentation import MAX_METHODS, METHODS
# Pruning, size counting and ONNX export
from compression import count_macs_params, export_onnx, prune_model
# Dataset folder default and DataLoader construction (incl. lane marking)
from data_loader import DEFAULT_DATA_DIR, has_split_folders, make_loaders
# Network registry, building, rebuilding from weights and .json handling
from loader import ARCHS, build_model, detect_arch, read_meta, rebuild_from_state_dict, write_meta
# MSE and pixel-error metrics
from metrics import evaluate
# Training recipe description and the selectable losses
from training import LOSSES, RECIPE
# Description of the lane-marking preprocessing, saved in every .json
from preprocessing import CONFIG as PREPROCESSING
# Weights fingerprint and sanity frames for the robot-side check
from sanity import file_sha256, make_sanity

# Metrics summarised across seeds
SUMMARY_KEYS = ("mse", "mean_px", "mae_x_px", "mae_y_px", "p95_px")


# Text describing the augmentation, used in the dropdown label
def _variant_label(aug):
    # "aug=aug" or "aug=none"
    return "aug=%s" % aug.tag()


# Short text of a metrics dict for the log
def _fmt(m):
    # MSE and mean / p95 pixel error
    return "mse %.5f, %.1f px (p95 %.1f px)" % (m["mse"], m["mean_px"], m["p95_px"])


# Write a checkpoint, optionally its ONNX file, and its .json
def save(model, path, meta, onnx, sanity_paths):
    # Reference frames + outputs for the robot-side check
    meta["sanity"] = make_sanity(model, sanity_paths)
    # Save the weights (state_dict)
    torch.save(model.state_dict(), path)
    # Fingerprint of the saved weights
    meta["weights_sha256"] = file_sha256(path)
    # Also export ONNX if requested
    if onnx:
        # Export next to the .pth and record the file name in the metadata
        meta["onnx"] = os.path.basename(export_onnx(model, os.path.splitext(path)[0] + ".onnx"))
    # Write the .json sidecar
    write_meta(path, meta)
    # Report the file
    print("saved %s" % path)


# Metrics on the held-out test split, evaluated once per finished model
def report_test(model, test_loader, device, tag):
    """Test metrics of the final (best-validation) weights, or None without a test split."""
    # Nothing to evaluate without a test/ folder
    if test_loader is None:
        # Stored as null in the .json
        return None
    # Metrics over the test images
    m = evaluate(model, test_loader, device)
    # Show them next to the training log
    print("[%s] test %s  (%d images, never used for training or model selection)"
          % (tag, _fmt(m), len(test_loader.dataset)))
    # Values for the .json
    return m


# Train (or load) the baseline and prune it, for one seed
def run_seed(args, arch, seed, name, loaders, common, aug):
    """Baseline + pruned models for one seed. Returns {tag: {"val": .., "test": ..}}."""
    # Unpack the loaders
    train_loader, val_loader, test_loader = loaders
    # Training device
    device = torch.device(args.device)
    # Seed the weight initialisation and the augmentation randomness
    torch.manual_seed(seed)
    # Seed the shuffling order (the split itself stays fixed by --split-seed)
    train_loader.generator.manual_seed(seed)
    # Validation images whose outputs are stored for the robot-side check
    sanity_paths = val_loader.dataset.image_paths
    # Metadata shared by this seed's models
    common = dict(common, seed=seed)
    # Results per model tag for the seed summary
    results = {}

    # ---- baseline --------------------------------------------------------- #
    # Prune an existing baseline
    if args.base:
        # Rebuild the baseline network from its weights
        model = rebuild_from_state_dict(torch.load(args.base, map_location="cpu")).to(device)
        # Its metadata
        base_meta = read_meta(args.base)
        # Its file
        base_path = args.base
        # Show its validation metrics on the current split
        print("loaded baseline %s (val %s)" % (args.base, _fmt(evaluate(model, val_loader, device))))
        # Also show its test metrics when there is a test split
        report_test(model, test_loader, device, "base")
    # Train a new baseline
    else:
        # Fresh network (ImageNet weights unless --no-pretrained)
        model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
        # The network's own training routine
        best, history = arch.train(model, train_loader, val_loader, device, args.epochs, args.lr, "base",
                                   loss=args.loss)
        # Test metrics of the chosen weights
        test = report_test(model, test_loader, device, "base")
        # Its size
        macs, params = count_macs_params(model)
        # Metadata of the baseline
        base_meta = dict(common, label="%s | %s | no compression" % (arch.LABEL, _variant_label(aug)),
                         compression=None, epochs=args.epochs, lr=args.lr,
                         best_val_loss=best["mse"], val_metrics=best,
                         test_loss=test["mse"] if test else None, test_metrics=test,
                         macs=macs, params=params, history=history)
        # File of the baseline
        base_path = os.path.join(args.out, name + "_base.pth")
        # Save it
        save(model, base_path, base_meta, args.onnx, sanity_paths)
        # Remember the result for the seed summary
        results["base"] = {"val": best, "test": test}

    # ---- DepGraph pruning + finetune -------------------------------------- #
    # One pruned model per ratio
    for ratio in args.prune:
        # Prune a copy of the baseline
        pruned, info = prune_model(model, ratio, round_to=args.round_to)
        # Make sure it is on the training device
        pruned = pruned.to(device)
        # Validation MSE right after pruning, before fine-tuning
        before = evaluate(pruned, val_loader, device)["mse"]
        # Report the size reduction
        print("pruned %.0f%%: MACs %.2fG -> %.2fG, params %.2fM -> %.2fM, val before finetune %.5f"
              % (ratio * 100, info["macs_before"] / 1e9, info["macs_after"] / 1e9,
                 info["params_before"] / 1e6, info["params_after"] / 1e6, before))
        # File suffix, e.g. "pruned30"
        tag = "pruned%02d" % round(ratio * 100)
        # Fine-tune with the network's own training routine
        best, history = arch.train(pruned, train_loader, val_loader, device,
                                   args.finetune_epochs, args.finetune_lr, tag, loss=args.loss)
        # Test metrics of the chosen weights
        test = report_test(pruned, test_loader, device, tag)
        # Metadata of the pruned model
        meta = dict(common, label="%s | %s | DepGraph %d%%" % (arch.LABEL, _variant_label(aug),
                                                              round(ratio * 100)),
                    compression=dict(info, method="DepGraph", importance="GroupMagnitude(p=2)"),
                    base=os.path.basename(base_path), base_val_loss=base_meta.get("best_val_loss"),
                    val_loss_before_finetune=before, epochs=args.finetune_epochs, lr=args.finetune_lr,
                    best_val_loss=best["mse"], val_metrics=best,
                    test_loss=test["mse"] if test else None, test_metrics=test,
                    macs=info["macs_after"], params=info["params_after"], history=history)
        # Save it
        save(pruned, os.path.join(args.out, "%s_%s.pth" % (name, tag)), meta, args.onnx, sanity_paths)
        # Remember the result for the seed summary
        results[tag] = {"val": best, "test": test}
    # Results of this seed
    return results


# Mean / std of every metric across seeds
def summarise(per_seed):
    """``{tag: {"val": {metric: {mean, std, values}}, "test": ...}}`` from per-seed results."""
    # Summary per model tag
    out = {}
    # Every model tag (base, pruned30, ...) of the first seed
    for tag in per_seed[0][1]:
        # Summary of this tag
        out[tag] = {}
        # Validation and test separately
        for split in ("val", "test"):
            # Metrics of every seed that has them
            runs = [r[tag][split] for _, r in per_seed if r.get(tag, {}).get(split)]
            # Skip a split that was not evaluated (no test folder)
            if not runs:
                # Nothing to summarise
                continue
            # Mean, sample standard deviation and raw values per metric
            out[tag][split] = {k: {"mean": float(np.mean([m[k] for m in runs])),
                                   "std": float(np.std([m[k] for m in runs], ddof=1)) if len(runs) > 1 else 0.0,
                                   "values": [m[k] for m in runs]} for k in SUMMARY_KEYS}
    # Summary of all tags
    return out


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
    # The only augmentation switch
    ap.add_argument("--augment", action="store_true",
                    help="subsets of at most 2 of flip / rotate / crop / color / noise / blur / compress (x29 per epoch)")
    # Baseline training epochs
    ap.add_argument("--epochs", type=int, default=None, help="default: the network's DEFAULTS")
    # Baseline learning rate (head; the trained backbone parts get it x backbone_lr_scale)
    ap.add_argument("--lr", type=float, default=None, help="head lr, default: the network's DEFAULTS")
    # Coordinate loss
    ap.add_argument("--loss", choices=LOSSES, default="mse", help="coordinate loss (huber: SmoothL1, beta 0.1)")
    # Batch size
    ap.add_argument("--batch-size", type=int, default=16)
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
    # Training seeds: one full run (baseline + pruning) per seed
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="training seeds, e.g. 0 1 2: one run each + mean/std summary")
    # Seed of the random / session split (kept fixed across --seeds)
    ap.add_argument("--split-seed", type=int, default=0, help="seed of the train/val split")
    # Existing baseline to prune instead of training a new one
    ap.add_argument("--base", default=None, help="existing baseline .pth: skip baseline training")
    # Pruning ratios (one pruned model per ratio)
    ap.add_argument("--prune", type=float, nargs="*", default=[], help="pruning ratios, e.g. 0.3 0.5")
    # Channel counts after pruning are rounded to a multiple of this
    ap.add_argument("--round-to", type=int, default=8)
    # Fine-tuning epochs after pruning
    ap.add_argument("--finetune-epochs", type=int, default=None, help="default: the network's DEFAULTS")
    # Fine-tuning learning rate
    ap.add_argument("--finetune-lr", type=float, default=None, help="default: the network's DEFAULTS")
    # Also export ONNX files
    ap.add_argument("--onnx", action="store_true", help="also export ONNX (opset 11)")
    # Testing only: random initialisation instead of ImageNet weights
    ap.add_argument("--no-pretrained", action="store_true", help="(testing) skip ImageNet weights")
    # Training device
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Parse the command line
    args = ap.parse_args()

    # One existing baseline cannot stand for several seeds
    if args.base and len(args.seeds) > 1:
        # Explain the conflict
        ap.error("--base uses one existing baseline; give a single --seeds value")
    # Create the output folder if needed
    os.makedirs(args.out, exist_ok=True)

    # Settings inherited from an existing baseline
    if args.base:
        # The network type comes from the baseline's weights
        args.arch = detect_arch(torch.load(args.base, map_location="cpu"))
        # Reuse the baseline's augmentation switch and run name
        # unless overridden, so pruned files stay grouped with their baseline.
        base_aug = read_meta(args.base).get("augmentation", {})
        # Only when --augment was not given on the command line
        if not args.augment:
            # Baseline's switch (older files: any of the former general / perlin / compression switches)
            args.augment = bool(base_aug.get("enabled", any(base_aug.get(k) for k in
                                                            ("general", "perlin", "compression"))))
        # Baseline file name without extension
        stem = os.path.splitext(os.path.basename(args.base))[0]
        # Derive the run name from "<name>_base"
        if args.name is None and stem.endswith("_base"):
            # Strip the "_base" suffix
            args.name = stem[:-len("_base")]

    # Module of the chosen network (training routine, defaults, names)
    arch = ARCHS[args.arch]
    # Fill every training setting that was not given from the network's defaults
    for key in ("epochs", "lr", "finetune_epochs", "finetune_lr"):
        # Only settings left at None
        if getattr(args, key) is None:
            # Take the network's default
            setattr(args, key, arch.DEFAULTS[key])

    # The team's sub-folders decide the split; --split / --val-ratio do not apply
    if has_split_folders(args.data) and (args.split != "random" or args.val_ratio != 0.1):
        # Tell the user their split options are ignored
        print("note: %s has train/validation sub-folders: --split / --val-ratio are ignored" % args.data)
    # Training / validation / test loaders (lane marking happens inside)
    train_loader, val_loader, test_loader, aug = make_loaders(
        args.data, augment=args.augment,
        val_ratio=args.val_ratio, batch_size=args.batch_size, num_workers=args.workers, seed=args.split_seed,
        split=args.split, metadata_dir=args.metadata, segment_gap=args.segment_gap)
    # Run name, e.g. "mobilenetv2_aug"
    name = args.name or "%s_%s" % (arch.FILE_TAG, aug.tag())
    # Metadata shared by every model of this run
    common = {
        "arch": args.arch,
        "preprocessing": PREPROCESSING,
        "training": dict(RECIPE, loss=args.loss, **{k: arch.DEFAULTS[k] for k in
                                                    ("backbone_lr_scale", "weight_decay", "patience", "min_epochs")}),
        "augmentation": {"enabled": aug.enabled, "method": "powerset", "methods": list(METHODS), "max_methods": MAX_METHODS,
                         "variants_per_image": aug.n_variants, "tag": aug.tag()},
        "data": os.path.abspath(args.data),
        "n_train": len(train_loader.dataset.image_paths), "n_train_samples": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
        "n_test": len(test_loader.dataset) if test_loader is not None else 0,
        "val_ratio": args.val_ratio, "split_seed": args.split_seed, "batch_size": args.batch_size,
        "split": train_loader.split_info,
    }
    # Summary of the setup
    print("data: %d train (%d samples per epoch) / %d val / %d test, %s, %r, lane marking on, device %s"
          % (common["n_train"], common["n_train_samples"], common["n_val"], common["n_test"], arch.LABEL, aug,
             args.device))
    # How the data was split
    print("split: %s" % train_loader.split_info)
    # Training settings in use
    print("training: %d epochs lr %g, finetune %d epochs lr %g, loss %s, seeds %s"
          % (args.epochs, args.lr, args.finetune_epochs, args.finetune_lr, args.loss, args.seeds))
    # (seed, results) of every run
    per_seed = []
    # One full run per seed
    for seed in args.seeds:
        # Several seeds: separate file names per seed
        run_name = name if len(args.seeds) == 1 else "%s_s%d" % (name, seed)
        # Mark the start of the seed in the log
        if len(args.seeds) > 1:
            # Header line
            print("==== seed %d (%s) ====" % (seed, run_name))
        # Baseline + pruning for this seed
        results = run_seed(args, arch, seed, run_name, (train_loader, val_loader, test_loader),
                           common, aug)
        # Keep the results
        per_seed.append((seed, results))

    # Mean / std across seeds when there are several
    if len(args.seeds) > 1:
        # Summary per model
        summary = summarise(per_seed)
        # Summary file next to the models
        path = os.path.join(args.out, name + "_seeds.json")
        # Write it with the run settings
        with open(path, "w", encoding="utf-8") as f:
            # Seeds, shared metadata and the statistics
            json.dump({"name": name, "seeds": args.seeds, "summary": summary,
                       "settings": {k: v for k, v in common.items()}}, f, indent=2, ensure_ascii=False)
        # Readable table: model, split, mean +- std of MSE and pixel error
        for tag, splits in summary.items():
            # Validation and test rows
            for split, stats in splits.items():
                # One row
                print("%-10s %-4s mse %.5f +- %.5f | mean %.1f +- %.1f px | p95 %.1f +- %.1f px"
                      % (tag, split, stats["mse"]["mean"], stats["mse"]["std"],
                         stats["mean_px"]["mean"], stats["mean_px"]["std"],
                         stats["p95_px"]["mean"], stats["p95_px"]["std"]))
        # Where the summary went
        print("seed summary -> %s" % path)

    # Machine-readable summary of where the files went
    print(json.dumps({"out": args.out, "name": name}, indent=None))


# Run the pipeline when executed as a script
if __name__ == "__main__":
    # Start training
    main()
