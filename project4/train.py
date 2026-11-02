"""PC training entry point: baseline -> DepGraph pruning -> finetune -> export.

Examples
--------
    # baseline only, no augmentation (course-equivalent, minus the hflip bug)
    python train.py

    # general + Perlin augmentation, then prune 30 % and 50 %, export ONNX
    python train.py --general --perlin --prune 0.3 0.5 --onnx

    # preprocessing switches: crop the top 30 %, fix label y at 0.65 (defaults
    # when no value is given), drop bad labels
    python train.py --roi-top --fixed-y --exclude bad_labels.txt --general
    python train.py --roi-top 0.4 --general          # custom crop fraction

    # --no-preprocess forces every preprocessing step off (also overrides --base)

    # prune an existing baseline without retraining it
    # (augmentation switches and run name are taken from its .json)
    python train.py --base models/resnet18_none_base.pth --prune 0.3 --onnx

Outputs in ``--out`` (default ``models/``), per run name ``resnet18_<aug>``
(``resnet18_<pre>_<aug>`` when preprocessing is on):
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

from compression import count_macs_params, export_onnx, prune_model
from data_loader import DEFAULT_DATA_DIR, make_loaders
from model import build_model, read_meta, rebuild_resnet18_from_state_dict, write_meta
from preprocessing import FIXED_Y_DEFAULT, ROI_TOP_DEFAULT, Preprocessor


def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            total += F.mse_loss(model(images), labels, reduction="sum").item() / 2.0
            n += images.size(0)
    return total / max(n, 1)


def fit(model, train_loader, val_loader, device, epochs, lr, tag):
    """Adam + MSE, keep the weights with the lowest validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_state, history = float("inf"), None, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(images), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * images.size(0)
            n += images.size(0)
        train_loss = total / max(n, 1)
        val_loss = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        mark = ""
        if val_loss < best_loss:
            best_loss, best_state, mark = val_loss, copy.deepcopy(model.state_dict()), " *"
        print("[%s] epoch %3d/%d  train %.5f  val %.5f  (%.1fs)%s"
              % (tag, epoch, epochs, train_loss, val_loss, time.time() - t0, mark))
    model.load_state_dict(best_state)
    return best_loss, history


def _variant_label(prep, aug):
    return ("pre=%s | aug=%s" % (prep.tag(), aug.tag())) if prep.tag() else "aug=%s" % aug.tag()


def save(model, path, meta, onnx):
    torch.save(model.state_dict(), path)
    if onnx:
        meta["onnx"] = os.path.basename(export_onnx(model, os.path.splitext(path)[0] + ".onnx"))
    write_meta(path, meta)
    print("saved %s" % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA_DIR, help="folder with xy_*.jpg")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
    ap.add_argument("--name", default=None, help="run name (default resnet18_<aug>)")
    pg = ap.add_argument_group("preprocessing switches (all off = none; also applied on the robot)")
    pg.add_argument("--roi-top", type=float, nargs="?", const=ROI_TOP_DEFAULT, default=0.0, metavar="FRAC",
                    help="crop the top of the image (no value = %.1f)" % ROI_TOP_DEFAULT)
    pg.add_argument("--fixed-y", type=float, nargs="?", const=FIXED_Y_DEFAULT, default=None, metavar="FRAC",
                    help="fix label y at this height fraction (no value = %.2f)" % FIXED_Y_DEFAULT)
    pg.add_argument("--exclude", default=None, help="file with image names to drop (bad labels)")
    pg.add_argument("--no-preprocess", action="store_true",
                    help="turn every preprocessing step off (overrides the switches above and --base)")
    g = ap.add_argument_group("augmentation switches (all off = none)")
    g.add_argument("--general", action="store_true", help="flip / rotate / crop / colour jitter")
    g.add_argument("--perlin", action="store_true", help="Perlin-noise lighting")
    g.add_argument("--compression", action="store_true", help="JPEG / 144p artefacts")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--split", choices=["random", "session"], default="random",
                    help="session: keep shooting segments together (needs the metadata folder)")
    ap.add_argument("--metadata", default=None, help="metadata folder (default: next to --data)")
    ap.add_argument("--segment-gap", type=float, default=60.0,
                    help="seconds between captures that start a new segment (--split session)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default=None, help="existing baseline .pth: skip baseline training")
    ap.add_argument("--prune", type=float, nargs="*", default=[], help="pruning ratios, e.g. 0.3 0.5")
    ap.add_argument("--round-to", type=int, default=8)
    ap.add_argument("--finetune-epochs", type=int, default=15)
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--onnx", action="store_true", help="also export ONNX (opset 11)")
    ap.add_argument("--no-pretrained", action="store_true", help="(testing) skip ImageNet weights")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    if args.base:
        # Reuse the baseline's preprocessing, augmentation switches and run name
        # unless overridden, so pruned files stay grouped with their baseline.
        base_pre = read_meta(args.base).get("preprocessing") or {}
        if (not args.no_preprocess and not args.roi_top and args.fixed_y is None
                and args.exclude is None):
            args.roi_top = base_pre.get("roi_top") or 0.0
            args.fixed_y = base_pre.get("fixed_y")
            ex = base_pre.get("exclude_path")
            if ex and os.path.exists(ex):
                args.exclude = ex
            elif base_pre.get("exclude"):
                print("warning: baseline used exclude file %r, not found; pass --exclude"
                      % base_pre["exclude"])
        base_aug = read_meta(args.base).get("augmentation", {})
        if not (args.general or args.perlin or args.compression):
            args.general = bool(base_aug.get("general"))
            args.perlin = bool(base_aug.get("perlin"))
            args.compression = bool(base_aug.get("compression"))
        stem = os.path.splitext(os.path.basename(args.base))[0]
        if args.name is None and stem.endswith("_base"):
            args.name = stem[:-len("_base")]

    if args.no_preprocess:
        if args.base and (read_meta(args.base).get("preprocessing") or {}).get("roi_top"):
            print("warning: the baseline was trained with ROI cropping; --no-preprocess feeds it full frames")
        args.roi_top, args.fixed_y, args.exclude = 0.0, None, None
    prep = Preprocessor(roi_top=args.roi_top, fixed_y=args.fixed_y, exclude=args.exclude)
    train_loader, val_loader, aug = make_loaders(
        args.data, general=args.general, perlin=args.perlin, compression=args.compression,
        val_ratio=args.val_ratio, batch_size=args.batch_size, num_workers=args.workers, seed=args.seed,
        preprocessor=prep, split=args.split, metadata_dir=args.metadata, segment_gap=args.segment_gap)
    variant = "%s_%s" % (prep.tag(), aug.tag()) if prep.tag() else aug.tag()
    name = args.name or "resnet18_%s" % variant
    common = {
        "arch": "resnet18",
        "preprocessing": prep.config(),
        "augmentation": {"general": aug.general, "perlin": aug.perlin,
                         "compression": aug.compression, "tag": aug.tag()},
        "data": os.path.abspath(args.data),
        "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset),
        "val_ratio": args.val_ratio, "seed": args.seed, "batch_size": args.batch_size,
        "split": train_loader.split_info,
    }
    print("data: %d train / %d val, %r, %r, device %s"
          % (common["n_train"], common["n_val"], prep, aug, device))
    print("split: %s" % train_loader.split_info)

    # ---- baseline --------------------------------------------------------- #
    if args.base:
        model = rebuild_resnet18_from_state_dict(torch.load(args.base, map_location="cpu")).to(device)
        base_meta = read_meta(args.base)
        base_path = args.base
        print("loaded baseline %s (val %.5f)" % (args.base, evaluate(model, val_loader, device)))
    else:
        model = build_model(pretrained=not args.no_pretrained).to(device)
        best, history = fit(model, train_loader, val_loader, device, args.epochs, args.lr, "base")
        macs, params = count_macs_params(model)
        base_meta = dict(common, label="ResNet18 | %s | no compression" % _variant_label(prep, aug),
                         compression=None, epochs=args.epochs, lr=args.lr,
                         best_val_loss=best, macs=macs, params=params, history=history)
        base_path = os.path.join(args.out, name + "_base.pth")
        save(model, base_path, base_meta, args.onnx)

    # ---- DepGraph pruning + finetune -------------------------------------- #
    for ratio in args.prune:
        pruned, info = prune_model(model, ratio, round_to=args.round_to)
        pruned = pruned.to(device)
        before = evaluate(pruned, val_loader, device)
        print("pruned %.0f%%: MACs %.2fG -> %.2fG, params %.2fM -> %.2fM, val before finetune %.5f"
              % (ratio * 100, info["macs_before"] / 1e9, info["macs_after"] / 1e9,
                 info["params_before"] / 1e6, info["params_after"] / 1e6, before))
        tag = "pruned%02d" % round(ratio * 100)
        best, history = fit(pruned, train_loader, val_loader, device,
                            args.finetune_epochs, args.finetune_lr, tag)
        meta = dict(common, label="ResNet18 | %s | DepGraph %d%%" % (_variant_label(prep, aug), round(ratio * 100)),
                    compression=dict(info, method="DepGraph", importance="GroupMagnitude(p=2)"),
                    base=os.path.basename(base_path), base_val_loss=base_meta.get("best_val_loss"),
                    val_loss_before_finetune=before, epochs=args.finetune_epochs, lr=args.finetune_lr,
                    best_val_loss=best, macs=info["macs_after"], params=info["params_after"],
                    history=history)
        save(pruned, os.path.join(args.out, "%s_%s.pth" % (name, tag)), meta, args.onnx)

    print(json.dumps({"out": args.out, "name": name}, indent=None))


if __name__ == "__main__":
    main()
