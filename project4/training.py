"""Shared training loop (the teammate's recipe, flow 1 of model_training_methods).

Small-data fine-tuning of an ImageNet-pretrained network:

* partial freezing: only the parameters whose names start with one of the
  network's ``TRAINABLE`` prefixes are trained (ResNet-18: conv1, layer3,
  layer4, fc); the rest keeps its ImageNet values
* BatchNorm frozen: every BatchNorm layer stays in eval mode, so its running
  mean / variance are not changed by the small batches
* AdamW with two learning rates: ``lr`` for the head (``HEAD`` prefixes),
  ``lr * backbone_lr_scale`` for the trained backbone parts; weight decay
* cosine learning-rate schedule from ``lr`` down to 1e-6 over ``epochs``
* gradient clipping (norm 5)
* loss: MSE (default) or Huber / SmoothL1 with beta 0.1 on (x, y)
* model selection on the validation *mean pixel distance*, early stop when
  it has not improved for ``patience`` epochs (after at least ``min_epochs``)

Parameters that are already frozen (``requires_grad=False``, e.g. DSNT's
fixed upsampling / expectation layers) are never unfrozen.
The teammate's experiments.json was not in the zip, so the values in each
network's ``DEFAULTS`` (backbone lr x0.1, weight decay 1e-4, patience 5,
min 10 epochs) are common choices, not the teammate's.
Python 3.6 compatible.
"""
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Validation metrics (MSE and pixel errors)
from metrics import evaluate

# Public names of this module
__all__ = ["fit", "coordinate_loss", "LOSSES", "RECIPE"]

# Coordinate losses that can be chosen with --loss
LOSSES = ("mse", "huber")
# Lowest learning rate at the end of the cosine schedule
MIN_LR = 1e-6
# Largest gradient norm
CLIP_NORM = 5.0
# Huber / SmoothL1 transition point (normalised units, 0.1 = 11 px)
HUBER_BETA = 0.1
# Description saved in every model's .json
RECIPE = {"name": "teammate_flow1", "optimizer": "AdamW", "schedule": "cosine", "min_lr": MIN_LR,
          "clip_norm": CLIP_NORM, "batchnorm": "frozen", "selection": "validation mean_px",
          "huber_beta": HUBER_BETA}


# Loss between predicted and labelled (x, y)
def coordinate_loss(name):
    """``loss(pred, labels)`` for ``name`` in LOSSES."""
    # Mean squared error
    if name == "mse":
        # Plain MSE
        return F.mse_loss
    # Huber: quadratic below beta, linear above (less pulled by outlier labels)
    if name == "huber":
        # SmoothL1 with the teammate's beta
        return lambda pred, labels: F.smooth_l1_loss(pred, labels, beta=HUBER_BETA)
    # Anything else is a typo
    raise ValueError("unknown loss %r (choose from %s)" % (name, ", ".join(LOSSES)))


# Put every BatchNorm layer into eval mode (running statistics frozen)
def _freeze_batchnorm(model):
    # Every sub-module
    for module in model.modules():
        # BatchNorm of any dimension
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            # Use and keep the stored statistics
            module.eval()


# Train with the teammate's recipe and keep the best weights
def fit(model, train_loader, val_loader, device, epochs, lr, tag, trainable, head,
        backbone_lr_scale=0.1, weight_decay=1e-4, patience=5, min_epochs=10, loss="mse", loss_fn=None):
    """Returns (best validation metrics, per-epoch history); the model ends with the best weights.

    ``trainable`` / ``head``: tuples of parameter-name prefixes. ``loss``:
    one of LOSSES. ``loss_fn(model, images, labels)`` replaces the plain
    coordinate loss (DSNT adds its heatmap term).
    """
    # Coordinate loss unless the network brings its own
    criterion = loss_fn or (lambda m, images, labels, f=coordinate_loss(loss): f(m(images), labels))
    # Original trainable flags, restored at the end
    original = [(p, p.requires_grad) for p in model.parameters()]
    # Head parameters (full learning rate) and trained backbone parameters (scaled)
    head_params, backbone_params = [], []
    # Decide for every parameter
    for name, p in model.named_parameters():
        # Already-frozen parameters stay frozen; others only if their prefix is listed
        p.requires_grad_(p.requires_grad and name.startswith(tuple(trainable)))
        # Skip the frozen ones
        if not p.requires_grad:
            # Not handed to the optimiser
            continue
        # Head or backbone group
        (head_params if name.startswith(tuple(head)) else backbone_params).append(p)
    # AdamW with one learning rate per group (empty groups are left out)
    groups = [g for g in ({"params": backbone_params, "lr": lr * backbone_lr_scale},
                          {"params": head_params, "lr": lr}) if g["params"]]
    # Optimiser with weight decay
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    # Cosine decay of every group's learning rate over the planned epochs
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=MIN_LR)
    # Number of trained parameters, for the log
    n_trained = sum(p.numel() for g in groups for p in g["params"])
    # Show what is trained
    print("[%s] training %.2fM of %.2fM parameters (head lr %g, backbone lr %g), loss %s"
          % (tag, n_trained / 1e6, sum(p.numel() for p in model.parameters()) / 1e6, lr,
             lr * backbone_lr_scale, "custom" if loss_fn else loss))
    # Best validation metrics so far, their weights, their epoch, and the per-epoch log
    best, best_state, best_epoch, history = None, None, 0, []
    # One iteration per epoch
    for epoch in range(1, epochs + 1):
        # Start time of the epoch
        t0 = time.time()
        # Training mode (dropout on) ...
        model.train()
        # ... but BatchNorm statistics frozen
        _freeze_batchnorm(model)
        # Summed training loss and number of images
        total, n = 0.0, 0
        # Every training batch
        for images, labels in train_loader:
            # Move the batch to the training device
            images, labels = images.to(device), labels.to(device)
            # Clear the gradients of the previous step
            optimizer.zero_grad()
            # Loss of the batch
            batch_loss = criterion(model, images, labels)
            # A NaN / inf loss would silently ruin the weights
            if not torch.isfinite(batch_loss):
                # Stop with a clear message
                raise RuntimeError("[%s] non-finite training loss at epoch %d" % (tag, epoch))
            # Back-propagate
            batch_loss.backward()
            # Limit the gradient norm
            nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], CLIP_NORM)
            # Update the weights
            optimizer.step()
            # Accumulate the loss weighted by batch size
            total += batch_loss.item() * images.size(0)
            # Count the images
            n += images.size(0)
        # Head learning rate used in this epoch
        epoch_lr = optimizer.param_groups[-1]["lr"]
        # Next epoch's learning rates
        schedule.step()
        # Mean training loss of the epoch
        train_loss = total / max(n, 1)
        # Validation metrics after the epoch (MSE and pixel errors)
        val = evaluate(model, val_loader, device)
        # Log the losses, the mean pixel error and the learning rate
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["mse"],
                        "val_mean_px": val["mean_px"], "lr": epoch_lr})
        # Marker printed next to a new best epoch
        mark = ""
        # New best validation mean pixel distance
        if best is None or val["mean_px"] < best["mean_px"]:
            # Remember the metrics, a copy of the weights and the epoch
            best, best_state, best_epoch, mark = val, copy.deepcopy(model.state_dict()), epoch, " *"
        # Progress line
        print("[%s] epoch %3d/%d  train %.5f  val %.5f (%.1f px)  lr %.2g  (%.1fs)%s"
              % (tag, epoch, epochs, train_loss, val["mse"], val["mean_px"], epoch_lr, time.time() - t0, mark))
        # Early stop: no improvement for `patience` epochs, after the minimum
        if epoch >= min_epochs and epoch - best_epoch >= patience:
            # Say why training ends early
            print("[%s] early stop: no better validation px for %d epochs (best epoch %d)"
                  % (tag, patience, best_epoch))
            # Leave the epoch loop
            break
    # Restore the best weights (if any epoch ran)
    if best_state is not None:
        # Load them back into the model
        model.load_state_dict(best_state)
    # Without epochs, report the current weights instead
    else:
        # Metrics of the untrained / loaded weights
        best = evaluate(model, val_loader, device)
    # Original flags back (pruning and later fine-tuning decide for themselves)
    for p, flag in original:
        # Fixed-by-design layers stay frozen, everything else trainable again
        p.requires_grad_(flag)
    # Best epoch, recorded with the metrics
    best = dict(best, best_epoch=best_epoch)
    # Best validation metrics and the full log
    return best, history
