"""MobileNetV2: ImageNet-pretrained, ``classifier[1] = Linear(1280, 2)``.

The "other model" for the comparison: about 1/6 of ResNet-18's MACs (0.3 G vs
1.8 G) and 1/5 of its parameters, built for mobile / embedded GPUs like the
Nano. Depthwise convolutions are handled by DepGraph pruning and by
``loader.rebuild_from_state_dict``; ReLU6 exports to ONNX opset 11 as Clip.

``train()`` holds this network's own training routine (``train.py`` calls
it); shared loading / rebuilding after pruning lives in ``loader.py``.
Python 3.6 / old-torchvision compatible (runs on the Jetson Nano).
"""
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Validation metrics (MSE and pixel errors)
from metrics import evaluate

# Value of --arch and of meta["arch"] for this network
NAME = "mobilenet_v2"
# Prefix of the checkpoint file names (mobilenetv2_<aug>_base.pth, ...)
FILE_TAG = "mobilenetv2"
# Name shown in the live demo dropdown
LABEL = "MobileNetV2"
# state_dict keys this network has; used by loader.detect_arch
KEYS = ("classifier.1.weight",)


# Create the stock torchvision MobileNetV2, with or without ImageNet weights
def _torchvision(pretrained):
    """torchvision mobilenet_v2 for both the old (pretrained=) and new (weights=) API."""
    # New torchvision API (>= 0.13) takes a weights enum
    try:
        # Pick the ImageNet weights, or None for random initialisation
        weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        # Build the network with the new API
        return torchvision.models.mobilenet_v2(weights=weights)
    # Old torchvision (Nano) has no weights enum / rejects weights=None
    except (AttributeError, TypeError):
        # Fall back to the old pretrained= flag
        return torchvision.models.mobilenet_v2(pretrained=pretrained)


# Network used for training: MobileNetV2 with a 2-output regression head
def build(pretrained=True, num_outputs=2):
    # Start from the stock MobileNetV2
    model = _torchvision(pretrained)
    # Replace the 1000-class Linear (after Dropout) with one predicting (x, y)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_outputs)
    # Hand back the ready-to-train network
    return model


# Training settings for MobileNetV2; train.py uses them unless given on the command line
DEFAULTS = {"epochs": 10, "lr": 1e-3, "finetune_epochs": 15, "finetune_lr": 1e-4}


# Train MobileNetV2 and keep its best weights
def train(model, train_loader, val_loader, device, epochs, lr, tag="mobilenet_v2"):
    """MobileNetV2 training: Adam + MSE, keep the epoch with the lowest validation MSE.

    Used for the baseline and for fine-tuning after pruning (train.py passes
    the matching epochs / lr). Change the optimiser, loss or schedule here to
    customise MobileNetV2 without affecting the other networks.
    Returns (best validation metrics, per-epoch history).
    """
    # Adam optimiser, as in the course
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Best validation metrics so far, their weights, and the per-epoch log
    best, best_state, history = None, None, []
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
        # Validation metrics after the epoch (MSE and pixel errors)
        val = evaluate(model, val_loader, device)
        # Log the losses and the mean pixel error for the report
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["mse"],
                        "val_mean_px": val["mean_px"]})
        # Marker printed next to a new best epoch
        mark = ""
        # New best validation MSE
        if best is None or val["mse"] < best["mse"]:
            # Remember the metrics and a copy of the weights
            best, best_state, mark = val, copy.deepcopy(model.state_dict()), " *"
        # Progress line
        print("[%s] epoch %3d/%d  train %.5f  val %.5f (%.1f px)  (%.1fs)%s"
              % (tag, epoch, epochs, train_loss, val["mse"], val["mean_px"], time.time() - t0, mark))
    # Restore the best weights (if any epoch ran)
    if best_state is not None:
        # Load them back into the model
        model.load_state_dict(best_state)
    # Without epochs, report the current weights instead
    else:
        # Metrics of the untrained / loaded weights
        best = evaluate(model, val_loader, device)
    # Best validation metrics and the full log
    return best, history
