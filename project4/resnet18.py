"""ResNet-18 (slides P.39 / P.45): ImageNet-pretrained, ``fc = Linear(512, 2)``.

``train()`` holds this network's training settings (the loop is ``training.fit``; ``train.py`` calls
it); shared loading / rebuilding after pruning lives in ``loader.py``.
Python 3.6 / old-torchvision compatible (runs on the Jetson Nano).
"""
import torch
import torch.nn as nn
import torchvision

# Shared training loop (teammate's recipe)
import training

# Value of --arch and of meta["arch"] for this network
NAME = "resnet18"
# Prefix of the checkpoint file names (resnet18_<aug>_base.pth, ...)
FILE_TAG = "resnet18"
# Name shown in the live demo dropdown
LABEL = "ResNet18"
# state_dict keys this network has; loader.detect_arch picks the network whose keys
# are all present, preferring the one with the most keys (the variants add keys)
KEYS = ("fc.weight",)


# Create the stock torchvision ResNet-18, with or without ImageNet weights
def _torchvision(pretrained):
    """torchvision resnet18 for both the old (pretrained=) and new (weights=) API."""
    # New torchvision API (>= 0.13) takes a weights enum
    try:
        # Pick the ImageNet weights, or None for random initialisation
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        # Build the network with the new API
        return torchvision.models.resnet18(weights=weights)
    # torchvision < 0.13 (Nano): no ResNet18_Weights (AttributeError) when
    # pretrained=True, or resnet18() rejects weights=None (TypeError) when False.
    except (AttributeError, TypeError):
        # Fall back to the old pretrained= flag
        return torchvision.models.resnet18(pretrained=pretrained)


# Network used for training: ResNet-18 with a 2-output regression head
def build(pretrained=True, num_outputs=2):
    # Start from the stock ResNet-18
    model = _torchvision(pretrained)
    # Replace the 1000-class classifier with a Linear layer predicting (x, y)
    model.fc = nn.Linear(model.fc.in_features, num_outputs)
    # Hand back the ready-to-train network
    return model


# Training settings for ResNet-18; train.py uses them unless given on the command line
# (lr = head lr; the trained backbone parts get lr * backbone_lr_scale)
DEFAULTS = {"epochs": 10, "lr": 1e-3, "finetune_epochs": 15, "finetune_lr": 1e-4,
            "backbone_lr_scale": 0.1, "weight_decay": 1e-4, "patience": 5, "min_epochs": 10}
# Parameters that are trained (teammate's partial freezing); layer1 / layer2 / bn1 keep ImageNet values
TRAINABLE = ("conv1", "layer3", "layer4", "fc")
# Parameters that get the full learning rate
HEAD = ("fc",)


# Train ResNet-18 and keep its best weights
def train(model, train_loader, val_loader, device, epochs, lr, tag="resnet18", loss="mse", loss_fn=None,
          trainable=TRAINABLE, head=HEAD):
    """ResNet-18 training with the teammate's recipe (see ``training.py``).

    Used for the baseline and for fine-tuning after pruning (train.py passes
    the matching epochs / lr). ``loss``: "mse" or "huber".
    ``loss_fn(model, images, labels) -> loss`` replaces the coordinate loss
    (DSNT's heatmap loss); ``trainable`` / ``head`` are parameter-name
    prefixes (the variants pass their own). Model selection: validation
    mean pixel distance, with early stopping.
    Returns (best validation metrics, per-epoch history).
    """
    # Shared loop with this network's settings
    return training.fit(model, train_loader, val_loader, device, epochs, lr, tag, trainable, head,
                        backbone_lr_scale=DEFAULTS["backbone_lr_scale"], weight_decay=DEFAULTS["weight_decay"],
                        patience=DEFAULTS["patience"], min_epochs=DEFAULTS["min_epochs"], loss=loss,
                        loss_fn=loss_fn)
