"""ResNet-18 (slides P.39 / P.45): ImageNet-pretrained, ``fc = Linear(512, 2)``.

Shared loading / rebuilding after pruning lives in ``loader.py``.
Python 3.6 / old-torchvision compatible (runs on the Jetson Nano).
"""
import torch.nn as nn
import torchvision

# Value of --arch and of meta["arch"] for this network
NAME = "resnet18"
# Prefix of the checkpoint file names (resnet18_<aug>_base.pth, ...)
FILE_TAG = "resnet18"
# Name shown in the live demo dropdown
LABEL = "ResNet18"
# state_dict key that only this network has; used to detect the architecture
HEAD_KEY = "fc.weight"


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
