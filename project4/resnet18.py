"""ResNet-18 (slides P.39 / P.45): ImageNet-pretrained, ``fc = Linear(512, 2)``.

Shared loading / rebuilding after pruning lives in ``loader.py``.
Python 3.6 / old-torchvision compatible (runs on the Jetson Nano).
"""
import torch.nn as nn
import torchvision

NAME = "resnet18"          # --arch value, meta["arch"]
FILE_TAG = "resnet18"      # prefix of the checkpoint file names
LABEL = "ResNet18"         # shown in the live demo dropdown
HEAD_KEY = "fc.weight"     # state_dict key that identifies this network


def _torchvision(pretrained):
    """torchvision resnet18 for both the old (pretrained=) and new (weights=) API."""
    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return torchvision.models.resnet18(weights=weights)
    except (AttributeError, TypeError):
        # torchvision < 0.13 (Nano): no ResNet18_Weights (AttributeError) when
        # pretrained=True, or resnet18() rejects weights=None (TypeError) when False.
        return torchvision.models.resnet18(pretrained=pretrained)


def build(pretrained=True, num_outputs=2):
    model = _torchvision(pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_outputs)
    return model
