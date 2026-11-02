"""MobileNetV2: ImageNet-pretrained, ``classifier[1] = Linear(1280, 2)``.

The "other model" for the comparison: about 1/6 of ResNet-18's MACs (0.3 G vs
1.8 G) and 1/5 of its parameters, built for mobile / embedded GPUs like the
Nano. Depthwise convolutions are handled by DepGraph pruning and by
``loader.rebuild_from_state_dict``; ReLU6 exports to ONNX opset 11 as Clip.

Shared loading / rebuilding after pruning lives in ``loader.py``.
Python 3.6 / old-torchvision compatible (runs on the Jetson Nano).
"""
import torch.nn as nn
import torchvision

# Value of --arch and of meta["arch"] for this network
NAME = "mobilenet_v2"
# Prefix of the checkpoint file names (mobilenetv2_<aug>_base.pth, ...)
FILE_TAG = "mobilenetv2"
# Name shown in the live demo dropdown
LABEL = "MobileNetV2"
# state_dict key that only this network has; used to detect the architecture
HEAD_KEY = "classifier.1.weight"


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
