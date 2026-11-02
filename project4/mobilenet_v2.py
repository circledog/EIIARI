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

NAME = "mobilenet_v2"
FILE_TAG = "mobilenetv2"
LABEL = "MobileNetV2"
HEAD_KEY = "classifier.1.weight"


def _torchvision(pretrained):
    """torchvision mobilenet_v2 for both the old (pretrained=) and new (weights=) API."""
    try:
        weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        return torchvision.models.mobilenet_v2(weights=weights)
    except (AttributeError, TypeError):
        return torchvision.models.mobilenet_v2(pretrained=pretrained)


def build(pretrained=True, num_outputs=2):
    model = _torchvision(pretrained)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_outputs)
    return model
