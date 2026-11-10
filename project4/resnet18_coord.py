"""ResNet-18 with coordinate channels (CoordConv), as in the teammate's ``course_coordconv``.

Two fixed maps -- x (left -1 .. right +1) and y (top -1 .. bottom +1) -- are
appended to the 3 normalised colour channels *inside* the network, so the
camera, preprocessing, sanity check and TensorRT conversion all still feed
a (1, 3, 224, 224) image. ``conv1`` takes 5 channels; its weights for the two
extra channels start at zero, so the ImageNet-pretrained network starts out
exactly like the plain ResNet-18 and learns how much to use the positions.

The maps are a buffer (``coords``, shape (2, H, W)), saved in the
state_dict, so PC and robot use identical values. At batch 1 (inference,
ONNX export, torch2trt) the buffer gets its batch dimension with ``view`` and
is concatenated; only training batches ``expand`` it. Checked offline with
``check_trt_offline.py`` on torch2trt 2019-10 .. 2022-07: ``view`` converts on
all of them, while concatenating a stored (1, 2, H, W) constant fails on
torch2trt before 2020-09 (constants keep their batch dimension there) and
``expand`` on a constant fails from 2021-01 on.
State_dict keys are the plain ResNet-18 keys plus ``coords`` (the teammate's
weights load with ``strict=False``).

Python 3.6 / old-torchvision compatible (runs on the Jetson Nano).
"""
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet

# Plain ResNet-18: pretrained weights and the shared training routine
import resnet18

# Value of --arch and of meta["arch"] for this network
NAME = "resnet18_coord"
# Prefix of the checkpoint file names
FILE_TAG = "resnet18coord"
# Name shown in the live demo dropdown
LABEL = "ResNet18 CoordConv"
# state_dict keys this network has (the coordinate buffer marks the variant)
KEYS = ("fc.weight", "coords")
# Input image size the coordinate maps are made for (camera.preprocess always resizes to it)
SIZE = 224
# Training settings: same as the plain ResNet-18 (teammate recipe, see training.py)
DEFAULTS = dict(resnet18.DEFAULTS)


# Fixed coordinate maps: channel 0 = x (varies along the width), channel 1 = y (along the height)
def coordinate_maps(size=SIZE):
    """(2, size, size) float tensor, values -1 .. 1 (same as the teammate's meshgrid)."""
    # Evenly spaced positions from -1 to 1
    line = torch.linspace(-1.0, 1.0, size)
    # x varies along the width
    xs = line.view(1, 1, 1, size).expand(1, 1, size, size)
    # y varies along the height
    ys = line.view(1, 1, size, 1).expand(1, 1, size, size)
    # Stack into one contiguous tensor, without a batch dimension
    return torch.cat([xs, ys], dim=1)[0].contiguous()


# Append the coordinate maps to a batch of images
def add_coords(images, coords):
    """(B, 3, H, W) + (2, H, W) buffer -> (B, 5, H, W)."""
    # (1, 2, H, W) via view: converted by every torch2trt version
    grid = coords.view(1, coords.shape[0], coords.shape[1], coords.shape[2])
    # Training batches (PyTorch only): one copy of the maps per image
    if images.shape[0] != 1:
        # Expanded view, no data copied
        grid = grid.expand(images.shape[0], -1, -1, -1)
    # Colour channels first, then x and y
    return torch.cat([images, grid], dim=1)


# 5-channel first convolution initialised from a 3-channel one
def coord_conv1(conv3=None):
    """Conv2d(5, 64, 7, 2, 3) with the RGB weights of ``conv3`` and zeros for x / y."""
    # Same geometry as ResNet-18's stem, two more input channels
    conv = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Copy the pretrained weights when given
    if conv3 is not None:
        # No gradient tracking while editing weights
        with torch.no_grad():
            # Coordinate channels start unused
            conv.weight.zero_()
            # Colour channels keep the pretrained filters
            conv.weight[:, :3].copy_(conv3.weight)
    # The new layer
    return conv


# ResNet-18 whose forward appends the coordinate maps first
class ResNet18Coord(ResNet):
    """torchvision ResNet-18 layout (same keys) + ``coords`` buffer + 5-channel ``conv1``."""

    # Build the layers, the 5-channel stem and the coordinate buffer
    def __init__(self, num_outputs=2):
        # Standard ResNet-18 blocks
        super(ResNet18Coord, self).__init__(BasicBlock, [2, 2, 2, 2])
        # Stem that takes colour + coordinates
        self.conv1 = coord_conv1()
        # Regression head for (x, y)
        self.fc = nn.Linear(512, num_outputs)
        # Coordinate maps, saved with the weights
        self.register_buffer("coords", coordinate_maps())

    # Same layer order as torchvision's ResNet, written out (old torchvision has no _forward_impl)
    def forward(self, x):
        # 3 colour channels + 2 coordinate channels
        x = add_coords(x, self.coords)
        # Stem
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        # Residual stages
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        # Global average pooling
        x = self.avgpool(x)
        # (B, 512)
        x = torch.flatten(x, 1)
        # (B, 2)
        return self.fc(x)


# Network used for training
def build(pretrained=True, num_outputs=2):
    # Empty CoordConv ResNet-18
    model = ResNet18Coord(num_outputs)
    # Start from the ImageNet weights
    if pretrained:
        # Stock pretrained ResNet-18
        stock = resnet18._torchvision(True)
        # Its weights except the stem and the 1000-class head
        weights = {k: v for k, v in stock.state_dict().items() if not k.startswith(("conv1.", "fc."))}
        # Load them (conv1, fc and coords stay as built)
        model.load_state_dict(weights, strict=False)
        # Stem: pretrained RGB filters, zero for the coordinates
        model.conv1 = coord_conv1(stock.conv1)
    # Ready-to-train network
    return model


# Same training routine as the plain ResNet-18 (same parameter names, so the same layers are trained)
def train(model, train_loader, val_loader, device, epochs, lr, tag=NAME, loss="mse"):
    """See ``resnet18.train``. Returns (best validation metrics, per-epoch history)."""
    # Shared routine
    return resnet18.train(model, train_loader, val_loader, device, epochs, lr, tag, loss=loss)
