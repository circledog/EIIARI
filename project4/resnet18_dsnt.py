"""ResNet-18 with a heatmap head and soft-argmax output (DSNT), from the teammate's ``dsnt``.

Instead of regressing (x, y) from a pooled vector, the network predicts a
28x28 heatmap of where the target is and outputs its expected position:

    ResNet-18, layer4 stride 1  -> 512 x 14 x 14 features
    head: 1x1 conv 512->64, ReLU, 1x1 conv 64->1  -> 14 x 14 logits
    bilinear upsampling to 28 x 28, softmax over the 784 cells -> heatmap
    (x, y) = sum(heatmap * x-grid), sum(heatmap * y-grid)   (grids -1 .. 1)

Training loss (teammate): MSE(x, y) + 0.1 * Jensen-Shannon divergence between
the heatmap and a Gaussian (sigma 1.5 cells) at the label.

TensorRT-friendly form, same values as the teammate's code: the bilinear
upsampling (``F.interpolate``) and the expectation are both fixed linear
maps, so they are stored as frozen ``nn.Linear`` layers -- ``upsample``
(196 -> 784, built by interpolating each of the 196 unit images) and
``expect`` (784 -> 2, rows = flattened x / y grids). The network then only
uses conv / BN / ReLU / pooling / view / Linear / F.softmax, which
torch2trt 2019-10 .. 2022-07 converts (checked offline with
``check_trt_offline.py``) and TensorRT's ONNX parser supports. Avoided:
the interpolate converter (missing before TensorRT 7.1 without the plugin;
TensorRT 7 resize has no half-pixel mode, so values could differ) and
``torch.softmax`` (no converter in any tested version: torch2trt would
silently freeze its value into a constant).
Output stays (B, 2), so camera, preprocessing, sanity, control and the UI
are unchanged.

Python 3.6 / old-torchvision compatible (runs on the Jetson Nano).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Plain ResNet-18: backbone weights and the shared training routine
import resnet18

# Value of --arch and of meta["arch"] for this network
NAME = "resnet18_dsnt"
# Prefix of the checkpoint file names
FILE_TAG = "resnet18dsnt"
# Name shown in the live demo dropdown
LABEL = "ResNet18 DSNT"
# state_dict keys this network has (the expectation layer marks DSNT)
KEYS = ("expect.weight",)
# Side length of the backbone's feature map for a 224 input with layer4 stride 1
FEATURE = 14
# Side length of the heatmap
HEATMAP = 28
# Training settings: shared ones + the teammate's heatmap-loss weight and Gaussian width
DEFAULTS = dict(resnet18.DEFAULTS, js_weight=0.1, sigma_bins=1.5)


# Cell-centre coordinates of a size x size heatmap, -1 .. 1
def heatmap_grids(size=HEATMAP):
    """(x-grid, y-grid), each (1, 1, size, size), same as the teammate's coordinate_grid."""
    # Evenly spaced positions from -1 to 1
    line = torch.linspace(-1.0, 1.0, size)
    # x varies along the width
    xs = line.view(1, 1, 1, size).expand(1, 1, size, size).contiguous()
    # y varies along the height
    ys = line.view(1, 1, size, 1).expand(1, 1, size, size).contiguous()
    # Both grids
    return xs, ys


# Matrix of F.interpolate(bilinear, align_corners=False) from FEATURE^2 to HEATMAP^2 cells
def upsample_matrix(src=FEATURE, dst=HEATMAP):
    """(dst*dst, src*src) matrix M with M @ logits.flatten() == interpolate(logits).flatten()."""
    # One unit image per input cell
    eye = torch.eye(src * src).view(src * src, 1, src, src)
    # Interpolate each unit image exactly as the teammate's model does
    up = F.interpolate(eye, size=(dst, dst), mode="bilinear", align_corners=False)
    # Column j = response to input cell j
    return up.view(src * src, dst * dst).t().contiguous()


# Frozen Linear layer holding a fixed matrix
def fixed_linear(matrix):
    """nn.Linear whose weight is ``matrix`` and bias zero, never trained.

    The (zero) bias is kept because torch2trt's Linear converter before
    2020 requires one.
    """
    # Layer of the right size
    layer = nn.Linear(matrix.shape[1], matrix.shape[0])
    # No gradient tracking while setting the values
    with torch.no_grad():
        # Fixed weights
        layer.weight.copy_(matrix)
        # No offset
        layer.bias.zero_()
    # Neither is trained (Adam skips parameters without gradients)
    for p in layer.parameters():
        # Frozen
        p.requires_grad_(False)
    # The layer
    return layer


# Heatmap regressor on a ResNet-18 backbone
class ResNet18DSNT(nn.Module):
    """Backbone (``backbone.*``), heatmap head (``head.*``), fixed ``upsample`` / ``expect``."""

    # Build backbone, head and the fixed layers
    def __init__(self, backbone, num_outputs=2):
        # Initialise nn.Module
        super(ResNet18DSNT, self).__init__()
        # Only (x, y) outputs are supported by the expectation
        if num_outputs != 2:
            # Tell the caller
            raise ValueError("DSNT predicts exactly (x, y)")
        # The 1000-class classifier is not used
        backbone.fc = nn.Identity() if hasattr(nn, "Identity") else nn.Sequential()
        # layer4 keeps the 14x14 resolution (weight shapes unchanged, pretrained values kept)
        backbone.layer4[0].conv1.stride = (1, 1)
        # Its shortcut must use the same stride
        backbone.layer4[0].downsample[0].stride = (1, 1)
        # ResNet-18 trunk
        self.backbone = backbone
        # 1x1 conv head: 512 features -> 1 logit per cell
        self.head = nn.Sequential(nn.Conv2d(512, 64, 1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        # Fixed bilinear upsampling 14x14 -> 28x28 as a linear map
        self.upsample = fixed_linear(upsample_matrix())
        # Heatmap grids (also used by the training loss)
        xs, ys = heatmap_grids()
        # Fixed expectation: (x, y) = heatmap . grids
        self.expect = fixed_linear(torch.cat([xs.reshape(1, -1), ys.reshape(1, -1)], dim=0))

    # Network input as the backbone's first convolution expects it (overridden by CoordConv)
    def stem_input(self, images):
        # Plain colour images
        return images

    # Heatmap probabilities, flattened: (B, 784)
    def heatmaps(self, images):
        # Backbone without pooling / classifier
        b = self.backbone
        # Stem
        x = b.maxpool(b.relu(b.bn1(b.conv1(self.stem_input(images)))))
        # Residual stages -> (B, 512, 14, 14)
        x = b.layer4(b.layer3(b.layer2(b.layer1(x))))
        # Logits per cell, flattened (view converts on every torch2trt version)
        logits = self.head(x).view(-1, FEATURE * FEATURE)
        # Bilinear upsampling to 28x28, flattened
        logits = self.upsample(logits)
        # Probabilities over the 784 cells (F.softmax has a converter, torch.softmax did not)
        return F.softmax(logits, dim=1)

    # Expected target position: (B, 2)
    def forward(self, images):
        # Expectation of the heatmap
        return self.expect(self.heatmaps(images))

    # Prediction plus heatmap for the training loss
    def training_output(self, images):
        """((B, 2) prediction, (B, 1, 28, 28) heatmap)."""
        # Heatmap probabilities
        heat = self.heatmaps(images)
        # Prediction and the heatmap as an image
        return self.expect(heat), heat.view(-1, 1, HEATMAP, HEATMAP)


# Backbone with or without ImageNet weights
def backbone(pretrained):
    # Stock torchvision ResNet-18
    return resnet18._torchvision(pretrained)


# Network used for training
def build(pretrained=True, num_outputs=2):
    # DSNT on a (pretrained) ResNet-18
    return ResNet18DSNT(backbone(pretrained), num_outputs)


# Jensen-Shannon divergence between heatmaps and Gaussians at the labels (teammate's gaussian_js)
def gaussian_js(heatmap, target, sigma_bins=1.5):
    """Mean JS divergence; ``heatmap`` (B, 1, 28, 28) probabilities, ``target`` (B, 2) in -1 .. 1."""
    # Grids on the heatmap's device
    xs, ys = [g.to(heatmap.device, heatmap.dtype) for g in heatmap_grids(heatmap.shape[-1])]
    # Gaussian width in normalised units
    sigma = 2.0 / (heatmap.shape[-1] - 1) * sigma_bins
    # Squared distance of every cell to the label
    squared = (xs - target[:, 0, None, None, None]) ** 2 + (ys - target[:, 1, None, None, None]) ** 2
    # Unnormalised Gaussian
    gaussian = torch.exp(-squared / (2 * sigma * sigma))
    # Normalise to a probability map
    gaussian = gaussian / gaussian.sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)
    # Avoid log(0)
    p, q = heatmap.clamp_min(1e-8), gaussian.clamp_min(1e-8)
    # Mixture distribution
    middle = (p + q) / 2
    # 0.5 * (KL(p || m) + KL(q || m)), averaged over the batch
    return (0.5 * (p * (p.log() - middle.log()) + q * (q.log() - middle.log())).sum(dim=(1, 2, 3))).mean()


# Loss function for resnet18.train: MSE on (x, y) + weighted heatmap divergence
def make_loss(js_weight=DEFAULTS["js_weight"], sigma_bins=DEFAULTS["sigma_bins"]):
    """``loss(model, images, labels)`` as in the teammate's spatial comparison."""
    # The loss
    def loss(model, images, labels):
        # Prediction and heatmap
        prediction, heat = model.training_output(images)
        # Coordinate error + heatmap shape term
        return F.mse_loss(prediction, labels) + js_weight * gaussian_js(heat, labels, sigma_bins)
    # Hand it to the training loop
    return loss


# Shared ResNet-18 training loop with the DSNT loss
def train(model, train_loader, val_loader, device, epochs, lr, tag=NAME):
    """Adam, MSE + 0.1 JS loss, keep the epoch with the lowest validation MSE (teammate's selection)."""
    # Shared routine with the heatmap loss
    return resnet18.train(model, train_loader, val_loader, device, epochs, lr, tag,
                          loss_fn=make_loss(DEFAULTS["js_weight"], DEFAULTS["sigma_bins"]))
