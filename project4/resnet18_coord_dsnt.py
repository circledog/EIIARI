"""CoordConv + DSNT on ResNet-18, the teammate's ``coordconv_dsnt``.

``resnet18_dsnt`` whose backbone stem takes the 3 colour channels plus the
two coordinate maps of ``resnet18_coord`` (added inside the network, zero
initial weights for them). Input stays (1, 3, 224, 224), output (B, 2).

Python 3.6 / old-torchvision compatible (runs on the Jetson Nano).
"""
# Coordinate maps and the 5-channel stem
import resnet18_coord
# Heatmap head, loss and training routine
import resnet18_dsnt

# Value of --arch and of meta["arch"] for this network
NAME = "resnet18_coord_dsnt"
# Prefix of the checkpoint file names
FILE_TAG = "resnet18coorddsnt"
# Name shown in the live demo dropdown
LABEL = "ResNet18 CoordConv+DSNT"
# state_dict keys this network has
KEYS = ("expect.weight", "coords")
# Same settings as DSNT
DEFAULTS = dict(resnet18_dsnt.DEFAULTS)


# DSNT whose stem also sees the coordinate maps
class ResNet18CoordDSNT(resnet18_dsnt.ResNet18DSNT):
    """``resnet18_dsnt.ResNet18DSNT`` + ``coords`` buffer + 5-channel ``backbone.conv1``."""

    # Build DSNT, then widen the stem
    def __init__(self, backbone, num_outputs=2):
        # Plain DSNT on the given backbone
        super(ResNet18CoordDSNT, self).__init__(backbone, num_outputs)
        # 5-channel stem: pretrained RGB filters, zero for the coordinates
        self.backbone.conv1 = resnet18_coord.coord_conv1(self.backbone.conv1)
        # Coordinate maps, saved with the weights
        self.register_buffer("coords", resnet18_coord.coordinate_maps())

    # Colour + coordinate channels for the stem
    def stem_input(self, images):
        # Same helper as resnet18_coord (constant at batch 1)
        return resnet18_coord.add_coords(images, self.coords)


# Network used for training
def build(pretrained=True, num_outputs=2):
    # CoordConv DSNT on a (pretrained) ResNet-18
    return ResNet18CoordDSNT(resnet18_dsnt.backbone(pretrained), num_outputs)


# Same training as DSNT
def train(model, train_loader, val_loader, device, epochs, lr, tag=NAME):
    """See ``resnet18_dsnt.train``."""
    # Shared DSNT routine
    return resnet18_dsnt.train(model, train_loader, val_loader, device, epochs, lr, tag)
