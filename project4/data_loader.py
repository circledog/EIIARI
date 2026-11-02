"""Dataset / DataLoader for the road-following (x, y) data.

Per-sample pipeline:
    load jpg + label  ->  Preprocessor (deterministic, also used on the robot)
                      ->  Augmenter (random, training split only)  ->  tensor

File name convention (from the data-collection notebook):
    xy_<x>_<y>_<uuid>.jpg      x, y = pixel coordinates in that image

Targets are normalised exactly like the course ``train_model.ipynb``:
    x_n = (x - W/2) / (W/2),  y_n = (y - H/2) / (H/2)      (both in [-1, 1])

Tensor pre-processing also follows the course so that models stay compatible
with the slide's ``preprocess()`` on the robot: resize 224x224, to_tensor,
RGB -> BGR channel flip (the JetBot camera delivers BGR), ImageNet normalise.
"""
import glob
import os

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF

from augmentation import Augmenter
from preprocessing import Preprocessor

__all__ = ["DEFAULT_DATA_DIR", "XYDataset", "parse_xy", "image_to_tensor",
           "split_indices", "make_loaders", "IMAGENET_MEAN", "IMAGENET_STD"]

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(_HERE, "data", "dataset_xy_2026-10-21", "dataset_xy")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = (224, 224)


def parse_xy(path):
    """Pixel (x, y) from ``xy_<x>_<y>_<uuid>.jpg``."""
    parts = os.path.basename(path).split("_")
    return float(int(parts[1])), float(int(parts[2]))


def image_to_tensor(image):
    """PIL RGB image -> normalised BGR tensor (3, 224, 224), as in the course."""
    image = TF.resize(image, INPUT_SIZE)
    t = TF.to_tensor(image)
    t = t.flip(0)  # RGB -> BGR, equivalent to numpy()[::-1]
    return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)


class XYDataset(torch.utils.data.Dataset):
    """Road-following dataset.

    Parameters
    ----------
    directory : str
        Folder with ``xy_*.jpg``. Defaults to ``DEFAULT_DATA_DIR``.
    preprocessor : Preprocessor or None
        Deterministic transform applied first (ROI crop, fixed y); images in
        its exclude list are dropped. None -> no preprocessing.
    augmenter : Augmenter or None
        Applied to (image, x, y) after preprocessing. None -> no augmentation.
    indices : list[int] or None
        Use only these files (indices into the sorted list of *all* files, so
        the split does not change when the exclude list does).
    """

    def __init__(self, directory=None, augmenter=None, indices=None, preprocessor=None):
        self.directory = directory or DEFAULT_DATA_DIR
        paths = sorted(glob.glob(os.path.join(self.directory, "xy_*.jpg")))
        if not paths:
            raise FileNotFoundError("no xy_*.jpg found in %s" % self.directory)
        self.all_paths = paths
        chosen = paths if indices is None else [paths[i] for i in indices]
        self.preprocessor = preprocessor
        if preprocessor is not None:
            chosen = [p for p in chosen if preprocessor.keep(p)]
        self.image_paths = chosen
        self.augmenter = augmenter

    def __len__(self):
        return len(self.image_paths)

    def load(self, idx):
        """Raw sample: (PIL image, x_px, y_px) after augmentation, before tensor."""
        path = self.image_paths[idx]
        image = PIL.Image.open(path).convert("RGB")
        x, y = parse_xy(path)
        if self.preprocessor is not None:
            image, x, y = self.preprocessor(image, x, y)
        if self.augmenter is not None:
            image, x, y = self.augmenter(image, x, y)
        return image, x, y

    def __getitem__(self, idx):
        image, x, y = self.load(idx)
        w, h = image.size
        xn = (x - w / 2.0) / (w / 2.0)
        yn = (y - h / 2.0) / (h / 2.0)
        return image_to_tensor(image), torch.tensor([xn, yn], dtype=torch.float32)


def split_indices(n, val_ratio=0.1, seed=0):
    """Random but reproducible train/val index split."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(n).tolist()
    n_val = max(1, int(round(n * val_ratio)))
    return sorted(order[n_val:]), sorted(order[:n_val])


def make_loaders(directory=None, general=False, perlin=False, compression=False,
                 val_ratio=0.1, batch_size=8, num_workers=4, seed=0, augmenter=None,
                 preprocessor=None):
    """Train/val DataLoaders.

    Preprocessing is applied to both splits; augmentation to the training split
    only. Pass the three augmentation switches, or a pre-configured
    ``augmenter`` (overrides them). Returns (train_loader, val_loader, augmenter).
    """
    directory = directory or DEFAULT_DATA_DIR
    if augmenter is None:
        augmenter = Augmenter(general=general, perlin=perlin, compression=compression)
    n = len(XYDataset(directory))
    train_idx, val_idx = split_indices(n, val_ratio, seed)
    train_ds = XYDataset(directory, augmenter=augmenter, indices=train_idx, preprocessor=preprocessor)
    val_ds = XYDataset(directory, augmenter=None, indices=val_idx, preprocessor=preprocessor)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, augmenter
