"""Dataset / DataLoader for the road-following (x, y) data.

Per-sample pipeline:
    load jpg + label  ->  Augmenter (random, training split only)
                      ->  preprocessing.mark_lanes_pil (lane marking, always, live)  ->  tensor

Lane marking comes after augmentation so that Perlin / compression noise
affects the detection exactly as the "input disturbance" does on the robot.

File name convention (from the data-collection notebook):
    xy_<x>_<y>_<uuid>.jpg      x, y = pixel coordinates in that image

Targets are normalised exactly like the course ``train_model.ipynb``:
    x_n = (x - W/2) / (W/2),  y_n = (y - H/2) / (H/2)      (both in [-1, 1])

Tensor pre-processing also follows the course so that models stay compatible
with the slide's ``preprocess()`` on the robot: resize 224x224, to_tensor,
RGB -> BGR channel flip (the JetBot camera delivers BGR), ImageNet normalise.
"""
import datetime
import glob
import json
import os

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF

from augmentation import Augmenter
from preprocessing import mark_lanes_pil

__all__ = ["DEFAULT_DATA_DIR", "XYDataset", "parse_xy", "image_to_tensor",
           "split_indices", "split_by_session", "default_metadata_dir", "make_loaders",
           "IMAGENET_MEAN", "IMAGENET_STD"]

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
    augmenter : Augmenter or None
        Applied to (image, x, y) before lane marking. None -> no augmentation.
    indices : list[int] or None
        Use only these files (indices into the sorted list of all files).
    """

    def __init__(self, directory=None, augmenter=None, indices=None):
        self.directory = directory or DEFAULT_DATA_DIR
        paths = sorted(glob.glob(os.path.join(self.directory, "xy_*.jpg")))
        if not paths:
            raise FileNotFoundError("no xy_*.jpg found in %s" % self.directory)
        self.all_paths = paths
        chosen = paths if indices is None else [paths[i] for i in indices]
        self.image_paths = chosen
        self.augmenter = augmenter

    def __len__(self):
        return len(self.image_paths)

    def load(self, idx):
        """Raw sample: (PIL image, x_px, y_px) after augmentation + lane marking, before tensor."""
        path = self.image_paths[idx]
        image = PIL.Image.open(path).convert("RGB")
        x, y = parse_xy(path)
        if self.augmenter is not None:
            image, x, y = self.augmenter(image, x, y)
        return mark_lanes_pil(image), x, y

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


def default_metadata_dir(directory):
    """``<parent>/metadata`` next to ``<parent>/dataset_xy`` (collection notebook layout)."""
    return os.path.join(os.path.dirname(os.path.normpath(directory)), "metadata")


def _parse_time(text):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(text, fmt)
        except (TypeError, ValueError):
            pass
    return None


def split_by_session(paths, metadata_dir, val_ratio=0.1, seed=0, segment_gap=60.0):
    """Split by shooting segment instead of by image.

    A segment = consecutive captures of one ``session_id`` (one notebook run)
    with less than ``segment_gap`` seconds between them. Whole segments go to
    the validation split until it holds about ``val_ratio`` of the images, so
    near-identical neighbouring shots never end up on both sides.

    Images without a metadata JSON become segments of their own.
    Returns (train_idx, val_idx, info).
    """
    records = []
    missing = 0
    for i, p in enumerate(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        meta_path = os.path.join(metadata_dir, stem + ".json")
        session, t = None, None
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            session = meta.get("session_id")
            t = _parse_time(meta.get("captured_at") or meta.get("saved_at"))
        if session is None or t is None:
            missing += 1
            records.append((i, "single-%d" % i, None))
        else:
            records.append((i, session, t))

    groups = []
    by_session = {}
    for i, session, t in records:
        if t is None:
            groups.append([i])
        else:
            by_session.setdefault(session, []).append((t, i))
    for session in sorted(by_session):
        items = sorted(by_session[session])
        current = [items[0][1]]
        for (t0, _), (t1, i1) in zip(items, items[1:]):
            if (t1 - t0).total_seconds() > segment_gap:
                groups.append(current)
                current = []
            current.append(i1)
        groups.append(current)

    if len(groups) < 2:
        raise ValueError("only %d shooting segment(s): cannot split by session; use split='random' "
                         "or a smaller segment_gap" % len(groups))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(groups)).tolist()
    target = max(1, int(round(len(paths) * val_ratio)))
    val = []
    for g in order:
        if len(val) >= target:
            break
        if len(val) + len(groups[g]) >= len(paths):
            continue  # always leave something for training
        val.extend(groups[g])
    if not val:
        raise ValueError("every segment is too large for the validation split; use split='random' "
                         "or a smaller segment_gap")
    val_set = set(val)
    train = [i for i in range(len(paths)) if i not in val_set]
    info = {"split": "session", "segments": len(groups), "sessions": len(by_session),
            "images_without_metadata": missing, "segment_gap": segment_gap,
            "metadata_dir": os.path.abspath(metadata_dir)}
    return sorted(train), sorted(val), info


def make_loaders(directory=None, general=False, perlin=False, compression=False,
                 val_ratio=0.1, batch_size=8, num_workers=4, seed=0, augmenter=None,
                 split="random", metadata_dir=None, segment_gap=60.0):
    """Train/val DataLoaders.

    Augmentation is applied to the training split only; lane marking to both.
    Pass the three augmentation switches, or a pre-configured ``augmenter``
    (overrides them).

    ``split='random'`` (default) splits images at random; ``split='session'``
    keeps shooting segments together (needs the collection ``metadata`` folder,
    default ``default_metadata_dir(directory)``).

    Returns (train_loader, val_loader, augmenter); the split details are in
    ``train_loader.split_info``.
    """
    directory = directory or DEFAULT_DATA_DIR
    if augmenter is None:
        augmenter = Augmenter(general=general, perlin=perlin, compression=compression)
    all_paths = XYDataset(directory).all_paths
    n = len(all_paths)
    if split == "session":
        metadata_dir = metadata_dir or default_metadata_dir(directory)
        if not os.path.isdir(metadata_dir):
            raise FileNotFoundError("split='session' needs the metadata folder: %s" % metadata_dir)
        train_idx, val_idx, split_info = split_by_session(all_paths, metadata_dir, val_ratio, seed, segment_gap)
    elif split == "random":
        train_idx, val_idx = split_indices(n, val_ratio, seed)
        split_info = {"split": "random"}
    else:
        raise ValueError("split must be 'random' or 'session'")
    train_ds = XYDataset(directory, augmenter=augmenter, indices=train_idx)
    val_ds = XYDataset(directory, augmenter=None, indices=val_idx)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loader.split_info = split_info
    return train_loader, val_loader, augmenter
