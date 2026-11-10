"""Dataset / DataLoader for the road-following (x, y) data.

Per-sample pipeline:
    load jpg + label  ->  Augmenter (one powerset subset, training split only)
                      ->  preprocessing.mark_lanes_pil (lane marking, always, live)  ->  tensor

With augmentation on, the training set has len(images) x 29 samples: sample
``i`` is image ``i % n_images`` with method subset ``SUBSETS[i // n_images]``
(see ``augmentation.py``), so one epoch sees every image with every subset.
Lane marking comes after augmentation, as on the robot.

Folder layouts (``make_loaders`` picks automatically):
    <dir>/xy_*.jpg                       one folder, split here (random / session)
    <dir>/train/  <dir>/validation/  [<dir>/test/]
                                         already split by the team: used as is

Empty or broken image files are skipped with a warning.

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

# Powerset training augmentation (on / off)
from augmentation import Augmenter
# Lane marking, applied to every sample after augmentation
from preprocessing import mark_lanes_pil

# Public names of this module
__all__ = ["DEFAULT_DATA_DIR", "XYDataset", "parse_xy", "image_to_tensor",
           "has_split_folders", "split_indices", "split_by_session", "default_metadata_dir", "make_loaders",
           "IMAGENET_MEAN", "IMAGENET_STD"]

# Folder this file lives in (project4/)
_HERE = os.path.dirname(os.path.abspath(__file__))
# Dataset used when no --data folder is given
DEFAULT_DATA_DIR = os.path.join(_HERE, "data", "dataset_xy_2026-10-21", "dataset_xy")

# ImageNet per-channel mean (the networks are ImageNet-pretrained)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
# ImageNet per-channel standard deviation
IMAGENET_STD = [0.229, 0.224, 0.225]
# Network input size (height, width)
INPUT_SIZE = (224, 224)
# Sub-folder names of a dataset that is already split
SPLIT_FOLDERS = ("train", "validation", "test")

# Broken files already reported, so each warning is printed only once
_warned = set()


# Read the pixel label stored in a file name
def parse_xy(path):
    """Pixel (x, y) from ``xy_<x>_<y>_<uuid>.jpg``."""
    # "xy_112_130_<uuid>.jpg" -> ["xy", "112", "130", "<uuid>.jpg"]
    parts = os.path.basename(path).split("_")
    # x and y as floats
    return float(int(parts[1])), float(int(parts[2]))


# Drop image files that cannot be decoded (empty or truncated)
def _readable(paths):
    """The paths whose images decode completely; the others are reported once."""
    # Paths that passed the check
    good = []
    # Check every file
    for p in paths:
        # Try a full decode, not just the header, so truncated files are caught
        try:
            # Open the file and close it again after decoding
            with PIL.Image.open(p) as im:
                # Decode every pixel
                im.load()
            # Decoded fine
            good.append(p)
        # Any decoding error means the file is unusable
        except Exception as e:
            # Report each broken file only once per process
            if p not in _warned:
                # Remember that it was reported
                _warned.add(p)
                # Warn with the reason
                print("warning: skipping unreadable image %s (%s)" % (p, e))
    # Usable files in the original order
    return good


# Convert an image into the network input tensor, exactly as the course does
def image_to_tensor(image):
    """PIL RGB image -> normalised BGR tensor (3, 224, 224), as in the course."""
    # Resize to 224x224
    image = TF.resize(image, INPUT_SIZE)
    # HWC uint8 -> CHW float in [0, 1]
    t = TF.to_tensor(image)
    # RGB -> BGR, equivalent to numpy()[::-1]; the robot's camera delivers BGR
    t = t.flip(0)
    # ImageNet normalisation
    return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)


# PyTorch dataset of (image tensor, normalised (x, y)) pairs
class XYDataset(torch.utils.data.Dataset):
    """Road-following dataset.

    Parameters
    ----------
    directory : str
        Folder with ``xy_*.jpg``. Defaults to ``DEFAULT_DATA_DIR``.
    augmenter : Augmenter or None
        Applied to (image, x, y) before lane marking; when enabled, every image
        appears once per method subset. None -> no augmentation.
    indices : list[int] or None
        Use only these files (indices into the sorted list of all files).
    """

    # Find the images and remember which ones this split uses
    def __init__(self, directory=None, augmenter=None, indices=None):
        # Folder with the images
        self.directory = directory or DEFAULT_DATA_DIR
        # All labelled images in a stable order
        found = sorted(glob.glob(os.path.join(self.directory, "xy_*.jpg")))
        # Only the files that can actually be decoded
        paths = _readable(found)
        # Number of files dropped as unreadable
        self.n_skipped = len(found) - len(paths)
        # An empty folder is almost certainly a wrong path
        if not paths:
            # Report the folder that was searched
            raise FileNotFoundError("no readable xy_*.jpg found in %s" % self.directory)
        # Every image in the folder (used for splitting)
        self.all_paths = paths
        # Only the images of this split, or all of them
        chosen = paths if indices is None else [paths[i] for i in indices]
        # Images this dataset returns
        self.image_paths = chosen
        # Augmentation for this split (None for validation)
        self.augmenter = augmenter

    # Variants per image: 29 with augmentation on, else 1
    @property
    def n_variants(self):
        # Powerset size of an enabled augmenter
        return self.augmenter.n_variants if self.augmenter is not None else 1

    # Number of samples
    def __len__(self):
        # Every image once per variant
        return len(self.image_paths) * self.n_variants

    # Load one sample as a PIL image plus pixel label
    def load(self, idx):
        """Raw sample: (PIL image, x_px, y_px) after augmentation + lane marking, before tensor."""
        # Image index and method subset of this sample
        subset, image_idx = divmod(idx, len(self.image_paths))
        # File of this sample
        path = self.image_paths[image_idx]
        # Decode the image as RGB
        image = PIL.Image.open(path).convert("RGB")
        # Label from the file name
        x, y = parse_xy(path)
        # Training split: this sample's method subset (moves the label with the image)
        if self.augmenter is not None and self.augmenter.enabled:
            # Augmented image and label
            image, x, y = self.augmenter(image, x, y, subset)
        # Mark the lane lines last, as on the robot
        return mark_lanes_pil(image), x, y

    # Sample in the form the training loop needs
    def __getitem__(self, idx):
        # Augmented, lane-marked image and pixel label
        image, x, y = self.load(idx)
        # Image size
        w, h = image.size
        # Pixel x -> [-1, 1] (course normalisation)
        xn = (x - w / 2.0) / (w / 2.0)
        # Pixel y -> [-1, 1]
        yn = (y - h / 2.0) / (h / 2.0)
        # Network input tensor and target tensor
        return image_to_tensor(image), torch.tensor([xn, yn], dtype=torch.float32)


# Whether a dataset folder is already split into sub-folders
def has_split_folders(directory):
    """True if ``directory`` holds ``train/`` and ``validation/`` sub-folders."""
    # Both required sub-folders must exist (test/ is optional)
    return all(os.path.isdir(os.path.join(directory, d)) for d in SPLIT_FOLDERS[:2])


# Random train / validation split that is the same every run
def split_indices(n, val_ratio=0.1, seed=0):
    """Random but reproducible train/val index split."""
    # Seeded generator for reproducibility
    rng = np.random.RandomState(seed)
    # Shuffled image indices
    order = rng.permutation(n).tolist()
    # Size of the validation split (at least one image)
    n_val = max(1, int(round(n * val_ratio)))
    # Train indices, validation indices (both sorted)
    return sorted(order[n_val:]), sorted(order[:n_val])


# Default location of the capture metadata
def default_metadata_dir(directory):
    """``<parent>/metadata`` next to ``<parent>/dataset_xy`` (collection notebook layout)."""
    # Sibling folder called "metadata"
    return os.path.join(os.path.dirname(os.path.normpath(directory)), "metadata")


# Parse a capture timestamp, with or without fractional seconds
def _parse_time(text):
    # Try both timestamp formats
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        # strptime raises when the format does not match
        try:
            # Parsed timestamp
            return datetime.datetime.strptime(text, fmt)
        # Wrong format or missing value
        except (TypeError, ValueError):
            # Try the next format
            pass
    # No format matched
    return None


# Split so that near-identical neighbouring shots stay on the same side
def split_by_session(paths, metadata_dir, val_ratio=0.1, seed=0, segment_gap=60.0):
    """Split by shooting segment instead of by image.

    A segment = consecutive captures of one ``session_id`` (one notebook run)
    with less than ``segment_gap`` seconds between them. Whole segments go to
    the validation split until it holds about ``val_ratio`` of the images, so
    near-identical neighbouring shots never end up on both sides.

    Images without a metadata JSON become segments of their own.
    Returns (train_idx, val_idx, info).
    """
    # (index, session id, capture time) of every image
    records = []
    # Number of images without usable metadata
    missing = 0
    # Read the metadata of every image
    for i, p in enumerate(paths):
        # File name without extension
        stem = os.path.splitext(os.path.basename(p))[0]
        # Matching metadata file
        meta_path = os.path.join(metadata_dir, stem + ".json")
        # Unknown until the metadata is read
        session, t = None, None
        # Only images with a metadata file have a session and time
        if os.path.exists(meta_path):
            # Open the metadata
            with open(meta_path, encoding="utf-8") as f:
                # Parsed metadata
                meta = json.load(f)
            # Which notebook run captured it
            session = meta.get("session_id")
            # When it was captured
            t = _parse_time(meta.get("captured_at") or meta.get("saved_at"))
        # Without session or time the image becomes its own segment
        if session is None or t is None:
            # Count it
            missing += 1
            # Unique pseudo-session, no time
            records.append((i, "single-%d" % i, None))
        # Normal case
        else:
            # Keep session and time
            records.append((i, session, t))

    # List of segments, each a list of image indices
    groups = []
    # Session id -> list of (time, index)
    by_session = {}
    # Sort the records into sessions
    for i, session, t in records:
        # Images without time are single-image segments
        if t is None:
            # Segment of one
            groups.append([i])
        # Images with time go to their session
        else:
            # Append to that session's list
            by_session.setdefault(session, []).append((t, i))
    # Cut every session into segments at large time gaps
    for session in sorted(by_session):
        # Captures of this session in time order
        items = sorted(by_session[session])
        # The first capture starts the first segment
        current = [items[0][1]]
        # Walk over consecutive pairs of captures
        for (t0, _), (t1, i1) in zip(items, items[1:]):
            # A long pause starts a new segment
            if (t1 - t0).total_seconds() > segment_gap:
                # Close the current segment
                groups.append(current)
                # Start a new one
                current = []
            # Add this capture to the current segment
            current.append(i1)
        # Close the last segment of the session
        groups.append(current)

    # Splitting by segment needs at least two segments
    if len(groups) < 2:
        # Suggest the alternatives
        raise ValueError("only %d shooting segment(s): cannot split by session; use split='random' "
                         "or a smaller segment_gap" % len(groups))
    # Seeded generator for reproducibility
    rng = np.random.RandomState(seed)
    # Segments in random order
    order = rng.permutation(len(groups)).tolist()
    # Desired number of validation images
    target = max(1, int(round(len(paths) * val_ratio)))
    # Validation indices collected so far
    val = []
    # Take whole segments until the target is reached
    for g in order:
        # Enough validation images
        if len(val) >= target:
            # Stop adding segments
            break
        # Never move so much that training would be empty
        if len(val) + len(groups[g]) >= len(paths):
            # Skip this segment; always leave something for training
            continue
        # Move the whole segment to validation
        val.extend(groups[g])
    # Every segment was too large to move
    if not val:
        # Suggest the alternatives
        raise ValueError("every segment is too large for the validation split; use split='random' "
                         "or a smaller segment_gap")
    # Set for fast membership tests
    val_set = set(val)
    # Everything not in validation is training
    train = [i for i in range(len(paths)) if i not in val_set]
    # Description of the split, saved in the model's .json
    info = {"split": "session", "segments": len(groups), "sessions": len(by_session),
            "images_without_metadata": missing, "segment_gap": segment_gap,
            "metadata_dir": os.path.abspath(metadata_dir)}
    # Train indices, validation indices, description
    return sorted(train), sorted(val), info


# Build the training, validation and test DataLoaders
def make_loaders(directory=None, augment=False,
                 val_ratio=0.1, batch_size=16, num_workers=4, seed=0, augmenter=None,
                 split="random", metadata_dir=None, segment_gap=60.0):
    """Train/val/test DataLoaders.

    Augmentation is applied to the training split only; lane marking to all.
    Pass ``augment`` (on / off), or a pre-configured ``augmenter`` (overrides it).

    If ``directory`` has ``train/`` and ``validation/`` sub-folders (and
    optionally ``test/``) they are used as they are and ``split`` /
    ``val_ratio`` are ignored. Otherwise the single folder is split here:
    ``split='random'`` (default) at random, ``split='session'`` keeping
    shooting segments together (needs the collection ``metadata`` folder,
    default ``default_metadata_dir(directory)``); there is no test split then.

    Returns (train_loader, val_loader, test_loader, augmenter); test_loader is
    None without a ``test/`` folder. The split details are in
    ``train_loader.split_info``.
    """
    # Fall back to the default dataset
    directory = directory or DEFAULT_DATA_DIR
    # Build the augmenter from the switch unless one is passed in
    if augmenter is None:
        # Powerset augmenter, on or off
        augmenter = Augmenter(enabled=augment)
    # No test split unless a test/ folder exists
    test_ds = None
    # Dataset already split into sub-folders by the team
    if has_split_folders(directory):
        # Training folder, with augmentation
        train_ds = XYDataset(os.path.join(directory, "train"), augmenter=augmenter)
        # Validation folder, never augmented
        val_ds = XYDataset(os.path.join(directory, "validation"), augmenter=None)
        # Test folder, if present
        test_dir = os.path.join(directory, "test")
        # Load it only when it exists
        if os.path.isdir(test_dir):
            # Test folder, never augmented
            test_ds = XYDataset(test_dir, augmenter=None)
        # Every dataset that was loaded
        parts = [d for d in (train_ds, val_ds, test_ds) if d is not None]
        # Description of the split, saved in the model's .json
        split_info = {"split": "folders", "train": len(train_ds.image_paths), "validation": len(val_ds),
                      "test": len(test_ds) if test_ds is not None else 0,
                      "skipped_unreadable": sum(d.n_skipped for d in parts)}
    # Single folder: split it here
    else:
        # Every readable image in the folder
        full = XYDataset(directory)
        # Number of images
        n = len(full.all_paths)
        # Split by shooting segment
        if split == "session":
            # Metadata folder next to the data unless given
            metadata_dir = metadata_dir or default_metadata_dir(directory)
            # Session split is impossible without metadata
            if not os.path.isdir(metadata_dir):
                # Report the missing folder
                raise FileNotFoundError("split='session' needs the metadata folder: %s" % metadata_dir)
            # Segment-based split
            train_idx, val_idx, split_info = split_by_session(full.all_paths, metadata_dir, val_ratio, seed,
                                                              segment_gap)
        # Split images at random
        elif split == "random":
            # Seeded random split
            train_idx, val_idx = split_indices(n, val_ratio, seed)
            # Description of the split
            split_info = {"split": "random"}
        # Anything else is a typo
        else:
            # Report the valid choices
            raise ValueError("split must be 'random' or 'session'")
        # Record how many broken files were left out
        split_info["skipped_unreadable"] = full.n_skipped
        # Training split, with augmentation
        train_ds = XYDataset(directory, augmenter=augmenter, indices=train_idx)
        # Validation split, never augmented
        val_ds = XYDataset(directory, augmenter=None, indices=val_idx)
    # Seeded generator so the shuffling order is reproducible
    g = torch.Generator()
    # Seed it
    g.manual_seed(seed)
    # Shuffled training batches
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g)
    # Validation batches in a fixed order
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # Test batches in a fixed order, if there is a test split
    test_loader = None if test_ds is None else torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # Attach the split description for train.py to save
    train_loader.split_info = split_info
    # The loaders and the augmenter that was used
    return train_loader, val_loader, test_loader, augmenter
