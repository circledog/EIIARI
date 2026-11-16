"""Shared helpers for the tests: import path and a small synthetic dataset.

The tests need no real data, no GPU and no downloads (networks are built
with ``pretrained=False``).
"""
import os
import sys

import cv2
import numpy as np

# project4/ folder (parent of tests/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Make the project modules importable from any working directory
if ROOT not in sys.path:
    # Put project4/ first on the import path
    sys.path.insert(0, ROOT)


# Camera-like 224x224 BGR frame: grey floor with two white tape lines
def track_frame(seed=0, size=224):
    """Synthetic track picture the lane marking reacts to."""
    # Reproducible randomness per frame
    rng = np.random.RandomState(seed)
    # Dark grey floor with a little noise
    frame = np.clip(rng.normal(70, 6, (size, size, 3)), 0, 255).astype(np.uint8)
    # Horizontal offset of the lane, different per frame
    shift = int(rng.randint(-30, 30))
    # Left tape line from the bottom towards the top
    cv2.line(frame, (40 + shift, size - 1), (90 + shift, 60), (235, 235, 235), 9)
    # Right tape line
    cv2.line(frame, (184 + shift, size - 1), (134 + shift, 60), (235, 235, 235), 9)
    # The finished frame
    return frame


# Folder of xy_<x>_<y>_<id>.jpg files like the team's dataset
def make_dataset(folder, n=8, seed=0):
    """Write ``n`` synthetic labelled images into ``folder``; returns their paths."""
    # Create the folder if needed
    os.makedirs(folder, exist_ok=True)
    # Reproducible labels
    rng = np.random.RandomState(seed)
    # Written files
    paths = []
    # One image per index
    for i in range(n):
        # Label inside the image, upper half (like real targets)
        x, y = int(rng.randint(20, 204)), int(rng.randint(40, 150))
        # File name in the dataset's format
        path = os.path.join(folder, "xy_%03d_%03d_%04d.jpg" % (x, y, i))
        # Save the synthetic frame
        cv2.imwrite(path, track_frame(seed * 100 + i))
        # Remember the file
        paths.append(path)
    # All files
    return paths


# Team layout: train/ validation/ test/ sub-folders
def make_split_dataset(folder, n=8):
    """Synthetic dataset with the three split folders; returns ``folder``."""
    # One sub-folder per split, different images in each
    for k, split in enumerate(("train", "validation", "test")):
        # Write the split's images
        make_dataset(os.path.join(folder, split), n, seed=k + 1)
    # Root of the dataset
    return folder
