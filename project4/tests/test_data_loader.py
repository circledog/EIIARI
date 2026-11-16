"""Data loading: lane-marking switch, augmentation length, train path == robot path."""
import os
import shutil
import tempfile
import unittest

import cv2
import torch

# Import path and synthetic data
import _helpers
# Robot-side tensor conversion
import camera as cam_mod
# Code under test
from data_loader import XYDataset, make_loaders
# Augmenter switch
from augmentation import Augmenter
# Robot-side lane marking
from preprocessing import apply_frame


# Tests of data_loader.py
class DataLoaderTest(unittest.TestCase):

    # One synthetic dataset for all tests of the class
    @classmethod
    def setUpClass(cls):
        # Temporary folder
        cls.tmp = tempfile.mkdtemp()
        # train/ validation/ test/ with 8 images each
        cls.root = _helpers.make_split_dataset(cls.tmp, n=8)

    # Remove the temporary folder
    @classmethod
    def tearDownClass(cls):
        # Delete everything that was written
        shutil.rmtree(cls.tmp)

    # Split folders are used as they are; augmentation multiplies only the training set
    def test_lengths(self):
        # Loaders with augmentation, no worker processes
        train, val, test, aug = make_loaders(self.root, augment=True, batch_size=4, num_workers=0)
        # 8 images x 29 subsets
        self.assertEqual(len(train.dataset), 8 * 29)
        # Validation and test never augmented
        self.assertEqual((len(val.dataset), len(test.dataset)), (8, 8))
        # The split description counts images, not samples
        self.assertEqual(train.split_info["train"], 8)

    # lanes=False feeds the plain image, lanes=True the marked one
    def test_lane_switch(self):
        # Validation folder
        folder = os.path.join(self.root, "validation")
        # Same image with and without lane marking
        marked = XYDataset(folder, lanes=True)[0][0]
        # Plain version
        plain = XYDataset(folder, lanes=False)[0][0]
        # The synthetic tape must have been painted, so the tensors differ
        self.assertGreater((marked - plain).abs().max().item(), 0.5)
        # make_loaders passes the switch to every split
        loaders = make_loaders(self.root, batch_size=4, num_workers=0, lanes=False)[:3]
        # All three datasets have marking off
        self.assertEqual([l.dataset.lanes for l in loaders], [False, False, False])

    # The training tensor equals the robot's tensor for the same picture
    def test_train_path_equals_robot_path(self):
        # Dataset without augmentation, lane marking on
        ds = XYDataset(os.path.join(self.root, "test"), augmenter=Augmenter(False))
        # Every image
        for i, path in enumerate(ds.image_paths):
            # Robot path: BGR frame -> lane marking -> camera.preprocess
            robot = cam_mod.preprocess(apply_frame(cv2.resize(cv2.imread(path), (224, 224))), "cpu",
                                       torch.float32)[0]
            # Training path
            train = ds[i][0]
            # Identical up to float rounding
            self.assertLess((robot - train).abs().max().item(), 1e-5, path)


# Run with python3 -m unittest
if __name__ == "__main__":
    # Standard test runner
    unittest.main()
