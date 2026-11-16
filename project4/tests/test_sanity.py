"""Sanity frames: stored with the model's preprocessing and passed by the same model."""
import base64
import shutil
import tempfile
import unittest

import cv2
import numpy as np
import torch

# Import path and synthetic data
import _helpers
# Code under test
import sanity
# Network building and the inference wrapper
from loader import LoadedModel, build_model


# Tests of sanity.py
class SanityTest(unittest.TestCase):

    # Small dataset and a random-init network
    def setUp(self):
        # Temporary folder
        self.tmp = tempfile.mkdtemp()
        # Three images
        self.paths = _helpers.make_dataset(self.tmp, n=3)
        # Fixed initialisation
        torch.manual_seed(0)
        # Evaluation-mode network
        self.model = build_model("mobilenet_v2", pretrained=False).eval()

    # Remove the temporary folder
    def tearDown(self):
        # Delete everything that was written
        shutil.rmtree(self.tmp)

    # Decode the stored frames
    @staticmethod
    def frames(data):
        # BGR arrays from the PNG strings
        return [cv2.imdecode(np.frombuffer(base64.b64decode(s["png_b64"]), np.uint8), cv2.IMREAD_COLOR)
                for s in data["samples"]]

    # lanes=True stores marked frames, lanes=False the plain ones; both pass the check
    def test_lanes_and_check(self):
        # Sanity data both ways
        marked, plain = (sanity.make_sanity(self.model, self.paths, lanes=v) for v in (True, False))
        # Plain frames are the resized pictures themselves
        for frame, path in zip(self.frames(plain), self.paths):
            # Same pixels (PNG is lossless)
            self.assertTrue(np.array_equal(frame, cv2.resize(cv2.imread(path), (224, 224))))
        # Marked frames differ from plain ones
        self.assertFalse(all(np.array_equal(a, b) for a, b in zip(self.frames(marked), self.frames(plain))))
        # Each set passes when checked with the same FP32 model
        for data in (marked, plain):
            # Wrapper like load_model returns
            m = LoadedModel(self.model, "torch", torch.float32, torch.device("cpu"), "x.pth", {"sanity": data})
            # Comparison result
            result = sanity.check(m)
            # Passed, with an ASCII message
            self.assertEqual(result["status"], "passed", result["message"])
            # Message is English / ASCII
            self.assertTrue(all(ord(c) < 128 for c in result["message"]))


# Run with python3 -m unittest
if __name__ == "__main__":
    # Standard test runner
    unittest.main()
