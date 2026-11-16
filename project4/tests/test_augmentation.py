"""Augmentation: the subsets, and labels that move with flip / rotate / crop."""
import random
import unittest

import numpy as np
import PIL.Image

# Import path for the project modules
import _helpers  # noqa: F401
# Code under test
from augmentation import MAX_METHODS, METHODS, SUBSETS, Augmenter


# Black image with one white dot (the label) of radius 3
def dot_image(x, y, size=224):
    # Black canvas
    a = np.zeros((size, size, 3), np.uint8)
    # Pixel grid
    yy, xx = np.mgrid[0:size, 0:size]
    # White disc around (x, y)
    a[(xx - x) ** 2 + (yy - y) ** 2 <= 9] = 255
    # As a PIL image (the augmenter works on PIL)
    return PIL.Image.fromarray(a)


# Brightness-weighted centre of the dot after augmentation
def dot_centre(image):
    # Grey values as float
    g = np.asarray(image.convert("L"), np.float64)
    # Only clearly bright pixels count
    g[g < 60] = 0
    # Pixel grid
    yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    # Weighted mean position (+0.5 would be the pixel centre; labels use the same convention as the dot)
    return (g * xx).sum() / g.sum(), (g * yy).sum() / g.sum()


# Tests of augmentation.py
class AugmentationTest(unittest.TestCase):

    # 1 original + 7 single methods + 21 pairs
    def test_subsets(self):
        # 29 subsets in total
        self.assertEqual(len(SUBSETS), 1 + 7 + 21)
        # The original image comes first
        self.assertEqual(SUBSETS[0], ())
        # None has more than MAX_METHODS methods, all are known names
        self.assertTrue(all(len(s) <= MAX_METHODS and set(s) <= set(METHODS) for s in SUBSETS))
        # No duplicates
        self.assertEqual(len(set(SUBSETS)), len(SUBSETS))
        # Switch on -> 29 variants, off -> 1
        self.assertEqual((Augmenter(True).n_variants, Augmenter(False).n_variants), (29, 1))

    # The label follows the picture under every geometric subset
    def test_geometry_moves_label(self):
        # Fixed randomness
        random.seed(0)
        # Augmenter with the default ranges
        aug = Augmenter(enabled=True)
        # Subsets made only of methods that move pixels (blur / 144p would just smear the dot)
        for subset in [s for s in SUBSETS[1:] if set(s) <= {"flip", "rotate", "crop"}]:
            # Several random parameters per subset
            for trial in range(5):
                # Label away from the border so the dot is not cut off
                x, y = random.uniform(30, 194), random.uniform(30, 194)
                # Apply the augmentation to the dot image
                out, nx, ny = aug(dot_image(x, y), x, y, subset)
                # Where the dot really is now
                cx, cy = dot_centre(out)
                # Label and dot agree within 1.5 px (flip is ~1 px off: w - x vs w - 1 - x pixel-centre convention)
                self.assertLess(np.hypot(cx - nx, cy - ny), 1.5, "%s trial %d" % (subset, trial))

    # Every subset runs and keeps the size and a label inside the image
    def test_every_subset_runs(self):
        # Fixed randomness
        random.seed(1)
        # Augmenter with the default ranges
        aug = Augmenter(enabled=True)
        # Each of the 29 subsets by index
        for i in range(len(SUBSETS)):
            # Augment a dot image
            out, nx, ny = aug(dot_image(112, 80), 112.0, 80.0, i)
            # Same size as the input
            self.assertEqual(out.size, (224, 224))
            # Label still inside the image
            self.assertTrue(0 <= nx < 224 and 0 <= ny < 224, SUBSETS[i])


# Run with python3 -m unittest
if __name__ == "__main__":
    # Standard test runner
    unittest.main()
