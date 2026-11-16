"""Code version: comparison rules used by live_demo."""
import unittest

# Import path for the project modules
import _helpers  # noqa: F401
# Code under test
from version import code_version, compare

# Two made-up commits
A, B = "a" * 40, "b" * 40


# Tests of version.py
class VersionTest(unittest.TestCase):

    # Same / different / unknown / dirty
    def test_compare(self):
        # Identical clean commits
        self.assertEqual(compare({"commit": A, "dirty": False}, {"commit": A, "dirty": False})[0], "same")
        # Different commits
        self.assertEqual(compare({"commit": A, "dirty": False}, {"commit": B, "dirty": False})[0], "different")
        # Same commit with local edits
        self.assertEqual(compare({"commit": A, "dirty": True}, {"commit": A, "dirty": False})[0], "different")
        # Old .json without a version
        self.assertEqual(compare(None, {"commit": A, "dirty": False})[0], "unknown")
        # Code copied without git
        self.assertEqual(compare({"commit": A, "dirty": False}, {"commit": None, "dirty": None})[0], "unknown")

    # code_version always returns both keys
    def test_code_version_shape(self):
        # Version of this checkout (or unknown without git)
        v = code_version()
        # Both keys present
        self.assertEqual(sorted(v), ["commit", "dirty"])


# Run with python3 -m unittest
if __name__ == "__main__":
    # Standard test runner
    unittest.main()
