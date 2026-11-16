"""File-level rules: ASCII-only code and JSON (the JetBot container has an ASCII locale),
and a notebook that matches its generator."""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

# Import path and project folder
import _helpers
# Code under test
from loader import read_meta, write_meta


# Tests over the project's files
class FilesTest(unittest.TestCase):

    # Every .py file is pure ASCII (comments included)
    def test_python_files_ascii(self):
        # Project modules, tests and tools
        files = glob.glob(os.path.join(_helpers.ROOT, "*.py")) + glob.glob(os.path.join(_helpers.ROOT, "*", "*.py"))
        # Files with a byte above 127 (fake_tensorrt is a third-party stand-in, still checked)
        bad = [f for f in files if any(b > 127 for b in open(f, "rb").read())]
        # None allowed, except the notebook generator (its Chinese markdown goes into the notebook only)
        self.assertEqual([os.path.relpath(f, _helpers.ROOT) for f in bad if not f.endswith("make_train_experiments.py")],
                         [])

    # .json files are written as ASCII even with non-ASCII content
    def test_meta_ascii(self):
        # Temporary folder
        tmp = tempfile.mkdtemp()
        # Clean up afterwards
        self.addCleanup(shutil.rmtree, tmp)
        # Checkpoint path whose sidecar is written
        path = os.path.join(tmp, "m.pth")
        # Metadata with Chinese text
        write_meta(path, {"message": "\u6700\u5927\u5dee"})
        # Raw bytes of the .json
        raw = open(os.path.join(tmp, "m.json"), "rb").read()
        # Only ASCII bytes
        self.assertTrue(all(b < 128 for b in raw))
        # Reads back to the same text
        self.assertEqual(read_meta(path)["message"], "\u6700\u5927\u5dee")

    # train_experiments.ipynb is exactly what tools/make_train_experiments.py writes
    def test_notebook_matches_generator(self):
        # Temporary output
        tmp = tempfile.mkdtemp()
        # Clean up afterwards
        self.addCleanup(shutil.rmtree, tmp)
        # Generated notebook
        out = os.path.join(tmp, "nb.ipynb")
        # Run the generator
        subprocess.check_call([sys.executable, os.path.join(_helpers.ROOT, "tools", "make_train_experiments.py"), out],
                              stdout=subprocess.DEVNULL)
        # Cell sources of the generated notebook
        new = [c["source"] for c in json.load(open(out, encoding="utf-8"))["cells"]]
        # Cell sources of the committed notebook
        old = [c["source"] for c in json.load(open(os.path.join(_helpers.ROOT, "train_experiments.ipynb"),
                                                   encoding="utf-8"))["cells"]]
        # Identical (regenerate the notebook after editing the generator)
        self.assertEqual(new, old)


# Run with python3 -m unittest
if __name__ == "__main__":
    # Standard test runner
    unittest.main()
