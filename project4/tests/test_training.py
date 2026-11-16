"""Training recipe: which layers train, frozen BatchNorm, early-stop switch."""
import copy
import shutil
import tempfile
import unittest
from unittest import mock

import torch

# Import path and synthetic data
import _helpers
# Code under test
import training
# Network registry
from loader import ARCHS, build_model
# Small loaders for the tests
from data_loader import make_loaders


# Tests of training.py through each network's train()
class TrainingTest(unittest.TestCase):

    # One tiny dataset for all tests
    @classmethod
    def setUpClass(cls):
        # Temporary folder
        cls.tmp = tempfile.mkdtemp()
        # 4 images per split
        root = _helpers.make_split_dataset(cls.tmp, n=4)
        # Batches of 2, no worker processes, no augmentation
        cls.train, cls.val, _, _ = make_loaders(root, batch_size=2, num_workers=0)

    # Remove the temporary folder
    @classmethod
    def tearDownClass(cls):
        # Delete everything that was written
        shutil.rmtree(cls.tmp)

    # Train one epoch and report which parameters / BN statistics changed
    def _train_once(self, arch):
        # Fixed initialisation
        torch.manual_seed(0)
        # Random-init network (no download)
        model = build_model(arch, pretrained=False)
        # Weights and buffers before training
        before = copy.deepcopy(model.state_dict())
        # requires_grad flags before training
        flags = {n: p.requires_grad for n, p in model.named_parameters()}
        # One epoch with the network's own routine
        ARCHS[arch].train(model, self.train, self.val, torch.device("cpu"), 1, 1e-2, "test")
        # State after training
        after = model.state_dict()
        # Names of the entries that changed
        changed = {k for k in before if not torch.equal(before[k], after[k])}
        # Model, changed names and the flags before
        return model, changed, flags

    # Only TRAINABLE parameters change, BN statistics never, flags are restored
    def test_partial_freezing(self):
        # Every network
        for arch, module in sorted(ARCHS.items()):
            # Train one epoch
            model, changed, flags = self._train_once(arch)
            # Prefixes this network trains
            prefixes = tuple(module.TRAINABLE)
            # Some parameters were trained
            self.assertTrue(changed, arch)
            # Every changed entry belongs to a trained prefix
            self.assertTrue(all(k.startswith(prefixes) for k in changed),
                            (arch, sorted(k for k in changed if not k.startswith(prefixes))[:5]))
            # BatchNorm running statistics are frozen
            self.assertFalse([k for k in changed if "running_" in k or "num_batches" in k], arch)
            # requires_grad flags are the same as before training
            self.assertEqual(flags, {n: p.requires_grad for n, p in model.named_parameters()}, arch)

    # DSNT's fixed upsample / expect layers stay fixed and frozen
    def test_dsnt_fixed_layers(self):
        # Both DSNT variants
        for arch in ("resnet18_dsnt", "resnet18_coord_dsnt"):
            # Train one epoch
            model, changed, _ = self._train_once(arch)
            # The fixed matrices did not change
            self.assertFalse([k for k in changed if k.startswith(("upsample", "expect"))], arch)
            # And they are still frozen
            self.assertFalse(any(p.requires_grad for n, p in model.named_parameters()
                                 if n.startswith(("upsample", "expect"))), arch)

    # early_stop=False trains every epoch; True stops after `patience` epochs without progress
    def test_early_stop_switch(self):
        # Validation result that never improves after epoch 1
        flat = {"mse": 1.0, "mean_px": 10.0, "mae_x_px": 5.0, "mae_y_px": 5.0, "p95_px": 20.0,
                "angle_deg": 1.0, "p95_angle_deg": 2.0}
        # Both settings
        for early_stop, expected in ((False, 6), (True, 3)):
            # Small network
            model = build_model("mobilenet_v2", pretrained=False)
            # Replace the validation pass by the constant result
            with mock.patch.object(training, "evaluate", return_value=flat):
                # 6 epochs, patience 2, minimum 1
                best, history = training.fit(model, self.train, self.val, torch.device("cpu"), 6, 1e-3, "test",
                                             ARCHS["mobilenet_v2"].TRAINABLE, ARCHS["mobilenet_v2"].HEAD,
                                             patience=2, min_epochs=1, early_stop=early_stop)
            # Off: all 6 epochs; on: epoch 1 best, stop at epoch 3
            self.assertEqual(len(history), expected, "early_stop=%s" % early_stop)
            # The best epoch is the first
            self.assertEqual(best["best_epoch"], 1)


# Run with python3 -m unittest
if __name__ == "__main__":
    # Standard test runner
    unittest.main()
