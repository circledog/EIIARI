"""Metrics: steering angle matches the robot's controller; perfect model scores 0."""
import unittest

import numpy as np
import torch

# Import path for the project modules
import _helpers  # noqa: F401
# Code under test
from metrics import evaluate, steering_angle
# The robot's steering formula
from control import SteeringController


# Network stand-in that returns fixed predictions
class Fixed(torch.nn.Module):

    # Remember the outputs
    def __init__(self, outputs):
        # Initialise nn.Module
        super(Fixed, self).__init__()
        # (N, 2) predictions, returned batch by batch
        self.outputs = outputs
        # Position in the outputs
        self.i = 0

    # Return the next batch of predictions
    def forward(self, images):
        # Batch size
        n = images.shape[0]
        # Slice of the stored outputs
        out = self.outputs[self.i:self.i + n]
        # Move on
        self.i += n
        # The predictions
        return out


# Tests of metrics.py
class MetricsTest(unittest.TestCase):

    # Vectorised angle == control.py's 'teammate' angle wherever the robot drives
    def test_angle_matches_controller(self):
        # Controller with the default (teammate) formula
        ctrl = SteeringController(formula="teammate")
        # Grid of targets over the image, including outside values
        for x in np.linspace(-1.2, 1.2, 25):
            # Rows from top to just above the stop line
            for y in np.linspace(-1.2, 0.9, 25):
                # Robot's angle and status
                ref, status = ctrl.raw_angle(float(x), float(y))
                # Same value from the metric
                self.assertAlmostEqual(float(steering_angle(np.array([x]), np.array([y]))[0]), ref, places=9)

    # Prediction == label gives zero error; a known x shift gives the expected pixels
    def test_evaluate(self):
        # Four labels
        labels = torch.tensor([[0.0, -0.5], [0.3, 0.0], [-0.4, 0.2], [0.1, -0.9]])
        # Loader of (dummy image, label) batches of 2
        loader = [(torch.zeros(2, 3, 4, 4), labels[:2]), (torch.zeros(2, 3, 4, 4), labels[2:])]
        # Perfect predictions
        m = evaluate(Fixed(labels.clone()), loader, "cpu")
        # Every error is zero
        self.assertEqual((m["mean_px"], m["angle_deg"], m["mse"]), (0.0, 0.0, 0.0))
        # Predictions shifted right by 0.1 (= 11.2 px)
        m = evaluate(Fixed(labels + torch.tensor([0.1, 0.0])), loader, "cpu")
        # x error 11.2 px, no y error
        self.assertAlmostEqual(m["mae_x_px"], 11.2, places=4)
        # y error zero
        self.assertAlmostEqual(m["mae_y_px"], 0.0, places=6)
        # A sideways shift changes the steering angle
        self.assertGreater(m["angle_deg"], 0.0)


# Run with python3 -m unittest
if __name__ == "__main__":
    # Standard test runner
    unittest.main()
