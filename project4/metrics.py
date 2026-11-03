"""Evaluation metrics for the (x, y) regression: MSE plus errors in pixels.

The labels are normalised to [-1, 1] on 224x224 images, so one normalised
unit is 112 px. Pixel errors are easier to judge than MSE: "the target point
is off by 15 px on average" means more than "MSE 0.02".

``evaluate`` returns
    mse         mean squared error per image over (x, y)  (the training loss)
    mae_x_px    mean absolute x error in pixels
    mae_y_px    mean absolute y error in pixels
    mean_px     mean Euclidean distance in pixels
    p95_px      95th percentile of the distance in pixels (the bad cases)

Python 3.6 compatible.
"""
import numpy as np
import torch

# Public names of this module
__all__ = ["evaluate", "PX_PER_UNIT"]

# Pixels per normalised unit: half the 224-px input size
PX_PER_UNIT = 112.0


# All metrics of a model on one DataLoader
def evaluate(model, loader, device):
    """Dict of mse / mae_x_px / mae_y_px / mean_px / p95_px over ``loader``."""
    # Evaluation mode (BatchNorm uses running statistics, no dropout)
    model.eval()
    # Per-image errors of every batch
    errors = []
    # No gradients needed for evaluation
    with torch.no_grad():
        # Every batch
        for images, labels in loader:
            # Prediction minus label, per image and coordinate
            diff = model(images.to(device)).float().cpu() - labels.float()
            # Keep them for the statistics
            errors.append(diff.numpy())
    # Guard against an empty loader
    if not errors:
        # No numbers to report
        return {"mse": float("nan"), "mae_x_px": float("nan"), "mae_y_px": float("nan"),
                "mean_px": float("nan"), "p95_px": float("nan")}
    # (N, 2) array of normalised errors
    e = np.concatenate(errors, axis=0)
    # Euclidean distance per image in pixels
    dist = np.hypot(e[:, 0], e[:, 1]) * PX_PER_UNIT
    # Every metric as a plain float (JSON friendly)
    return {"mse": float(np.mean(e ** 2)),
            "mae_x_px": float(np.mean(np.abs(e[:, 0])) * PX_PER_UNIT),
            "mae_y_px": float(np.mean(np.abs(e[:, 1])) * PX_PER_UNIT),
            "mean_px": float(np.mean(dist)),
            "p95_px": float(np.percentile(dist, 95))}
