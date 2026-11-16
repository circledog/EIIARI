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
    angle_deg   mean absolute steering-angle error in degrees
    p95_angle_deg  95th percentile of the steering-angle error

The steering angle uses the robot's default formula (control.py,
'teammate'): ``atan2(x, max(0.08, 2 * (0.95 - v)))`` with
``v = (y + 1) / 2``. It shows what an (x, y) error does to the steering:
a y error far ahead matters little, an x error close to the robot a lot.
Targets at the very bottom (where the robot would stop) are clamped to the
stop line v = 0.98, so every image gets an angle.

Python 3.6 compatible.
"""
import numpy as np
import torch

# Public names of this module
__all__ = ["evaluate", "steering_angle", "PX_PER_UNIT"]

# Pixels per normalised unit: half the 224-px input size
PX_PER_UNIT = 112.0


# Lowest image row (0 top .. 1 bottom) used for the angle; control.py stops below it
STOP_V = 0.98


# Steering angle (rad) of normalised targets, vectorised version of control.py's 'teammate' formula
def steering_angle(x, y):
    """``x``, ``y``: arrays in [-1, 1] -> angle in radians (positive = right)."""
    # Targets clamped into the image, as control.py does
    x, y = np.clip(x, -1.0, 1.0), np.clip(y, -1.0, 1.0)
    # Vertical position 0 (top) .. 1 (bottom), at most the stop line
    v = np.minimum((y + 1.0) / 2.0, STOP_V)
    # Angle between straight ahead and the target
    return np.arctan2(x, np.maximum(0.08, 2.0 * (0.95 - v)))


# All metrics of a model on one DataLoader
def evaluate(model, loader, device):
    """Dict of mse / mae_x_px / mae_y_px / mean_px / p95_px / angle_deg / p95_angle_deg over ``loader``."""
    # Evaluation mode (BatchNorm uses running statistics, no dropout)
    model.eval()
    # Per-image predictions and labels of every batch
    preds, targets = [], []
    # No gradients needed for evaluation
    with torch.no_grad():
        # Every batch
        for images, labels in loader:
            # Predictions as float32 on the CPU
            preds.append(model(images.to(device)).float().cpu().numpy())
            # Matching labels
            targets.append(labels.float().numpy())
    # Guard against an empty loader
    if not preds:
        # No numbers to report
        return {k: float("nan") for k in ("mse", "mae_x_px", "mae_y_px", "mean_px", "p95_px",
                                          "angle_deg", "p95_angle_deg")}
    # (N, 2) predictions and labels
    p, t = np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)
    # (N, 2) array of normalised errors
    e = p - t
    # Euclidean distance per image in pixels
    dist = np.hypot(e[:, 0], e[:, 1]) * PX_PER_UNIT
    # Steering-angle error per image in degrees
    angle = np.degrees(np.abs(steering_angle(p[:, 0], p[:, 1]) - steering_angle(t[:, 0], t[:, 1])))
    # Every metric as a plain float (JSON friendly)
    return {"mse": float(np.mean(e ** 2)),
            "mae_x_px": float(np.mean(np.abs(e[:, 0])) * PX_PER_UNIT),
            "mae_y_px": float(np.mean(np.abs(e[:, 1])) * PX_PER_UNIT),
            "mean_px": float(np.mean(dist)),
            "p95_px": float(np.percentile(dist, 95)),
            "angle_deg": float(np.mean(angle)),
            "p95_angle_deg": float(np.percentile(angle, 95))}
