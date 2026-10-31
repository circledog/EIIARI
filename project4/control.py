"""Motor control: PD + bias steering with differential ("antagonistic") wheel drive.

Follows the slides (P.67-P.75):

    angle    = arctan2(x, y)                                  # direction error θ
    pid      = angle * Kp + (angle - angle_last) * Kd         # P + D
    steering = pid + bias                                     # motor tolerance fix
    left     = clip(speed + steering, 0, 1)
    right    = clip(speed - steering, 0, 1)

The same steering term is *added* to one wheel and *subtracted* from the other,
so the two motors work against each other to turn the robot -- positive
steering turns right, negative turns left (P.73).

``x, y`` are the model outputs converted like the slides' ``execute()``:
    x = xy[0]
    y = (0.5 - xy[1]) / 2.0

Python 3.6 compatible (runs on the Jetson Nano).
"""
import math

__all__ = ["PDController", "model_output_to_xy"]


def model_output_to_xy(xy):
    """Model output (normalised x, y in [-1, 1]) -> (x, y) used for the angle."""
    return float(xy[0]), (0.5 - float(xy[1])) / 2.0


class PDController(object):
    """Stateful PD + bias controller.

    Parameters mirror the slide's sliders:
        speed  (speed gain, 0~1)      kp (steering gain, 0~1)
        kd     (steering kd, 0~0.5)   bias (steering bias, -0.3~0.3)
    Change them at any time (e.g. from widgets) by assigning the attributes.
    """

    def __init__(self, speed=0.0, kp=0.2, kd=0.0, bias=0.0, max_output=1.0):
        self.speed = speed
        self.kp = kp
        self.kd = kd
        self.bias = bias
        self.max_output = max_output
        self.reset()

    def reset(self):
        """Forget the previous angle (call when (re)starting or switching model)."""
        self.angle_last = 0.0
        self.last = {"x": 0.0, "y": 0.0, "angle": 0.0, "p": 0.0, "d": 0.0,
                     "steering": 0.0, "speed": 0.0, "left": 0.0, "right": 0.0}

    def update(self, x, y):
        """One control step from the target point. Returns (left, right) motor values.

        Full breakdown (angle, P term, D term, steering, ...) is kept in ``self.last``.
        """
        angle = math.atan2(x, y)
        p = angle * self.kp
        d = (angle - self.angle_last) * self.kd
        self.angle_last = angle
        steering = p + d + self.bias
        left = self._clip(self.speed + steering)
        right = self._clip(self.speed - steering)
        self.last = {"x": x, "y": y, "angle": angle, "p": p, "d": d,
                     "steering": steering, "speed": self.speed,
                     "left": left, "right": right}
        return left, right

    def step_from_output(self, xy):
        """Convenience: raw model output -> (left, right)."""
        x, y = model_output_to_xy(xy)
        return self.update(x, y)

    def _clip(self, v):
        return max(min(v, self.max_output), 0.0)
