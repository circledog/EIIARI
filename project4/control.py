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

Two optional extensions for sharp turns (hairpins); both are OFF by default,
and with them off the output is exactly the slide formula above:

* ``turn_slowdown`` k (0~1): base speed drops as the turn gets sharper,
      speed_eff = speed * max(0, 1 - k * |angle|)
  so the robot slows down in curves and keeps full speed on straights.
* ``allow_reverse``: the lower motor limit becomes ``-max_output`` instead of 0,
  so the inner wheel may spin backwards and the robot can pivot tightly.

``x, y`` are the model outputs converted like the slides' ``execute()``:
    x = xy[0]
    y = (0.5 - xy[1]) / 2.0

Python 3.6 compatible (runs on the Jetson Nano).
"""
import math

# Public names of this module
__all__ = ["PDController", "model_output_to_xy"]


# Convert the raw network output into the (x, y) pair used for the steering angle
def model_output_to_xy(xy):
    """Model output (normalised x, y in [-1, 1]) -> (x, y) used for the angle."""
    # x stays as is; y is remapped exactly like the slides' execute()
    return float(xy[0]), (0.5 - float(xy[1])) / 2.0


# Controller that turns a target point into left / right motor values
class PDController(object):
    """Stateful PD + bias controller.

    Parameters mirror the slide's sliders:
        speed  (speed gain, 0~1)      kp (steering gain, 0~1)
        kd     (steering kd, 0~0.5)   bias (steering bias, -0.3~0.3)
    Change them at any time (e.g. from widgets) by assigning the attributes.
    """

    # Store the gains and options, then clear the controller state
    def __init__(self, speed=0.0, kp=0.2, kd=0.0, bias=0.0, max_output=1.0,
                 turn_slowdown=0.0, allow_reverse=False):
        # Base forward speed (speed gain slider)
        self.speed = speed
        # Proportional gain on the angle (steering gain slider)
        self.kp = kp
        # Derivative gain on the change of angle (steering kd slider)
        self.kd = kd
        # Constant offset that compensates for unequal motors (steering bias slider)
        self.bias = bias
        # Largest absolute value sent to a motor
        self.max_output = max_output
        # Hairpin option: how much the speed drops in sharp turns (0 = off)
        self.turn_slowdown = turn_slowdown
        # Hairpin option: let the inner wheel spin backwards
        self.allow_reverse = allow_reverse
        # Start with a clean state
        self.reset()

    # Clear the remembered angle and the last readings
    def reset(self):
        """Forget the previous angle (call when (re)starting or switching model)."""
        # No previous angle yet, so the first D term is computed against 0
        self.angle_last = 0.0
        # Readings shown in the UI before the first update
        self.last = {"x": 0.0, "y": 0.0, "angle": 0.0, "p": 0.0, "d": 0.0,
                     "steering": 0.0, "speed": 0.0, "speed_eff": 0.0, "left": 0.0, "right": 0.0}

    # Run one control step for a target point
    def update(self, x, y):
        """One control step from the target point. Returns (left, right) motor values.

        Full breakdown (angle, P term, D term, steering, ...) is kept in ``self.last``.
        """
        # Direction error: angle between straight ahead and the target point
        angle = math.atan2(x, y)
        # Proportional term
        p = angle * self.kp
        # Derivative term: how fast the angle is changing
        d = (angle - self.angle_last) * self.kd
        # Remember this angle for the next D term
        self.angle_last = angle
        # Total steering = P + D + bias
        steering = p + d + self.bias
        # Start from the base speed
        speed = self.speed
        # Hairpin option: slow down when the turn is sharp
        if self.turn_slowdown > 0:
            # Scale the speed down with |angle|, never below 0
            speed = speed * max(0.0, 1.0 - self.turn_slowdown * abs(angle))
        # Left wheel gets speed + steering (positive steering turns right)
        left = self._clip(speed + steering)
        # Right wheel gets speed - steering
        right = self._clip(speed - steering)
        # Keep every intermediate value for the UI and the CSV log
        self.last = {"x": x, "y": y, "angle": angle, "p": p, "d": d,
                     "steering": steering, "speed": self.speed, "speed_eff": speed,
                     "left": left, "right": right}
        # Motor values to send
        return left, right

    # Shortcut that takes the raw network output directly
    def step_from_output(self, xy):
        """Convenience: raw model output -> (left, right)."""
        # Convert the network output to (x, y)
        x, y = model_output_to_xy(xy)
        # Run the normal control step
        return self.update(x, y)

    # Limit a motor value to the allowed range
    def _clip(self, v):
        # Lower limit is -max when reversing is allowed, otherwise 0
        low = -self.max_output if self.allow_reverse else 0.0
        # Clamp between the lower limit and max_output
        return max(min(v, self.max_output), low)
