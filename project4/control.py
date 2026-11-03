"""Motor control: target point -> steering angle -> PID -> differential wheel drive.

The network predicts a target point (x, y), both normalised to [-1, 1]
(x: left -1 .. right +1, y: top -1 .. bottom +1). The controller turns it into
left / right motor values:

1. Steering angle, one of two formulas (``formula``):

   * ``'teammate'`` (default, from the teammate's deployment bundle)::

         v     = (y + 1) / 2                        # 0 = top, 1 = bottom of the image
         angle = atan2(x, max(0.08, 2 * (0.95 - v)))

     The forward distance stays positive for every target, so the robot never
     steers the wrong way. A target at the very bottom (v >= 0.98) stops the
     robot ("target_too_close").
   * ``'slides'`` (course P.67-P.75)::

         angle = atan2(x, (0.5 - y) / 2)

     For a target below y = 0.5 the second argument turns negative: the
     angle goes past +-90 deg (far too much steering), and near x = 0 it jumps
     between +180 and -180 deg, so a tiny change of x flips the direction.

2. Low-pass filter on the angle: ``angle += alpha * (raw - angle)``
   (``angle_alpha`` = 1 switches it off).
3. PID: ``steering = kp*angle + ki*integral + kd*d(angle)/dt + bias``; the
   D term is per second, so it does not depend on the frame rate.
4. Speed: ``max_speed`` on straights, reduced in turns by ``turn_slowdown``,
   never below ``min_speed`` (and never above ``max_speed``).
5. Differential drive: ``left = speed + steering``, ``right = speed - steering``,
   limited to ``[0, max_output]`` (or ``[-max_output, max_output]`` with
   ``allow_reverse``: the inner wheel may spin backwards in hairpins).
6. Slew limit: a motor value changes by at most ``slew_per_second * dt`` per
   step, so the robot starts and turns smoothly (0 switches it off).

Bad predictions (NaN / inf, far outside the image) return status
``'invalid_prediction'`` / ``'out_of_range'`` and zero motors; ``live_demo``
then stops the robot. Every attribute can be changed at any time (sliders).

Python 3.6 compatible (runs on the Jetson Nano).
"""
import math

# Public names of this module
__all__ = ["SteeringController", "DEFAULTS", "FORMULAS"]

# Available steering-angle formulas
FORMULAS = ("teammate", "slides")
# Starting values (teammate's bundle); not tuned on the track yet
DEFAULTS = {"formula": "teammate", "kp": 0.18, "ki": 0.0, "kd": 0.008, "bias": 0.0,
            "min_speed": 0.10, "max_speed": 0.20, "turn_slowdown": 0.65, "angle_alpha": 0.35,
            "slew_per_second": 0.6, "max_output": 0.4, "allow_reverse": False}
# Largest |integral| so the I term cannot wind up
INTEGRAL_LIMIT = 0.3
# Predictions further outside the image than this are rejected
RANGE_LIMIT = 1.25
# Time step assumed for the first frame after a reset
FIRST_DT = 0.05
# Longer gaps between frames restart the controller state
MAX_GAP = 0.5


# Limit a value to [low, high]
def _clamp(v, low, high):
    # Lower bound first, then upper bound
    return max(low, min(high, v))


# Stateful controller that turns a predicted target point into motor values
class SteeringController(object):
    """Target point -> (left, right, status). See the module docstring."""

    # Store the settings, then clear the controller state
    def __init__(self, **settings):
        # Start from the defaults
        for key, value in DEFAULTS.items():
            # One attribute per setting
            setattr(self, key, value)
        # Apply the given overrides
        for key, value in settings.items():
            # Unknown names are almost certainly typos
            if key not in DEFAULTS:
                # Report the bad name
                raise TypeError("unknown controller setting %r" % key)
            # Override the default
            setattr(self, key, value)
        # Start with a clean state
        self.reset()

    # Forget filter, integral, motor ramp and timing
    def reset(self):
        """Clear the state (call when starting to drive or switching model)."""
        # Filtered angle, integral and current (ramped) motor values
        self.angle, self.integral, self.left, self.right = 0.0, 0.0, 0.0, 0.0
        # Time of the previous step (None: next step is the first)
        self.last_time = None
        # Readings shown in the UI before the first step
        self.last = {"x": 0.0, "y": 0.0, "raw_angle": 0.0, "angle": 0.0, "p": 0.0, "i": 0.0, "d": 0.0,
                     "steering": 0.0, "speed": 0.0, "left": 0.0, "right": 0.0, "dt": 0.0}

    # Steering angle of a target point with the selected formula
    def raw_angle(self, x, y):
        """(angle in rad, status) for a normalised target point."""
        # Teammate's formula: forward distance always positive
        if self.formula == "teammate":
            # Vertical position 0 (top) .. 1 (bottom), target clamped into the image
            v = (_clamp(y, -1.0, 1.0) + 1.0) / 2.0
            # Target right in front of the wheels: no sensible direction
            if v >= 0.98:
                # Ask the caller to stop
                return 0.0, "target_too_close"
            # Angle between straight ahead and the target
            return math.atan2(_clamp(x, -1.0, 1.0), max(0.08, 2.0 * (0.95 - v))), "ok"
        # Slides' formula (course P.67)
        return math.atan2(x, (0.5 - y) / 2.0), "ok"

    # One control step
    def update(self, xy, now):
        """Raw network output ``xy`` at time ``now`` (s) -> (left, right, status)."""
        # Both coordinates must be real numbers
        try:
            # Convert to floats
            x, y = float(xy[0]), float(xy[1])
        # Wrong shape or type
        except (TypeError, ValueError, IndexError):
            # Treat as a bad prediction
            return self._halt("invalid_prediction")
        # NaN / inf outputs
        if not (math.isfinite(x) and math.isfinite(y)):
            # Stop
            return self._halt("invalid_prediction")
        # Far outside the image: the network is confused
        if abs(x) > RANGE_LIMIT or abs(y) > RANGE_LIMIT:
            # Stop
            return self._halt("out_of_range")
        # Time since the previous step
        dt = FIRST_DT if self.last_time is None else now - self.last_time
        # Clock went backwards or a long pause: start over
        if dt <= 0 or dt > MAX_GAP:
            # Fresh state (motors ramp up again)
            self.reset()
            # Use the first-frame step
            dt = FIRST_DT
        # Angle of the target
        raw, status = self.raw_angle(x, y)
        # The formula asked to stop
        if status != "ok":
            # Stop
            return self._halt(status)
        # Low-pass filtered angle
        angle = self.angle + _clamp(self.angle_alpha, 0.0, 1.0) * (raw - self.angle)
        # Rate of change of the filtered angle (rad/s)
        derivative = (angle - self.angle) / dt
        # Integral of the angle, limited against wind-up
        self.integral = _clamp(self.integral + angle * dt, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
        # P, I and D terms
        p, i, d = self.kp * angle, self.ki * self.integral, self.kd * derivative
        # Total steering
        steering = p + i + d + self.bias
        # Speed reduced in turns
        speed = self.max_speed * (1.0 - self.turn_slowdown * min(1.0, abs(angle) / (math.pi / 2)))
        # Keep it within [min_speed, max_speed]
        speed = min(self.max_speed, max(self.min_speed, speed))
        # Lower motor limit: negative only when reversing is allowed
        low = -self.max_output if self.allow_reverse else 0.0
        # Wanted left motor value (positive steering turns right)
        target_left = _clamp(speed + steering, low, self.max_output)
        # Wanted right motor value
        target_right = _clamp(speed - steering, low, self.max_output)
        # Largest change allowed in this step (0 = no limit)
        limit = self.slew_per_second * dt if self.slew_per_second > 0 else float("inf")
        # Move the left motor towards its target
        self.left += _clamp(target_left - self.left, -limit, limit)
        # Move the right motor towards its target
        self.right += _clamp(target_right - self.right, -limit, limit)
        # Remember the angle and the time for the next step
        self.angle, self.last_time = angle, now
        # Every intermediate value for the UI and the CSV log
        self.last = {"x": x, "y": y, "raw_angle": raw, "angle": angle, "p": p, "i": i, "d": d,
                     "steering": steering, "speed": speed, "left": self.left, "right": self.right, "dt": dt}
        # Motor values and status
        return self.left, self.right, "ok"

    # Zero output with a reason
    def _halt(self, status):
        # Clear the state so the next valid frame starts fresh
        self.reset()
        # Stop both motors
        return 0.0, 0.0, status
