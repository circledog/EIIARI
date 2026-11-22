"""Last-resort motor stop when a runner process died without stopping.

The JetBot's motor driver (PCA9685 / Adafruit MotorHAT on I2C bus 1) keeps
the last speed when the program that set it dies, so a crashed runner would
leave the robot driving. The web process runs this module in a fresh
process when a runner exits unexpectedly::

    python3 -m studio.motor_guard

It talks to the driver directly (motors 1 and 2, like ``jetbot.Robot``),
which is fast and does not import torch; if that fails it falls back to
``jetbot.Robot().stop()``.

**Not yet tested on the robot** (sandbox has no I2C).

Python 3.6 compatible (runs on the Jetson Nano).
"""
import sys


# Stop both motors through the driver chip
def stop_direct():
    """Release motors 1 and 2 on the MotorHAT at I2C bus 1."""
    from Adafruit_MotorHAT import Adafruit_MotorHAT
    # Same bus as jetbot.Robot
    hat = Adafruit_MotorHAT(i2c_bus=1)
    # jetbot's default left / right channels
    for channel in (1, 2):
        # The motor on that channel
        motor = hat.getMotor(channel)
        # Speed 0
        motor.setSpeed(0)
        # Coast (no drive current)
        motor.run(Adafruit_MotorHAT.RELEASE)


# Stop the motors through the jetbot library
def stop_jetbot():
    """``jetbot.Robot().stop()`` (slower: imports the jetbot package)."""
    from jetbot import Robot
    # New driver instance and stop
    Robot().stop()


# Try the direct way, then the jetbot way
def emergency_stop():
    """Return ``(ok, message)``."""
    # Collected error messages
    errors = []
    # Both methods in order
    for name, fn in (("direct", stop_direct), ("jetbot", stop_jetbot)):
        # Each may fail (no I2C, no library)
        try:
            # Stop the motors
            fn()
            # Done
            return True, "motors stopped (%s)" % name
        # Remember why and try the next one
        except Exception as e:
            # Short reason
            errors.append("%s: %r" % (name, e))
    # Nothing worked
    return False, "; ".join(errors)


# Command-line entry point
def main():
    """Exit status 0 when the motors were stopped."""
    # Try to stop
    ok, message = emergency_stop()
    # Report on stdout (ends up in the server log)
    print(message)
    # Status for the caller
    return 0 if ok else 1


# Run as a script
if __name__ == "__main__":
    # Exit with the result
    sys.exit(main())
