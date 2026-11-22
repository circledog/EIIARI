"""studio: a password-protected web workspace for the JetBot.

Log in from a browser on the LAN, pick a task, set its parameters, start it
and watch the live picture. See ``studio/README.md``.

The web process (tornado) never imports torch; every task runs in its own
runner process that owns the camera and the motors (``studio/runner.py``).

Python 3.6 compatible (runs on the Jetson Nano).
"""
import os

# Version string shown in the page footer
__version__ = "0.1.0"
# Folder of this package
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
# Repository root (parent of studio/)
REPO_DIR = os.path.dirname(PACKAGE_DIR)
# project4/ folder whose modules the tasks reuse
PROJECT4_DIR = os.path.join(REPO_DIR, "project4")
