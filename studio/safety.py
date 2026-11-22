"""Safety stop inside the runner process (TASKS M1-7).

The runner is the only process that drives the motors, so the stop logic
lives here, close to the motors:

* **Link watchdog**: the web process forwards a heartbeat every 0.1 s while
  at least one open task page answers. When no heartbeat arrived for
  ``link_timeout`` seconds (page closed, Wi-Fi lost, web process hung) the
  motors are stopped. Motion commands are refused while the link is down.
  Default 5 s (user's decision, 2026-11-22): a short Wi-Fi stall must not
  stop a lane-following robot; set it lower for manual driving.
* **Web process gone**: the runner's socket closes -> stop at once, exit.
* **E-stop**: the page's emergency button (or the space bar) -> stop at once.
* **SIGTERM / SIGINT** -> stop, exit.

If the runner itself dies the web process starts ``studio.motor_guard``.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import threading
import time


# Heartbeat-based stop: no heartbeat for `timeout` seconds -> on_lost()
class LinkWatchdog(object):
    """Calls ``on_lost(seconds_since_heartbeat)`` once per link loss."""

    # Store the callback; the link counts as down until the first heartbeat
    def __init__(self, on_lost, timeout=5.0, period=0.02, clock=time.monotonic):
        # Called when the link is lost
        self.on_lost = on_lost
        # Seconds without heartbeat that count as lost (may change at runtime)
        self.timeout = timeout
        # Check interval
        self.period = period
        # Time source (replaceable in tests)
        self.clock = clock
        # Time of the last heartbeat (None = never)
        self.last = None
        # Whether on_lost was already called for the current loss
        self._reported = True
        # Set by stop()
        self._stop = threading.Event()
        # Checker thread
        self._thread = None

    # A heartbeat arrived
    def beat(self):
        """Record a heartbeat; re-arms the loss report."""
        # Remember the time
        self.last = self.clock()
        # The next loss will be reported again
        self._reported = False

    # Whether the link is currently up
    def alive(self):
        """True when a heartbeat arrived within ``timeout``."""
        # Never had a heartbeat: down
        if self.last is None:
            # Down
            return False
        # Recent enough
        return self.clock() - self.last <= self.timeout

    # One check (also called directly by tests)
    def check(self):
        """Call ``on_lost`` if the link just went down; returns True if it did."""
        # Down, and not reported yet for this loss
        if not self._reported and not self.alive():
            # Report only once per loss
            self._reported = True
            # Seconds since the last heartbeat
            self.on_lost(self.clock() - self.last)
            # Reported now
            return True
        # Nothing new
        return False

    # Thread body
    def _run(self):
        # Until stop()
        while not self._stop.wait(self.period):
            # One check
            self.check()

    # Start checking in the background
    def start(self):
        """Start the checker thread."""
        # Daemon thread
        self._thread = threading.Thread(target=self._run, name="link-watchdog")
        # Do not block interpreter shutdown
        self._thread.daemon = True
        # Run it
        self._thread.start()
        # Allow chaining
        return self

    # Stop checking
    def stop(self):
        """End the checker thread."""
        # Wake and end the thread
        self._stop.set()
