"""A one-item mailbox that only ever holds the newest value.

``put()`` overwrites the previous item and bumps a sequence number; readers
call ``get()`` or ``wait_newer(seq)`` and always receive the newest item.
Items that nobody picked up in time are simply dropped, so a slow reader can
never build a backlog (unlike a FIFO queue, whose delay and memory grow
while the reader is slower than the writer).

Used between the camera thread and the control loop, and between the
control loop and the display thread (``drive.py``, ``live_demo.ipynb``).

Python 3.6 compatible (runs on the Jetson Nano).
"""
import threading

# Public names of this module
__all__ = ["LatestSlot"]


# Single-slot mailbox: writers overwrite, readers get the newest item
class LatestSlot(object):
    """Holds only the newest item; see the module docstring."""

    # Start empty: sequence 0 means "nothing written yet"
    def __init__(self):
        # Condition variable guarding the slot and waking waiting readers
        self._cond = threading.Condition()
        # Number of items written so far
        self._seq = 0
        # The newest item
        self._item = None
        # Set by close(): waiting readers return at once
        self._closed = False

    # Replace the item with a newer one
    def put(self, item):
        """Store ``item`` (dropping the previous one); returns its sequence number."""
        # One writer / reader at a time
        with self._cond:
            # Next sequence number
            self._seq += 1
            # Overwrite: the old item is dropped here
            self._item = item
            # Wake every reader waiting for something newer
            self._cond.notify_all()
            # Sequence number of the stored item
            return self._seq

    # Read the newest item without waiting
    def get(self):
        """Return ``(seq, item)``; ``(0, None)`` before the first ``put``."""
        # Read both values consistently
        with self._cond:
            # Sequence number and item
            return self._seq, self._item

    # Wait until an item newer than `seq` exists
    def wait_newer(self, seq, timeout=None):
        """Return ``(seq, item)`` of the newest item once its seq > ``seq``.

        Returns ``None`` on timeout or after ``close()``. When several items
        were written meanwhile, only the newest one is returned.
        """
        # Wait under the condition's lock
        with self._cond:
            # Sleep until a newer item arrives, the slot is closed or the time is up
            self._cond.wait_for(lambda: self._seq > seq or self._closed, timeout)
            # Nothing newer (timeout or closed)
            if self._seq <= seq:
                # Tell the caller nothing new arrived
                return None
            # The newest item
            return self._seq, self._item

    # Wake all waiting readers for good (shutdown)
    def close(self):
        """Make every current and future ``wait_newer`` return immediately."""
        # Change the flag under the lock
        with self._cond:
            # Mark as closed
            self._closed = True
            # Wake every waiting reader
            self._cond.notify_all()

    # Whether close() was called
    @property
    def closed(self):
        """True after ``close()``."""
        # Current flag
        return self._closed
