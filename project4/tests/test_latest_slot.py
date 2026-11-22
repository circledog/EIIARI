"""LatestSlot: keeps only the newest item, never queues, wait_newer times out."""
import threading
import time
import unittest

# Shared test helpers (sets the import path)
import _helpers  # noqa: F401
# Module under test
from latest_slot import LatestSlot


# Tests for the single-slot mailbox
class LatestSlotTest(unittest.TestCase):
    # 1000 puts keep only the last item
    def test_only_newest_is_kept(self):
        # Fresh slot
        slot = LatestSlot()
        # Empty slot reports sequence 0 and no item
        self.assertEqual(slot.get(), (0, None))
        # Write 1000 items
        for i in range(1000):
            # Overwrite
            slot.put(i)
        # Only the last one is there, with sequence 1000
        self.assertEqual(slot.get(), (1000, 999))
        # A reader that saw nothing gets the newest item, not the oldest
        self.assertEqual(slot.wait_newer(0, timeout=0.1), (1000, 999))

    # wait_newer returns None when nothing newer arrives in time
    def test_wait_newer_times_out(self):
        # Slot with one item
        slot = LatestSlot()
        # Sequence 1
        seq = slot.put("a")
        # Start the timer
        t0 = time.monotonic()
        # Nothing newer than seq will come
        self.assertIsNone(slot.wait_newer(seq, timeout=0.15))
        # It really waited about the timeout
        self.assertGreaterEqual(time.monotonic() - t0, 0.14)

    # wait_newer wakes up as soon as a writer puts something
    def test_wait_newer_wakes_on_put(self):
        # Fresh slot
        slot = LatestSlot()
        # Writer that puts after 50 ms
        threading.Timer(0.05, slot.put, args=("x",)).start()
        # Start the timer
        t0 = time.monotonic()
        # Wait up to 2 s
        got = slot.wait_newer(0, timeout=2.0)
        # Got the item quickly, long before the timeout
        self.assertEqual(got, (1, "x"))
        # Woke within half a second
        self.assertLess(time.monotonic() - t0, 0.5)

    # close() releases waiting readers
    def test_close_releases_waiters(self):
        # Fresh slot
        slot = LatestSlot()
        # Close it after 50 ms
        threading.Timer(0.05, slot.close).start()
        # Start the timer
        t0 = time.monotonic()
        # Would wait 5 s without close()
        self.assertIsNone(slot.wait_newer(0, timeout=5.0))
        # Released quickly
        self.assertLess(time.monotonic() - t0, 1.0)
        # Flag is set
        self.assertTrue(slot.closed)

    # Concurrent writers and readers: a reader never loses the newest value
    def test_threads_never_lose_the_newest(self):
        # Fresh slot
        slot = LatestSlot()
        # Items each writer puts
        n = 2000
        # Writer body: put (writer id, counter) items
        def writer(wid):
            # Count up
            for i in range(n):
                # Overwrite
                slot.put((wid, i))
        # Highest sequence numbers seen by the reader, in order
        seen = []
        # Set when the writers are done
        done = threading.Event()
        # Reader body: keep waiting for newer items
        def reader():
            # Last sequence seen
            seq = 0
            # Until the writers finish and nothing newer is left
            while True:
                # Wait briefly for something newer
                got = slot.wait_newer(seq, timeout=0.05)
                # Nothing newer
                if got is None:
                    # Finished when the writers are done
                    if done.is_set():
                        # Stop reading
                        return
                    # Otherwise keep waiting
                    continue
                # Remember the newest sequence
                seq = got[0]
                # Record it
                seen.append(seq)
        # Reader thread
        r = threading.Thread(target=reader)
        # Start the reader first
        r.start()
        # Three writer threads
        ws = [threading.Thread(target=writer, args=(k,)) for k in range(3)]
        # Start them
        for w in ws:
            # Run
            w.start()
        # Wait for all writers
        for w in ws:
            # Join
            w.join()
        # Tell the reader the writers are done
        done.set()
        # Wait for the reader
        r.join(5.0)
        # Sequence numbers only grow
        self.assertEqual(seen, sorted(seen))
        # The reader saw the very last write
        self.assertEqual(seen[-1], 3 * n)
        # The slot holds a final item of one of the writers
        self.assertEqual(slot.get()[1][1], n - 1)


# Run the tests directly
if __name__ == "__main__":
    # unittest entry point
    unittest.main()
