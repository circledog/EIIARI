"""Background jobs, resources, the link watchdog, and the web process staying torch-free."""
import subprocess
import sys
import time
import unittest

# Test helpers (temporary STUDIO_HOME, import path)
from _base import REPO
# Modules under test
from studio import jobs
# Link watchdog
from studio.safety import LinkWatchdog


# Wait until a job is finished
def wait_job(q, job, timeout=20):
    # Deadline
    deadline = time.monotonic() + timeout
    # Tick until finished
    while job.status not in jobs.FINISHED and time.monotonic() < deadline:
        # Advance the queue
        q.tick()
        # Short pause
        time.sleep(0.05)
    # The job
    return job


# Job queue
class JobQueueTest(unittest.TestCase):
    # A job runs, its log is readable, progress is parsed
    def test_run_and_log(self):
        # Queue without device restrictions
        q = jobs.JobQueue()
        # Parser that counts "step" lines
        q.parsers["count"] = lambda job, text: job.progress.update(
            steps=job.progress.get("steps", 0) + text.count("step"))
        # Job printing three steps
        job = q.submit("t", [sys.executable, "-c", "print('step\\nstep\\nstep')"], kind="count")
        # Finished successfully
        self.assertEqual(wait_job(q, job).status, "done")
        # Log content
        text, offset = q.read_log(job.id)
        # Contains the output
        self.assertIn("step", text)
        # Progress parsed
        self.assertEqual(job.progress["steps"], 3)
        # Reading from the end gives nothing new
        self.assertEqual(q.read_log(job.id, offset)[0], "")

    # Failure and cancel
    def test_fail_and_cancel(self):
        # Queue
        q = jobs.JobQueue()
        # Failing job
        bad = q.submit("bad", [sys.executable, "-c", "raise SystemExit(3)"])
        # Failed with its code
        self.assertEqual(wait_job(q, bad).status, "failed")
        # Exit code kept
        self.assertEqual(bad.returncode, 3)
        # Long job
        slow = q.submit("slow", [sys.executable, "-c", "import time; time.sleep(30)"])
        # Running
        self.assertEqual(slow.status, "running")
        # Cancel it
        self.assertTrue(q.cancel(slow.id))
        # Cancelled and the process is gone
        self.assertEqual(slow.status, "cancelled")
        # Process ended
        self.assertIsNotNone(slow.proc.poll())

    # Jobs that need the GPU wait while a task runs
    def test_device_bound_job_waits(self):
        # Device state controlled by the test
        busy = [True]
        # Queue with that state
        q = jobs.JobQueue(device_free=lambda: not busy[0])
        # TensorRT-like job
        job = q.submit("trt", [sys.executable, "-c", "print(1)"], needs_device_free=True)
        # Still queued with a note
        self.assertEqual(job.status, "queued")
        # Reason shown
        self.assertIn("waiting", job.progress)
        # Task stopped
        busy[0] = False
        # Runs now
        self.assertEqual(wait_job(q, job).status, "done")

    # Jobs interrupted by a server restart are marked lost
    def test_restart_marks_lost(self):
        # Queue with a long job
        q = jobs.JobQueue()
        # Running job
        job = q.submit("long", [sys.executable, "-c", "import time; time.sleep(30)"])
        # New queue = server restart (reads job.json)
        q2 = jobs.JobQueue()
        # Marked lost
        self.assertEqual(q2.jobs[job.id].status, "lost")
        # Clean up the process
        q.cancel(job.id)


# Resources and quotas
class ResourceTest(unittest.TestCase):
    # A sample has the expected keys; quotas refuse a huge write
    def test_sample_and_quota(self):
        # Monitor
        mon = jobs.ResourceMonitor()
        # One sample
        s = mon.sample()
        # Keys
        for key in ("cpu_percent", "mem_avail_mb", "disk_free_mb", "web_rss_mb", "quotas", "warnings"):
            # Present
            self.assertIn(key, s)
        # Writing 10 GB of recordings is refused
        ok, msg = mon.quota_ok("recordings", extra_mb=10000)
        # Refused with a reason
        self.assertFalse(ok)
        # Explains
        self.assertIn("MB", msg)

    # quota_ok never walks folders in the caller after the first call; it counts accepted writes
    def test_quota_rescans_in_background(self):
        import os
        # Studio home folders
        from studio import config
        # Monitor
        mon = jobs.ResourceMonitor()
        # First call measures synchronously
        self.assertTrue(mon.quota_ok("maps", 1)[0])
        # Measured value (includes the accepted 1 MB)
        before = mon._used["maps"][0]
        # 3 MB appear on disk
        with open(os.path.join(config.home_dir("maps"), "big.bin"), "wb") as f:
            # Content
            f.write(b"x" * 3000000)
        # A stale cache: the call returns at once with the old value and starts a rescan
        self.assertTrue(mon.quota_ok("maps", 0, max_age=0.0)[0])
        # Wait for the background scan
        t0 = time.monotonic()
        # Until updated
        while mon._used["maps"][0] < before + 2.0 and time.monotonic() - t0 < 5:
            # Short wait
            time.sleep(0.02)
        # The new file is counted
        self.assertGreaterEqual(mon._used["maps"][0], before + 2.0)
        # And no scan is left running
        self.assertEqual(mon._scanning, set())


# Link watchdog with a fake clock
class LinkWatchdogTest(unittest.TestCase):
    # Loss is reported once, after the timeout, and again after a new heartbeat
    def test_reports_once_per_loss(self):
        # Fake clock
        now = [0.0]
        # Reported delays
        lost = []
        # Watchdog with a 0.5 s timeout
        wd = LinkWatchdog(lost.append, timeout=0.5, clock=lambda: now[0])
        # Down before the first heartbeat, but nothing to report
        self.assertFalse(wd.alive())
        # Nothing reported
        self.assertFalse(wd.check())
        # Heartbeat
        wd.beat()
        # 0.4 s later: alive
        now[0] = 0.4
        # No report
        self.assertFalse(wd.check())
        # 0.6 s: lost
        now[0] = 0.6
        # Reported
        self.assertTrue(wd.check())
        # Only once
        self.assertFalse(wd.check())
        # New heartbeat and a new loss
        wd.beat()
        # 1.2 s
        now[0] = 1.2
        # Reported again
        self.assertTrue(wd.check())
        # Two reports
        self.assertEqual(len(lost), 2)


# The web process must not load torch / OpenCV (RAM budget: web <= 150 MB)
class ImportBudgetTest(unittest.TestCase):
    # Import the server and load the tasks in a fresh interpreter
    def test_server_imports_no_torch(self):
        # Code run in a fresh interpreter
        code = ("import sys; from studio import server; server.Studio({'cookie_secret': 'x', 'role': 'pc'}); "
                "bad = [m for m in ('torch', 'cv2', 'numpy', 'torchvision') if m in sys.modules]; "
                "print(','.join(bad))")
        # Run it from the repository root
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=60)
        # Nothing heavy was imported
        self.assertEqual(out.stdout.decode().strip().splitlines()[-1] if out.stdout.strip() else "", "",
                         out.stdout.decode())


# Run directly
if __name__ == "__main__":
    # unittest entry point
    unittest.main()
