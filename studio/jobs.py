"""Jobs and resources (TASKS M1-8).

* **Device owner**: only one task runner at a time holds the camera and the
  motors. Starting a second task is refused with the reason.
* **Background jobs**: training (PC) and TensorRT conversion (car) run as
  subprocesses, one at a time, in a queue. Each job lives in
  ``STUDIO_HOME/jobs/<id>/`` (``job.json`` + ``log.txt``), so closing the
  browser does not matter. Jobs marked ``needs_device_free`` (TensorRT
  conversion: needs the GPU and most of the RAM) wait until no task runs.
* **Resources**: CPU, RAM, disk, GPU load and process memory, with warnings
  against ``config.BUDGET_MB``; disk quotas per category.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time

# studio paths and budgets
from studio import PROJECT4_DIR, config

# Server log
log = logging.getLogger("studio.jobs")
# Job states that are finished
FINISHED = ("done", "failed", "cancelled", "lost")


# A task is already holding the camera / motors
class DeviceBusy(Exception):
    """Message names the task that holds the devices."""


# --------------------------------------------------------------------------- #
# Background jobs
# --------------------------------------------------------------------------- #
# One background job (subprocess with a log file)
class Job(object):
    """State of a job; saved to ``job.json`` on every change."""

    # Fields stored in job.json
    FIELDS = ("id", "name", "kind", "cmd", "cwd", "status", "created", "started", "ended",
              "returncode", "needs_device_free", "meta", "progress", "user")

    # New job (or one loaded from job.json)
    def __init__(self, **kw):
        # Every field, None when not given
        for k in self.FIELDS:
            # Set the attribute
            setattr(self, k, kw.get(k))
        # Extra data (task id, outputs, ...) and progress default to dicts
        self.meta, self.progress = self.meta or {}, self.progress or {}
        # Running process (not saved)
        self.proc = None
        # Bytes of the log already parsed for progress
        self._parsed = 0

    # Folder of this job
    @property
    def folder(self):
        """``STUDIO_HOME/jobs/<id>``."""
        # Created when missing
        return config.home_dir("jobs", self.id)

    # Path of the log
    @property
    def log_path(self):
        """``.../log.txt``."""
        # Inside the job folder
        return os.path.join(self.folder, "log.txt")

    # Save the state
    def save(self):
        """Write ``job.json``."""
        # Temporary file then replace (never a half-written file)
        tmp = os.path.join(self.folder, "job.json.tmp")
        # ASCII JSON
        with open(tmp, "w", encoding="utf-8") as f:
            # Every stored field
            json.dump(self.to_dict(), f, indent=2)
        # Replace
        os.replace(tmp, os.path.join(self.folder, "job.json"))

    # JSON-able state
    def to_dict(self):
        """Every stored field."""
        # Field -> value
        return dict((k, getattr(self, k)) for k in self.FIELDS)


# Queue of background jobs
class JobQueue(object):
    """``submit`` / ``cancel`` / ``tick`` (call ``tick`` about once per second)."""

    # Load earlier jobs from disk
    def __init__(self, device_free=lambda: True):
        # Function telling whether no task runner holds the devices
        self.device_free = device_free
        # id -> Job
        self.jobs = {}
        # Progress parsers by job kind: fn(job, new_text)
        self.parsers = {}
        # Hooks by job kind called once when a job ends: fn(queue, job)
        self.on_finish = {}
        # Earlier jobs
        root = config.home_dir("jobs")
        # Every job folder
        for name in sorted(os.listdir(root)):
            # Its state file
            p = os.path.join(root, name, "job.json")
            # Skip folders without one
            if not os.path.exists(p):
                # Next one
                continue
            # A corrupt file must not stop the server
            try:
                # Read it
                with open(p, encoding="utf-8") as f:
                    # Rebuild the job
                    job = Job(**json.load(f))
            # Skip it
            except (OSError, ValueError, TypeError):
                # Next one
                continue
            # Jobs that were running when the server stopped are lost
            if job.status not in FINISHED:
                # Mark and save
                job.status, job.ended = "lost", job.ended or time.time()
                # Store the change
                job.save()
            # Keep it
            self.jobs[job.id] = job

    # Add a job to the queue
    def submit(self, name, cmd, cwd=None, kind="generic", needs_device_free=False, meta=None, user=None):
        """Queue ``cmd`` (list of arguments); returns the ``Job``."""
        # Unique, sortable id
        jid = time.strftime("%Y%m%d-%H%M%S") + "-%04d" % (len(self.jobs) % 10000)
        # Never reuse an id
        while jid in self.jobs:
            # Add a suffix
            jid += "x"
        # New job
        job = Job(id=jid, name=name, kind=kind, cmd=[str(c) for c in cmd], cwd=cwd or PROJECT4_DIR,
                  status="queued", created=time.time(), needs_device_free=bool(needs_device_free),
                  meta=meta or {}, user=user)
        # Save it
        job.save()
        # Register it
        self.jobs[jid] = job
        # Start right away if possible
        self.tick()
        # The job
        return job

    # The running job, if any
    def running(self):
        """The ``Job`` currently running or None."""
        # First running job
        for job in self.jobs.values():
            # Running
            if job.status == "running":
                # Found
                return job
        # None
        return None

    # Start a queued job
    def _start(self, job):
        # Log file (appending keeps earlier text after a restart of the same job)
        logf = open(job.log_path, "a", encoding="utf-8")
        # Unbuffered child output
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        # Own process group so cancel can stop the whole tree
        kwargs = {"start_new_session": True} if os.name == "posix" else {}
        # Start it
        try:
            # The process
            job.proc = subprocess.Popen(job.cmd, cwd=job.cwd, stdout=logf, stderr=subprocess.STDOUT,
                                        env=env, **kwargs)
            # Running
            job.status, job.started = "running", time.time()
        # Command not found, bad folder, ...
        except OSError as e:
            # Write the reason into the log
            logf.write("cannot start: %r\n" % (e,))
            # Failed
            job.status, job.ended = "failed", time.time()
        # The child has its own handle
        logf.close()
        # Store the change
        job.save()

    # Feed new log text to the job kind's progress parser
    def _parse(self, job):
        # Parser for this kind
        parser = self.parsers.get(job.kind)
        # None registered or no log yet
        if parser is None or not os.path.exists(job.log_path):
            # Nothing to do
            return
        # Read the part not parsed yet
        with open(job.log_path, "rb") as f:
            # Skip what was parsed
            f.seek(job._parsed)
            # New bytes
            data = f.read()
        # Only whole lines
        cut = data.rfind(b"\n") + 1
        # Nothing complete yet
        if cut <= 0:
            # Wait for more
            return
        # Remember the position
        job._parsed += cut
        # A parser bug must not stop the queue
        try:
            # Update job.progress
            parser(job, data[:cut].decode("utf-8", "replace"))
        # Log it
        except Exception:
            # Server log
            log.exception("progress parser failed for job %s", job.id)

    # Advance the queue
    def tick(self):
        """Check the running job, parse progress, start the next one."""
        # The running job
        job = self.running()
        # One is running
        if job is not None:
            # New progress lines
            self._parse(job)
            # Process lost (server restarted while it ran): cannot follow it
            if job.proc is None:
                # Mark lost
                job.status, job.ended = "lost", time.time()
                # Store
                job.save()
            # Finished
            elif job.proc.poll() is not None:
                # Last lines
                self._parse(job)
                # Exit code and end time
                job.returncode, job.ended = job.proc.returncode, time.time()
                # Done or failed (cancelled was set by cancel())
                if job.status == "running":
                    # By exit code
                    job.status = "done" if job.returncode == 0 else "failed"
                # Store
                job.save()
                # Kind-specific follow-up (e.g. retry with a smaller batch)
                hook = self.on_finish.get(job.kind)
                # Run it; a broken hook must not stop the queue
                if hook is not None:
                    # Guarded call
                    try:
                        # Follow-up
                        hook(self, job)
                    # Log it
                    except Exception:
                        # Server log
                        log.exception("finish hook failed for job %s", job.id)
            # Still running
            else:
                # Save progress now and then
                job.save()
                # Nothing else to do
                return
        # Oldest queued job that may start now
        for job in sorted(self.jobs.values(), key=lambda j: j.created):
            # Waiting
            if job.status != "queued":
                # Next one
                continue
            # Needs the GPU / RAM free and a task is running: wait
            if job.needs_device_free and not self.device_free():
                # Note why it waits
                job.progress["waiting"] = "a task is running; stop it to start this job"
                # Do not start others ahead of it
                return
            # Clear the note
            job.progress.pop("waiting", None)
            # Start it
            self._start(job)
            # One at a time
            return

    # Cancel a job
    def cancel(self, jid):
        """Cancel a queued or running job; returns False when it cannot be cancelled."""
        # The job
        job = self.jobs.get(jid)
        # Unknown or finished
        if job is None or job.status in FINISHED:
            # Nothing to cancel
            return False
        # Running: stop the process (whole group on Linux)
        if job.status == "running" and job.proc is not None and job.proc.poll() is None:
            # Polite stop first
            try:
                # Process group on POSIX
                if os.name == "posix":
                    # SIGTERM to the group
                    os.killpg(job.proc.pid, signal.SIGTERM)
                # Windows: the process itself
                else:
                    # Terminate
                    job.proc.terminate()
            # Already gone
            except OSError:
                # Nothing to stop
                pass
            # Wait up to 5 s, then kill
            try:
                # Wait
                job.proc.wait(5)
            # Still alive
            except subprocess.TimeoutExpired:
                # Kill
                job.proc.kill()
        # Cancelled
        job.status, job.ended = "cancelled", time.time()
        # Store
        job.save()
        # Next job may start
        self.tick()
        # Done
        return True

    # Part of a job's log
    def read_log(self, jid, offset=0, limit=65536):
        """Return ``(text, next_offset)``; ``(None, 0)`` for an unknown job."""
        # The job
        job = self.jobs.get(jid)
        # Unknown or no log yet
        if job is None:
            # Nothing
            return None, 0
        # No log yet
        if not os.path.exists(job.log_path):
            # Empty
            return "", 0
        # Read from the offset
        with open(job.log_path, "rb") as f:
            # Position
            f.seek(max(0, int(offset)))
            # Up to `limit` bytes
            data = f.read(limit)
        # Text and the next offset
        return data.decode("utf-8", "replace"), max(0, int(offset)) + len(data)

    # Jobs for the page, newest first
    def describe(self, limit=30):
        """List of job dicts."""
        # Newest first
        jobs = sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)[:limit]
        # Dicts
        return [j.to_dict() for j in jobs]


# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #
# Read a small text file, None on any error
def _read(path):
    # Missing files are normal (PC, Windows)
    try:
        # Read it
        with open(path, encoding="utf-8") as f:
            # Content
            return f.read()
    # Not available
    except (OSError, UnicodeDecodeError):
        # Unknown
        return None


# Resident memory of a process in MB (Linux)
def process_rss_mb(pid):
    """VmRSS of ``pid`` in MB, or None."""
    # /proc status file
    text = _read("/proc/%d/status" % pid)
    # Not Linux or no such process
    if not text:
        # Unknown
        return None
    # The VmRSS line: "VmRSS:   12345 kB"
    m = re.search(r"VmRSS:\s+(\d+)\s+kB", text)
    # MB
    return int(m.group(1)) / 1024.0 if m else None


# CPU, RAM, disk and GPU usage with warnings
class ResourceMonitor(object):
    """``sample()`` returns a dict (cached for ``max_age`` seconds)."""

    # Empty cache
    def __init__(self, max_age=1.0):
        # Seconds a sample stays valid
        self.max_age = max_age
        # Last sample and its time
        self._last, self._t = None, 0.0
        # Previous CPU counters (total, idle)
        self._cpu = None
        # Cached GPU value from nvidia-smi and its time (PC)
        self._gpu, self._gpu_t = None, 0.0
        # Cached quota usage and its time (walking folders is slow)
        self._quota, self._quota_t = None, 0.0

    # CPU usage since the previous call, percent
    def _cpu_percent(self):
        # First line of /proc/stat: "cpu user nice system idle iowait ..."
        text = _read("/proc/stat")
        # Not Linux
        if not text:
            # Unknown
            return None
        # Counters of the "cpu" line
        nums = [int(v) for v in text.splitlines()[0].split()[1:]]
        # Idle includes iowait
        total, idle = sum(nums), nums[3] + (nums[4] if len(nums) > 4 else 0)
        # Previous counters
        prev, self._cpu = self._cpu, (total, idle)
        # First call: no difference yet
        if prev is None or total == prev[0]:
            # Unknown
            return None
        # Busy share of the interval
        return 100.0 * (1.0 - (idle - prev[1]) / float(total - prev[0]))

    # Total and available RAM in MB
    def _memory(self):
        # /proc/meminfo
        text = _read("/proc/meminfo")
        # Not Linux
        if not text:
            # Unknown
            return None, None
        # "Key:   123 kB" lines
        vals = dict((m.group(1), int(m.group(2))) for m in re.finditer(r"(\w+):\s+(\d+)\s+kB", text))
        # Total and available (MemAvailable exists on every recent kernel)
        return vals.get("MemTotal", 0) / 1024.0, vals.get("MemAvailable", vals.get("MemFree", 0)) / 1024.0

    # GPU load in percent
    def _gpu_percent(self):
        # Jetson: load in tenths of a percent
        for p in ("/sys/devices/gpu.0/load", "/sys/devices/platform/host1x/57000000.gpu/load"):
            # Read it
            text = _read(p)
            # Found
            if text and text.strip().isdigit():
                # Percent
                return int(text.strip()) / 10.0
        # PC with an NVIDIA GPU: nvidia-smi at most every 5 s
        if time.monotonic() - self._gpu_t > 5.0:
            # Remember the time even on failure
            self._gpu_t = time.monotonic()
            # nvidia-smi may be missing
            try:
                # Utilisation in percent
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=2)
                # First GPU's value
                self._gpu = float(out.stdout.decode().split()[0])
            # No NVIDIA GPU / tool
            except Exception:
                # Unknown
                self._gpu = None
        # Cached value
        return self._gpu

    # Disk usage of every quota category
    def quotas(self, max_age=30.0):
        """``{category: {'used_mb', 'limit_mb', 'folders'}}`` (cached ``max_age`` s)."""
        # Fresh enough
        if self._quota is not None and time.monotonic() - self._quota_t < max_age:
            # Cached
            return self._quota
        # Folders counted in each category
        folders = {
            # PyTorch models and TensorRT engines
            "models": [os.path.join(PROJECT4_DIR, "models"), os.path.join(PROJECT4_DIR, "models_trt")],
            # Drive logs, recorded frames, collected labels
            "recordings": [config.home_dir("recordings"), os.path.join(PROJECT4_DIR, "runs")],
            # Maps of the search task
            "maps": [config.home_dir("maps")],
            # Server, runner and job logs
            "logs": [config.home_dir("logs"), config.home_dir("jobs")],
        }
        # Category -> usage
        out = {}
        # Every category
        for cat, dirs in folders.items():
            # Sum of the folders
            used = sum(config.folder_size(d) for d in dirs) / 1e6
            # Usage and limit
            out[cat] = {"used_mb": round(used, 1), "limit_mb": config.BUDGET_MB[cat], "folders": dirs}
        # Cache it
        self._quota, self._quota_t = out, time.monotonic()
        # Usage
        return out

    # Whether a category may grow by `extra_mb`
    def quota_ok(self, category, extra_mb=0.0):
        """Return ``(ok, message)`` for writing ``extra_mb`` more into ``category``."""
        # Current usage (fresh)
        q = self.quotas(max_age=5.0)[category]
        # Over the category limit
        if q["used_mb"] + extra_mb > q["limit_mb"]:
            # Refuse with numbers
            return False, "%s quota full: %.0f of %d MB used; delete old files first" % (
                category, q["used_mb"], q["limit_mb"])
        # Free disk after writing
        free = config.disk_free_mb() - extra_mb
        # Below the reserve
        if free < config.BUDGET_MB["disk_reserve"]:
            # Refuse
            return False, "only %.0f MB disk left (reserve %d MB)" % (free, config.BUDGET_MB["disk_reserve"])
        # Allowed
        return True, ""

    # One sample of everything
    def sample(self, runner_pid=None):
        """Dict with cpu/mem/disk/gpu/rss values (None when unknown) and ``warnings``."""
        # Fresh enough
        if self._last is not None and time.monotonic() - self._t < self.max_age and \
                self._last.get("runner_pid") == runner_pid:
            # Cached
            return self._last
        # RAM
        mem_total, mem_avail = self._memory()
        # Disk of the studio home
        import shutil
        # Usage of that file system
        disk = shutil.disk_usage(config.studio_home())
        # Values
        s = {"cpu_percent": self._cpu_percent(), "mem_total_mb": mem_total, "mem_avail_mb": mem_avail,
             "disk_total_mb": disk.total / 1e6, "disk_free_mb": disk.free / 1e6,
             "gpu_percent": self._gpu_percent(), "web_rss_mb": process_rss_mb(os.getpid()),
             "runner_pid": runner_pid, "runner_rss_mb": process_rss_mb(runner_pid) if runner_pid else None,
             "quotas": self.quotas(), "warnings": []}
        # Short name for the budgets
        b = config.BUDGET_MB
        # Disk below the reserve
        if s["disk_free_mb"] < b["disk_reserve"]:
            # Warn
            s["warnings"].append("disk: only %.0f MB free (keep %d MB)" % (s["disk_free_mb"], b["disk_reserve"]))
        # RAM low
        if mem_avail is not None and mem_avail < b["ram_warn"]:
            # Warn
            s["warnings"].append("memory: only %.0f MB available" % mem_avail)
        # Web process too big
        if s["web_rss_mb"] is not None and s["web_rss_mb"] > b["ram_web"]:
            # Warn
            s["warnings"].append("web process uses %.0f MB (budget %d MB)" % (s["web_rss_mb"], b["ram_web"]))
        # Runner too big
        if s["runner_rss_mb"] is not None and s["runner_rss_mb"] > b["ram_runner"]:
            # Warn
            s["warnings"].append("task process uses %.0f MB (budget %d MB)" % (s["runner_rss_mb"], b["ram_runner"]))
        # Quotas over the limit
        for cat, q in s["quotas"].items():
            # Over the limit
            if q["used_mb"] > q["limit_mb"]:
                # Warn
                s["warnings"].append("%s: %.0f MB used, quota %d MB" % (cat, q["used_mb"], q["limit_mb"]))
        # Cache it
        self._last, self._t = s, time.monotonic()
        # The sample
        return s


# --------------------------------------------------------------------------- #
# Everything together
# --------------------------------------------------------------------------- #
# Owner of the running task, the job queue and the resource monitor
class Manager(object):
    """One per web process."""

    # Create the parts
    def __init__(self, settings, link_factory=None):
        # Server settings (role, camera source, ...)
        self.settings = settings
        # Function creating a RunnerLink (replaceable in tests)
        if link_factory is None:
            # Imported here: tornado-based
            from studio.runlink import RunnerLink
            # Default factory
            link_factory = RunnerLink
        # Factory
        self.link_factory = link_factory
        # Running task's link (None = devices free)
        self.runner = None
        # Background jobs; device-bound jobs wait while a task runs
        self.jobs = JobQueue(device_free=self.device_free)
        # Resource monitor
        self.resources = ResourceMonitor()

    # Whether no task holds the devices
    def device_free(self):
        """True when no task runner is starting / running / stopping."""
        # No link or a finished one
        return self.runner is None or self.runner.status in ("stopped", "failed")

    # Start a task
    async def start_task(self, task, values, user=None):
        """Start the runner; raises ``DeviceBusy`` or ``runlink.RunnerError``."""
        # Another task holds the devices
        if not self.device_free():
            # Which one and since when
            r = self.runner
            # Refuse with the reason
            raise DeviceBusy("camera and motors are used by task '%s' (started %s by %s); stop it first" % (
                r.task["title"], time.strftime("%H:%M:%S", time.localtime(r.started)), r.user or "?"))
        # A device-bound job is running (TensorRT conversion needs the GPU / RAM)
        job = self.jobs.running()
        # Refuse while it runs
        if job is not None and job.needs_device_free:
            # Reason
            raise DeviceBusy("job '%s' is using the GPU; wait for it or cancel it" % job.name)
        # New link
        link = self.link_factory(task, values, self.settings, user=user)
        # Owner from now on (also while starting)
        self.runner = link
        # Start it (raises on failure; status becomes 'failed', devices free again)
        await link.start()
        # The link
        return link

    # Stop the running task
    async def stop_task(self):
        """Stop the runner if any; returns True when one was stopped."""
        # Nothing running
        if self.runner is None or self.runner.status in ("stopped", "failed"):
            # Nothing to do
            return False
        # Stop it
        await self.runner.stop()
        # A waiting job may start now
        self.jobs.tick()
        # Stopped
        return True

    # The running task's link for a task id
    def link_for(self, task_id):
        """The ``RunnerLink`` if ``task_id`` is the running task, else None."""
        # Same task and alive
        if self.runner is not None and self.runner.task["id"] == task_id and \
                self.runner.status in ("starting", "running"):
            # The link
            return self.runner
        # Not running
        return None

    # Everything the home page shows
    def status(self):
        """JSON-able status."""
        # Runner pid for its memory
        pid = self.runner.proc.pid if self.runner is not None and self.runner.proc is not None and \
            self.runner.status in ("starting", "running", "stopping") else None
        # Resources, runner and jobs
        return {"resources": self.resources.sample(pid),
                "runner": self.runner.describe() if self.runner is not None else None,
                "jobs": self.jobs.describe(), "role": self.settings.get("role")}


# Python executable for jobs (same as the server)
PYTHON = sys.executable
