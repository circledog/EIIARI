"""Which version of the code is running: git commit of this folder.

``train.py`` stores ``code_version()`` in every model's .json;
``live_demo`` compares it with the robot's own code and warns when they
differ (the weights are checked by SHA256, the code was not checked at all).

Without git (or outside a repository, e.g. files copied with scp) the
commit is None and the comparison is skipped.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import os
import subprocess

# Public names of this module
__all__ = ["code_version", "compare"]

# Folder of this file: the project4 code
HERE = os.path.dirname(os.path.abspath(__file__))


# Run one git command in this folder, None on any failure
def _git(*args):
    # git may be missing or the folder may not be a repository
    try:
        # Output of the command, without the trailing newline
        out = subprocess.check_output(("git",) + args, cwd=HERE, stderr=subprocess.DEVNULL)
    # Not available: no version information
    except (OSError, subprocess.CalledProcessError):
        # Caller treats None as unknown
        return None
    # Decode as ASCII (commit hashes and file names of this project are ASCII)
    return out.decode("ascii", "replace").strip()


# Commit of the code and whether tracked files were changed after it
def code_version():
    """``{"commit": <40-hex> or None, "dirty": bool or None}``."""
    # Commit currently checked out
    commit = _git("rev-parse", "HEAD")
    # Not a repository: nothing is known
    if not commit:
        # Unknown version
        return {"commit": None, "dirty": None}
    # Changed tracked files under this folder (untracked files such as models/ are ignored)
    changes = _git("status", "--porcelain", "--untracked-files=no", "--", ".")
    # Commit and whether it was modified
    return {"commit": commit, "dirty": bool(changes)}


# Compare the version a model was trained with against the running code
def compare(trained, running=None):
    """(status, text): status 'same', 'different' or 'unknown'; ``trained`` is meta["code_version"]."""
    # Version of the code running now
    running = running or code_version()
    # Commit the model's code had (older .json files have none)
    t = (trained or {}).get("commit")
    # Commit of the running code
    r = running.get("commit")
    # Either side unknown: cannot compare
    if not t or not r:
        # Say which side is missing
        return "unknown", "code version unknown (trained %s, running %s)" % (t and t[:7], r and r[:7])
    # Different commits: the robot may run other preprocessing / network code
    if t != r:
        # Both short hashes
        return "different", "code differs: trained with %s, running %s" % (t[:7], r[:7])
    # Same commit, but uncommitted edits on either side make it uncertain
    if trained.get("dirty") or running.get("dirty"):
        # Same commit with local edits
        return "different", "same commit %s but with uncommitted changes" % r[:7]
    # Identical code
    return "same", "same code (%s)" % r[:7]
