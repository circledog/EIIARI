"""Settings, paths and resource budgets.

Everything studio writes lives under ``STUDIO_HOME`` (default ``~/.studio``,
never inside the git repository):

    config.json        password hash, cookie secret, port, camera source, ...
    presets/<task>/    saved parameter sets
    jobs/<id>/         background jobs (training, TRT conversion): meta + log
    recordings/        drive logs and recorded frames
    logs/              rotating server log

Roles: ``car`` (the Jetson Nano: runs tasks, converts to TensorRT) and ``pc``
(trains models). The role is detected automatically and can be forced with
``--role``.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import json
import os
import secrets
import shutil

# Port of the web page (Jupyter uses 8888)
DEFAULT_PORT = 8890
# Valid roles
ROLES = ("car", "pc")

# Resource budgets for the Jetson Nano 4 GB (RAM is shared by CPU and GPU), in MB
BUDGET_MB = {
    # Disk: models (.pth + TensorRT engines)
    "models": 600,
    # Disk: datasets and recordings
    "recordings": 2000,
    # Disk: maps of the search task
    "maps": 300,
    # Disk: logs
    "logs": 100,
    # Disk: free space that must always remain (system, temp files, TRT build)
    "disk_reserve": 700,
    # RAM: free memory below which the home page warns
    "ram_warn": 500,
    # RAM: the web process must stay below this
    "ram_web": 150,
    # RAM: a runner process (camera + one model + control) should stay below this
    "ram_runner": 1500,
}


# Folder for everything studio writes
def studio_home():
    """``$STUDIO_HOME`` or ``~/.studio``."""
    # Environment override (tests, several installations)
    return os.environ.get("STUDIO_HOME") or os.path.join(os.path.expanduser("~"), ".studio")


# A folder below the studio home, created when missing
def home_dir(*parts):
    """``studio_home()/parts...`` as a folder that exists."""
    # Full path
    p = os.path.join(studio_home(), *parts)
    # Create it (and its parents) if needed
    os.makedirs(p, exist_ok=True)
    # The folder
    return p


# A file below the studio home; its folder is created when missing
def home_file(*parts):
    """``studio_home()/parts...`` as a file path whose folder exists."""
    # Folder part and file name
    return os.path.join(home_dir(*parts[:-1]), parts[-1])


# Guess whether this machine is the robot
def detect_role():
    """``'car'`` on a Jetson (``/etc/nv_tegra_release`` exists), else ``'pc'``."""
    # NVIDIA L4T writes this file on every Jetson
    return "car" if os.path.exists("/etc/nv_tegra_release") else "pc"


# Default settings written on first start
def _defaults():
    # Fresh settings
    return {
        # Web port
        "port": DEFAULT_PORT,
        # Address to listen on (all interfaces: the page is opened from the LAN)
        "address": "0.0.0.0",
        # 'car' / 'pc' / '' (auto)
        "role": "",
        # Frame source passed to project4/camera.py
        "camera_source": "jetbot",
        # Folder for camera_source = 'folder'
        "camera_folder": "",
        # Password record (set with `python3 -m studio set-password`)
        "password": None,
        # Secret for signing the login cookie
        "cookie_secret": secrets.token_hex(32),
        # Days a login stays valid
        "login_days": 7,
    }


# Load the settings, creating the file with defaults the first time
def load():
    """Settings dict from ``config.json`` (missing keys filled with defaults)."""
    # Settings file
    p = home_file("config.json")
    # Defaults for every key
    cfg = _defaults()
    # Merge the stored values over the defaults
    if os.path.exists(p):
        # Read the file
        with open(p, encoding="utf-8") as f:
            # Stored values win
            cfg.update(json.load(f))
    # First start: write the file so the cookie secret stays the same
    else:
        # Store the defaults
        save(cfg)
    # The settings
    return cfg


# Write the settings (readable by the owner only: it holds the password hash)
def save(cfg):
    """Write ``config.json`` atomically with mode 600."""
    # Settings file and a temporary file next to it
    p = home_file("config.json")
    # Temporary file in the same folder
    tmp = p + ".tmp"
    # Create the file readable / writable by the owner only
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    # Write ASCII-only JSON
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        # Pretty-printed
        json.dump(cfg, f, indent=2, sort_keys=True)
    # Replace the old file in one step
    os.replace(tmp, p)


# Role of this installation
def role(cfg):
    """Configured role, or the detected one when not set."""
    # Configured value, if valid
    r = cfg.get("role") or ""
    # Valid configured role wins; otherwise detect
    return r if r in ROLES else detect_role()


# Total size of the files below a folder
def folder_size(folder):
    """Bytes used by all files below ``folder`` (0 when missing)."""
    # Running total
    total = 0
    # Walk the tree
    for root, _, files in os.walk(folder):
        # Every file
        for name in files:
            # Size can fail for broken links / files removed meanwhile
            try:
                # Add the size
                total += os.path.getsize(os.path.join(root, name))
            # Ignore files that vanished
            except OSError:
                # Skip it
                pass
    # Bytes
    return total


# Free disk space at the studio home
def disk_free_mb():
    """Free MB on the file system holding ``studio_home()``."""
    # Usage of the file system
    usage = shutil.disk_usage(studio_home() if os.path.exists(studio_home()) else os.path.expanduser("~"))
    # Free megabytes
    return usage.free / 1e6
