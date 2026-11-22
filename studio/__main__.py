"""Command line: ``python3 -m studio [serve|set-password] [options]``.

    python3 -m studio set-password          # first time: choose the login password
    python3 -m studio                       # serve on port 8890
    python3 -m studio --role pc --camera dummy --port 8891

Run from the repository root (the folder that contains ``studio/``).

Python 3.6 compatible (runs on the Jetson Nano).
"""
import argparse
import getpass
import logging
import logging.handlers
import signal
import sys

# Settings and paths
from studio import config, __version__
# Password hashing
from studio import auth


# Ask for a new password and store its hash
def set_password(stdin=False):
    """Interactive (or one line from stdin with ``--stdin``); returns the exit code."""
    # Non-interactive: first line of stdin
    if stdin:
        # Password without the newline
        pw = sys.stdin.readline().rstrip("\n")
        # No confirmation in scripts
        again = pw
    # Interactive: ask twice without echo
    else:
        # First entry
        pw = getpass.getpass("New studio password: ")
        # Confirmation
        again = getpass.getpass("Repeat: ")
    # Entries differ
    if pw != again:
        # Report
        print("passwords differ, nothing changed")
        # Failure
        return 1
    # Too short
    if len(pw) < auth.MIN_LENGTH:
        # Report
        print("use at least %d characters, nothing changed" % auth.MIN_LENGTH)
        # Failure
        return 1
    # Current settings
    cfg = config.load()
    # New hash
    cfg["password"] = auth.hash_password(pw)
    # Store it
    config.save(cfg)
    # Report where
    print("password saved in %s" % config.home_file("config.json"))
    # Success
    return 0


# Log to the console and to a rotating file (budget: logs <= 100 MB)
def setup_logging():
    """Console + ``STUDIO_HOME/logs/studio.log`` (1 MB x 5)."""
    # Root logger
    root = logging.getLogger()
    # Info and above
    root.setLevel(logging.INFO)
    # Same format for both outputs
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Rotating file
    fh = logging.handlers.RotatingFileHandler(config.home_file("logs", "studio.log"), maxBytes=1 << 20,
                                              backupCount=5, encoding="utf-8")
    # Console
    sh = logging.StreamHandler()
    # Both handlers
    for h in (fh, sh):
        # Format
        h.setFormatter(fmt)
        # Attach
        root.addHandler(h)


# Run the web server
def serve(args):
    """Start tornado; returns the exit code."""
    import tornado.ioloop
    # Server application and shared state
    from studio import server
    # Settings
    cfg = config.load()
    # Command-line overrides (not stored)
    for key, value in (("port", args.port), ("address", args.address), ("role", args.role),
                       ("camera_source", args.camera), ("camera_folder", args.camera_folder)):
        # Only given options
        if value is not None:
            # Override
            cfg[key] = value
    # No password yet: refuse to serve an open page
    if not cfg.get("password"):
        # Explain
        print("no password set: run `python3 -m studio set-password` first")
        # Failure
        return 1
    # Logging
    setup_logging()
    # Shared state (loads the tasks)
    studio = server.Studio(cfg)
    # Application
    app = server.make_app(studio, debug=args.debug)
    # Listen
    app.listen(int(cfg["port"]), address=cfg["address"])
    # Job queue timer
    server.start_timers(studio)
    # Startup message
    logging.getLogger("studio").info("studio %s on http://%s:%s (role %s, camera %s, %d tasks)", __version__,
                                     cfg["address"], cfg["port"], studio.role, cfg["camera_source"],
                                     len(studio.tasks))
    # Event loop
    loop = tornado.ioloop.IOLoop.current()
    # Ctrl-C / SIGTERM: stop the task (motors off) before exiting
    def on_signal(signum, frame):
        # Schedule the shutdown on the loop
        loop.add_callback_from_signal(shutdown)
    # Orderly shutdown
    async def shutdown():
        # Stop the running task
        await studio.manager.stop_task()
        # End the loop
        loop.stop()
    # Register both signals
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Same handler
        signal.signal(sig, on_signal)
    # Run until stopped
    loop.start()
    # Success
    return 0


# Parse the command line and dispatch
def main(argv=None):
    """Entry point; returns the exit code."""
    # Options
    ap = argparse.ArgumentParser(prog="python3 -m studio", description="JetBot studio web workspace")
    # Sub-command
    ap.add_argument("command", nargs="?", default="serve", choices=("serve", "set-password"))
    # Port
    ap.add_argument("--port", type=int, help="web port (default %d)" % config.DEFAULT_PORT)
    # Address
    ap.add_argument("--address", help="address to listen on (default 0.0.0.0)")
    # Role override
    ap.add_argument("--role", choices=config.ROLES, help="force car / pc (default: detected)")
    # Camera override
    ap.add_argument("--camera", help="camera source: jetbot | csi | opencv | folder | dummy")
    # Folder for the folder camera
    ap.add_argument("--camera-folder", help="image folder for --camera folder")
    # Read the password from stdin
    ap.add_argument("--stdin", action="store_true", help="set-password: read it from stdin")
    # Template reload
    ap.add_argument("--debug", action="store_true", help="reload templates on change")
    # Parse
    args = ap.parse_args(argv)
    # Password
    if args.command == "set-password":
        # Ask and store
        return set_password(args.stdin)
    # Web server
    return serve(args)


# Run as a module
if __name__ == "__main__":
    # Exit with the result
    sys.exit(main())
