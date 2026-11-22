"""Every studio page in a real DOM (jsdom): no JavaScript errors, start / stop works.

Starts ``python3 -m studio`` with a temporary home and the dummy camera, then
runs ``browser/pages.js`` with Node. Skipped when ``node`` or the ``jsdom``
package is not available (``cd studio/tests/browser && npm install``, or set
NODE_PATH to a folder containing jsdom).
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

# Test helpers (import path)
from _base import REPO

# Folder of the browser script
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "browser")


# Whether node can load jsdom
def have_jsdom():
    """True when ``node`` exists and ``require('jsdom')`` works from the browser folder."""
    # Node itself
    if shutil.which("node") is None:
        # No node
        return False
    # Try to load jsdom
    r = subprocess.run(["node", "-e", "require('jsdom')"], cwd=HERE, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    # Loaded
    return r.returncode == 0


# Free TCP port on localhost
def free_port():
    """A port nobody listens on."""
    # Bind to port 0 and read the chosen port
    with socket.socket() as s:
        # Any free port
        s.bind(("127.0.0.1", 0))
        # The port
        return s.getsockname()[1]


# Pages in jsdom
@unittest.skipUnless(have_jsdom(), "node / jsdom not installed (cd studio/tests/browser && npm install)")
class BrowserPagesTest(unittest.TestCase):
    # Start a server, run the script, check every page
    def test_pages(self):
        # Temporary studio home
        home = tempfile.mkdtemp(prefix="studio-browser-")
        # Environment of the server
        env = dict(os.environ, STUDIO_HOME=home)
        # Password
        subprocess.run([sys.executable, "-m", "studio", "set-password", "--stdin"], cwd=REPO, env=env,
                       input=b"browser-test\n", check=True, stdout=subprocess.DEVNULL)
        # Port
        port = free_port()
        # Server with the dummy camera
        server = subprocess.Popen([sys.executable, "-m", "studio", "--camera", "dummy", "--role", "pc",
                                   "--port", str(port), "--address", "127.0.0.1"], cwd=REPO, env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Run the check, always stop the server
        try:
            # Wait for the port
            deadline = time.time() + 30
            # Until it accepts connections
            while time.time() < deadline:
                # Try to connect
                try:
                    # Connected: ready
                    socket.create_connection(("127.0.0.1", port), timeout=1).close()
                    # Done waiting
                    break
                # Not yet
                except OSError:
                    # Wait
                    time.sleep(0.2)
            # Run the browser script
            out = subprocess.run(["node", "pages.js", "http://127.0.0.1:%d" % port, "browser-test"], cwd=HERE,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        # Stop the server
        finally:
            # SIGTERM: stops the running task first
            server.terminate()
            # Wait for it
            server.wait(20)
        # One JSON line per page
        pages = [json.loads(l) for l in out.stdout.decode().splitlines() if l.startswith("{")]
        # Print for the HANDOFF record
        print("\n[browser] " + "; ".join("%s errors=%d" % (p["path"], len(p["errors"])) for p in pages))
        # Every page was checked
        self.assertEqual(len(pages), 8, out.stdout.decode()[-2000:])
        # No JavaScript errors anywhere
        self.assertEqual([(p["path"], p["errors"]) for p in pages if p["errors"]], [])
        # Pages by path
        by = dict((p["path"], p["info"]) for p in pages)
        # Home: four tasks, resource line filled
        self.assertEqual(by["/"]["tasks"], 4)
        # Resource bar
        self.assertIn("CPU", by["/"]["res"])
        # Every task started, streamed pictures and showed telemetry
        for t in ("drive_check", "lane_follow", "navigate", "search"):
            # Task page info
            info = by["/task/" + t]
            # Running badge seen
            self.assertEqual(info["state"], "\u57f7\u884c\u4e2d", t)
            # Frames received and acknowledged (more than one: the ack loop works)
            self.assertGreater(info["frames"], 5, t)
            # Parameter form built
            self.assertGreater(info["fields"], 3, t)
        # Training form built from the schema
        self.assertGreater(by["/lane/training"]["fields"], 15)


# Run directly
if __name__ == "__main__":
    # unittest entry point
    unittest.main()
