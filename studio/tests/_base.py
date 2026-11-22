"""Shared test helpers: a temporary STUDIO_HOME, a logged-in client, WebSockets.

The tests need no hardware: runners use the dummy camera and fake motors.
"""
import json
import os
import re
import sys
import tempfile

from tornado.httpclient import HTTPRequest
from tornado.testing import AsyncHTTPTestCase
from tornado.websocket import websocket_connect

# Repository root (parent of studio/)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Make `studio` importable from any working directory
if REPO not in sys.path:
    # Put the repository first
    sys.path.insert(0, REPO)

# Temporary home for every test run (set before studio reads it)
os.environ["STUDIO_HOME"] = tempfile.mkdtemp(prefix="studio-test-")

# Password hashing
from studio import auth  # noqa: E402
# Application and shared state
from studio import server  # noqa: E402

# Password used by the tests
PASSWORD = "test-password"


# Settings for a test server: dummy camera, PC role
def test_config(**over):
    """Config dict with a password and a fixed cookie secret."""
    # Fast hash for tests (fewer rounds)
    cfg = {"port": 0, "address": "127.0.0.1", "role": "pc", "camera_source": "dummy", "camera_folder": "",
           "password": auth.hash_password(PASSWORD, iterations=1000), "cookie_secret": "x" * 64, "login_days": 7}
    # Overrides
    cfg.update(over)
    # Settings
    return cfg


# Test case with a running studio app and login helpers
class StudioTestCase(AsyncHTTPTestCase):
    """``self.studio`` is the shared state; ``self.cookies`` after ``login()``."""

    # Settings overrides of a test class
    CONFIG = {}

    # Build the application
    def get_app(self):
        # Shared state with the test settings
        self.studio = server.Studio(test_config(**self.CONFIG))
        # The app
        return server.make_app(self.studio)

    # Stop a runner left over by a failing test
    def tearDown(self):
        # Running link, if any
        r = self.studio.manager.runner
        # Still alive: terminate it
        if r is not None and r.proc is not None and r.proc.poll() is None:
            # Kill it
            r.proc.kill()
        # tornado's cleanup
        AsyncHTTPTestCase.tearDown(self)

    # Cookie header from a response's Set-Cookie headers merged into a dict
    @staticmethod
    def _merge(cookies, response):
        # Every Set-Cookie header
        for header in response.headers.get_list("Set-Cookie"):
            # name=value before the first ';'
            name, _, value = header.split(";")[0].partition("=")
            # Store it
            cookies[name.strip()] = value.strip()
        # Updated dict
        return cookies

    # Cookie header string
    def cookie_header(self):
        """``name=value; ...`` of the stored cookies."""
        # Join them
        return "; ".join("%s=%s" % kv for kv in self.cookies.items())

    # Log in through the form
    def login(self, password=PASSWORD):
        """POST the login form; returns the response (cookies stored)."""
        # Fresh cookie jar
        self.cookies = {}
        # Login page sets the XSRF cookie
        r = self.fetch("/login")
        # Store it
        self._merge(self.cookies, r)
        # XSRF token from the form
        xsrf = re.search(r'name="_xsrf" value="([^"]+)"', r.body.decode()).group(1)
        # Submit the form
        r = self.fetch("/login", method="POST", follow_redirects=False, raise_error=False,
                       headers={"Cookie": self.cookie_header()},
                       body="_xsrf=%s&password=%s&next=/" % (xsrf, password))
        # Store the login cookie
        self._merge(self.cookies, r)
        # The response
        return r

    # Async login for use inside @gen_test (sync fetch cannot run there)
    async def login_async(self, password=PASSWORD):
        """Same as ``login()`` but awaitable."""
        # Fresh cookie jar
        self.cookies = {}
        # Login page sets the XSRF cookie
        r = await self.http_client.fetch(self.get_url("/login"))
        # Store it
        self._merge(self.cookies, r)
        # XSRF token from the form
        xsrf = re.search(r'name="_xsrf" value="([^"]+)"', r.body.decode()).group(1)
        # Submit the form
        r = await self.http_client.fetch(self.get_url("/login"), method="POST", follow_redirects=False,
                                         raise_error=False, headers={"Cookie": self.cookie_header()},
                                         body="_xsrf=%s&password=%s&next=/" % (xsrf, password))
        # Store the login cookie
        self._merge(self.cookies, r)
        # The response
        return r

    # Call the JSON API with the stored cookies and XSRF header
    def api(self, method, url, body=None):
        """Return ``(status, json)``."""
        # Body only for methods that have one
        data = json.dumps(body if body is not None else {}) if method in ("POST", "PUT") else None
        # Request with cookies and the XSRF header
        r = self.fetch(url, method=method, body=data, raise_error=False, follow_redirects=False,
                       headers={"Cookie": self.cookie_header(), "X-XSRFToken": self.cookies.get("_xsrf", ""),
                                "Content-Type": "application/json"})
        # Status and parsed body
        return r.code, (json.loads(r.body.decode()) if r.body else {})

    # Open a WebSocket with the stored cookies
    def ws(self, path):
        """Future of the connection."""
        # Request with the cookie header
        req = HTTPRequest("ws://127.0.0.1:%d%s" % (self.get_http_port(), path),
                          headers={"Cookie": self.cookie_header()})
        # Connect
        return websocket_connect(req)
