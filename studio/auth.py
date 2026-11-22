"""Password login.

* The password is stored as PBKDF2-HMAC-SHA256 with a random salt in
  ``config.json`` (never the password itself).
* After a successful login the browser gets a signed cookie (tornado's
  secure cookie, HMAC with ``cookie_secret``); a tampered cookie is invalid.
* 5 failed attempts from one address lock that address for 60 s.

The page is only used on the LAN (user's decision, 2026-11-22), so there is
no HTTPS; do not expose the port to the internet.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import base64
import hashlib
import hmac
import http.cookies
import os
import threading
import time

# PBKDF2 rounds: ~0.2 s on the Nano's CPU per login, slow enough for guessing
ITERATIONS = 100000
# Failed attempts allowed before a lock
MAX_FAILURES = 5
# Seconds an address stays locked
LOCK_SECONDS = 60.0
# Shortest password accepted by set-password
MIN_LENGTH = 6
# Name of the login cookie
COOKIE_NAME = "studio_user"


# Hash a password with a new (or given) salt
def hash_password(password, salt=None, iterations=ITERATIONS):
    """Return ``{'algo', 'iterations', 'salt', 'hash'}`` (base64 strings)."""
    # New random salt unless one is given
    salt = salt if salt is not None else os.urandom(16)
    # Derive the key
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    # Record that can be stored as JSON
    return {"algo": "pbkdf2_sha256", "iterations": iterations,
            "salt": base64.b64encode(salt).decode("ascii"), "hash": base64.b64encode(digest).decode("ascii")}


# Check a password against a stored record
def verify_password(password, record):
    """True when ``password`` matches ``record`` (constant-time compare)."""
    # No password set: nobody can log in
    if not record:
        # Reject
        return False
    # Recompute with the stored salt and rounds
    salt = base64.b64decode(record["salt"])
    # Same derivation as when the password was set
    again = hash_password(password, salt, int(record["iterations"]))
    # Constant-time comparison of the hashes
    return hmac.compare_digest(again["hash"], record["hash"])


# Counts failed logins per address and locks addresses for a while
class Lockout(object):
    """``check(addr)`` -> seconds left in a lock (0 = may try)."""

    # Empty table
    def __init__(self, max_failures=MAX_FAILURES, lock_seconds=LOCK_SECONDS, clock=time.monotonic):
        # Failures before a lock and lock duration
        self.max_failures, self.lock_seconds = max_failures, lock_seconds
        # Time source (replaceable in tests)
        self.clock = clock
        # addr -> [failure count, locked until]
        self.table = {}
        # Handlers may run in several threads
        self.lock = threading.Lock()

    # Seconds the address must still wait
    def check(self, addr):
        """0 when ``addr`` may try to log in, otherwise the seconds left."""
        # Read the entry under the lock
        with self.lock:
            # Failures and lock end of this address
            count, until = self.table.get(addr, (0, 0.0))
            # Seconds left (never negative)
            return max(0.0, until - self.clock())

    # Record a failed attempt
    def fail(self, addr):
        """Count a failure; locks the address after ``max_failures``."""
        # Update the entry under the lock
        with self.lock:
            # Previous failures
            count, until = self.table.get(addr, (0, 0.0))
            # One more
            count += 1
            # Too many: lock and start counting again
            if count >= self.max_failures:
                # Locked until now + lock time
                self.table[addr] = (0, self.clock() + self.lock_seconds)
            # Not yet: remember the count
            else:
                # Keep any running lock
                self.table[addr] = (count, until)

    # Forget the failures after a successful login
    def success(self, addr):
        """Clear the address's failures."""
        # Remove the entry under the lock
        with self.lock:
            # Forget it
            self.table.pop(addr, None)


# Mixin for tornado handlers: the logged-in user from the signed cookie
class CurrentUserMixin(object):
    """``get_current_user`` for RequestHandler and WebSocketHandler subclasses."""

    # Read and verify the signed cookie
    def get_current_user(self):
        """User name, or None when not logged in / the cookie is invalid or expired."""
        # tornado >= 6.3 renamed secure cookies to signed cookies
        getter = getattr(self, "get_signed_cookie", None) or self.get_secure_cookie
        # Verified value (None if missing, tampered or older than the login period)
        value = getter(COOKIE_NAME, max_age_days=self.settings.get("login_days", 7))
        # Bytes -> text
        return value.decode("utf-8") if value else None

    # Write the signed cookie after a successful login
    def set_login(self, user):
        """Sign ``user`` into the login cookie."""
        # tornado >= 6.3 name, older name otherwise
        setter = getattr(self, "set_signed_cookie", None) or self.set_secure_cookie
        # HttpOnly: scripts cannot read it
        extra = {"httponly": True}
        # SameSite (not sent by other sites' forms) only exists in Python >= 3.8's cookie module
        if "samesite" in http.cookies.Morsel._reserved:
            # Add it
            extra["samesite"] = "Strict"
        # Write the cookie
        setter(COOKIE_NAME, user, expires_days=self.settings.get("login_days", 7), **extra)
