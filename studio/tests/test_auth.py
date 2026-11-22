"""Login: password hashing, lockout, pages / API / WebSockets need a login, tampered cookies fail."""
import unittest

from tornado.testing import gen_test

# Test helpers (temporary STUDIO_HOME)
from _base import StudioTestCase, PASSWORD
# Module under test
from studio import auth


# Hashing and lockout without HTTP
class PasswordTest(unittest.TestCase):
    # Right / wrong password, salt differs every time
    def test_hash_and_verify(self):
        # Hash a password
        rec = auth.hash_password("hello123", iterations=1000)
        # Right password
        self.assertTrue(auth.verify_password("hello123", rec))
        # Wrong password
        self.assertFalse(auth.verify_password("hello124", rec))
        # The password itself is not stored
        self.assertNotIn("hello123", str(rec))
        # A new salt every time
        self.assertNotEqual(rec["salt"], auth.hash_password("hello123", iterations=1000)["salt"])
        # No password set: nobody gets in
        self.assertFalse(auth.verify_password("", None))

    # 5 failures lock for 60 s, then it opens again
    def test_lockout(self):
        # Fake clock
        now = [100.0]
        # Lockout with that clock
        lock = auth.Lockout(clock=lambda: now[0])
        # Four failures: still allowed
        for _ in range(4):
            # Fail
            lock.fail("1.2.3.4")
        # Not locked yet
        self.assertEqual(lock.check("1.2.3.4"), 0)
        # Fifth failure locks
        lock.fail("1.2.3.4")
        # About 60 s left
        self.assertAlmostEqual(lock.check("1.2.3.4"), 60.0)
        # Another address is not affected
        self.assertEqual(lock.check("5.6.7.8"), 0)
        # After 61 s it is open again
        now[0] += 61
        # Open
        self.assertEqual(lock.check("1.2.3.4"), 0)


# HTTP behaviour
class LoginHttpTest(StudioTestCase):
    # Pages redirect, the API answers 401
    def test_requires_login(self):
        # No cookies
        self.cookies = {}
        # Home page
        r = self.fetch("/", follow_redirects=False, raise_error=False)
        # Redirect to the login page
        self.assertEqual(r.code, 302)
        # With the login URL
        self.assertTrue(r.headers["Location"].startswith("/login"))
        # Every API route
        for url in ("/api/status", "/api/tasks", "/api/tasks/drive_check", "/api/jobs"):
            # Unauthorized as JSON
            self.assertEqual(self.api("GET", url)[0], 401, url)

    # Wrong password is refused, the right one logs in
    def test_wrong_and_right_password(self):
        # Wrong
        r = self.login("nope-nope")
        # Forbidden, no login cookie
        self.assertEqual(r.code, 403)
        # No cookie
        self.assertNotIn(auth.COOKIE_NAME, self.cookies)
        # Right
        r = self.login(PASSWORD)
        # Redirect home
        self.assertEqual(r.code, 302)
        # API works now
        self.assertEqual(self.api("GET", "/api/status")[0], 200)

    # Five wrong passwords lock the address, even for the right one
    def test_lockout_http(self):
        # Five failures
        for _ in range(5):
            # Wrong password
            self.login("wrong-wrong")
        # Right password while locked
        r = self.login(PASSWORD)
        # Too many requests
        self.assertEqual(r.code, 429)

    # A changed cookie is not accepted
    def test_tampered_cookie(self):
        # Log in
        self.login()
        # Valid cookie
        value = self.cookies[auth.COOKIE_NAME]
        # Change one character of the signed value
        self.cookies[auth.COOKIE_NAME] = value[:-2] + ("A" if value[-2] != "A" else "B") + value[-1]
        # Refused
        self.assertEqual(self.api("GET", "/api/status")[0], 401)
        # Forged unsigned value
        self.cookies[auth.COOKIE_NAME] = "operator"
        # Refused
        self.assertEqual(self.api("GET", "/api/status")[0], 401)

    # POST without the XSRF token is refused
    def test_xsrf_required(self):
        # Log in
        self.login()
        # POST without the header
        r = self.fetch("/api/estop", method="POST", body="{}", raise_error=False,
                       headers={"Cookie": self.cookie_header()})
        # Forbidden by tornado's XSRF check
        self.assertEqual(r.code, 403)

    # WebSockets close without a login
    @gen_test(timeout=10)
    async def test_websocket_requires_login(self):
        # No cookies
        self.cookies = {}
        # Connect
        conn = await self.ws("/ws/control/drive_check")
        # The server closes at once
        msg = await conn.read_message()
        # Closed (None) with the login code
        self.assertIsNone(msg)
        # Application close code
        self.assertEqual(conn.close_code, 4001)


# Run directly
if __name__ == "__main__":
    # unittest entry point
    unittest.main()
