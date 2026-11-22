"""Wire format between the web process and a runner process.

Each message is a small JSON header plus an optional binary payload
(a JPEG frame)::

    4 bytes  header length (big endian)
    4 bytes  payload length (big endian)
    header   JSON, ASCII only, must contain "type"
    payload  raw bytes

The runner side uses blocking sockets in threads (``Connection``); the web
side reads with tornado's IOStream (``read_message``).

Connections only go to 127.0.0.1 and the runner must present the random
token it got on its command line.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import json
import socket
import struct
import threading

# Two big-endian unsigned 32-bit lengths
_PREFIX = struct.Struct(">II")
# Largest header accepted (protects against garbage)
MAX_HEADER = 1 << 20
# Largest payload accepted (a 224x224 JPEG is ~15 kB)
MAX_PAYLOAD = 16 << 20


# Encode one message
def pack(header, payload=b""):
    """Bytes for ``header`` (dict with 'type') and ``payload``."""
    # ASCII-only JSON
    body = json.dumps(header, separators=(",", ":")).encode("ascii")
    # Prefix + header + payload
    return _PREFIX.pack(len(body), len(payload)) + body + payload


# Check the two lengths of a prefix
def unpack_prefix(prefix):
    """Return ``(header_len, payload_len)``; raises ``ValueError`` if too large."""
    # Decode the lengths
    hlen, plen = _PREFIX.unpack(prefix)
    # Reject absurd sizes
    if hlen > MAX_HEADER or plen > MAX_PAYLOAD:
        # Corrupt or hostile stream
        raise ValueError("message too large (%d / %d bytes)" % (hlen, plen))
    # Both lengths
    return hlen, plen


# Size of the prefix in bytes
PREFIX_SIZE = _PREFIX.size


# Blocking connection used by the runner process
class Connection(object):
    """Thread-safe ``send``; ``recv`` from one reader thread."""

    # Wrap a connected socket
    def __init__(self, sock):
        # The socket
        self.sock = sock
        # Small messages (heartbeats) must not wait for Nagle
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Several threads send (frames, telemetry, replies)
        self._send_lock = threading.Lock()

    # Open a connection to the web process
    @classmethod
    def connect(cls, port, host="127.0.0.1", timeout=10.0):
        """Connect to ``host:port``."""
        # TCP connection
        sock = socket.create_connection((host, port), timeout=timeout)
        # Blocking mode from now on
        sock.settimeout(None)
        # Wrapped connection
        return cls(sock)

    # Send one message
    def send(self, header, payload=b""):
        """Send atomically (whole message or exception)."""
        # Encode outside the lock
        data = pack(header, payload)
        # One message at a time
        with self._send_lock:
            # Send everything
            self.sock.sendall(data)

    # Read exactly n bytes
    def _read_exact(self, n):
        # Buffer for the bytes
        buf = bytearray(n)
        # View for partial reads
        view = memoryview(buf)
        # Bytes received so far
        got = 0
        # Until complete
        while got < n:
            # Receive into the rest of the buffer
            k = self.sock.recv_into(view[got:], n - got)
            # Peer closed the connection
            if k == 0:
                # Signal end of stream
                raise EOFError("connection closed")
            # Count them
            got += k
        # Complete bytes
        return bytes(buf)

    # Receive one message
    def recv(self):
        """Return ``(header, payload)``; raises ``EOFError`` when closed."""
        # Lengths
        hlen, plen = unpack_prefix(self._read_exact(PREFIX_SIZE))
        # Header JSON
        header = json.loads(self._read_exact(hlen).decode("ascii"))
        # Payload (may be empty)
        payload = self._read_exact(plen) if plen else b""
        # Both parts
        return header, payload

    # Close the socket
    def close(self):
        """Close; a blocked ``recv`` in another thread gets ``EOFError`` / ``OSError``."""
        # Shut down both directions first so a blocked recv returns
        try:
            # Wake the reader
            self.sock.shutdown(socket.SHUT_RDWR)
        # Already closed
        except OSError:
            # Nothing to do
            pass
        # Release the socket
        self.sock.close()


# Read one message from a tornado IOStream (web process)
async def read_message(stream):
    """Return ``(header, payload)`` from ``stream`` (tornado ``IOStream``)."""
    # Lengths
    hlen, plen = unpack_prefix(await stream.read_bytes(PREFIX_SIZE))
    # Header JSON
    header = json.loads((await stream.read_bytes(hlen)).decode("ascii"))
    # Payload (may be empty)
    payload = (await stream.read_bytes(plen)) if plen else b""
    # Both parts
    return header, payload
