"""Dataset thumbnails for the web pages (imports OpenCV / numpy only when used).

``render(path, mode, label, size)`` -> JPEG bytes:

* ``raw``: the image with its label (green dot, line from the robot);
* ``lanes``: the same after ``project4/preprocessing.py``'s lane marking, i.e.
  exactly what the network sees when trained with lane marking.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import sys

# project4 folder
from studio import PROJECT4_DIR


# Load OpenCV and the lane marking on first use
def _modules():
    # project4 on the import path
    if PROJECT4_DIR not in sys.path:
        # Put project4/ first
        sys.path.insert(0, PROJECT4_DIR)
    import cv2
    # Lane marking of project4 (cv2 / numpy / PIL only, no torch)
    from preprocessing import mark_lanes
    # Both
    return cv2, mark_lanes


# Image -> annotated JPEG
def render(path, mode="raw", label=None, size=224, quality=80):
    """JPEG bytes of ``path`` (BGR jpg) in ``mode`` with the pixel ``label`` drawn."""
    # Lazy imports
    cv2, mark_lanes = _modules()
    # Read the image
    img = cv2.imread(path)
    # Unreadable file
    if img is None:
        # Report it
        raise ValueError("cannot read %s" % path)
    # Lane marking like training / the robot
    if mode == "lanes":
        # Paint the tape red
        img = mark_lanes(img)
    # Label: dot on the target and a line from the bottom centre (the robot)
    if label is not None:
        # Image height and width
        h, w = img.shape[:2]
        # Pixel coordinates
        x, y = int(label[0]), int(label[1])
        # Line from the robot
        cv2.line(img, (w // 2, h - 1), (x, y), (0, 255, 0), 1)
        # Dot on the target
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
    # Scale for the page
    if size and size != img.shape[1]:
        # Keep the square shape
        img = cv2.resize(img, (int(size), int(size)))
    # Encode
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    # Bytes
    return bytes(buf)


# Find the images that cannot be decoded completely
def unreadable(paths):
    """Paths whose image does not decode (truncated / empty files), like data_loader._readable."""
    import PIL.Image
    # Broken files
    bad = []
    # Check each
    for p in paths:
        # Full decode, not only the header
        try:
            # Open and decode every pixel
            with PIL.Image.open(p) as im:
                # Decode
                im.load()
        # Any error: unusable
        except Exception:
            # Remember it
            bad.append(p)
    # Broken files
    return bad
