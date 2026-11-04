"""Camera input and pre-processing.

``open_camera(source)`` returns an object with ``read()`` (latest BGR uint8
frame, HxWx3) and ``stop()``:

* ``'jetbot'``  -- ``jetbot.Camera`` (CSI camera on the robot, 224x224).
* ``'csi'``     -- the same CSI camera through our own GStreamer pipeline with
                   ``appsink drop=true max-buffers=1``: GStreamer keeps only the
                   newest frame, so a stalled loop can never build a backlog.
                   Do not open it while a ``jetbot.Camera`` is running.
* ``'opencv'``  -- ``cv2.VideoCapture(index)`` (USB webcam / PC testing).
* ``'folder'``  -- loops over jpgs in a folder (offline testing with the dataset).
* ``'dummy'``   -- synthetic frames, no hardware needed.

``preprocess(frame)`` reproduces the slides' P.59 ``preprocess`` (BGR frame in,
normalised NCHW tensor out), with the dtype chosen per model instead of a
hard-coded ``.half()`` -- trtexec engines expect FP32 (handoff 4.2).

Every source also has ``flush()``: drop frames that queued up while the loop
was stalled (e.g. loading a model), so the next ``read()`` is live again.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import glob
import os
import time

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF

# Public names of this module
__all__ = ["open_camera", "preprocess", "draw_overlay", "to_jpeg"]

# ImageNet per-channel mean used for normalisation (same as training)
MEAN = [0.485, 0.456, 0.406]
# ImageNet per-channel standard deviation used for normalisation
STD = [0.229, 0.224, 0.225]
# Side length of the square network input
INPUT_SIZE = 224


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
# Frame source backed by the jetbot library's Camera (the default on the robot)
class _JetbotSource(object):
    # Open the robot's CSI camera through jetbot
    def __init__(self, width, height, fps):
        from jetbot import Camera
        # Older jetbot releases expose a singleton
        try:
            # Get (or create) the shared camera instance
            self.camera = Camera.instance(width=width, height=height, fps=fps)
        # Newer jetbot: plain constructor
        except AttributeError:
            # Create the camera directly
            self.camera = Camera(width=width, height=height, fps=fps)

    # Latest frame captured by jetbot's background thread
    def read(self):
        # BGR uint8 array (or None before the first frame)
        return self.camera.value

    # Drop frames that queued up while the loop was stalled
    def flush(self, timeout=3.0):
        """Restart the GStreamer pipeline to drop queued frames.

        jetbot's ``appsink`` has no ``drop=true``: if the capture thread is
        starved (e.g. the GIL is held while a TensorRT engine deserialises),
        frames pile up and are then delivered at the camera rate, so the
        delay never goes away. Restarting the pipeline empties the queue.
        Returns True once a fresh frame has arrived.
        """
        # Short name for the camera
        cam = self.camera
        # Remember the current frame object to detect when a new one arrives
        old = cam.value
        # Stop the pipeline, which throws away the queued frames
        cam.stop()
        # Start it again
        cam.start()
        # Time the wait for a fresh frame
        t0 = time.time()
        # Wait at most `timeout` seconds
        while time.time() - t0 < timeout:
            # A different frame object means the restarted pipeline delivered
            if cam.value is not old:
                # Flushed successfully
                return True
            # Check again shortly
            time.sleep(0.02)
        # No new frame in time
        return False

    # Release the camera
    def stop(self):
        # Stop jetbot's capture pipeline
        self.camera.stop()


# Frame source with our own GStreamer pipeline that never builds a backlog
class _CSISource(object):
    """CSI camera via GStreamer, keeping only the newest frame (no backlog)."""

    # Build and open the GStreamer pipeline
    def __init__(self, width, height, fps, capture_width=3280, capture_height=2464):
        # Sensor -> resize -> BGR -> appsink that keeps only the newest frame
        gst = ("nvarguscamerasrc sensor-mode=3 ! "
               "video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, "
               "framerate=(fraction)%d/1 ! nvvidconv ! "
               "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
               "videoconvert ! video/x-raw, format=(string)BGR ! "
               "appsink drop=true max-buffers=1 sync=false"
               % (capture_width, capture_height, fps, width, height))
        # Open the pipeline through OpenCV's GStreamer backend
        self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        # Opening fails when another program holds the camera
        if not self.cap.isOpened():
            # Explain the most likely cause
            raise RuntimeError("cannot open CSI camera; is another notebook / jetbot.Camera using it?")

    # Grab the newest frame
    def read(self):
        # Blocking read of the next frame
        ok, frame = self.cap.read()
        # None when the read failed
        return frame if ok else None

    # Nothing to flush
    def flush(self):
        # appsink drop=true max-buffers=1 never queues
        return True

    # Release the camera
    def stop(self):
        # Close the GStreamer pipeline
        self.cap.release()


# Frame source for a USB webcam or a PC camera
class _OpenCVSource(object):
    # Open the camera by index
    def __init__(self, index, width, height):
        # OpenCV capture device
        self.cap = cv2.VideoCapture(index)
        # Ask the driver to buffer as few frames as possible
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Size every frame is resized to
        self.size = (width, height)

    # Grab the next frame, resized
    def read(self):
        # Blocking read
        ok, frame = self.cap.read()
        # Read failed (camera unplugged, ...)
        if not ok:
            # Signal "no frame"
            return None
        # Resize to the network's input size
        return cv2.resize(frame, self.size)

    # Throw away frames buffered by the driver
    def flush(self, max_frames=30):
        """Grab until a grab actually waits for the camera (= queue is empty)."""
        # Try at most max_frames grabs
        for _ in range(max_frames):
            # Time this grab
            t0 = time.time()
            # Grab (and discard) one frame
            self.cap.grab()
            # A slow grab means we had to wait for the camera: the buffer is empty
            if time.time() - t0 > 0.015:
                # Stop flushing
                break
        # Flushing never fails here
        return True

    # Release the camera
    def stop(self):
        # Close the capture device
        self.cap.release()


# Frame source that replays the jpgs of a folder (offline testing)
class _FolderSource(object):
    # Collect the images and set the replay rate
    def __init__(self, folder, width, height, fps):
        # All jpgs of the folder in a stable order
        self.paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        # An empty folder cannot be replayed
        if not self.paths:
            # Report the folder that has no images
            raise FileNotFoundError("no jpg in %s" % folder)
        # Size every frame is resized to
        self.size = (width, height)
        # Seconds each image is shown
        self.period = 1.0 / fps
        # Start time of the replay
        self.t0 = time.time()

    # Image that is "live" at the current time
    def read(self):
        # Index advances with time and wraps around
        i = int((time.time() - self.t0) / self.period) % len(self.paths)
        # Load and resize that image
        return cv2.resize(cv2.imread(self.paths[i]), self.size)

    # Nothing to flush
    def flush(self):
        # Files never queue up
        return True

    # Nothing to release
    def stop(self):
        # No hardware to close
        pass


# Synthetic frame source for running the loop without any camera
class _DummySource(object):
    """A moving synthetic 'road' so the whole loop can run without hardware."""

    # Remember the frame size and the start time
    def __init__(self, width, height):
        # Size of the generated frames
        self.size = (width, height)
        # Start time, drives the road's sideways motion
        self.t0 = time.time()

    # Draw a frame with two white lines that sway left and right
    def read(self):
        # Frame width and height
        w, h = self.size
        # Dark grey background
        img = np.full((h, w, 3), 60, np.uint8)
        # Sideways offset that follows a sine wave over time
        off = int(0.25 * w * np.sin(time.time() - self.t0))
        # Current road centre
        cx = w // 2 + off
        # Left lane line, converging towards the top
        cv2.line(img, (cx - w // 3, h - 1), (cx - w // 12, h // 3), (255, 255, 255), 3)
        # Right lane line
        cv2.line(img, (cx + w // 3, h - 1), (cx + w // 12, h // 3), (255, 255, 255), 3)
        # The synthetic frame
        return img

    # Nothing to flush
    def flush(self):
        # Generated frames never queue up
        return True

    # Nothing to release
    def stop(self):
        # No hardware to close
        pass


# Create the requested frame source
def open_camera(source="jetbot", width=224, height=224, fps=21, index=0, folder=None):
    """Open a frame source; see module docstring."""
    # Robot camera through the jetbot library
    if source == "jetbot":
        # jetbot.Camera wrapper
        return _JetbotSource(width, height, fps)
    # Robot camera through our drop-old-frames pipeline
    if source == "csi":
        # Own GStreamer pipeline
        return _CSISource(width, height, fps)
    # USB / PC camera
    if source == "opencv":
        # cv2.VideoCapture wrapper
        return _OpenCVSource(index, width, height)
    # Replay a folder of images
    if source == "folder":
        # Two images per second
        return _FolderSource(folder, width, height, fps=2)
    # Synthetic frames
    if source == "dummy":
        # Moving fake road
        return _DummySource(width, height)
    # Any other name is a typo
    raise ValueError("unknown camera source %r" % source)


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #
# Mean / std tensors already placed on a device, keyed by (device, dtype)
_norm_cache = {}


# Mean / std tensors for a device and dtype, created once and reused
def _mean_std(device, dtype):
    # Cache key
    key = (str(device), dtype)
    # Build the tensors the first time this combination is used
    if key not in _norm_cache:
        # Shape (3, 1, 1) so they broadcast over a CHW image
        _norm_cache[key] = (torch.tensor(MEAN, device=device, dtype=dtype)[:, None, None],
                            torch.tensor(STD, device=device, dtype=dtype)[:, None, None])
    # Cached tensors
    return _norm_cache[key]


# Turn a camera frame into the network's input tensor (slides P.59)
def preprocess(frame, device="cpu", dtype=torch.float32):
    """BGR uint8 HxWx3 -> normalised (1, 3, 224, 224) tensor (channels stay BGR).

    Matches training (``data_loader.image_to_tensor``): the course flips RGB to
    BGR when training, and the JetBot camera already delivers BGR.
    """
    # Resize only when the frame is not already 224x224
    if frame.shape[0] != INPUT_SIZE or frame.shape[1] != INPUT_SIZE:
        # Resize to the network input size
        frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    # Same PIL path as training, so the result is bit-identical
    image = PIL.Image.fromarray(frame)
    # HWC uint8 -> CHW float in [0, 1], on the target device and dtype
    t = TF.to_tensor(image).to(device=device, dtype=dtype)
    # Normalisation tensors for this device / dtype
    mean, std = _mean_std(t.device, dtype)
    # Normalise in place: (t - mean) / std
    t.sub_(mean).div_(std)
    # Add the batch dimension
    return t[None, ...]


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
# Draw the prediction and the steering on a copy of the frame
def draw_overlay(frame, xy=None, steering=None):
    """Copy of ``frame`` with the predicted target point and a steering bar.

    ``xy`` is in normalised coordinates ([-1, 1]).
    """
    # Never draw on the original frame
    out = frame.copy()
    # Frame height and width
    h, w = out.shape[:2]
    # Draw the target point when there is a usable prediction (NaN / inf cannot be drawn)
    if xy is not None and np.all(np.isfinite(np.asarray(xy, dtype=np.float64)[:2])):
        # Normalised x -> pixel column
        px = int(round((float(xy[0]) + 1) * w / 2.0))
        # Normalised y -> pixel row
        py = int(round((float(xy[1]) + 1) * h / 2.0))
        # Green line from the bottom centre (the robot) to the target
        cv2.line(out, (w // 2, h - 1), (px, py), (0, 255, 0), 2)
        # Green dot on the target
        cv2.circle(out, (px, py), 5, (0, 255, 0), -1)
    # Draw the steering bar when a steering value is given
    if steering is not None and np.isfinite(steering):
        # Bar starts at the horizontal centre
        cx = w // 2
        # Bar length proportional to steering, clamped to [-1, 1]
        bar = int(max(-1.0, min(1.0, steering)) * (w // 2 - 4))
        # Orange bar at the top: right of centre = turning right
        cv2.rectangle(out, (cx, 4), (cx + bar, 10), (0, 165, 255), -1)
        # White tick marking zero steering
        cv2.line(out, (cx, 2), (cx, 12), (255, 255, 255), 1)
    # Annotated copy
    return out


# Encode a frame as JPEG for the notebook's image widget
def to_jpeg(frame, quality=80):
    """BGR frame -> JPEG bytes for ``ipywidgets.Image`` (like jetbot.bgr8_to_jpeg)."""
    # JPEG-encode with the given quality
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    # Raw bytes, or empty bytes if encoding failed
    return bytes(buf) if ok else b""
