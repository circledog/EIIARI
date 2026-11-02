"""Data augmentation for road-following (x, y) regression.

Three independently switchable groups:

* ``general``     -- geometric (h-flip, rotation, crop/zoom) + colour jitter.
                     Geometric ops move the (x, y) label together with the image.
* ``perlin``      -- smooth Perlin-noise brightness field (uneven lighting / shadows).
* ``compression`` -- JPEG artefacts or "YouTube 144p" style down-scale + re-encode.

With every switch off, ``Augmenter`` returns the input unchanged.

Labels are passed in *pixel* coordinates of the given image, exactly like the
``xy_<x>_<y>_<uuid>.jpg`` file names.

Randomness uses Python's ``random`` module, which PyTorch re-seeds in every
DataLoader worker, so workers do not produce identical augmentations.
"""
import io
import math
import random

import numpy as np
import PIL.Image
import PIL.ImageEnhance

# Public names of this module
__all__ = ["Augmenter", "perlin_noise", "jpeg_compress", "low_res_compress"]


# --------------------------------------------------------------------------- #
# Perlin noise
# --------------------------------------------------------------------------- #
# Perlin's smoothstep curve 6t^5 - 15t^4 + 10t^3 for blending between lattice points
def _fade(t):
    # Polynomial with zero 1st and 2nd derivative at 0 and 1 (no visible seams)
    return t * t * t * (t * (t * 6 - 15) + 10)


# One octave of 2-D Perlin noise
def perlin_noise(height, width, cells, rng=None):
    """2-D gradient (Perlin) noise in roughly [-1, 1], shape (height, width).

    ``cells`` is the number of lattice cells along the longer side; small values
    give large, soft blobs (like shadows), large values give finer texture.
    """
    # Use numpy's global generator unless one is given
    rng = rng if rng is not None else np.random
    # At least one lattice cell
    cells = max(1, int(cells))
    # Number of cells vertically, keeping square cells
    cy = max(1, int(round(cells * height / float(max(height, width)))))
    # Number of cells horizontally
    cx = max(1, int(round(cells * width / float(max(height, width)))))
    # Random gradient direction at every lattice corner
    angles = rng.uniform(0, 2 * np.pi, size=(cy + 1, cx + 1))
    # Unit gradient vectors (cos, sin) from the angles
    grad = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    # Lattice-space y coordinate of every pixel row
    ys = np.linspace(0, cy, height, endpoint=False)
    # Lattice-space x coordinate of every pixel column
    xs = np.linspace(0, cx, width, endpoint=False)
    # Full grids of lattice coordinates
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    # Lattice cell row of every pixel
    y0 = yy.astype(int)
    # Lattice cell column of every pixel
    x0 = xx.astype(int)
    # Position inside the cell, vertically (0..1)
    fy = yy - y0
    # Position inside the cell, horizontally (0..1)
    fx = xx - x0

    # Dot product of a corner's gradient with the offset to the pixel
    def dot(iy, ix, dy, dx):
        # Gradient vectors of the selected corners
        g = grad[iy, ix]
        # g . (dx, dy)
        return g[..., 0] * dx + g[..., 1] * dy

    # Contribution of the top-left corner
    n00 = dot(y0, x0, fy, fx)
    # Contribution of the top-right corner
    n01 = dot(y0, x0 + 1, fy, fx - 1)
    # Contribution of the bottom-left corner
    n10 = dot(y0 + 1, x0, fy - 1, fx)
    # Contribution of the bottom-right corner
    n11 = dot(y0 + 1, x0 + 1, fy - 1, fx - 1)
    # Smoothed horizontal blend weight
    u = _fade(fx)
    # Smoothed vertical blend weight
    v = _fade(fy)
    # Blend the top two corners
    nx0 = n00 + u * (n01 - n00)
    # Blend the bottom two corners
    nx1 = n10 + u * (n11 - n10)
    # Blend top and bottom, scaled to roughly [-1, 1]
    return (nx0 + v * (nx1 - nx0)) * math.sqrt(2.0)


# Sum of several Perlin octaves (coarse blobs plus finer detail)
def _fractal_perlin(height, width, base_cells, octaves, rng):
    # Accumulated noise
    total = np.zeros((height, width), dtype=np.float64)
    # Current octave's amplitude and the sum of all amplitudes
    amp, norm = 1.0, 0.0
    # One pass per octave
    for o in range(octaves):
        # Each octave doubles the frequency
        total += amp * perlin_noise(height, width, base_cells * (2 ** o), rng)
        # Track the total amplitude for normalisation
        norm += amp
        # Each octave has half the amplitude of the previous one
        amp *= 0.5
    # Normalise back to roughly [-1, 1]
    return total / norm


# --------------------------------------------------------------------------- #
# Compression artefacts
# --------------------------------------------------------------------------- #
# Simulate heavy JPEG compression
def jpeg_compress(image, quality):
    """Re-encode a PIL image as JPEG with the given quality (1-95)."""
    # In-memory file
    buf = io.BytesIO()
    # Encode as JPEG at the requested quality
    image.save(buf, format="JPEG", quality=int(quality))
    # Rewind to read it back
    buf.seek(0)
    # Decode the compressed image
    out = PIL.Image.open(buf)
    # Force decoding now, before the buffer goes away
    out.load()
    # Return as RGB
    return out.convert("RGB")


# Simulate a low-resolution video stream
def low_res_compress(image, short_side, quality):
    """'144p video' look: shrink, JPEG re-encode, then scale back up."""
    # Original size
    w, h = image.size
    # Scale factor that brings the short side down to `short_side` pixels
    scale = float(short_side) / min(w, h)
    # Shrink the image
    small = image.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                         PIL.Image.BILINEAR)
    # Add JPEG artefacts at the low resolution
    small = jpeg_compress(small, quality)
    # Blow it back up to the original size (blurry + blocky)
    return small.resize((w, h), PIL.Image.BILINEAR)


# --------------------------------------------------------------------------- #
# Augmenter
# --------------------------------------------------------------------------- #
# Random augmentation applied to (image, label) pairs during training
class Augmenter(object):
    """Callable ``(PIL image, x_px, y_px) -> (PIL image, x_px, y_px)``.

    Parameters
    ----------
    general, perlin, compression : bool
        The three on/off switches. All False -> identity.
    hflip_p : float
        Probability of a horizontal flip (label x is mirrored).
    max_rotate : float
        Max rotation in degrees (uniform in [-max_rotate, max_rotate]).
    crop_scale : (float, float)
        Range of crop side length relative to the image (then resized back).
    color_jitter : float
        Strength of brightness / contrast / saturation jitter (0.3 like the course).
    perlin_p, perlin_strength, perlin_cells, perlin_octaves
        Probability, max brightness amplitude, lattice-cell range and octaves.
    compression_p, jpeg_quality, low_res_p, low_res_side
        Probability, JPEG quality range, share of the "144p" variant among
        compression ops, and its short-side range in pixels.
    """

    # Store the switches and all tuning parameters
    def __init__(self, general=False, perlin=False, compression=False,
                 hflip_p=0.5, max_rotate=10.0, crop_scale=(0.8, 1.0), color_jitter=0.3,
                 perlin_p=0.7, perlin_strength=0.5, perlin_cells=(2, 5), perlin_octaves=2,
                 compression_p=0.7, jpeg_quality=(5, 35), low_res_p=0.5, low_res_side=(48, 112)):
        # Switch: flip / rotate / crop + colour jitter
        self.general = bool(general)
        # Switch: Perlin-noise lighting
        self.perlin = bool(perlin)
        # Switch: JPEG / 144p artefacts
        self.compression = bool(compression)
        # Probability of a horizontal flip
        self.hflip_p = hflip_p
        # Largest rotation in degrees
        self.max_rotate = max_rotate
        # Range of the crop size relative to the image
        self.crop_scale = crop_scale
        # Strength of the brightness / contrast / saturation jitter
        self.color_jitter = color_jitter
        # Probability of adding Perlin noise
        self.perlin_p = perlin_p
        # Largest brightness change caused by the noise
        self.perlin_strength = perlin_strength
        # Range of lattice cells (blob size) for the noise
        self.perlin_cells = perlin_cells
        # Number of noise octaves
        self.perlin_octaves = perlin_octaves
        # Probability of a compression artefact
        self.compression_p = compression_p
        # JPEG quality range
        self.jpeg_quality = jpeg_quality
        # Share of the "144p" variant among compression artefacts
        self.low_res_p = low_res_p
        # Short-side range in pixels for the "144p" variant
        self.low_res_side = low_res_side

    # True when at least one switch is on
    @property
    def enabled(self):
        # Any of the three groups enabled
        return self.general or self.perlin or self.compression

    # Short name of the enabled groups, used in file names
    def tag(self):
        """Short name for file names, e.g. 'gen+perlin' or 'none'."""
        # Names of the switches that are on, in a fixed order
        parts = [n for n, on in (("gen", self.general), ("perlin", self.perlin),
                                 ("comp", self.compression)) if on]
        # Joined with '+', or 'none' when all are off
        return "+".join(parts) if parts else "none"

    # Readable representation for log output
    def __repr__(self):
        # Show the three switches
        return "Augmenter(general=%s, perlin=%s, compression=%s)" % (
            self.general, self.perlin, self.compression)

    # Augment one sample
    def __call__(self, image, x, y):
        # All switches off: identity
        if not self.enabled:
            # Unchanged sample
            return image, x, y
        # Work on RGB images
        image = image.convert("RGB")
        # General group: geometry first (moves the label), then colour
        if self.general:
            # Flip / rotate / crop, label moved along
            image, x, y = self._geometric(image, x, y)
            # Brightness / contrast / saturation
            image = self._color(image)
        # Perlin group, applied with probability perlin_p
        if self.perlin and random.random() < self.perlin_p:
            # Uneven lighting
            image = self._perlin(image)
        # Compression last: it mimics what happens between the lens and the model.
        if self.compression and random.random() < self.compression_p:
            # JPEG or 144p artefacts
            image = self._compress(image)
        # Augmented sample
        return image, x, y

    # -- general ----------------------------------------------------------- #
    # Random flip, rotation and crop; the label follows the image
    def _geometric(self, image, x, y):
        # Image size
        w, h = image.size
        # Horizontal flip with probability hflip_p
        if random.random() < self.hflip_p:
            # Mirror the image left-right
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            # Mirror the label too (matches the course's normalised x -> -x)
            x = w - x

        # Small random rotation
        if self.max_rotate > 0:
            # Up to 10 tries to find an angle that keeps the label inside the image
            for _ in range(10):
                # Random angle in [-max_rotate, max_rotate]
                deg = random.uniform(-self.max_rotate, self.max_rotate)
                # Where the label lands after this rotation
                nx, ny = _rotate_point(x, y, deg, w / 2.0, h / 2.0)
                # Accept the angle only if the label stays in the frame
                if 0 <= nx < w and 0 <= ny < h:
                    # Rotate the image about its centre
                    image = image.rotate(deg, resample=PIL.Image.BILINEAR)
                    # Use the rotated label
                    x, y = nx, ny
                    # Done with rotation
                    break

        # Smallest and largest crop size
        lo, hi = self.crop_scale
        # Crop only if smaller crops are allowed
        if lo < 1.0:
            # Up to 10 tries to find a crop that contains the label
            for _ in range(10):
                # Random relative crop size
                s = random.uniform(lo, hi)
                # Crop width and height in pixels
                cw, ch = w * s, h * s
                # Random left edge
                left = random.uniform(0, w - cw)
                # Random top edge
                top = random.uniform(0, h - ch)
                # Accept the crop only if the label is inside it
                if left <= x < left + cw and top <= y < top + ch:
                    # Cut the box out and scale it back to the full size (zoom in)
                    image = image.resize((w, h), PIL.Image.BILINEAR,
                                         box=(left, top, left + cw, top + ch))
                    # Label x in the zoomed image
                    x = (x - left) * w / cw
                    # Label y in the zoomed image
                    y = (y - top) * h / ch
                    # Done with cropping
                    break
        # Transformed image and label
        return image, x, y

    # Random brightness, contrast and saturation (in random order)
    def _color(self, image):
        # Jitter strength
        j = self.color_jitter
        # No jitter requested
        if j <= 0:
            # Unchanged image
            return image
        # The three PIL enhancers
        ops = [PIL.ImageEnhance.Brightness, PIL.ImageEnhance.Contrast, PIL.ImageEnhance.Color]
        # Apply them in a random order
        random.shuffle(ops)
        # Apply each enhancer
        for op in ops:
            # Random factor in [1 - j, 1 + j]
            image = op(image).enhance(random.uniform(1 - j, 1 + j))
        # Colour-jittered image
        return image

    # -- perlin ------------------------------------------------------------ #
    # Multiply the image by a smooth random brightness field
    def _perlin(self, image):
        # Image size
        w, h = image.size
        # numpy generator seeded from Python's random (re-seeded per DataLoader worker)
        rng = np.random.RandomState(random.getrandbits(32))
        # Random blob size
        cells = random.randint(self.perlin_cells[0], self.perlin_cells[1])
        # Noise field in roughly [-1, 1]
        field = _fractal_perlin(h, w, cells, self.perlin_octaves, rng)
        # Random amplitude between half and full strength
        amp = random.uniform(0.5, 1.0) * self.perlin_strength
        # Per-pixel brightness factor around 1
        gain = 1.0 + amp * field
        # Apply the factor to every colour channel
        arr = np.asarray(image, dtype=np.float32) * gain[..., None]
        # Clip to valid pixel values and convert back to an image
        return PIL.Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # -- compression ------------------------------------------------------- #
    # Add JPEG or 144p artefacts
    def _compress(self, image):
        # Random JPEG quality
        q = random.randint(self.jpeg_quality[0], self.jpeg_quality[1])
        # Sometimes use the low-resolution variant
        if random.random() < self.low_res_p:
            # Random short side for the shrunken image
            side = random.randint(self.low_res_side[0], self.low_res_side[1])
            # Shrink, compress, enlarge
            return low_res_compress(image, side, q)
        # Plain JPEG re-encode
        return jpeg_compress(image, q)


# Position of a point after rotating the image about (cx, cy)
def _rotate_point(x, y, deg, cx, cy):
    """Where (x, y) lands after PIL ``Image.rotate(deg)`` (counter-clockwise, y down)."""
    # Degrees to radians
    t = math.radians(deg)
    # Offset from the rotation centre
    dx, dy = x - cx, y - cy
    # Rotate the offset (y axis points down) and add the centre back
    return (cx + dx * math.cos(t) + dy * math.sin(t),
            cy - dx * math.sin(t) + dy * math.cos(t))
