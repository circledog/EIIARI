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

__all__ = ["Augmenter", "perlin_noise", "jpeg_compress", "low_res_compress"]


# --------------------------------------------------------------------------- #
# Perlin noise
# --------------------------------------------------------------------------- #
def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin_noise(height, width, cells, rng=None):
    """2-D gradient (Perlin) noise in roughly [-1, 1], shape (height, width).

    ``cells`` is the number of lattice cells along the longer side; small values
    give large, soft blobs (like shadows), large values give finer texture.
    """
    rng = rng if rng is not None else np.random
    cells = max(1, int(cells))
    cy = max(1, int(round(cells * height / float(max(height, width)))))
    cx = max(1, int(round(cells * width / float(max(height, width)))))
    angles = rng.uniform(0, 2 * np.pi, size=(cy + 1, cx + 1))
    grad = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    ys = np.linspace(0, cy, height, endpoint=False)
    xs = np.linspace(0, cx, width, endpoint=False)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    y0 = yy.astype(int)
    x0 = xx.astype(int)
    fy = yy - y0
    fx = xx - x0

    def dot(iy, ix, dy, dx):
        g = grad[iy, ix]
        return g[..., 0] * dx + g[..., 1] * dy

    n00 = dot(y0, x0, fy, fx)
    n01 = dot(y0, x0 + 1, fy, fx - 1)
    n10 = dot(y0 + 1, x0, fy - 1, fx)
    n11 = dot(y0 + 1, x0 + 1, fy - 1, fx - 1)
    u = _fade(fx)
    v = _fade(fy)
    nx0 = n00 + u * (n01 - n00)
    nx1 = n10 + u * (n11 - n10)
    return (nx0 + v * (nx1 - nx0)) * math.sqrt(2.0)


def _fractal_perlin(height, width, base_cells, octaves, rng):
    total = np.zeros((height, width), dtype=np.float64)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        total += amp * perlin_noise(height, width, base_cells * (2 ** o), rng)
        norm += amp
        amp *= 0.5
    return total / norm


# --------------------------------------------------------------------------- #
# Compression artefacts
# --------------------------------------------------------------------------- #
def jpeg_compress(image, quality):
    """Re-encode a PIL image as JPEG with the given quality (1-95)."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    out = PIL.Image.open(buf)
    out.load()
    return out.convert("RGB")


def low_res_compress(image, short_side, quality):
    """'144p video' look: shrink, JPEG re-encode, then scale back up."""
    w, h = image.size
    scale = float(short_side) / min(w, h)
    small = image.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                         PIL.Image.BILINEAR)
    small = jpeg_compress(small, quality)
    return small.resize((w, h), PIL.Image.BILINEAR)


# --------------------------------------------------------------------------- #
# Augmenter
# --------------------------------------------------------------------------- #
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

    def __init__(self, general=False, perlin=False, compression=False,
                 hflip_p=0.5, max_rotate=10.0, crop_scale=(0.8, 1.0), color_jitter=0.3,
                 perlin_p=0.7, perlin_strength=0.5, perlin_cells=(2, 5), perlin_octaves=2,
                 compression_p=0.7, jpeg_quality=(5, 35), low_res_p=0.5, low_res_side=(48, 112)):
        self.general = bool(general)
        self.perlin = bool(perlin)
        self.compression = bool(compression)
        self.hflip_p = hflip_p
        self.max_rotate = max_rotate
        self.crop_scale = crop_scale
        self.color_jitter = color_jitter
        self.perlin_p = perlin_p
        self.perlin_strength = perlin_strength
        self.perlin_cells = perlin_cells
        self.perlin_octaves = perlin_octaves
        self.compression_p = compression_p
        self.jpeg_quality = jpeg_quality
        self.low_res_p = low_res_p
        self.low_res_side = low_res_side

    @property
    def enabled(self):
        return self.general or self.perlin or self.compression

    def tag(self):
        """Short name for file names, e.g. 'gen+perlin' or 'none'."""
        parts = [n for n, on in (("gen", self.general), ("perlin", self.perlin),
                                 ("comp", self.compression)) if on]
        return "+".join(parts) if parts else "none"

    def __repr__(self):
        return "Augmenter(general=%s, perlin=%s, compression=%s)" % (
            self.general, self.perlin, self.compression)

    def __call__(self, image, x, y):
        if not self.enabled:
            return image, x, y
        image = image.convert("RGB")
        if self.general:
            image, x, y = self._geometric(image, x, y)
            image = self._color(image)
        if self.perlin and random.random() < self.perlin_p:
            image = self._perlin(image)
        # Compression last: it mimics what happens between the lens and the model.
        if self.compression and random.random() < self.compression_p:
            image = self._compress(image)
        return image, x, y

    # -- general ----------------------------------------------------------- #
    def _geometric(self, image, x, y):
        w, h = image.size
        if random.random() < self.hflip_p:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            x = w - x  # matches the course's normalised x -> -x

        if self.max_rotate > 0:
            for _ in range(10):
                deg = random.uniform(-self.max_rotate, self.max_rotate)
                nx, ny = _rotate_point(x, y, deg, w / 2.0, h / 2.0)
                if 0 <= nx < w and 0 <= ny < h:
                    image = image.rotate(deg, resample=PIL.Image.BILINEAR)
                    x, y = nx, ny
                    break

        lo, hi = self.crop_scale
        if lo < 1.0:
            for _ in range(10):
                s = random.uniform(lo, hi)
                cw, ch = w * s, h * s
                left = random.uniform(0, w - cw)
                top = random.uniform(0, h - ch)
                if left <= x < left + cw and top <= y < top + ch:
                    image = image.resize((w, h), PIL.Image.BILINEAR,
                                         box=(left, top, left + cw, top + ch))
                    x = (x - left) * w / cw
                    y = (y - top) * h / ch
                    break
        return image, x, y

    def _color(self, image):
        j = self.color_jitter
        if j <= 0:
            return image
        ops = [PIL.ImageEnhance.Brightness, PIL.ImageEnhance.Contrast, PIL.ImageEnhance.Color]
        random.shuffle(ops)
        for op in ops:
            image = op(image).enhance(random.uniform(1 - j, 1 + j))
        return image

    # -- perlin ------------------------------------------------------------ #
    def _perlin(self, image):
        w, h = image.size
        rng = np.random.RandomState(random.getrandbits(32))
        cells = random.randint(self.perlin_cells[0], self.perlin_cells[1])
        field = _fractal_perlin(h, w, cells, self.perlin_octaves, rng)
        amp = random.uniform(0.5, 1.0) * self.perlin_strength
        gain = 1.0 + amp * field
        arr = np.asarray(image, dtype=np.float32) * gain[..., None]
        return PIL.Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # -- compression ------------------------------------------------------- #
    def _compress(self, image):
        q = random.randint(self.jpeg_quality[0], self.jpeg_quality[1])
        if random.random() < self.low_res_p:
            side = random.randint(self.low_res_side[0], self.low_res_side[1])
            return low_res_compress(image, side, q)
        return jpeg_compress(image, q)


def _rotate_point(x, y, deg, cx, cy):
    """Where (x, y) lands after PIL ``Image.rotate(deg)`` (counter-clockwise, y down)."""
    t = math.radians(deg)
    dx, dy = x - cx, y - cy
    return (cx + dx * math.cos(t) + dy * math.sin(t),
            cy - dx * math.sin(t) + dy * math.cos(t))
