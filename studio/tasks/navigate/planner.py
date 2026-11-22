"""Target direction + free sectors -> target point for project4's controller (TASKS M3-4).

The output is a normalised point ``(x, y)`` in [-1, 1] -- exactly what the
lane model predicts -- so ``project4/control.py``'s ``SteeringController``
(steering formula, PID, speed, slew) is reused unchanged:

* ``x``: centre of the chosen sector;
* ``y``: higher in the picture the more free space the sector has, so the
  controller drives faster where it is free and slows down near obstacles
  (the teammate formula stops by itself when the point reaches the bottom).

Choice of the sector: the target's sector when it is free enough, otherwise
the free sector closest to it.

Thresholds in **metres** when a ``distance`` function (sector score ->
metres, from the camera model) is given -- use this on the robot: with a
wide, tilted camera the bottom 40 % of the picture covers only the first few
centimetres of floor, so fractions of the picture height are misleading
(found by the exploration simulation: a 0.12 fraction meant ~4 cm). Without
it the thresholds are fractions of the analysed height (unit tests).

**Last protection, independent of any model**: when the sectors straight
ahead are blocked closer than ``stop_distance``, or any sector closer than
``side_distance`` (an obstacle beside the front, which the centre sectors do
not see -- the simulation's robot scraped along a box corner without it), the
planner returns ``'too_close'`` and the robot stops.

Python 3.6 compatible (runs on the Jetson Nano).
"""


# Sector choice and target point
class Planner(object):
    """``step(target_u, scores)`` -> ``(xy or None, status, sector)``."""

    # Store the settings
    def __init__(self, free_threshold=0.35, stop_distance=0.12, horizon=0.45, centre_sectors=3, distance=None,
                 side_distance=None):
        # Free distance a sector needs to be driven into
        self.free_threshold = float(free_threshold)
        # Free distance straight ahead below which the robot must stop
        self.stop_distance = float(stop_distance)
        # Horizon of the free-space estimate (fraction of the height)
        self.horizon = float(horizon)
        # Sectors in the middle that count as "straight ahead"
        self.centre_sectors = int(centre_sectors)
        # Sector score -> metres (None: thresholds are fractions)
        self.distance = distance
        # Closest any sector may be (default: 8 cm in metres mode, off in fraction mode)
        self.side_distance = side_distance if side_distance is not None else (0.08 if distance is not None else 0.0)

    # Free distance of a sector in the thresholds' unit
    def _free(self, score):
        # Metres via the camera model, or the fraction itself
        return self.distance(score) if self.distance is not None else score

    # Sector index of a horizontal position u (0..1)
    @staticmethod
    def sector_of(u, n):
        """Index of the sector containing ``u``."""
        # Clamp into the last sector for u = 1
        return min(n - 1, max(0, int(u * n)))

    # One planning step
    def step(self, target_u, scores):
        """``target_u``: target's horizontal position (0..1) or None (go straight ahead).

        Returns ``(xy, status, sector)``; ``xy`` is None when the robot must stop.
        Status: ``ok``, ``detour`` (target blocked, driving around), ``too_close``, ``blocked``.
        """
        # Number of sectors
        n = len(scores)
        # Middle sectors
        mid = n // 2
        # Half of the centre band
        half = self.centre_sectors // 2
        # Free distance straight ahead (worst of the centre sectors)
        ahead = min(self._free(s) for s in scores[max(0, mid - half):mid + half + 1])
        # Closest obstacle in any sector (beside the front)
        nearest = min(self._free(s) for s in scores)
        # Last protection: obstacle right in front, or very close at the side
        if ahead < self.stop_distance or nearest < self.side_distance:
            # Stop
            return None, "too_close", None
        # Sector of the target (straight ahead without a target)
        want = self.sector_of(target_u, n) if target_u is not None else mid
        # Sectors free enough to drive into
        free = [i for i in range(n) if self._free(scores[i]) >= self.free_threshold]
        # Nowhere to go
        if not free:
            # Stop
            return None, "blocked", None
        # Target sector free: go there; otherwise the closest free one (ties: towards the middle)
        chosen = want if want in free else min(free, key=lambda i: (abs(i - want), abs(i - mid)))
        # Horizontal position: the target itself when its sector is chosen, else the sector centre
        u = target_u if (chosen == want and target_u is not None) else (chosen + 0.5) / n
        # Free distance of the chosen sector -> how high the point is in the picture
        s = scores[chosen]
        # Row as a fraction of the height: bottom (1.0) minus the free part up to the horizon
        v = 1.0 - s * (1.0 - self.horizon)
        # Normalised point for the controller
        xy = (2.0 * u - 1.0, 2.0 * v - 1.0)
        # Result
        return xy, ("ok" if chosen == want else "detour"), chosen
