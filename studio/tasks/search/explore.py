"""Frontier exploration (TASKS M4-5): where to drive next.

Pick a frontier cluster of the grid (free next to unknown), preferring big
and near ones, and turn it into a direction for the robot:

* goal inside the camera's view -> a target column ``u`` for M3's planner
  (which keeps the obstacle avoidance and the model-free stop rule);
* goal beside / behind the robot -> turn on the spot towards it.

Python 3.6 compatible (runs on the Jetson Nano).
"""
import math


# Choose the next goal
def choose_goal(frontiers, pose, visited=(), min_dist=0.3, visit_radius=0.3):
    """Best frontier ``(x, y)`` or None.

    Score = cells / (1 + distance): big clusters win, far ones lose. Frontiers
    closer than ``min_dist`` (already being seen) and goals already reached
    (``visited``) are skipped.
    """
    # Robot position
    x, y, _ = pose
    # Best goal and score
    best, best_score = None, -1.0
    # Every cluster
    for fx, fy, cells in frontiers:
        # Distance from the robot
        d = math.hypot(fx - x, fy - y)
        # Too close: it is being seen right now
        if d < min_dist:
            # Skip
            continue
        # Already reached before (it did not disappear: unreachable / unobservable)
        if any(math.hypot(fx - vx, fy - vy) < visit_radius for vx, vy in visited):
            # Skip
            continue
        # Score
        score = cells / (1.0 + d)
        # Keep the best
        if score > best_score:
            # New best
            best, best_score = (fx, fy), score
    # Goal or None (nothing left to explore)
    return best


# Direction to a goal
def steer_to(goal, pose, hfov_deg):
    """Return ``('view', u)`` with the image column fraction, or ``('turn', +1 | -1)`` (left / right)."""
    # Robot pose
    x, y, yaw = pose
    # Bearing of the goal relative to the heading (left positive)
    b = math.atan2(goal[1] - y, goal[0] - x) - yaw
    # Wrap to [-pi, pi]
    b = math.atan2(math.sin(b), math.cos(b))
    # Usable part of the view (70 % of the half field of view)
    half = math.radians(hfov_deg) / 2.0 * 0.7
    # Outside the view: turn towards it
    if abs(b) > half:
        # Left for positive bearings
        return "turn", (1 if b > 0 else -1)
    # Image column: left of the centre for positive bearings (pinhole: tan mapping)
    u = 0.5 - 0.5 * math.tan(b) / math.tan(math.radians(hfov_deg) / 2.0)
    # Column fraction
    return "view", min(1.0, max(0.0, u))
