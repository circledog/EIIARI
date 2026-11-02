"""Lane detection and automatic target labelling for data collection.

Moved out of ``project4_data_collection_gamepad.ipynb`` so it can be tested
offline against the existing dataset (``python lane_label.py eval``).

* ``detect(image, profile)``      -- lane detection (indoor tape / outdoor profile 3).
* ``compute_midpoint(...)``       -- target point at a fixed look-ahead height.
* ``draw_guides(frame, ...)``     -- display-only overlay for the click image.

Notes from evaluating on the existing 200 images (``python lane_label.py eval``)
------------------------------------------------------------------------------
* The detector finds both tapes at the look-ahead height in ~37 % of images;
  agreement with the manual labels there is ~14 px (median).
* It fails mainly in curves: the tape ahead runs almost horizontally across
  the image and the inner tape is out of view, so no midpoint exists at a
  fixed height. Tracking the tapes from the bottom up instead of splitting at
  the image centre did not help (72 vs 74 found), and estimating the target
  from one tape +/- half a lane width pointed the wrong way in curves (median
  error 60 px), so neither is used. Curves are labelled by hand, with the
  click's y locked to the look-ahead line (see the collection notebook).

Python 3.6 compatible (runs on the Jetson Nano).
"""
import numpy as np
import cv2

__all__ = ["OUTDOOR_PROFILE_3", "INDOOR_PROFILE", "CENTERLINE_CONFIG", "detect",
           "detect_indoor", "detect_profile3", "compute_midpoint", "draw_midpoint",
           "draw_guides"]

OUTDOOR_PROFILE_3 = {'roi': [[0, 0.985], [0.465, 0.565], [0.51, 0.565], [1, 0.965], [1, 1], [0, 1]],
                     'vanishing_point': [0.49, 0.55],
                     'vp_tolerance': 0.025,
                     'contrast': 12,
                     'min_gray': 100,
                     'canny': [35, 100]}

# Indoor mat + tape: tolerate colour casts and curved markings, no fixed vanishing point.
INDOOR_PROFILE = {
    'name': 'indoor_tape',
    'working_width': 480,
    'roi': [[0.0, 0.98], [0.0, 0.45], [0.28, 0.12],
            [0.72, 0.12], [1.0, 0.45], [1.0, 0.98]],
    'contrast': 8,
    'min_gray': 60,
    'background_sigma_x': 15,
    'background_sigma_y': 3,
    'component_min_height': 0.10,
    'component_min_area': 0.0004,
    'min_row_width': 3,
    'max_row_width': 42,
    'fit_tolerance': 8,
    'min_lane_height': 0.25,
    'max_draw_gap': 0.025,
    'hough_threshold': 12,
    'hough_min_length': 12,
    'hough_max_gap': 10,
}

# Fractions of the image: y grows towards the robot.
CENTERLINE_CONFIG = {
    'near_x': 0.5,
    'near_y': 0.95,
    'far_y': 0.65,
    'min_lane_width': 0.10,
}


# --------------------------------------------------------------------------- #
# Outdoor profile (unchanged from the notebook, kept for comparison)
# --------------------------------------------------------------------------- #
def detect_profile3(image, profile, use_morphology=True):
    """Original Project 3 detector. Returns (overlay, stages, metadata)."""
    original_h, original_w = image.shape[:2]
    w = 1000
    h = round(original_h * w / original_w)
    frame = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA if original_w > w else cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    local = cv2.GaussianBlur(smooth, (31, 31), 0)
    contrast = cv2.subtract(smooth, local)
    white = (smooth >= profile['min_gray']) & (hsv[:, :, 1] <= 110)
    yellow = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 95)
    binary = np.uint8((contrast >= profile['contrast']) & (white | yellow)) * 255
    roi = np.zeros_like(gray)
    polygon = np.rint(np.array(profile['roi']) * [w - 1, h - 1]).astype(np.int32)
    cv2.fillPoly(roi, [polygon], 255)
    broad_white = np.uint8((smooth > 175) & (hsv[:, :, 1] < 70)) * 255
    broad_white = cv2.morphologyEx(broad_white, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(broad_white)
    occlusion = np.zeros_like(gray)
    for bx, by, bw, bh, area in stats[1:]:
        if 12 <= bw <= .12 * w and 8 <= bh <= .12 * h and area >= 80 and profile['vanishing_point'][1] * h - 8 <= by <= .62 * h:
            cv2.rectangle(occlusion, (max(0, bx - 9), max(0, by - 6)), (min(w - 1, bx + bw + 12), min(h - 1, by + bh + 10)), 255, -1)
    roi = cv2.bitwise_and(roi, cv2.bitwise_not(occlusion))
    binary = cv2.bitwise_and(binary, roi)
    morphology = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) if use_morphology else binary.copy()
    edges = cv2.Canny(smooth, *profile['canny'])
    candidates = cv2.bitwise_and(edges, cv2.dilate(morphology, np.ones((3, 3), np.uint8)))
    candidates = cv2.bitwise_and(candidates, roi)
    paint_support = cv2.dilate(morphology, np.ones((5, 5), np.uint8))
    raw = cv2.HoughLinesP(candidates, 1, np.pi / 180, 12, minLineLength=12, maxLineGap=10)
    raw = [] if raw is None else raw.reshape(-1, 4).tolist()
    vx, vy = np.array(profile['vanishing_point']) * [w, h]
    segments = []
    candidate_view = frame.copy()
    for x1, y1, x2, y2 in raw:
        cv2.line(candidate_view, (x1, y1), (x2, y2), (0, 165, 255), 1, cv2.LINE_AA)
        dx, dy = x2 - x1, y2 - y1
        if abs(dy) < 4 or abs(dy) / max(abs(dx), 1) < .15:
            continue
        a = dx / dy
        b = x1 - a * y1
        if abs(a * vy + b - vx) > w * profile['vp_tolerance']:
            continue
        n = max(2, int(np.hypot(dx, dy)))
        xs = np.clip(np.rint(np.linspace(x1, x2, n)).astype(int), 0, w - 1)
        ys = np.clip(np.rint(np.linspace(y1, y2, n)).astype(int), 0, h - 1)
        if np.mean(paint_support[ys, xs] > 0) < .65:
            continue
        segments.append({'points': [x1, y1, x2, y2], 'bottom': a * (h - 1) + b, 'slope': a, 'length': float(np.hypot(dx, dy))})

    groups = []

    def same_marking(s, anchor):
        if abs(s['bottom'] - anchor['bottom']) < 55 and abs(s['slope'] - anchor['slope']) < .50:
            return True
        ax, ay = anchor['points'][:2]
        endpoints = np.array(s['points']).reshape(2, 2)
        distance = np.abs(endpoints[:, 0] - ax - anchor['slope'] * (endpoints[:, 1] - ay)) / np.sqrt(1 + anchor['slope'] ** 2)
        return np.max(distance) < 8 and abs(np.arctan(s['slope']) - np.arctan(anchor['slope'])) < .10

    for s in sorted(segments, key=lambda s: -s['length']):
        match = next((g for g in groups if same_marking(s, g[0])), None)
        if match is None:
            groups.append([s])
        else:
            match.append(s)
    overlay = image.copy()
    detected_mask = np.zeros((original_h, original_w), np.uint8)
    lanes = []
    for group in groups:
        points = np.array([s['points'] for s in group]).reshape(-1, 2)
        if np.ptp(points[:, 1]) < 12 or sum(s['length'] for s in group) < 60:
            continue
        coeff = np.polyfit(points[:, 1], points[:, 0], 1)
        ys_all, xs_all = np.nonzero(morphology)
        distance = np.abs(xs_all - np.polyval(coeff, ys_all)) / np.sqrt(1 + coeff[0] ** 2)
        near = (distance < 8) & (ys_all >= vy + 5) & (ys_all <= points[:, 1].max() + 5)
        xs, ys = xs_all[near], ys_all[near]
        if len(xs) < 20:
            continue
        row_y = np.unique(ys)
        row_x = np.array([np.median(xs[ys == y]) for y in row_y])
        degree = 2 if np.ptp(row_y) > 45 and len(row_y) > 30 else 1
        coeff = np.polyfit(row_y, row_x, degree)
        row_parts = np.split(row_y, np.flatnonzero(np.diff(row_y) > 3) + 1)
        paths = []
        for part in row_parts:
            if len(part) < 3 or part[-1] - part[0] < 2:
                continue
            y = np.arange(part[0], part[-1] + 1, dtype=float)
            x = np.polyval(coeff, y)
            px = np.clip(np.rint(x).astype(int), 0, w - 1)
            py = y.astype(int)
            valid = (x >= 0) & (x < w) & (roi[py, px] > 0) & (paint_support[py, px] > 0)
            ids = np.flatnonzero(valid)
            for run in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1):
                if len(run) < 3:
                    continue
                xy = np.column_stack([x[run] * original_w / w, y[run] * original_h / h])
                xy = np.rint(xy).astype(np.int32)
                xy[:, 0] = np.clip(xy[:, 0], 0, original_w - 1)
                xy[:, 1] = np.clip(xy[:, 1], 0, original_h - 1)
                thickness = max(2, round(original_w / 400))
                cv2.polylines(overlay, [xy], False, (0, 0, 255), thickness, cv2.LINE_AA)
                cv2.polylines(detected_mask, [xy], False, 255, thickness)
                paths.append(xy.tolist())
        if paths:
            lanes.append({'paths': paths, 'hough_support': len(group)})
    stages = {'gray': gray, 'filter': smooth, 'contrast': contrast, 'bin': binary, 'roi': roi, 'occlusion': occlusion,
              'morphology': morphology, 'edge': edges, 'candidate_edge': candidates, 'hough': candidate_view,
              'detected_mask': detected_mask}
    metadata = {'input_size': [original_w, original_h], 'working_size': [w, h], 'profile': profile,
                'use_morphology': use_morphology, 'raw_hough_count': len(raw), 'selected_hough_count': len(segments),
                'lane_group_count': len(lanes), 'drawn_pixel_count': int(np.count_nonzero(detected_mask)), 'lanes': lanes}
    return overlay, stages, metadata


# --------------------------------------------------------------------------- #
# Indoor tape detector (original notebook algorithm)
# ---------------------------------------------------------------------------
def detect_indoor(image, profile, use_morphology=True):
    """以局部亮帶、連通區域與穩健二次擬合偵測左右膠帶；回傳格式與 detect 相同。"""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('輸入必須是 BGR 彩色影像')
    original_h, original_w = image.shape[:2]
    w = int(profile['working_width'])
    h = max(2, int(round(original_h * w / float(original_w))))
    frame = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA if original_w > w else cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    background = cv2.GaussianBlur(smooth, (0, 0),
                                  profile['background_sigma_x'], sigmaY=profile['background_sigma_y'])
    contrast = cv2.subtract(smooth, background)
    # 不使用 HSV 白色低飽和限制：畫面色偏時膠帶仍比鄰近地墊亮。
    binary = np.uint8((contrast >= profile['contrast']) &
                      (smooth >= profile['min_gray'])) * 255
    roi = np.zeros_like(gray)
    polygon = np.rint(np.asarray(profile['roi']) * [w - 1, h - 1]).astype(np.int32)
    cv2.fillPoly(roi, [polygon], 255)
    binary = cv2.bitwise_and(binary, roi)
    morphology = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) if use_morphology else binary.copy()
    # 地墊網格的亮點不應成為標線，先排除矮小／零碎區域。
    _, labels, stats, _ = cv2.connectedComponentsWithStats(morphology)
    clean = np.zeros_like(gray)
    components = 0
    for label in range(1, len(stats)):
        x, y, bw, bh, area = stats[label]
        if bh >= profile['component_min_height'] * h and area >= profile['component_min_area'] * w * h:
            clean[labels == label] = 255
            components += 1

    # PPT 必要流程：二值候選邊缘 -> HoughLinesP -> 依線段支持篩選。
    # linesP 直接保存 Hough 的原始輸出；它實際決定哪些候選列可進入曲線擬合。
    edge = cv2.Canny(clean, 50, 150)
    linesP = cv2.HoughLinesP(edge, 1, np.pi / 180, profile['hough_threshold'],
                             minLineLength=profile['hough_min_length'],
                             maxLineGap=profile['hough_max_gap'])
    raw_lines = [] if linesP is None else linesP.reshape(-1, 4).tolist()
    hough_support = np.zeros_like(gray)
    selected_lines = []
    hough_preview = frame.copy()
    for x1, y1, x2, y2 in raw_lines:
        dx, dy = x2 - x1, y2 - y1
        if abs(dy) < 4 or abs(dy) / max(abs(dx), 1) < 0.15:
            continue
        selected_lines.append([x1, y1, x2, y2])
        cv2.line(hough_support, (x1, y1), (x2, y2), 255, 9)
        cv2.line(hough_preview, (x1, y1), (x2, y2), (0, 165, 255), 2, cv2.LINE_AA)

    overlay = image.copy()
    lane_mask = np.zeros((original_h, original_w), np.uint8)
    lanes = []
    selected_rows = 0
    for side in ('left', 'right'):
        samples = []
        for y in range(h):
            xs = np.flatnonzero(clean[y])
            xs = xs[xs < w / 2.0] if side == 'left' else xs[xs >= w / 2.0]
            if not len(xs):
                continue
            runs = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
            runs = [run for run in runs if profile['min_row_width'] <= len(run) <= profile['max_row_width']]
            runs = [run for run in runs if np.any(hough_support[y, run] > 0)]
            if not runs:
                continue
            # 選道路內側的連續亮帶，避免選到更外側的牆面反光。
            run = max(runs, key=lambda r: r.mean()) if side == 'left' else min(runs, key=lambda r: r.mean())
            samples.append((float(np.median(run)), y))
        if len(samples) < max(20, int(.15 * h)):
            continue
        points = np.asarray(samples)
        x, y = points[:, 0], points[:, 1]
        if np.ptp(y) < profile['min_lane_height'] * h:
            continue
        # x=f(y) 可保留近垂直邊線；反覆剔除離群列，避免牆角／地墊接縫拉歪曲線。
        keep = np.ones(len(y), dtype=bool)
        for _ in range(4):
            if np.count_nonzero(keep) < 20:
                break
            coef = np.polyfit(y[keep] / h, x[keep], 2)
            residual = np.abs(x - np.polyval(coef, y / h))
            keep = residual <= profile['fit_tolerance']
        if np.count_nonzero(keep) < max(20, int(.15 * h)) or np.ptp(y[keep]) < profile['min_lane_height'] * h:
            continue
        coef = np.polyfit(y[keep] / h, x[keep], 2)
        support_y = y[keep].astype(int)
        paths = []
        max_gap = max(2, int(round(profile['max_draw_gap'] * h)))
        for rows in np.split(support_y, np.flatnonzero(np.diff(support_y) > max_gap) + 1):
            if len(rows) < 8:
                continue
            yy = np.arange(rows[0], rows[-1] + 1)
            xx = np.polyval(coef, yy / float(h))
            px = np.clip(np.rint(xx).astype(int), 0, w - 1)
            # 只在 ROI 內画線；不越過長遮擋區間補出一整條假線。
            valid = (xx >= 0) & (xx < w) & (roi[yy, px] > 0)
            indices = np.flatnonzero(valid)
            for chunk in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
                if len(chunk) < 8:
                    continue
                xy = np.rint(np.column_stack((xx[chunk] * original_w / w,
                                             yy[chunk] * original_h / h))).astype(np.int32)
                xy[:, 0] = np.clip(xy[:, 0], 0, original_w - 1)
                xy[:, 1] = np.clip(xy[:, 1], 0, original_h - 1)
                thickness = max(2, int(round(original_w / 220.0)))
                cv2.polylines(overlay, [xy], False, (0, 0, 255), thickness, cv2.LINE_AA)
                cv2.polylines(lane_mask, [xy], False, 255, thickness)
                paths.append(xy.tolist())
        if paths:
            selected_rows += int(np.count_nonzero(keep))
            lanes.append({'side': side, 'paths': paths, 'supported_rows': int(np.count_nonzero(keep))})
    stages = {'gray': gray, 'filter': smooth, 'contrast': contrast, 'bin': binary,
              'roi': roi, 'morphology': morphology, 'candidate_mask': clean,
              'detected_mask': lane_mask, 'edge': edge,
              'hough_support': hough_support, 'hough': hough_preview}
    metadata = {'input_size': [original_w, original_h], 'working_size': [w, h],
                'profile': profile, 'lane_group_count': len(lanes), 'lanes': lanes,
                'candidate_component_count': components, 'selected_row_count': selected_rows,
                'drawn_pixel_count': int(np.count_nonzero(lane_mask)),
                'raw_hough_count': len(raw_lines), 'selected_hough_count': len(selected_lines),
                'raw_hough_lines': raw_lines}
    return overlay, stages, metadata


def detect(image, profile, use_morphology=True):
    if profile.get('name') == 'indoor_tape':
        return detect_indoor(image, profile, use_morphology=use_morphology)
    return detect_profile3(image, profile, use_morphology=use_morphology)


# --------------------------------------------------------------------------- #
# Target point
# --------------------------------------------------------------------------- #
def _lane_x_at(lane, y_ref):
    samples = []
    for path in lane.get('paths', []):
        points = np.asarray(path, dtype=float)
        if points.ndim != 2 or len(points) < 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            continue
        points = points[np.argsort(points[:, 1])]
        if points[0, 1] <= y_ref <= points[-1, 1]:
            samples.append(float(np.interp(y_ref, points[:, 1], points[:, 0])))
    return float(np.median(samples)) if samples else None


def compute_midpoint(metadata, width, height, config=None):
    """Target at height ``far_y``: midpoint of the innermost left and right tape.

    Fills and returns ``metadata['centerline']`` (same fields as the original
    notebook's ``draw_lane_midpoint``). No extrapolation from previous frames.
    """
    cfg = dict(CENTERLINE_CONFIG)
    if config:
        cfg.update(config)
    near = [int(round((width - 1) * cfg['near_x'])), int(round((height - 1) * cfg['near_y']))]
    y_ref = int(round((height - 1) * cfg['far_y']))
    state = {'valid': False, 'near_point': near, 'far_point': None,
             'left_point': None, 'right_point': None, 'lookahead_y': y_ref,
             'reason': 'missing_lanes_at_lookahead'}
    metadata['centerline'] = state
    left, right = [], []
    for lane in metadata.get('lanes', []):
        x = _lane_x_at(lane, y_ref)
        if x is None or not 0 <= x < width:
            continue
        # The indoor detector already tags sides; do not reclassify by the image centre.
        side = lane.get('side') or ('left' if x < width / 2.0 else 'right')
        (left if side == 'left' else right).append(x)
    if not left or not right:
        return state
    lx, rx = max(left), min(right)
    if rx - lx < cfg['min_lane_width'] * width:
        state['reason'] = 'crossed_or_too_narrow'
        return state
    state.update(valid=True, far_point=[int(round((lx + rx) / 2.0)), y_ref],
                 left_point=[lx, y_ref], right_point=[rx, y_ref], reason='both_lanes_detected')
    return state


def draw_midpoint(image, state):
    """Draw the near point (blue), target point (green) and the line between them."""
    height, width = image.shape[:2]
    radius = max(3, width // 100)
    near = tuple(state['near_point'])
    cv2.circle(image, near, radius, (255, 0, 0), -1)
    if not state.get('valid'):
        return None
    far = tuple(state['far_point'])
    cv2.line(image, near, far, (0, 0, 255), max(2, width // 250), cv2.LINE_AA)
    cv2.circle(image, far, radius, (0, 255, 0), -1)
    cv2.circle(image, near, radius, (255, 0, 0), -1)
    return far


def draw_guides(frame, far_y, lock_y=True):
    """Copy of ``frame`` for the click widget only (never saved for training).

    Draws the look-ahead line at ``far_y``: with ``lock_y`` the click's y is
    replaced by this height, so only the left/right position matters.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    y = int(round((h - 1) * far_y))
    color = (0, 255, 255) if lock_y else (128, 128, 128)
    for x0 in range(0, w, 8):
        cv2.line(out, (x0, y), (min(w - 1, x0 + 4), y), color, 1)
    for frac in (0.25, 0.5, 0.75):
        x = int(round((w - 1) * frac))
        cv2.line(out, (x, y - 3), (x, y + 3), color, 1)
    return out


# --------------------------------------------------------------------------- #
# Offline evaluation against an existing dataset
# --------------------------------------------------------------------------- #
def _evaluate(args):
    import glob
    import os
    paths = sorted(glob.glob(os.path.join(args.data, "xy_*.jpg")))
    stats, diffs, tiles = {}, [], []
    for p in paths:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        overlay, _, meta = detect(img, INDOOR_PROFILE)
        state = compute_midpoint(meta, w, h)
        stats[state['reason']] = stats.get(state['reason'], 0) + 1
        parts = os.path.basename(p).split('_')
        x, y = int(parts[1]), int(parts[2])
        if state['valid'] and abs(y - state['lookahead_y']) <= args.y_window:
            diffs.append(abs(state['far_point'][0] - x))
        if args.sheet:
            draw_midpoint(overlay, state)
            cv2.circle(overlay, (x, y), 5, (255, 0, 255), 2)
            tiles.append(cv2.resize(overlay, (160, 160)))
    n = len(paths)
    both = stats.get('both_lanes_detected', 0)
    print("images %d | midpoint found %d (%.0f%%) | %s" % (n, both, 100.0 * both / n, stats))
    if diffs:
        print("|x_auto - x_manual| (labels with |y - lookahead| <= %d): n=%d median %.1f px, mean %.1f, >20px %d"
              % (args.y_window, len(diffs), np.median(diffs), np.mean(diffs), sum(v > 20 for v in diffs)))
    if args.sheet:
        cols = 10
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(args.sheet, sheet)
        print("wrote", args.sheet, "(green = auto target, magenta circle = manual label)")


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser(description="lane_label tools")
    sub = ap.add_subparsers(dest="cmd")
    e = sub.add_parser("eval", help="run the auto-labeller on a dataset and compare with its labels")
    e.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "data", "dataset_xy_2026-10-21", "dataset_xy"))
    e.add_argument("--y-window", type=int, default=20)
    e.add_argument("--sheet", default=None, help="write a contact sheet jpg")
    a = ap.parse_args()
    if a.cmd == "eval":
        _evaluate(a)
    else:
        ap.print_help()
