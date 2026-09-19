"""Part B -- localise an arbitrary crop and drive the camera until the marker is visible.

Movement convention (one convention, used everywhere)
-----------------------------------------------------
A command describes a **translation of the camera parallel to the port plane**,
in reference-image axes::

    RIGHT = +x_img      DOWN = +y_img

That model is not a convenience: translating a pinhole camera by
``t = (tx, ty, 0)`` while it keeps looking straight at a plane at depth ``d``
induces the homography ``H = K (I + t n^T / d) K^-1`` with ``n = (0, 0, 1)``,
which is an **exact pure image translation** of ``-f t / d``.  So "the field of
view slides over the reference image" is physically exact for this motion.
(A pan/tilt would *not* be a crop: ``H = K R K^-1`` is a projective warp.  No
pan/tilt angles are emitted by this module.)

The field-of-view window therefore translates by the same vector as the camera,
and the image content inside it appears to move the opposite way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config as C
from .detect import PortDetection
from .io import to_gray

__all__ = [
    "CropLocalization",
    "MovementCommand",
    "NavigationPlan",
    "extract_crop",
    "pixels_per_cm",
    "localize_crop",
    "marker_visible",
    "marker_visibility_target",
    "plan_crop_movements",
    "simulate_crop_navigation",
    "explore_until_localized",
]


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

@dataclass
class CropLocalization:
    """Where a crop sits in the reference image, and whether that is trustworthy."""

    status: str                       #: 'ok' | 'ambiguous' | 'featureless'
    x: Optional[int] = None
    y: Optional[int] = None
    w: int = 0
    h: int = 0
    score: float = float("nan")
    second_peak: float = float("nan")
    second_peak_ratio: float = float("nan")
    plateau_extent_px: Tuple[int, int] = (0, 0)
    uncertainty_px: Tuple[float, float] = (float("nan"), float("nan"))
    crop_std: float = float("nan")
    reasons: List[str] = field(default_factory=list)

    @property
    def window(self) -> Optional[Tuple[int, int, int, int]]:
        if self.x is None or self.y is None:
            return None
        return (int(self.x), int(self.y), int(self.w), int(self.h))


@dataclass
class MovementCommand:
    step: int
    direction: str                    #: 'RIGHT' | 'LEFT' | 'UP' | 'DOWN'
    dx_px: float
    dy_px: float
    distance_px: float
    distance_cm: float
    window_after: Tuple[int, int, int, int]

    def describe(self) -> str:
        return ("step {:>3d}: move camera {:<5s} {:6.1f} px  ({:5.2f} cm)  "
                "-> window (x={:4d}, y={:4d})"
                .format(self.step, self.direction, self.distance_px,
                        self.distance_cm, self.window_after[0], self.window_after[1]))


@dataclass
class NavigationPlan:
    status: str                       #: 'ok' | 'already_visible' | 'ambiguous' | 'featureless' | 'unreachable'
    commands: List[MovementCommand] = field(default_factory=list)
    start_window: Optional[Tuple[int, int, int, int]] = None
    target_window: Optional[Tuple[int, int, int, int]] = None
    goal: str = ""                    #: 'full_disk' | 'marker_centre'
    step_px: float = float("nan")
    total_px: Tuple[float, float] = (0.0, 0.0)
    total_cm: Tuple[float, float] = (0.0, 0.0)
    optimal_steps: int = 0
    reasons: List[str] = field(default_factory=list)
    localization: Optional[CropLocalization] = None

    @property
    def n_steps(self) -> int:
        return len(self.commands)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def extract_crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Crop with bounds checking (no padding: the window must lie inside)."""
    H, W = image.shape[:2]
    x, y = int(x), int(y)
    if x < 0 or y < 0 or x + w > W or y + h > H:
        raise ValueError("crop window ({}, {}, {}, {}) leaves the {}x{} image"
                         .format(x, y, w, h, W, H))
    return image[y:y + h, x:x + w].copy()


def pixels_per_cm(detection: PortDetection) -> Tuple[float, float]:
    """Per-axis raster scale, px/cm, from the detected outer square.

    Uses the canonical quad (marker corner first, then clockwise), so side 0/2
    are the port's top/bottom and side 1/3 its right/left.  These are the same
    per-axis scales that the texture homography ``G`` encodes, so the cm figures
    printed next to the pixel movements are consistent with Parts C/D.
    """
    q = np.asarray(detection.quad, float)
    lens = np.array([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)], float)
    width_px = 0.5 * (lens[0] + lens[2])
    height_px = 0.5 * (lens[1] + lens[3])
    return (width_px / C.OUTER_SIDE_CM, height_px / C.OUTER_SIDE_CM)


# ---------------------------------------------------------------------------
# Localisation
# ---------------------------------------------------------------------------

def localize_crop(crop: np.ndarray,
                  reference: np.ndarray,
                  min_score: Optional[float] = None) -> CropLocalization:
    """Locate ``crop`` inside ``reference`` and judge whether that is identifiable.

    Three independent identifiability tests, all scaled to the crop's own size:

    1. **Texture**: a crop whose grey-level standard deviation is below
       :data:`config.CROP_MIN_STD` carries no information -> ``featureless``.
    2. **Aperture / plateau**: the connected set of scores within
       :data:`config.MATCH_PLATEAU_TOL` of the peak must not span more than
       :data:`config.MATCH_PLATEAU_MAX_FRAC` of the crop along either axis.  A
       crop showing only one straight border slides freely along it and fails
       here on exactly one axis.
    3. **Second peak**: after suppressing a neighbourhood of the best match, a
       competing peak within :data:`config.MATCH_SECOND_PEAK_RATIO` of it means
       the crop is repeated in the image -> ``ambiguous``.

    A non-``ok`` status is a *refusal to guess*: :func:`plan_crop_movements`
    emits no commands for it.
    """
    if min_score is None:
        min_score = C.MATCH_MIN_SCORE

    cg = to_gray(crop).astype(np.float32)
    rg = to_gray(reference).astype(np.float32)
    h, w = cg.shape[:2]
    H, W = rg.shape[:2]
    std = float(cg.std())

    if h > H or w > W:
        return CropLocalization(status="ambiguous", w=w, h=h, crop_std=std,
                                reasons=["crop is larger than the reference image"])
    if std < C.CROP_MIN_STD:
        return CropLocalization(status="featureless", w=w, h=h, crop_std=std,
                                reasons=["crop grey-level std {:.2f} < {:.2f}: "
                                         "no structure to match".format(std, C.CROP_MIN_STD)])

    res = cv2.matchTemplate(rg, cg, cv2.TM_CCOEFF_NORMED)
    if not np.isfinite(res).all():
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, peak, _, loc = cv2.minMaxLoc(res)
    px, py = int(loc[0]), int(loc[1])

    reasons: List[str] = []

    # --- plateau extent around the peak ------------------------------------
    plateau = (res >= peak - C.MATCH_PLATEAU_TOL).astype(np.uint8)
    n_lab, labels = cv2.connectedComponents(plateau, connectivity=8)
    lab = int(labels[py, px])
    ys, xs = np.nonzero(labels == lab)
    ext_x = int(xs.max() - xs.min() + 1)
    ext_y = int(ys.max() - ys.min() + 1)
    max_x = max(1.0, C.MATCH_PLATEAU_MAX_FRAC * w)
    max_y = max(1.0, C.MATCH_PLATEAU_MAX_FRAC * h)

    # --- second peak after NMS ---------------------------------------------
    sup = res.copy()
    rx = max(1, int(round(C.MATCH_NMS_FRAC * w)))
    ry = max(1, int(round(C.MATCH_NMS_FRAC * h)))
    x0, x1 = max(0, px - rx), min(res.shape[1], px + rx + 1)
    y0, y1 = max(0, py - ry), min(res.shape[0], py + ry + 1)
    sup[y0:y1, x0:x1] = -np.inf
    finite = np.isfinite(sup)
    second = float(sup[finite].max()) if finite.any() else float("-inf")
    ratio = (second / peak) if peak > 1e-6 else float("inf")

    status = "ok"
    if peak < min_score:
        status = "ambiguous"
        reasons.append("best match score {:.3f} < {:.2f}".format(peak, min_score))
    if ext_x > max_x or ext_y > max_y:
        status = "ambiguous"
        reasons.append("score plateau spans {}x{} px (limit {:.0f}x{:.0f}); the "
                       "crop slides along at least one direction (aperture problem)"
                       .format(ext_x, ext_y, max_x, max_y))
    if np.isfinite(second) and second >= min_score and second >= C.MATCH_SECOND_PEAK_RATIO * peak:
        status = "ambiguous"
        reasons.append("a competing match at {:.3f} rivals the best {:.3f} "
                       "(ratio {:.3f}); the crop is repeated in the image"
                       .format(second, peak, ratio))

    return CropLocalization(
        status=status,
        x=px if status == "ok" else px,
        y=py if status == "ok" else py,
        w=w, h=h,
        score=float(peak),
        second_peak=float(second) if np.isfinite(second) else float("nan"),
        second_peak_ratio=float(ratio) if np.isfinite(ratio) else float("nan"),
        plateau_extent_px=(ext_x, ext_y),
        uncertainty_px=(0.5 * (ext_x - 1), 0.5 * (ext_y - 1)),
        crop_std=std,
        reasons=reasons or ["unique, sharply-peaked match"],
    )


# ---------------------------------------------------------------------------
# Visibility goal
# ---------------------------------------------------------------------------

def _marker_box(detection: PortDetection, margin_frac: Optional[float] = None
                ) -> Tuple[float, float, float, float]:
    if margin_frac is None:
        margin_frac = C.NAV_MARKER_MARGIN_FRAC
    c = np.asarray(detection.marker_center, float)
    r = float(detection.marker_radius_px) * (1.0 + margin_frac)
    return (c[0] - r, c[1] - r, c[0] + r, c[1] + r)


def marker_visible(window: Sequence[int],
                   detection: PortDetection,
                   goal: str = "auto") -> bool:
    """Is the marker inside ``window = (x, y, w, h)``?

    ``goal='full_disk'`` needs the whole disk (plus margin) inside the window,
    ``goal='marker_centre'`` only its centre.  ``'auto'`` falls back to the
    centre criterion when the window is too small to ever contain the disk.
    """
    x, y, w, h = [int(v) for v in window]
    if not detection.marker_found:
        raise ValueError("detection carries no marker")
    x0, y0, x1, y1 = _marker_box(detection)
    if goal == "auto":
        goal = "full_disk" if (w >= (x1 - x0) and h >= (y1 - y0)) else "marker_centre"
    if goal == "full_disk":
        return bool(x <= x0 and y <= y0 and x + w >= x1 and y + h >= y1)
    c = np.asarray(detection.marker_center, float)
    return bool(x <= c[0] <= x + w and y <= c[1] <= y + h)


def marker_visibility_target(window: Sequence[int],
                             detection: PortDetection,
                             image_shape: Sequence[int],
                             goal: str = "auto") -> Tuple[Tuple[int, int], str]:
    """Nearest window top-left that satisfies the visibility goal (clamped to the image)."""
    x, y, w, h = [int(v) for v in window]
    H, W = int(image_shape[0]), int(image_shape[1])
    x0, y0, x1, y1 = _marker_box(detection)
    if goal == "auto":
        goal = "full_disk" if (w >= (x1 - x0) and h >= (y1 - y0)) else "marker_centre"

    if goal == "full_disk":
        lo_x, hi_x = x1 - w, x0
        lo_y, hi_y = y1 - h, y0
    else:
        c = np.asarray(detection.marker_center, float)
        lo_x, hi_x = c[0] - w, c[0]
        lo_y, hi_y = c[1] - h, c[1]

    # The window origin is an integer, so tighten the real-valued admissible
    # interval to integers *before* clamping.  Rounding afterwards would let the
    # chosen origin fall a fraction of a pixel outside the interval and leave a
    # sliver of the marker off-screen.
    lo_xi, hi_xi = int(np.ceil(lo_x - 1e-9)), int(np.floor(hi_x + 1e-9))
    lo_yi, hi_yi = int(np.ceil(lo_y - 1e-9)), int(np.floor(hi_y + 1e-9))
    if (hi_xi < lo_xi or hi_yi < lo_yi) and goal == "full_disk":
        # The window cannot contain the whole disk plus its margin; fall back.
        return marker_visibility_target(window, detection, image_shape, "marker_centre")

    tx = int(np.clip(int(x), lo_xi, hi_xi))
    ty = int(np.clip(int(y), lo_yi, hi_yi))
    tx = int(np.clip(tx, 0, max(0, W - w)))
    ty = int(np.clip(ty, 0, max(0, H - h)))
    return (tx, ty), goal


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_crop_movements(localization: CropLocalization,
                        detection: PortDetection,
                        image_shape: Sequence[int],
                        step_px: Optional[float] = None,
                        goal: str = "auto") -> NavigationPlan:
    """Turn a *confident* localisation into a list of camera translations.

    If ``localization.status`` is not ``'ok'`` the plan is empty -- the correct
    answer to "where should the camera go?" when the crop cannot be located is
    "unknown", not a best guess.
    """
    if localization.status != "ok":
        return NavigationPlan(
            status=localization.status,
            reasons=(["localisation is not confident, so no movement commands "
                      "are issued"] + list(localization.reasons)),
            localization=localization,
        )

    win = localization.window
    assert win is not None
    x, y, w, h = win
    (tx, ty), goal_used = marker_visibility_target(win, detection, image_shape, goal)

    sx, sy = pixels_per_cm(detection)

    if (tx, ty) == (x, y) and marker_visible(win, detection, goal_used):
        return NavigationPlan(status="already_visible", commands=[],
                              start_window=win, target_window=win, goal=goal_used,
                              step_px=0.0, total_px=(0.0, 0.0), total_cm=(0.0, 0.0),
                              optimal_steps=0,
                              reasons=["the circular marker is already inside the view"],
                              localization=localization)

    if step_px is None:
        step_px = max(1.0, C.NAV_STEP_FRAC * min(w, h))

    dx_total = float(tx - x)
    dy_total = float(ty - y)

    # Dominant axis first, 4-connected moves.
    axes = [("x", dx_total), ("y", dy_total)]
    axes.sort(key=lambda kv: -abs(kv[1]))

    commands: List[MovementCommand] = []
    cx, cy = float(x), float(y)
    n = 0
    for axis, total in axes:
        remaining = total
        while abs(remaining) > 1e-9:
            if n >= C.NAV_MAX_STEPS:                      # pragma: no cover
                break
            d = float(np.sign(remaining)) * min(step_px, abs(remaining))
            if axis == "x":
                cx += d
                direction = "RIGHT" if d > 0 else "LEFT"
                dx, dy = d, 0.0
                dist_cm = abs(d) / sx
            else:
                cy += d
                direction = "DOWN" if d > 0 else "UP"
                dx, dy = 0.0, d
                dist_cm = abs(d) / sy
            remaining -= d
            n += 1
            commands.append(MovementCommand(
                step=n, direction=direction, dx_px=dx, dy_px=dy,
                distance_px=abs(d), distance_cm=dist_cm,
                window_after=(int(round(cx)), int(round(cy)), w, h)))

    optimal = int(np.ceil(abs(dx_total) / step_px)) + int(np.ceil(abs(dy_total) / step_px))
    return NavigationPlan(
        status="ok",
        commands=commands,
        start_window=win,
        target_window=(int(tx), int(ty), w, h),
        goal=goal_used,
        step_px=float(step_px),
        total_px=(dx_total, dy_total),
        total_cm=(dx_total / sx, dy_total / sy),
        optimal_steps=optimal,
        reasons=["camera translation parallel to the port plane, under the "
                 "frontal pinhole model"],
        localization=localization,
    )


def simulate_crop_navigation(reference: np.ndarray,
                             plan: NavigationPlan) -> List[Dict[str, object]]:
    """Re-render the view at every step of ``plan``.

    Each frame is cut fresh from the reference image -- nothing is warped from
    the previous frame, so there is no accumulation of resampling error.
    """
    frames: List[Dict[str, object]] = []
    if plan.start_window is None:
        return frames
    H, W = reference.shape[:2]

    def _grab(win):
        x, y, w, h = win
        x = int(np.clip(x, 0, max(0, W - w)))
        y = int(np.clip(y, 0, max(0, H - h)))
        return (x, y, w, h), extract_crop(reference, x, y, w, h)

    win, img = _grab(plan.start_window)
    frames.append({"step": 0, "window": win, "image": img, "direction": "START",
                   "distance_px": 0.0, "distance_cm": 0.0})
    for cmd in plan.commands:
        win, img = _grab(cmd.window_after)
        frames.append({"step": cmd.step, "window": win, "image": img,
                       "direction": cmd.direction,
                       "distance_px": cmd.distance_px,
                       "distance_cm": cmd.distance_cm})
    return frames


# ---------------------------------------------------------------------------
# OPTIONAL extension (not part of the core Part B deliverable)
# ---------------------------------------------------------------------------

def explore_until_localized(reference: np.ndarray,
                            detection: PortDetection,
                            start_window: Sequence[int],
                            step_px: Optional[float] = None,
                            max_steps: Optional[int] = None
                            ) -> Dict[str, object]:
    """OPTIONAL: closed-loop square-spiral search for an *ambiguous* start view.

    This is an extension, not part of the Part B answer.  The assignment asks
    for movement commands given a crop; when the crop cannot be localised the
    core pipeline reports that and stops.  This routine shows what an agent
    *could* do instead: walk an outward square spiral, re-localising the live
    view at every stop, and switch to the deterministic plan the moment the view
    becomes identifiable.
    """
    H, W = reference.shape[:2]
    x, y, w, h = [int(v) for v in start_window]
    if step_px is None:
        # A *search* stride, not a tracking stride: consecutive stops should
        # barely overlap, otherwise a small view spends thousands of moves
        # re-examining almost the same pixels.  Half the view's larger side
        # covers the image in O(area / view area) stops.
        step_px = max(1.0, C.EXPLORE_STEP_FRAC * max(w, h))
    if max_steps is None:
        max_steps = C.NAV_MAX_STEPS
    step = int(max(1, round(step_px)))

    trail = [(x, y)]
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    leg, di, taken = 1, 0, 0
    stalled = 0

    while taken < max_steps:
        loc = localize_crop(extract_crop(reference, x, y, w, h), reference)
        if loc.status == "ok":
            plan = plan_crop_movements(loc, detection, reference.shape)
            return {"status": "localized", "steps_explored": taken,
                    "step_px": float(step), "window": (x, y, w, h),
                    "localization": loc, "plan": plan, "trail": trail}
        dx, dy = directions[di % 4]
        moved = False
        for _ in range(leg):
            if taken >= max_steps:
                break
            nx = int(np.clip(x + dx * step, 0, max(0, W - w)))
            ny = int(np.clip(y + dy * step, 0, max(0, H - h)))
            if (nx, ny) != (x, y):
                x, y = nx, ny
                trail.append((x, y))
                taken += 1
                moved = True
        di += 1
        if di % 2 == 0:
            leg += 1
        stalled = 0 if moved else stalled + 1
        if stalled >= 4:        # the spiral has grown past the image on all sides
            break

    return {"status": "exhausted", "steps_explored": taken, "step_px": float(step),
            "window": (x, y, w, h), "localization": None, "plan": None,
            "trail": trail}
