"""Part B -- localise an arbitrary crop and drive the camera until the marker is visible.

Movement convention (one convention, used everywhere)
-----------------------------------------------------
A command describes a **translation of the camera parallel to the port plane**,
in reference-image axes::

    RIGHT = +x_img      DOWN = +y_img

That model is not a convenience: translating a pinhole camera by
``delta_C = (tx, ty, 0)`` while it keeps looking straight at a plane at depth ``d``
induces ``H = K (I - delta_C n^T / d) K^-1`` with ``n = (0, 0, 1)``,
an **exact pure image translation** of ``-f delta_C / d``. The relative
extrinsic translation is ``t_rel = -delta_C``, consistent with the PLUS form
used in Parts C/D. So "the field of
view slides over the reference image" is physically exact for this motion.
(A pan/tilt would *not* be a crop: ``H = K R K^-1`` is a projective warp.  No
pan/tilt angles are emitted by this module.)

The field-of-view window therefore translates by the same vector as the camera,
and the image content inside it appears to move the opposite way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
    "navigate_until_visible",
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

def _window_values(window: Sequence[int], image_shape=None) -> Tuple[int, int, int, int]:
    """Validate pixel windows instead of silently truncating coordinates."""
    try:
        x, y, w, h = (operator.index(v) for v in window)
    except (TypeError, ValueError) as exc:
        raise ValueError("window must contain four integer coordinates/dimensions") from exc
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError("window origin must be non-negative and dimensions positive")
    if image_shape is not None:
        H, W = image_shape[:2]
        if x + w > W or y + h > H:
            raise ValueError("crop window leaves the reference image")
    return x, y, w, h


def extract_crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Crop with bounds checking (no padding: the window must lie inside)."""
    x, y, w, h = _window_values((x, y, w, h), image.shape)
    return image[y:y + h, x:x + w].copy()


def pixels_per_cm(detection: PortDetection) -> Tuple[float, float]:
    """Per-axis raster scale, px/cm, from the detected outer square.

    Uses the canonical quad (marker corner first, then clockwise), so side 0/2
    are the port's top/bottom and side 1/3 its right/left.  These are the same
    mean raster side lengths used by Parts C/D. These constant scales give
    approximate centimetre movements; the schematic quad's projective ``G``
    has spatially varying scale and is not exactly described by two constants.
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

    Identifiability tests combine texture, correlation ties and spatial spread:

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
    if not np.isfinite(min_score) or not -1.0 <= min_score <= 1.0:
        raise ValueError("min_score must be finite and between -1 and 1")

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
    # NMS deliberately hides nearby peaks. Check near-ties BEFORE suppression:
    # otherwise a nearly uniform strip can be displaced many pixels while
    # retaining a numerically perfect score and still be labelled 'unique'.
    tied = int(np.count_nonzero(res >= peak - C.MATCH_TIE_TOL))
    if tied > 1:
        status = "ambiguous"
        reasons.append("{} positions have indistinguishable match scores; "
                       "no unique position can be claimed".format(tied))
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
        x=px,
        y=py,
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
    if not detection.marker_found or detection.marker_center is None:
        raise ValueError("detection carries no marker")
    c = np.asarray(detection.marker_center, float)
    r = float(detection.marker_radius_px) * (1.0 + margin_frac)
    if c.shape != (2,) or not np.isfinite(c).all() or not np.isfinite(r) or r <= 0:
        raise ValueError("marker centre and positive radius must be finite")
    return (c[0] - r, c[1] - r, c[0] + r, c[1] + r)


class _UnreachableGoal(ValueError):
    """A valid visibility request cannot fit inside the reference image."""


def _validate_goal(goal: str) -> None:
    if goal not in ("auto", "full_disk", "marker_centre"):
        raise ValueError("goal must be 'auto', 'full_disk', or 'marker_centre'")


def marker_visible(window: Sequence[int],
                   detection: PortDetection,
                   goal: str = "auto") -> bool:
    """Is the marker inside ``window = (x, y, w, h)``?

    ``goal='full_disk'`` needs the whole disk (plus margin) inside the window,
    ``goal='marker_centre'`` only its centre.  ``'auto'`` falls back to the
    centre criterion when the window is too small to ever contain the disk.
    """
    _validate_goal(goal)
    x, y, w, h = _window_values(window)
    x0, y0, x1, y1 = _marker_box(detection)
    if goal == "auto":
        goal = "full_disk" if (w >= (x1 - x0) and h >= (y1 - y0)) else "marker_centre"
    if goal == "full_disk":
        return bool(x <= x0 and y <= y0 and x1 < x + w and y1 < y + h)
    c = np.asarray(detection.marker_center, float)
    return bool(x <= c[0] < x + w and y <= c[1] < y + h)


def marker_visibility_target(window: Sequence[int],
                             detection: PortDetection,
                             image_shape: Sequence[int],
                             goal: str = "auto") -> Tuple[Tuple[int, int], str]:
    """Nearest window top-left that satisfies the visibility goal (clamped to the image)."""
    _validate_goal(goal)
    x, y, w, h = _window_values(window, image_shape)
    H, W = int(image_shape[0]), int(image_shape[1])
    x0, y0, x1, y1 = _marker_box(detection)
    auto = goal == "auto"
    if auto:
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
    # Crops are half-open: their right/bottom boundary is not in the view.
    lo_xi, hi_xi = max(0, int(np.floor(lo_x)) + 1), min(W - w, int(np.floor(hi_x)))
    lo_yi, hi_yi = max(0, int(np.floor(lo_y)) + 1), min(H - h, int(np.floor(hi_y)))
    if hi_xi < lo_xi or hi_yi < lo_yi:
        if auto and goal == "full_disk":
            return marker_visibility_target(window, detection, image_shape, "marker_centre")
        raise _UnreachableGoal("visibility goal {!r} cannot fit in this crop/reference".format(goal))

    tx = int(np.clip(int(x), lo_xi, hi_xi))
    ty = int(np.clip(int(y), lo_yi, hi_yi))
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
    _validate_goal(goal)
    if step_px is not None:
        step_px = float(step_px)
        if not np.isfinite(step_px) or step_px <= 0.0:
            raise ValueError("step_px must be a finite positive number")
    if localization.status != "ok":
        return NavigationPlan(
            status=localization.status,
            reasons=(["localisation is not confident, so no movement commands "
                      "are issued"] + list(localization.reasons)),
            localization=localization,
        )

    win = localization.window
    if win is None:
        raise ValueError("an ok localisation must have a window")
    x, y, w, h = _window_values(win, image_shape)
    try:
        (tx, ty), goal_used = marker_visibility_target(win, detection, image_shape, goal)
    except _UnreachableGoal as exc:
        return NavigationPlan(status="unreachable", start_window=win, goal=goal,
                              reasons=[str(exc)], localization=localization)

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
    step_px = float(step_px)
    if not np.isfinite(step_px) or step_px <= 0.0:
        raise ValueError("step_px must be a finite positive number")

    dx_total = float(tx - x)
    dy_total = float(ty - y)
    # Integer arithmetic avoids overflow for even the smallest positive float.
    numerator, denominator = step_px.as_integer_ratio()
    def _count(distance):
        whole, remainder = divmod(abs(int(distance)) * denominator, numerator)
        return whole + int(remainder / denominator > 1e-9)
    optimal = _count(tx - x) + _count(ty - y)

    # Never emit a partial route and call it successful.  Decide reachability
    # before producing any movement command.
    if optimal > C.NAV_MAX_STEPS:
        return NavigationPlan(
            status="unreachable", commands=[], start_window=win,
            target_window=(int(tx), int(ty), w, h), goal=goal_used,
            step_px=step_px, total_px=(dx_total, dy_total),
            total_cm=(dx_total / sx, dy_total / sy),
            optimal_steps=optimal,
            reasons=["the route needs {} steps at {:.6g} px per step, exceeding "
                     "the safety cap of {}; no partial route was emitted"
                     .format(optimal, step_px, C.NAV_MAX_STEPS)],
            localization=localization,
        )

    # Dominant axis first, 4-connected moves.
    axes = [("x", dx_total), ("y", dy_total)]
    axes.sort(key=lambda kv: -abs(kv[1]))

    commands: List[MovementCommand] = []
    cx, cy = float(x), float(y)
    n = 0
    for axis, total in axes:
        remaining = total
        while abs(remaining) > 1e-9:
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
    def _grab(win):
        x, y, w, h = win
        _window_values(win, reference.shape)
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

def explore_until_localized(initial_crop: np.ndarray,
                            reference: np.ndarray,
                            detection: PortDetection,
                            move_and_capture: Callable[[float, float], np.ndarray],
                            step_px: Optional[float] = None,
                            max_steps: Optional[int] = None,
                            accept_localization: Optional[Callable[[CropLocalization], bool]] = None
                            ) -> Dict[str, object]:
    """OPTIONAL feedback-driven search for an initially ambiguous live view.

    ``move_and_capture(dx_px, dy_px)`` is the only connection to the camera.  It
    requests a relative camera translation and returns the newly captured view.
    This function is never told the crop's true ``(x, y)`` in the reference:
    every localisation decision comes from observed pixels.

    The callback makes the same algorithm usable with hardware, a simulator, or
    recorded frames.  Search stops as soon as a fresh view can be localised, then
    the deterministic movement planner takes over. An optional predicate can
    require a particular visible goal before accepting a confident localisation.
    """
    view = np.asarray(initial_crop)
    if view.ndim not in (2, 3) or view.size == 0:
        raise ValueError("initial_crop must be a non-empty image")
    h, w = view.shape[:2]
    if h > reference.shape[0] or w > reference.shape[1]:
        raise ValueError("initial_crop must fit inside the reference image")

    if step_px is None:
        # Search stride, not tracking stride: consecutive stops should only
        # partly overlap so the search does not re-examine the same pixels.
        step_px = max(1.0, C.EXPLORE_STEP_FRAC * max(w, h))
    step_px = float(step_px)
    if not np.isfinite(step_px) or step_px <= 0.0:
        raise ValueError("step_px must be a finite positive number")
    step = int(max(1, round(step_px)))

    if max_steps is None:
        max_steps = C.NAV_MAX_STEPS
    try:
        max_steps = operator.index(max_steps)
    except TypeError as exc:
        raise ValueError("max_steps must be a non-negative integer") from exc
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    # Relative requested offsets only: a simulation may record its true
    # positions externally, but the algorithm never receives them.
    trail = [(0, 0)]
    commands: List[Dict[str, object]] = []
    offset_x = offset_y = 0

    def acceptable(loc):
        return loc.status == "ok" and (accept_localization is None or accept_localization(loc))

    def _finish(loc: CropLocalization, taken: int) -> Dict[str, object]:
        if acceptable(loc):
            plan = plan_crop_movements(loc, detection, reference.shape)
            return {"status": "localized", "steps_explored": taken,
                    "step_px": float(step), "window": loc.window,
                    "localization": loc, "plan": plan, "trail": trail,
                    "commands": commands, "last_view": view.copy()}
        return {"status": "exhausted", "steps_explored": taken,
                "step_px": float(step), "window": None,
                "localization": loc, "plan": None, "trail": trail,
                "commands": commands, "last_view": view.copy()}

    loc = localize_crop(view, reference)
    if acceptable(loc) or max_steps == 0:
        return _finish(loc, 0)

    directions = [(1, 0, "RIGHT"), (0, 1, "DOWN"),
                  (-1, 0, "LEFT"), (0, -1, "UP")]
    direction_index = taken = 0
    leg = 1

    while taken < max_steps:
        dx_unit, dy_unit, direction = directions[direction_index % 4]
        for _ in range(leg):
            if taken >= max_steps:
                break
            dx, dy = dx_unit * step, dy_unit * step
            fresh = np.asarray(move_and_capture(float(dx), float(dy)))
            if fresh.ndim not in (2, 3) or fresh.size == 0 or fresh.shape[:2] != (h, w):
                raise ValueError("move_and_capture must return a non-empty view "
                                 "with the same height and width as initial_crop")
            view = fresh.copy()
            taken += 1
            offset_x += dx
            offset_y += dy
            trail.append((offset_x, offset_y))
            commands.append({"step": taken, "direction": direction,
                             "dx_px": float(dx), "dy_px": float(dy),
                             "requested_offset_px": (offset_x, offset_y)})

            loc = localize_crop(view, reference)
            if acceptable(loc):
                return _finish(loc, taken)

        direction_index += 1
        if direction_index % 2 == 0:
            leg += 1

    return _finish(loc, taken)


def navigate_until_visible(initial_crop: np.ndarray,
                           reference: np.ndarray,
                           detection: PortDetection,
                           move_and_capture: Callable[[float, float], np.ndarray],
                           step_px: Optional[float] = None,
                           search_step_px: Optional[float] = None,
                           max_steps: int = C.NAV_MAX_STEPS,
                           goal: str = "auto") -> Dict[str, object]:
    """Search, execute camera translations, and verify arrival from fresh frames.

    One budget covers both search and navigation. After localisation, only the
    first command of each plan is executed, then the new view is localised and
    the route is recalculated. This tolerates imperfect camera motion. Lost
    localisation triggers another bounded search; an unchanged inferred window
    reports a stall. Requesting a move is never evidence that it happened.
    ``visible`` means the observed window satisfies the returned visibility
    goal under the same-scale reference-image model, not a hardware guarantee.
    """
    _validate_goal(goal)
    for value in (step_px, search_step_px):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError("step sizes must be finite positive numbers")
    try:
        budget = operator.index(max_steps)
    except TypeError as exc:
        raise ValueError("max_steps must be a non-negative integer") from exc
    if budget < 0:
        raise ValueError("max_steps must be non-negative")
    _marker_box(detection)
    view = np.asarray(initial_crop)
    to_gray(view)  # Validate before the first camera command.
    view_shape = view.shape[:2]
    moves = []

    def capture(dx, dy):
        fresh = np.asarray(move_and_capture(dx, dy))
        to_gray(fresh)
        if fresh.shape[:2] != view_shape:
            raise ValueError("move_and_capture must preserve the view height and width")
        moves.append({"step": len(moves) + 1, "dx_px": dx, "dy_px": dy})
        return fresh

    loc = localize_crop(view, reference)
    explored = 0
    plan = None

    def finish(status, reason):
        return {"status": status, "reason": reason, "steps_taken": len(moves),
                "steps_explored": explored, "commands": moves,
                "localization": loc, "window": loc.window if loc.status == "ok" else None,
                "plan": plan, "goal": plan.goal if plan is not None else goal,
                "last_view": view.copy()}

    # Feasibility depends on the view dimensions, not its unknown true origin.
    # Check it before asking the camera to move anywhere.
    try:
        marker_visibility_target((0, 0, view_shape[1], view_shape[0]),
                                 detection, reference.shape, goal)
    except _UnreachableGoal as exc:
        return finish("unreachable", str(exc))
    search = explore_until_localized(view, reference, detection, capture,
                                    step_px=search_step_px, max_steps=budget)
    explored = search["steps_explored"]
    loc, view = search["localization"], search["last_view"]
    if search["status"] != "localized":
        return finish("exhausted", "movement budget exhausted before localisation")
    while True:
        plan = plan_crop_movements(loc, detection, reference.shape, step_px, goal)
        if plan.status == "already_visible":
            return finish("visible", "fresh camera view satisfies the visibility goal")
        if plan.status != "ok":
            return finish(plan.status, "; ".join(plan.reasons))
        if len(moves) >= budget:
            return finish("exhausted", "movement budget exhausted before verified arrival")
        previous = loc.window
        command = plan.commands[0]
        view = capture(command.dx_px, command.dy_px)
        loc = localize_crop(view, reference)
        if loc.status != "ok":
            recovery = explore_until_localized(view, reference, detection, capture,
                                              step_px=search_step_px,
                                              max_steps=budget - len(moves),
                                              accept_localization=lambda found:
                                              marker_visible(found.window, detection, goal))
            explored += recovery["steps_explored"]
            loc, view = recovery["localization"], recovery["last_view"]
            if recovery["status"] != "localized":
                return finish("exhausted", "movement budget exhausted recovering localisation")
            continue
        if loc.window == previous:
            return finish("stalled", "camera view did not move to a new pixel window")
