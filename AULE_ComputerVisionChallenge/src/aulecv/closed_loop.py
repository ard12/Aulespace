"""Follow-up: closed-loop versions of Parts B and D that never trust their own motion.

The original Part B localised the crop once and then trusted every move it
commanded; the original Part D rendered poses it already knew.  Here the
camera's true position is hidden inside a simulator, every move is executed with
errors the navigator is not told about, and the position is re-estimated from
the image after every move.

Part B (redo)
    The view is re-localised by template matching after every move.  Some views
    cannot be localised (a blank patch, a single straight edge), so the
    navigator needs a way through them:

    * with zoom    -- zoom out until the view contains enough structure, travel
                      zoomed out, and zoom back in once the marker is in view;
    * without zoom -- map in advance which views CAN be localised, plan a route
                      that only passes through them, and if a view is still
                      ambiguous, undo the last move and route around that spot.

Part D (redo)
    The camera's pose is never assumed.  Every frame: find the outer corners,
    run PnP for a first estimate, refine it by aligning the whole image to the
    reference (ECC, every pixel instead of four corners), then command the next
    move along the 100 cm orbit from that estimate.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import camera as cam
from . import config as C
from . import navigation as nav
from .detect import PortDetection, detect_port
from .io import to_gray

__all__ = [
    "MotionError",
    "PanCamera",
    "ViewFix",
    "Localizer",
    "LocalizabilityMap",
    "NavResult",
    "navigate_with_zoom",
    "navigate_without_zoom",
    "navigate_trusting_motion",
    "PoseError",
    "OrbitCamera",
    "PoseEstimate",
    "PoseEstimator",
    "pose_from_raster_homography",
    "ReturnResult",
    "return_to_front_closed_loop",
]


# ===========================================================================
# Part B -- the simulated camera
# ===========================================================================

@dataclass
class MotionError:
    """How a commanded move differs from the real one.  The navigator never sees this."""

    gain: Tuple[float, float] = (1.0, 1.0)     #: real / commanded distance, per axis
    bias_px: Tuple[float, float] = (0.0, 0.0)  #: added to every move
    noise_px: float = 0.0                      #: random error per axis per move (std dev)
    seed: int = C.RANDOM_SEED


def _background_colour(image: np.ndarray) -> Tuple[float, float, float]:
    r = max(1, int(round(C.BG_RING_FRAC * min(image.shape[:2]))))
    ring = np.concatenate([image[:r].reshape(-1, 3), image[-r:].reshape(-1, 3),
                           image[:, :r].reshape(-1, 3), image[:, -r:].reshape(-1, 3)])
    return tuple(float(v) for v in np.median(ring, axis=0))


def _padded_world(reference: np.ndarray, pad: int) -> np.ndarray:
    """The scene: the reference, surrounded by plain background if ``pad > 0``."""
    if pad <= 0:
        return reference
    return cv2.copyMakeBorder(reference, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                              value=_background_colour(reference))


class PanCamera:
    """A camera that translates parallel to the port, with an optical zoom.

    Its true position and zoom live only inside this object.  The navigator
    talks to it through :meth:`capture`, :meth:`move_cm` and :meth:`set_zoom`
    and gets nothing back but images.  Positions are window centres in
    reference pixels; zoom ``s`` means the view covers ``s`` times the native
    width and height, squeezed into the same number of pixels.
    """

    def __init__(self, reference: np.ndarray, start_center: Sequence[float],
                 view_size: Tuple[int, int], px_per_cm: Sequence[float],
                 error: Optional[MotionError] = None, pad: int = 0,
                 max_zoom: float = 1.0):
        self.w, self.h = int(view_size[0]), int(view_size[1])
        self.H, self.W = reference.shape[:2]
        self.pad = int(pad)
        self.world = _padded_world(reference, self.pad)
        if max_zoom * self.w > self.W + 2 * self.pad or max_zoom * self.h > self.H + 2 * self.pad:
            raise ValueError("max_zoom needs more padding around the reference")
        self.max_zoom = float(max_zoom)
        self.px_per_cm = np.asarray(px_per_cm, float)
        self.error = error or MotionError()
        self._rng = np.random.default_rng(self.error.seed)
        self._c = np.asarray(start_center, float).copy()
        self._zoom = 1.0
        self._clamp()
        self.trail: List[np.ndarray] = [self._c.copy()]

    # -- ground truth (for evaluation only; the navigator never calls these) --
    @property
    def true_center(self) -> np.ndarray:
        return self._c.copy()

    @property
    def true_zoom(self) -> float:
        return self._zoom

    def true_window(self) -> Tuple[int, int, int, int]:
        """The native-zoom window around the true centre, clamped into the reference."""
        x = int(np.clip(round(self._c[0] - self.w / 2), 0, self.W - self.w))
        y = int(np.clip(round(self._c[1] - self.h / 2), 0, self.H - self.h))
        return (x, y, self.w, self.h)

    # -- the interface the navigator uses --------------------------------------
    def capture(self) -> np.ndarray:
        s = self._zoom
        ww, hh = int(round(s * self.w)), int(round(s * self.h))
        x0 = int(round(self._c[0] - ww / 2)) + self.pad
        y0 = int(round(self._c[1] - hh / 2)) + self.pad
        x0 = int(np.clip(x0, 0, self.world.shape[1] - ww))
        y0 = int(np.clip(y0, 0, self.world.shape[0] - hh))
        view = self.world[y0:y0 + hh, x0:x0 + ww]
        if (ww, hh) != (self.w, self.h):
            view = cv2.resize(view, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return view.copy()

    def move_cm(self, dx_cm: float, dy_cm: float) -> None:
        commanded = np.array([dx_cm, dy_cm], float) * self.px_per_cm
        e = self.error
        real = (commanded * np.asarray(e.gain, float) + np.asarray(e.bias_px, float)
                + self._rng.normal(0.0, e.noise_px, 2))
        self._c = self._c + real
        self._clamp()
        self.trail.append(self._c.copy())

    def set_zoom(self, zoom: float) -> None:
        self._zoom = float(np.clip(zoom, 1.0, self.max_zoom))
        self._clamp()

    def _clamp(self) -> None:
        # The centre stays over the reference; the zoomed window stays inside the world.
        half = 0.5 * self._zoom * np.array([self.w, self.h], float)
        size = np.array([self.W, self.H], float)
        lo = np.maximum(0.0, half - self.pad)
        hi = np.minimum(size, size + self.pad - half)
        self._c = np.clip(self._c, lo, hi)


# ===========================================================================
# Part B -- localisation from pixels only
# ===========================================================================

@dataclass
class ViewFix:
    """Where a view came from, judged from its pixels alone."""

    status: str                      #: 'ok' | 'ambiguous' | 'featureless'
    center: Optional[np.ndarray]     #: window centre in reference pixels (None if not located)
    zoom: float
    score: float
    uncertainty_px: float
    reasons: List[str] = field(default_factory=list)


class Localizer:
    """Template-matches a view against the scene at the view's zoom.

    A view at zoom ``s`` shows the scene shrunk by ``s``, so it is matched
    against a copy of the scene shrunk by the same factor.  The zoom setting is
    assumed known (a zoom lens reports its position); a scale search would
    remove that assumption at a higher cost.
    """

    def __init__(self, reference: np.ndarray, view_size: Tuple[int, int], pad: int = 0):
        self.w, self.h = int(view_size[0]), int(view_size[1])
        self.pad = int(pad)
        self.world = _padded_world(reference, self.pad)
        self._scaled: Dict[float, np.ndarray] = {}

    def _world_at(self, zoom: float) -> np.ndarray:
        key = round(float(zoom), 4)
        if key not in self._scaled:
            if key == 1.0:
                self._scaled[key] = self.world
            else:
                hw, ww = self.world.shape[:2]
                size = (int(round(ww / key)), int(round(hw / key)))
                self._scaled[key] = cv2.resize(self.world, size, interpolation=cv2.INTER_AREA)
        return self._scaled[key]

    def locate(self, view: np.ndarray, zoom: float = 1.0) -> ViewFix:
        scene = self._world_at(zoom)
        loc = nav.localize_crop(view, scene)
        if loc.x is None:
            return ViewFix(loc.status, None, zoom, float("nan"), float("inf"), list(loc.reasons))
        kx = self.world.shape[1] / scene.shape[1]
        ky = self.world.shape[0] / scene.shape[0]
        center = np.array([(loc.x + self.w / 2.0) * kx - self.pad,
                           (loc.y + self.h / 2.0) * ky - self.pad])
        unc = zoom * (1.0 + max(loc.uncertainty_px)) if np.all(np.isfinite(loc.uncertainty_px)) else float("inf")
        return ViewFix(loc.status, center, zoom, float(loc.score), float(unc), list(loc.reasons))


# ===========================================================================
# Part B -- navigators
# ===========================================================================

@dataclass
class NavResult:
    status: str                              #: 'visible' | 'exhausted' | 'no_route'
    moves: int
    zoom_changes: int
    lost_views: int
    backoffs: int
    log: List[Dict[str, object]]
    estimate: Optional[np.ndarray]           #: last estimated window centre

    def final_error_px(self, camera: PanCamera) -> float:
        if self.estimate is None:
            return float("nan")
        return float(np.linalg.norm(self.estimate - camera.true_center))


def _native_window(center: np.ndarray, w: int, h: int, ref_shape) -> Tuple[int, int, int, int]:
    H, W = ref_shape[:2]
    x = int(np.clip(round(center[0] - w / 2), 0, W - w))
    y = int(np.clip(round(center[1] - h / 2), 0, H - h))
    return (x, y, w, h)


def _goal_center(center: np.ndarray, detection: PortDetection, w: int, h: int, ref_shape) -> np.ndarray:
    """The window centre to aim for: the nearest one whose native view holds the marker,
    pulled a few pixels inside the allowed range.

    Aiming exactly at the edge of the allowed range means any small motion error
    lands just outside it, and the camera dithers there forever.
    """
    H, W = ref_shape[:2]
    x0, y0, x1, y1 = nav._marker_box(detection)
    if w >= x1 - x0 and h >= y1 - y0:          # the whole disc fits: window must contain it
        lo = np.array([x1 - w, y1 - h], float)
        hi = np.array([x0, y0], float)
    else:                                      # too small for the disc: hold its centre
        c = np.asarray(detection.marker_center, float)
        lo, hi = c - np.array([w, h], float), c.copy()
    lo = np.maximum(lo, 0.0)
    hi = np.minimum(hi, np.array([W - w, H - h], float))
    margin = np.minimum(C.CLOSED_LOOP_GOAL_MARGIN_PX, np.maximum(hi - lo, 0.0) / 2.0)
    corner = np.clip(np.asarray(center, float) - np.array([w, h]) / 2.0, lo + margin, hi - margin)
    return corner + np.array([w, h]) / 2.0


def _spiral(step: float):
    """Outward square spiral of relative moves: R, D, L, U, L, U ... legs 1, 1, 2, 2, 3, 3."""
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    leg, k = 1, 0
    while True:
        for _ in range(2):
            dx, dy = dirs[k % 4]
            for _ in range(leg):
                yield np.array([dx * step, dy * step], float)
            k += 1
        leg += 1


def _command(camera: PanCamera, vec_px: np.ndarray, px_per_cm: np.ndarray) -> Tuple[float, float]:
    cmd = vec_px / px_per_cm
    camera.move_cm(float(cmd[0]), float(cmd[1]))
    return float(cmd[0]), float(cmd[1])


def navigate_with_zoom(camera: PanCamera, localizer: Localizer, detection: PortDetection,
                       ref_shape, step_px: float = C.CLOSED_LOOP_STEP_PX,
                       zoom_levels: Sequence[float] = C.CLOSED_LOOP_ZOOM_LEVELS,
                       max_moves: int = C.NAV_MAX_STEPS, axis_first: bool = False) -> NavResult:
    """Re-localise after every move; zoom out whenever the view cannot be placed.

    Travel happens at whatever zoom last gave a confident fix.  Once the
    estimated native window holds the marker, the camera zooms back in and the
    arrival is confirmed from a native-zoom fix.  ``zoom_levels=(1.0,)`` turns
    zoom off, which shows the problem this solves.  ``axis_first=True`` moves
    one axis at a time, longer distance first, like the original Part B plan.
    """
    levels = [float(z) for z in zoom_levels if z <= camera.max_zoom + 1e-9] or [1.0]
    ppc = nav.pixels_per_cm(detection)
    ppc = np.array(ppc, float)
    w, h = localizer.w, localizer.h
    level, moves, zoom_changes, lost = 0, 0, 0, 0
    camera.set_zoom(levels[0])
    spiral = _spiral(C.EXPLORE_STEP_FRAC * max(w, h))
    log: List[Dict[str, object]] = []
    estimate: Optional[np.ndarray] = None

    def record(event, fix, command=None):
        log.append({"capture": len(log), "moves": moves, "zoom": levels[level],
                    "status": fix.status, "event": event,
                    "est_x": None if fix.center is None else round(float(fix.center[0]), 2),
                    "est_y": None if fix.center is None else round(float(fix.center[1]), 2),
                    "true_x": round(float(camera.true_center[0]), 2),
                    "true_y": round(float(camera.true_center[1]), 2),
                    "command_cm": command})

    while moves <= max_moves:
        fix = localizer.locate(camera.capture(), levels[level])
        if fix.status != "ok":
            lost += 1
            if level < len(levels) - 1:
                record("zoom out", fix)
                level += 1
                camera.set_zoom(levels[level])
                zoom_changes += 1
                continue
            if moves >= max_moves:
                record("give up", fix)
                break
            command = _command(camera, next(spiral), ppc)
            record("explore", fix, command)
            moves += 1
            continue

        estimate = fix.center
        window = _native_window(estimate, w, h, ref_shape)
        if nav.marker_visible(window, detection, "auto"):
            if level == 0:
                record("arrived", fix)
                return NavResult("visible", moves, zoom_changes, lost, 0, log, estimate)
            record("zoom in to confirm", fix)
            level = 0
            camera.set_zoom(levels[0])
            zoom_changes += 1
            continue
        if moves >= max_moves:
            record("give up", fix)
            break
        vec = _goal_center(estimate, detection, w, h, ref_shape) - estimate
        if axis_first:
            major = int(np.argmax(np.abs(vec)))
            vec = np.where(np.arange(2) == major, vec, 0.0)
        dist = float(np.linalg.norm(vec))
        step = vec if dist <= step_px else vec / dist * step_px
        command = _command(camera, step, ppc)
        record("move", fix, command)
        moves += 1
    return NavResult("exhausted", moves, zoom_changes, lost, 0, log, estimate)


class LocalizabilityMap:
    """Which native-zoom views of the reference can be localised, on a grid.

    Computed once from the reference: crop the view at every grid position and
    ask the localiser.  ``safe`` cells are localisable AND surrounded by
    localisable cells, so a move that lands a little off target still lands on
    a view that can be placed.
    """

    def __init__(self, reference: np.ndarray, localizer: Localizer, detection: PortDetection,
                 stride: int = C.LOCALIZABILITY_STRIDE_PX):
        self.stride = int(stride)
        self.w, self.h = localizer.w, localizer.h
        self.ref_shape = reference.shape
        H, W = reference.shape[:2]
        self.xs = np.arange(0, W - self.w + 1, self.stride)
        self.ys = np.arange(0, H - self.h + 1, self.stride)
        self.ok = np.zeros((len(self.ys), len(self.xs)), bool)
        self.goal = np.zeros_like(self.ok)
        for j, y in enumerate(self.ys):
            for i, x in enumerate(self.xs):
                crop = nav.extract_crop(reference, int(x), int(y), self.w, self.h)
                self.ok[j, i] = localizer.locate(crop, 1.0).status == "ok"
                self.goal[j, i] = nav.marker_visible((int(x), int(y), self.w, self.h), detection, "auto")
        padded = np.pad(self.ok, 1, constant_values=False)
        neighbours = np.ones_like(self.ok)
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                neighbours &= padded[1 + dj:1 + dj + self.ok.shape[0], 1 + di:1 + di + self.ok.shape[1]]
        self.safe = self.ok & neighbours
        # Clearance: how many cells away the nearest unlocalisable view is.  Routes
        # prefer high clearance, so a move that lands a little off still lands on
        # a view that can be placed.
        self.clearance = cv2.distanceTransform(np.pad(self.ok, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
        self.blocked = np.zeros_like(self.ok)

    @property
    def fraction_ok(self) -> float:
        return float(self.ok.mean())

    def cell_of(self, center: np.ndarray) -> Tuple[int, int]:
        x = center[0] - self.w / 2.0
        y = center[1] - self.h / 2.0
        i = int(np.clip(round(x / self.stride), 0, len(self.xs) - 1))
        j = int(np.clip(round(y / self.stride), 0, len(self.ys) - 1))
        return j, i

    def center_of(self, cell: Tuple[int, int]) -> np.ndarray:
        j, i = cell
        return np.array([self.xs[i] + self.w / 2.0, self.ys[j] + self.h / 2.0])

    def route(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Cheapest 8-connected route to ``goal``.

        Each step costs its length, plus a penalty for passing close to views
        that cannot be localised, plus a large cost for stepping onto one (a
        "blind" step, or one that already failed in this run).  So the route
        stays where the camera can see where it is, and when the localisable
        areas are separate islands it crosses the gap where it is narrowest.
        """
        rows, cols = self.ok.shape
        seeing = self.ok & ~self.blocked
        penalty = np.where(seeing, C.LOCALIZABILITY_CLEARANCE_WEIGHT / np.maximum(self.clearance, 0.5),
                           C.LOCALIZABILITY_BLIND_STEP_COST)
        best = {start: 0.0}
        prev = {start: None}
        heap = [(0.0, start)]
        while heap:
            cost, cell = heapq.heappop(heap)
            if cell == goal:
                break
            if cost > best[cell]:
                continue
            j, i = cell
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    nxt = (j + dj, i + di)
                    if not (dj or di) or not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                        continue
                    step = float(np.hypot(dj, di)) + float(penalty[nxt])
                    if cost + step < best.get(nxt, np.inf):
                        best[nxt] = cost + step
                        prev[nxt] = cell
                        heapq.heappush(heap, (cost + step, nxt))
        path = [goal]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
        return path[::-1]

    def sees(self, cell: Tuple[int, int]) -> bool:
        return bool(self.ok[cell] and not self.blocked[cell])


def navigate_without_zoom(camera: PanCamera, localizer: Localizer, lmap: LocalizabilityMap,
                          detection: PortDetection, max_moves: int = C.NAV_MAX_STEPS) -> NavResult:
    """No zoom: plan a route that keeps the camera able to see where it is.

    After every move the view is re-localised and the route re-planned from that
    fix.  The route avoids views that cannot be localised; where it has to cross
    them (separate islands), only those few steps are executed without a fix,
    counted on the commanded moves, and the next fix takes over again.  A view
    the map promised but that turns out ambiguous (the move landed somewhere
    unexpected) is undone once and avoided.  If the very first view cannot be
    placed, the camera searches outward until one can.
    """
    camera.set_zoom(1.0)
    ppc = np.array(nav.pixels_per_cm(detection), float)
    w, h = localizer.w, localizer.h
    spiral = _spiral(C.EXPLORE_STEP_FRAC * max(w, h))
    lmap.blocked[:] = False
    moves, lost, backoffs, blind = 0, 0, 0, 0
    log: List[Dict[str, object]] = []
    estimate: Optional[np.ndarray] = None      # last position from a fix
    believed: Optional[np.ndarray] = None      # dead-reckoned position during a planned blind crossing
    expected_blind = False                     # did the plan say the next view would be unlocalisable?
    last_move: Optional[np.ndarray] = None
    target_cell: Optional[Tuple[int, int]] = None

    def record(event, fix, command=None):
        log.append({"capture": len(log), "moves": moves, "zoom": 1.0, "status": fix.status,
                    "event": event,
                    "est_x": None if fix.center is None else round(float(fix.center[0]), 2),
                    "est_y": None if fix.center is None else round(float(fix.center[1]), 2),
                    "true_x": round(float(camera.true_center[0]), 2),
                    "true_y": round(float(camera.true_center[1]), 2),
                    "command_cm": command})

    def step_towards(position):
        """Next move from ``position``: along the planned route, or straight to the exact goal."""
        goal = _goal_center(position, detection, w, h, lmap.ref_shape)
        limit = float(lmap.stride)
        if float(np.linalg.norm(goal - position)) <= limit:
            return goal - position, None
        path = lmap.route(lmap.cell_of(position), lmap.cell_of(goal))
        cell = path[1] if len(path) > 1 else path[0]
        vec = lmap.center_of(cell) - position
        dist = float(np.linalg.norm(vec))
        return (vec if dist <= limit else vec / dist * limit), cell

    while moves <= max_moves:
        fix = localizer.locate(camera.capture(), 1.0)
        if fix.status == "ok":
            estimate, believed, blind = fix.center, None, 0
            if nav.marker_visible(_native_window(estimate, w, h, lmap.ref_shape), detection, "auto"):
                record("arrived", fix)
                return NavResult("visible", moves, 0, lost, backoffs, log, estimate)
            if moves >= max_moves:
                record("give up", fix)
                break
            vec, target_cell = step_towards(estimate)
            expected_blind = target_cell is not None and not lmap.sees(target_cell)
            command = _command(camera, vec, ppc)
            record("blind step (planned)" if expected_blind else "move", fix, command)
            last_move = vec
            believed = estimate + vec if expected_blind else None
            moves += 1
            continue

        lost += 1
        if moves >= max_moves:
            record("give up", fix)
            break
        if estimate is None:                                   # never had a fix: search
            command = _command(camera, next(spiral), ppc)
            record("explore", fix, command)
        elif believed is not None and blind < C.LOCALIZABILITY_MAX_BLIND_STEPS:
            vec, target_cell = step_towards(believed)          # planned crossing: keep going
            command = _command(camera, vec, ppc)
            record("blind step (planned)", fix, command)
            believed = believed + vec
            last_move = vec
            blind += 1
        elif last_move is not None and not expected_blind:     # surprise: undo once, avoid that cell
            if target_cell is not None:
                lmap.blocked[target_cell] = True
            command = _command(camera, -last_move, ppc)
            record("lost: undo last move", fix, command)
            backoffs += 1
            last_move, believed = None, None
        else:                                                   # crossing went wrong: search
            command = _command(camera, next(spiral), ppc)
            record("explore", fix, command)
            last_move, believed = None, None
        moves += 1
    return NavResult("exhausted", moves, 0, lost, backoffs, log, estimate)


def navigate_trusting_motion(camera: PanCamera, localizer: Localizer, detection: PortDetection,
                             ref_shape, step_px: float = C.CLOSED_LOOP_STEP_PX,
                             max_moves: int = C.NAV_MAX_STEPS) -> NavResult:
    """The baseline being replaced: localise once, then execute the whole route blindly."""
    camera.set_zoom(1.0)
    ppc = np.array(nav.pixels_per_cm(detection), float)
    w, h = localizer.w, localizer.h
    spiral = _spiral(C.EXPLORE_STEP_FRAC * max(w, h))
    moves, lost = 0, 0
    log: List[Dict[str, object]] = []
    fix = localizer.locate(camera.capture(), 1.0)
    while fix.status != "ok" and moves < max_moves:      # same start-up search as the others
        lost += 1
        _command(camera, next(spiral), ppc)
        moves += 1
        fix = localizer.locate(camera.capture(), 1.0)
    if fix.status != "ok":
        return NavResult("exhausted", moves, 0, lost, 0, log, None)
    believed = fix.center.copy()
    goal = _goal_center(believed, detection, w, h, ref_shape)
    while moves < max_moves:
        vec = goal - believed
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            break
        step = vec if dist <= step_px else vec / dist * step_px
        _command(camera, step, ppc)
        believed = believed + step            # trusting that the move happened exactly
        moves += 1
        log.append({"moves": moves, "believed_x": float(believed[0]), "believed_y": float(believed[1]),
                    "true_x": float(camera.true_center[0]), "true_y": float(camera.true_center[1])})
    believed_ok = nav.marker_visible(_native_window(believed, w, h, ref_shape), detection, "auto")
    return NavResult("visible" if believed_ok else "exhausted", moves, 0, lost, 0, log, believed)


# ===========================================================================
# Part D -- the simulated camera on (or near) the orbit
# ===========================================================================

@dataclass
class PoseError:
    """How a commanded camera move differs from the real one.  The controller never sees this."""

    gain: float = 1.0               #: real / commanded translation
    noise_cm: float = 0.0           #: random translation error per axis per move
    rot_noise_deg: float = 0.0      #: random rotation error per axis per move
    sensor_noise: float = 0.0       #: image noise, grey levels (std dev)
    seed: int = C.RANDOM_SEED


class OrbitCamera:
    """A camera that can be anywhere near the port.  Its true pose stays hidden.

    The controller sends relative moves in the camera's own frame (a
    translation and a rotation); they are executed with errors.  Images are
    rendered from the TRUE pose through the same plane model as Part C.
    """

    def __init__(self, reference: np.ndarray, renderer: cam.ViewRenderer,
                 start_theta_deg: float = C.PART_C_ANGLE_DEG,
                 radius_cm: float = C.CAMERA_DISTANCE_CM,
                 error: Optional[PoseError] = None):
        if renderer.model != "metric":
            raise ValueError("OrbitCamera renders with the metric (40 cm square) model")
        self.reference = reference
        self.K, self.G, self.canvas = renderer.K, renderer.G, renderer.canvas
        self.error = error or PoseError()
        self._rng = np.random.default_rng(self.error.seed)
        pose = cam.camera_pose_for_angle(start_theta_deg, radius_cm)
        self._R, self._C = pose["R"].copy(), pose["C"].copy()

    # -- ground truth (evaluation only) --
    @property
    def true_center(self) -> np.ndarray:
        return self._C.copy()

    @property
    def true_theta_deg(self) -> float:
        return float(np.degrees(np.arctan2(self._C[0], -self._C[2])))

    @property
    def true_distance_cm(self) -> float:
        return float(np.linalg.norm(self._C))

    @property
    def true_pointing_error_deg(self) -> float:
        to_centre = -self._C / np.linalg.norm(self._C)
        return float(np.degrees(np.arccos(np.clip(self._R[2] @ to_centre, -1.0, 1.0))))

    # -- the interface the controller uses --
    def capture(self) -> np.ndarray:
        t = -self._R @ self._C
        H = cam.plane_to_image_homography(self.K, self._R, t) @ self.G
        img = cam.render_port_view(self.reference, H, self.canvas)
        if self.error.sensor_noise > 0:
            noisy = img.astype(np.float32) + self._rng.normal(0.0, self.error.sensor_noise, img.shape)
            img = np.clip(noisy, 0, 255).astype(np.uint8)
        return img

    def move(self, delta_body_cm: Sequence[float], rotation: np.ndarray) -> None:
        e = self.error
        d = np.asarray(delta_body_cm, float) * e.gain + self._rng.normal(0.0, e.noise_cm, 3)
        self._C = self._C + self._R.T @ d
        rvec = np.radians(self._rng.normal(0.0, e.rot_noise_deg, 3))
        R_err, _ = cv2.Rodrigues(rvec)
        self._R = R_err @ np.asarray(rotation, float) @ self._R


# ===========================================================================
# Part D -- pose from the image
# ===========================================================================

def pose_from_raster_homography(H: np.ndarray, K: np.ndarray, G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Recover (R, t) from the raster->image homography ``H = K [r1 r2 t] G``.

    ``K^-1 H G^-1`` is ``[r1 r2 t]`` up to scale.  The scale is fixed by r1 and
    r2 having unit length, its sign by the port being in front of the camera,
    and the rotation is snapped to the nearest true rotation matrix.
    """
    M = np.linalg.inv(K) @ np.asarray(H, float) @ np.linalg.inv(G)
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    if M[2, 2] * lam < 0:
        lam = -lam
    r1, r2, t = M[:, 0] * lam, M[:, 1] * lam, M[:, 2] * lam
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R)
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return R, t


@dataclass
class PoseEstimate:
    R: np.ndarray
    t: np.ndarray
    C: np.ndarray
    theta_deg: float
    distance_cm: float
    method: str                      #: 'dense' | 'corners'
    alignment_score: float           #: ECC correlation (nan when not used)
    corner_theta_deg: float          #: the 4-corner PnP answer, for comparison
    corner_distance_cm: float


class PoseEstimator:
    """Pose from one image: four-corner PnP first, then dense alignment of every pixel.

    The four corners give a quick, robust first guess.  Near the front view
    that guess is weak (a slightly turned square looks almost unchanged).  So
    the guess seeds ECC, which aligns the reference to the whole image and gives
    a homography; the pose is read out of that homography.  If alignment fails
    or scores poorly, the corner answer is used and flagged.
    """

    def __init__(self, reference: np.ndarray, K: np.ndarray, G: np.ndarray,
                 coarse_scale: float = C.DENSE_COARSE_SCALE,
                 iterations: Tuple[int, int] = C.DENSE_ITERATIONS,
                 min_score: float = C.DENSE_MIN_SCORE):
        self.K, self.G = np.asarray(K, float), np.asarray(G, float)
        self.ref = to_gray(reference).astype(np.float32)
        self.s = float(coarse_scale)
        self.ref_small = cv2.resize(self.ref, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA)
        self.iterations = iterations
        self.min_score = float(min_score)

    @staticmethod
    def _readout(R, t, method, score, corner_theta, corner_dist) -> PoseEstimate:
        Cc = -R.T @ t
        return PoseEstimate(R, t, Cc, float(np.degrees(np.arctan2(Cc[0], -Cc[2]))),
                            float(np.linalg.norm(Cc)), method, score, corner_theta, corner_dist)

    def estimate(self, image: np.ndarray, refine: bool = True) -> PoseEstimate:
        det = detect_port(image)
        pnp = cam.recover_pose_pnp(det.quad, self.K)
        R0, t0 = pnp["R"], pnp["t"]
        if not refine:
            return self._readout(R0, t0, "corners", float("nan"), pnp["theta_deg"], pnp["distance_cm"])
        H0 = cam.plane_to_image_homography(self.K, R0, t0) @ self.G
        H0 = H0 / H0[2, 2]
        img = to_gray(image).astype(np.float32)
        S = np.diag([self.s, self.s, 1.0])
        try:
            small = cv2.resize(img, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA)
            warp = (S @ H0 @ np.linalg.inv(S)).astype(np.float32)
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.iterations[0], 1e-6)
            _, warp = cv2.findTransformECC(self.ref_small, small, warp, cv2.MOTION_HOMOGRAPHY, crit, None, 1)
            warp = (np.linalg.inv(S) @ warp.astype(np.float64) @ S).astype(np.float32)
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.iterations[1], 1e-7)
            score, warp = cv2.findTransformECC(self.ref, img, warp, cv2.MOTION_HOMOGRAPHY, crit, None, 1)
        except cv2.error:
            score = float("nan")
        if not np.isfinite(score) or score < self.min_score:
            return self._readout(R0, t0, "corners", float(score), pnp["theta_deg"], pnp["distance_cm"])
        R, t = pose_from_raster_homography(warp.astype(np.float64), self.K, self.G)
        return self._readout(R, t, "dense", float(score), pnp["theta_deg"], pnp["distance_cm"])


# ===========================================================================
# Part D -- the controller
# ===========================================================================

@dataclass
class ReturnResult:
    status: str                       #: 'converged' | 'not_converged'
    iterations: int
    log: List[Dict[str, float]]


def return_to_front_closed_loop(camera: OrbitCamera, estimator: PoseEstimator,
                                step_deg: float = C.PART_D_STEP_DEG,
                                radius_cm: float = C.CAMERA_DISTANCE_CM,
                                tol_deg: float = C.CLOSED_LOOP_TOL_DEG,
                                tol_cm: float = C.CLOSED_LOOP_TOL_CM,
                                max_iterations: int = C.CLOSED_LOOP_MAX_ITERATIONS,
                                refine: bool = True) -> ReturnResult:
    """Walk back to the front view using only pose estimates from images.

    Each iteration: estimate the pose from the image; if the estimate is at the
    front (angle and distance within tolerance) on two frames in a row, stop.
    Otherwise aim for the orbit point up to ``step_deg`` closer to 0, at
    ``radius_cm``, looking at the centre, and send the move needed to get
    there FROM THE ESTIMATED pose.  Distance errors are corrected on the way.
    """
    log: List[Dict[str, float]] = []
    settled = 0
    for k in range(max_iterations):
        est = estimator.estimate(camera.capture(), refine=refine)
        log.append({"iteration": k, "true_theta_deg": camera.true_theta_deg,
                    "est_theta_deg": est.theta_deg, "corner_theta_deg": est.corner_theta_deg,
                    "true_distance_cm": camera.true_distance_cm, "est_distance_cm": est.distance_cm,
                    "true_pointing_err_deg": camera.true_pointing_error_deg,
                    "method": est.method, "alignment_score": est.alignment_score})
        at_front = abs(est.theta_deg) < tol_deg and abs(est.distance_cm - radius_cm) < tol_cm
        settled = settled + 1 if at_front else 0
        if settled >= 2:
            return ReturnResult("converged", k + 1, log)
        theta_next = est.theta_deg - np.sign(est.theta_deg) * min(step_deg, abs(est.theta_deg))
        target = cam.camera_pose_for_angle(theta_next, radius_cm)
        delta_body = est.R @ (target["C"] - est.C)
        rotation = target["R"] @ est.R.T
        camera.move(delta_body, rotation)
    return ReturnResult("not_converged", max_iterations, log)
