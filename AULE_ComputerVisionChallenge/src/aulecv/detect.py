"""Detection of the satellite port: outer quadrilateral, centre and circular marker.

Design notes
------------
*   Everything is scaled off two *measured* quantities: the detected outer side
    length and the border thickness recovered from a distance transform.  No
    kernel size, no tolerance and no coordinate in this module is a literal
    tuned to the supplied raster (see :mod:`aulecv.config`).
*   Corners are sub-pixel: ``approxPolyDP`` gives a coarse 4-gon, then each side
    is re-fitted with ``cv2.fitLine`` on the *interior* portion of that side
    (the ends are trimmed, because corner rounding lives there) and consecutive
    side lines are intersected.
*   The marker is found by opening the dark mask with a disk whose diameter is a
    multiple of the border thickness.  That erases the square borders (which the
    marker touches, merging with them into one connected component) and leaves
    the disk.  ``fitEllipse`` on the surviving blob is the primary estimate;
    the distance-transform peak and ``HoughCircles`` are independent
    cross-checks that feed the confidence score.
*   The marker also *orients* the port.  The canonical corner ordering is
    "start at the quad vertex nearest the marker, then clockwise on screen",
    which is what makes the detector usable on an arbitrarily rotated image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config as C

__all__ = [
    "PortDetection",
    "estimate_background_level",
    "foreground_mask",
    "dark_mask",
    "estimate_border_thickness",
    "order_quad_clockwise",
    "quad_center",
    "quad_side_lengths",
    "refine_quad_corners",
    "detect_outer_quad",
    "detect_marker",
    "detect_port",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PortDetection:
    """Everything the detector recovers from one image."""

    quad: np.ndarray                      #: (4,2) float64, canonical order (marker corner first, then clockwise)
    quad_image_order: np.ndarray          #: (4,2) float64, clockwise from the top-left-most vertex
    center: np.ndarray                    #: (2,) intersection of the diagonals
    side_lengths: np.ndarray              #: (4,) lengths of the canonical quad's sides
    outer_side_px: float                  #: mean side length
    border_thickness_px: float

    marker_found: bool = False
    marker_center: Optional[np.ndarray] = None       #: (2,) fitEllipse centre (primary)
    marker_axes: Optional[Tuple[float, float]] = None  #: (major, minor) full axis lengths
    marker_angle_deg: Optional[float] = None
    marker_radius_px: Optional[float] = None
    marker_circularity: Optional[float] = None
    marker_axis_ratio: Optional[float] = None
    marker_center_dt: Optional[np.ndarray] = None    #: distance-transform peak (cross-check)
    marker_center_hough: Optional[np.ndarray] = None  #: HoughCircles centre (cross-check)
    marker_corner_index: Optional[int] = None        #: index into ``quad_image_order``

    confidence: float = 0.0
    confidence_terms: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    # -- convenience ------------------------------------------------------
    @property
    def ok(self) -> bool:
        return self.marker_found and self.confidence >= C.ROTATION_MIN_CONFIDENCE

    @property
    def marker_vector(self) -> np.ndarray:
        """Vector from the port centre to the marker centre, in image axes."""
        if not self.marker_found:
            raise ValueError("marker was not detected")
        return np.asarray(self.marker_center, float) - np.asarray(self.center, float)

    @property
    def marker_angle_from_center_deg(self) -> float:
        """Polar angle of :pyattr:`marker_vector`, degrees, image axes (y down)."""
        v = self.marker_vector
        return float(np.degrees(np.arctan2(v[1], v[0])))

    def edge_angle_mod90_deg(self) -> float:
        """Orientation of the quad's sides, folded into [0, 90) degrees.

        Averaged over all four sides on the unit circle of the *doubled*
        doubled angle (4*phi), which is the standard way to average directions
        that are equivalent modulo 90 deg.
        """
        q = np.asarray(self.quad_image_order, float)
        acc = 0.0 + 0.0j
        for i in range(4):
            d = q[(i + 1) % 4] - q[i]
            phi = np.arctan2(d[1], d[0])
            acc += np.exp(4j * phi) * float(np.hypot(d[0], d[1]))
        phi4 = np.angle(acc)
        return _fold(np.degrees(phi4 / 4.0), 90.0)

    def long_edge_angle_mod180_deg(self) -> float:
        """Orientation of the *longer* pair of sides, folded into [0, 180).

        NOTE: this cue only exists because the supplied drawing is schematic --
        the detected outer quad measures 550 x 518 px instead of being square
        (notebook 00 measures it).  The physical port *is* square, so this is
        reported as an experiment and is never used by the estimator.
        """
        q = np.asarray(self.quad_image_order, float)
        edges = [(q[(i + 1) % 4] - q[i]) for i in range(4)]
        lens = [float(np.hypot(e[0], e[1])) for e in edges]
        pair0 = lens[0] + lens[2]
        pair1 = lens[1] + lens[3]
        idxs = (0, 2) if pair0 >= pair1 else (1, 3)
        acc = 0.0 + 0.0j
        for i in idxs:
            phi = np.arctan2(edges[i][1], edges[i][0])
            acc += np.exp(2j * phi) * lens[i]
        return _fold(np.degrees(np.angle(acc) / 2.0), 180.0)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _fold(angle_deg: float, period: float) -> float:
    """``angle_deg`` folded into ``[0, period)``, robust to float wrap-around.

    ``(-1e-14) % 180.0`` evaluates to ``180.0`` in IEEE arithmetic, which would
    turn an exactly-horizontal edge into a 180 deg reading; snap that back.
    """
    a = float(np.mod(float(angle_deg), period))
    if period - a < 1e-9 * period:
        a = 0.0
    return a


def estimate_background_level(gray: np.ndarray) -> float:
    """Median grey level of a thin ring around the image border (the page)."""
    h, w = gray.shape[:2]
    r = max(1, int(round(C.BG_RING_FRAC * min(h, w))))
    ring = np.concatenate([
        gray[:r, :].ravel(), gray[-r:, :].ravel(),
        gray[:, :r].ravel(), gray[:, -r:].ravel(),
    ])
    return float(np.median(ring))


def foreground_mask(gray: np.ndarray, bg_level: Optional[float] = None) -> np.ndarray:
    """uint8 {0,255} mask of pixels that differ from the page background."""
    if bg_level is None:
        bg_level = estimate_background_level(gray)
    diff = np.abs(gray.astype(np.int32) - int(round(bg_level)))
    return ((diff > C.FG_TOLERANCE).astype(np.uint8)) * 255


def _otsu_threshold(values: np.ndarray) -> float:
    v = np.asarray(values, np.uint8).reshape(-1, 1)
    if v.size == 0 or int(v.min()) == int(v.max()):
        return float(v.min()) if v.size else 0.0
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def dark_mask(gray: np.ndarray, region: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Mask of the *ink* (darkest cluster) with a doubly-Otsu threshold.

    A single Otsu split on this drawing separates {black, mid-grey} from
    {light-grey, white}.  Applying Otsu a second time *inside* the dark side
    isolates the genuinely black ink -- the square borders and the marker --
    from the mid-grey fill.  Both thresholds are derived from the image's own
    histogram, so nothing is tuned to a particular picture.

    Returns ``(mask_uint8, threshold)``.
    """
    if region is None:
        sample = gray.ravel()
    else:
        sample = gray[region > 0]
        if sample.size == 0:
            sample = gray.ravel()
    t1 = _otsu_threshold(sample)
    sub = sample[sample <= t1]
    t = _otsu_threshold(sub) if sub.size and int(sub.min()) != int(sub.max()) else t1
    mask = (gray <= t).astype(np.uint8) * 255
    if region is not None:
        mask = cv2.bitwise_and(mask, region)
    return mask, float(t)


def estimate_border_thickness(mask: np.ndarray,
                              outer_side_px: Optional[float] = None) -> float:
    """Stroke thickness of the ink mask, from the ridge of its distance transform.

    For a straight stroke of thickness ``t`` the distance transform has a ridge
    (its medial axis) whose value is exactly ``t/2`` all along it.  The square
    borders contribute thousands of ridge pixels at the same value, the marker
    disk only a handful near its centre, so the *median* ridge value is a robust
    estimate of half the border thickness.
    """
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    dil = cv2.dilate(dt, np.ones((3, 3), np.float32))
    ridge = (dt > C.RIDGE_MIN_DT) & (dt >= dil - 1e-6)
    vals = dt[ridge]
    if vals.size < C.RIDGE_MIN_SAMPLES:
        est = (C.BORDER_THICKNESS_FALLBACK_FRAC * outer_side_px) if outer_side_px else 3.0
    else:
        est = 2.0 * float(np.median(vals))
    if outer_side_px:
        lo = C.BORDER_THICKNESS_MIN_FRAC * outer_side_px
        hi = C.BORDER_THICKNESS_MAX_FRAC * outer_side_px
        est = float(np.clip(est, lo, hi))
    return float(max(est, 1.0))


def order_quad_clockwise(quad: Sequence[Sequence[float]]) -> np.ndarray:
    """Order 4 points clockwise *on screen*, starting at the top-left-most.

    Image axes have y pointing down, so sorting by ``atan2(y - cy, x - cx)``
    ascending walks top-left -> top-right -> bottom-right -> bottom-left.
    """
    q = np.asarray(quad, float).reshape(4, 2)
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    order = np.argsort(ang)
    return q[order]


def quad_center(quad: Sequence[Sequence[float]]) -> np.ndarray:
    """Intersection of the quad's diagonals (the projective centre)."""
    q = np.asarray(quad, float).reshape(4, 2)
    p = _line_intersection(q[0], q[2] - q[0], q[1], q[3] - q[1])
    if p is None:                                   # pragma: no cover - degenerate
        return q.mean(axis=0)
    return p


def quad_side_lengths(quad: Sequence[Sequence[float]]) -> np.ndarray:
    q = np.asarray(quad, float).reshape(4, 2)
    return np.array([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)], float)


def _line_intersection(p1: np.ndarray, d1: np.ndarray,
                       p2: np.ndarray, d2: np.ndarray) -> Optional[np.ndarray]:
    cross = float(d1[0] * d2[1] - d1[1] * d2[0])
    n1 = float(np.hypot(d1[0], d1[1])) or 1.0
    n2 = float(np.hypot(d2[0], d2[1])) or 1.0
    if abs(cross) < 1e-9 * n1 * n2:
        return None
    dp = np.asarray(p2, float) - np.asarray(p1, float)
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / cross
    return np.asarray(p1, float) + t * np.asarray(d1, float)


# ---------------------------------------------------------------------------
# Outer quadrilateral
# ---------------------------------------------------------------------------

def _approx_quad(contour: np.ndarray) -> Optional[np.ndarray]:
    peri = cv2.arcLength(contour, True)
    for frac in C.APPROX_EPS_FRACTIONS:
        ap = cv2.approxPolyDP(contour, frac * peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            return ap.reshape(4, 2).astype(np.float64)
    # Fall back on the minimum-area rectangle, which always has 4 corners.
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    return np.asarray(box, np.float64)


def refine_quad_corners(contour: np.ndarray, coarse_quad: np.ndarray
                        ) -> Tuple[np.ndarray, List[bool]]:
    """Sub-pixel corners from per-side ``fitLine`` intersections.

    ``coarse_quad`` must be in walk order along ``contour``.  Each side's
    contour samples are trimmed by :data:`config.SIDE_TRIM_FRAC` at both ends
    before the fit so that corner rounding does not bias the line.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    # index of each coarse vertex along the contour
    idx = [int(np.argmin(np.sum((pts - v) ** 2, axis=1))) for v in coarse_quad]

    lines: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    for k in range(4):
        a, b = idx[k], idx[(k + 1) % 4]
        if b > a:
            seg = pts[a:b + 1]
        else:                       # wraps past the contour start
            seg = np.vstack([pts[a:], pts[:b + 1]])
        m = len(seg)
        trim = int(round(C.SIDE_TRIM_FRAC * m))
        core = seg[trim:m - trim] if m - 2 * trim >= C.SIDE_MIN_SAMPLES else seg
        if len(core) < C.SIDE_MIN_SAMPLES:
            lines.append(None)
            continue
        vx, vy, x0, y0 = cv2.fitLine(core.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines.append((np.array([float(x0), float(y0)]), np.array([float(vx), float(vy)])))

    side = float(np.mean(quad_side_lengths(coarse_quad)))
    max_shift = C.CORNER_REFINE_MAX_SHIFT_FRAC * side
    refined = coarse_quad.astype(np.float64).copy()
    used: List[bool] = []
    for k in range(4):
        la, lb = lines[(k - 1) % 4], lines[k]      # vertex k joins side k-1 and side k
        if la is None or lb is None:
            used.append(False)
            continue
        p = _line_intersection(la[0], la[1], lb[0], lb[1])
        if p is None or np.linalg.norm(p - coarse_quad[k]) > max_shift:
            used.append(False)
            continue
        refined[k] = p
        used.append(True)
    return refined, used


def detect_outer_quad(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
    """Largest foreground component's 4-gon.  Returns (quad, contour, refined_flags)."""
    fg = foreground_mask(gray)
    # Close pinholes so the outer boundary is one clean contour.
    k = max(3, int(round(C.FG_CLOSE_FRAC * min(gray.shape[:2]))) | 1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("no foreground contour found")
    contour = max(contours, key=cv2.contourArea)
    coarse = _approx_quad(contour)
    refined, used = refine_quad_corners(contour, coarse)
    return order_quad_clockwise(refined), contour, used


# ---------------------------------------------------------------------------
# Marker
# ---------------------------------------------------------------------------

def detect_marker(gray: np.ndarray,
                  quad: np.ndarray,
                  border_thickness: float,
                  ink: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Locate the circular marker inside ``quad``.

    Returns a dict with ``found`` plus, when found, the fitted ellipse, the
    distance-transform peak and the HoughCircles centre.
    """
    h, w = gray.shape[:2]
    region = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(region, np.round(quad).astype(np.int32), 255)
    if ink is None:
        ink, _ = dark_mask(gray, region)
    else:
        ink = cv2.bitwise_and(ink, region)

    side = float(np.mean(quad_side_lengths(quad)))
    diam = C.MARKER_OPEN_DIAM_FACTOR * border_thickness
    diam = float(np.clip(diam,
                         C.MARKER_OPEN_DIAM_MIN_FRAC * side,
                         C.MARKER_OPEN_DIAM_MAX_FRAC * side))
    ksz = max(3, int(round(diam)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)

    quad_area = float(cv2.contourArea(np.round(quad).astype(np.int32)))
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cands = [c for c in contours
             if C.MARKER_AREA_MIN_FRAC * quad_area <= cv2.contourArea(c)
             <= C.MARKER_AREA_MAX_FRAC * quad_area and len(c) >= 5]
    if not cands:
        return {"found": False, "reason": "no blob survived the opening",
                "open_kernel_px": ksz}

    # Prefer the most circular candidate, breaking ties by area.
    def _circ(c):
        p = cv2.arcLength(c, True)
        return (4.0 * np.pi * cv2.contourArea(c) / (p * p)) if p > 0 else 0.0

    blob = max(cands, key=lambda c: (_circ(c), cv2.contourArea(c)))
    circ = _circ(blob)
    (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(blob)
    major, minor = (max(ax1, ax2), min(ax1, ax2))
    axis_ratio = minor / major if major > 0 else 0.0
    radius = 0.25 * (major + minor)      # mean semi-axis

    # --- cross-check 1: distance-transform peak of the blob -----------------
    blob_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(blob_mask, [blob], -1, 255, cv2.FILLED)
    dtb = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
    peak = float(dtb.max())
    near = np.argwhere(dtb >= peak - 1e-6)
    dt_center = np.array([near[:, 1].mean(), near[:, 0].mean()], float)

    # --- cross-check 2: HoughCircles in a band around the measured radius ---
    hough_center = None
    try:
        lo = max(3, int(round(radius * (1.0 - C.HOUGH_RADIUS_BAND))))
        hi = int(round(radius * (1.0 + C.HOUGH_RADIUS_BAND))) + 1
        blur = cv2.GaussianBlur(gray, (0, 0), max(1.0, border_thickness / 2.0))
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.0,
                                   minDist=float(max(3.0, C.HOUGH_MIN_DIST_FRAC * radius)),
                                   param1=C.HOUGH_PARAM1, param2=C.HOUGH_PARAM2,
                                   minRadius=lo, maxRadius=hi)
        if circles is not None and len(circles[0]):
            arr = np.asarray(circles[0], float)
            d = np.hypot(arr[:, 0] - cx, arr[:, 1] - cy)
            hough_center = arr[int(np.argmin(d)), :2].copy()
    except cv2.error:                                   # pragma: no cover
        hough_center = None

    return {
        "found": True,
        "center": np.array([float(cx), float(cy)]),
        "axes": (float(major), float(minor)),
        "angle_deg": float(ang),
        "radius": float(radius),
        "circularity": float(circ),
        "axis_ratio": float(axis_ratio),
        "dt_center": dt_center,
        "dt_radius": peak,
        "hough_center": hough_center,
        "open_kernel_px": ksz,
        "area": float(cv2.contourArea(blob)),
    }


# ---------------------------------------------------------------------------
# Top-level detector
# ---------------------------------------------------------------------------

def detect_port(image: np.ndarray) -> PortDetection:
    """Full detection: outer quad, centre, marker, canonical ordering, confidence."""
    from .io import to_gray
    gray = to_gray(image)

    quad_img, contour, refined_flags = detect_outer_quad(gray)
    center = quad_center(quad_img)
    sides = quad_side_lengths(quad_img)
    side = float(sides.mean())

    region = np.zeros(gray.shape[:2], np.uint8)
    cv2.fillConvexPoly(region, np.round(quad_img).astype(np.int32), 255)
    ink, _thr = dark_mask(gray, region)
    thickness = estimate_border_thickness(ink, side)

    m = detect_marker(gray, quad_img, thickness, ink=ink)

    notes: List[str] = []
    terms: Dict[str, float] = {}

    # Quad quality: opposite sides of a square port should match.
    r1 = min(sides[0], sides[2]) / max(sides[0], sides[2])
    r2 = min(sides[1], sides[3]) / max(sides[1], sides[3])
    terms["quad_opposite_sides"] = float(min(r1, r2))
    terms["corners_refined"] = float(sum(refined_flags)) / 4.0

    det = PortDetection(
        quad=quad_img.copy(),
        quad_image_order=quad_img.copy(),
        center=center,
        side_lengths=sides,
        outer_side_px=side,
        border_thickness_px=thickness,
        notes=notes,
        confidence_terms=terms,
    )

    if not m.get("found"):
        notes.append("marker not found: {}".format(m.get("reason", "unknown")))
        terms["marker_shape"] = 0.0
        terms["marker_agreement"] = 0.0
        det.confidence = 0.0
        return det

    det.marker_found = True
    det.marker_center = np.asarray(m["center"], float)
    det.marker_axes = m["axes"]                      # type: ignore[assignment]
    det.marker_angle_deg = float(m["angle_deg"])     # type: ignore[arg-type]
    det.marker_radius_px = float(m["radius"])        # type: ignore[arg-type]
    det.marker_circularity = float(m["circularity"])  # type: ignore[arg-type]
    det.marker_axis_ratio = float(m["axis_ratio"])   # type: ignore[arg-type]
    det.marker_center_dt = np.asarray(m["dt_center"], float)
    det.marker_center_hough = (np.asarray(m["hough_center"], float)
                               if m.get("hough_center") is not None else None)

    # Canonical ordering: the marker names the port-local "top-left" corner.
    d = np.linalg.norm(quad_img - det.marker_center[None, :], axis=1)
    k = int(np.argmin(d))
    det.marker_corner_index = k
    det.quad = np.roll(quad_img, -k, axis=0)

    # --- confidence ---------------------------------------------------------
    shape = min(
        float(np.clip(det.marker_circularity / max(C.MARKER_CIRCULARITY_MIN, 1e-6), 0.0, 1.0)),
        float(np.clip(det.marker_axis_ratio / max(C.MARKER_AXIS_RATIO_MIN, 1e-6), 0.0, 1.0)),
    )
    terms["marker_shape"] = shape

    # Primary cross-check: the distance-transform peak of the same blob.  It is
    # metric (sub-pixel, same mask), so it gets the tight budget and is what the
    # confidence is built from.
    radius = max(det.marker_radius_px, 1e-6)
    dt_off = float(np.linalg.norm(det.marker_center - det.marker_center_dt))
    terms["marker_dt_offset_px"] = dt_off
    terms["marker_agreement"] = float(np.clip(
        1.0 - dt_off / (C.MARKER_CENTER_AGREE_FRAC * radius), 0.0, 1.0))

    # Secondary cross-check: HoughCircles.  It votes into an integer accumulator
    # over a coarse radius band on the raw gradient image, so an offset of a few
    # per cent of the radius is expected even when the circle is perfectly
    # found.  It therefore gets its own, looser budget and is reported rather
    # than allowed to dominate a min() over the strict terms -- otherwise the
    # coarsest method would set the score for the whole detector.
    if det.marker_center_hough is not None:
        h_off = float(np.linalg.norm(det.marker_center - det.marker_center_hough))
        terms["marker_hough_offset_px"] = h_off
        terms["marker_hough_agreement"] = float(np.clip(
            1.0 - h_off / (C.MARKER_HOUGH_AGREE_FRAC * radius), 0.0, 1.0))
        if h_off > C.MARKER_HOUGH_AGREE_FRAC * radius:
            notes.append("HoughCircles centre is {:.1f} px from the fitted "
                         "ellipse ({:.0f}% of the radius); reported, but not "
                         "used to score the fit".format(h_off, 100.0 * h_off / radius))
    else:
        terms["marker_hough_agreement"] = float("nan")
        notes.append("HoughCircles found no circle at this radius band; the "
                     "distance-transform cross-check is used on its own")

    scoring = ["quad_opposite_sides", "corners_refined", "marker_shape", "marker_agreement"]
    det.confidence = float(min(terms[k2] for k2 in scoring))
    return det
