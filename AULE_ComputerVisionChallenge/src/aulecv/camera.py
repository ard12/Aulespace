r"""Parts C and D -- pinhole camera model, plane homographies and rendering.

Frames
------
World origin at the **port centre**, axes ``X`` right, ``Y`` **down**, ``Z``
pointing away from the frontal camera *into* the port, so the port plane is
``Z = 0``.  The reference (frontal) camera sits at ``C0 = (0, 0, -100)`` cm and
the orbit is

    C(theta) = (R sin(theta), 0, -R cos(theta)),      R = 100 cm

with ``theta > 0`` meaning the camera has moved to the **viewer's right**, which
is what "a view that's from a slightly right side" asks for.

Look-at rotation (derivation)
----------------------------------------
Given the camera centre ``C``, the target ``O`` and a ``world_down`` direction
(the zero-roll choice: "looking directly at it" pins the optical axis but not
the roll about it, so we keep the camera level):

    z_c = (O - C) / ||O - C||
    x_c = normalize(world_down x z_c)
    y_c = z_c x x_c
    R_wc = [x_c^T; y_c^T; z_c^T],       t = -R_wc C

Hand check at ``theta = 0``: ``z_c = (0,0,1)``, ``x_c = (1,0,0)``,
``y_c = (0,1,0)`` so ``R = I``.  For general theta,
``R = [[c,0,s],[0,1,0],[-s,0,c]]`` and ``t = (0, 0, 100)`` for **every** theta,
which is just the statement that the camera is always 100 cm away along its own
optical axis.  These are asserted numerically, not assumed.

Plane-induced homography: the sign, derived
-------------------------------------------
Write a plane point in the **reference** camera frame as ``X0 = X_w + t0`` with
``t0 = (0, 0, 100)``.  The port plane ``Z_w = 0`` becomes

    n^T X0 = d,      n = (0, 0, 1),   d = +100.

In the rotated camera frame ``X_th = R X_w + t = R X0 + (t - R t0)``.  Put
``t_rel = t - R t0``.  On the plane ``n^T X0 / d = 1``, so

    X_th = (R + t_rel n^T / d) X0

and therefore

    **H = K (R + t_rel n^T / d) K^-1**     -- a PLUS sign.

The familiar minus sign (Hartley & Zisserman) belongs to the *other* plane
convention, ``n^T X + d = 0``.  :func:`plane_induced_homography` implements the
plus form and :func:`homography_sign_check` verifies it numerically against the
independently composed ``K[r1 r2 t](theta) (K[r1 r2 t](0))^-1`` -- the identity
is checked, never assumed.

Two rendering models
--------------------
``model='metric'`` (**Model A, primary**)
    The port *is* a 40 x 40 cm square.  ``G`` maps raster pixels to plane
    centimetres by sending the four detected outer corners to ``(+-20, +-20)``.
    The output camera has genuinely square pixels
    (``f = 2.5 * sqrt(w_px * h_px)``), so ``H_A(theta) = K [r1 r2 t](theta) G``.
    At ``theta = 0`` this renders the *rectified* (square) port.

    This is deliberately **not** the tempting shortcut of back-projecting the
    raw raster corners with ``fx = fy``: the raster's outer region measures
    about 553 x 516 px, so that shortcut silently turns the port into a
    ~41 x 39 cm rectangle and contradicts the 40 x 40 cm geometry we were given.

``model='image_faithful'`` (**Model B, secondary**)
    Takes the supplied raster *as* the theta = 0 camera image and warps it with
    the plane-induced homography using ``fx = 2.5 w_px``, ``fy = 2.5 h_px``
    (non-square pixels absorb the drawing's rectangularity).  Then
    ``H_B(0) = I`` **exactly**, which is a sharp check, at the price of the
    implied object not being exactly square.

The two are reported side by side, with the corner displacement between them
measured, so the modelling choice is visible rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config as C

__all__ = [
    "PortGeometry",
    "ViewRenderer",
    "estimate_camera_intrinsics",
    "look_at_rotation",
    "camera_pose_for_angle",
    "plane_to_image_homography",
    "raster_to_plane_homography",
    "plane_induced_homography",
    "homography_for_camera_pose",
    "homography_sign_check",
    "make_renderer",
    "render_port_view",
    "generate_return_sequence",
    "return_sequence_table",
    "recover_pose_pnp",
    "project_points",
    "normalize_homography",
    "homography_distance",
    "predicted_edge_depths_cm",
    "predicted_edge_height_ratio",
    "measure_edge_height_ratio",
    "similarity_fit_residual",
    "arc_and_chord_cm",
]


# ---------------------------------------------------------------------------
# Geometry container
# ---------------------------------------------------------------------------

@dataclass
class PortGeometry:
    """The physical port plus the raster that textures it."""

    raster_corners: np.ndarray          #: (4,2) canonical order: marker corner first, then clockwise
    raster_center: np.ndarray           #: (2,) port centre in raster pixels
    width_px: float                     #: mean of the two horizontal side lengths
    height_px: float                    #: mean of the two vertical side lengths
    raster_shape: Tuple[int, int]       #: (H, W) of the source image
    side_cm: float = C.OUTER_SIDE_CM
    distance_cm: float = C.CAMERA_DISTANCE_CM

    @classmethod
    def from_detection(cls, detection, raster_shape) -> "PortGeometry":
        q = np.asarray(detection.quad, float).reshape(4, 2)
        lens = np.array([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)])
        return cls(raster_corners=q,
                   raster_center=np.asarray(detection.center, float),
                   width_px=float(0.5 * (lens[0] + lens[2])),
                   height_px=float(0.5 * (lens[1] + lens[3])),
                   raster_shape=(int(raster_shape[0]), int(raster_shape[1])))

    @property
    def half_cm(self) -> float:
        return 0.5 * self.side_cm

    @property
    def plane_corners_cm(self) -> np.ndarray:
        """(4,2) plane coordinates matching :pyattr:`raster_corners`, in cm."""
        h = self.half_cm
        return np.array([[-h, -h], [h, -h], [h, h], [-h, h]], float)

    @property
    def object_points_cm(self) -> np.ndarray:
        """(4,3) float32 object points for ``solvePnP`` -- the *physical* square."""
        p = self.plane_corners_cm
        return np.hstack([p, np.zeros((4, 1))]).astype(np.float32)

    @property
    def metric_focal_px(self) -> float:
        """f = 2.5 * sqrt(w*h): square pixels, theta=0 at the raster's mean size."""
        return float(C.FOCAL_SCALE * np.sqrt(self.width_px * self.height_px))

    @property
    def image_faithful_focal_px(self) -> Tuple[float, float]:
        """(fx, fy) = 2.5 * (w, h): non-square pixels, H_B(0) = I exactly."""
        return (float(C.FOCAL_SCALE * self.width_px),
                float(C.FOCAL_SCALE * self.height_px))


# ---------------------------------------------------------------------------
# Intrinsics, pose, homographies
# ---------------------------------------------------------------------------

def estimate_camera_intrinsics(scale_px,
                               canvas: Optional[Sequence[int]] = None,
                               principal_point: Optional[Sequence[float]] = None
                               ) -> np.ndarray:
    """Build ``K`` from a pixel scale and a canvas.

    ``scale_px`` is either a scalar (square pixels, Model A) or ``(fx, fy)``
    (Model B).  With no explicit ``principal_point`` the optical centre is put
    at the canvas centre, ``((W-1)/2, (H-1)/2)``.

    ``f`` is a *display scale only*: replacing ``K`` by ``S K`` with ``S`` a
    similarity replaces ``H`` by ``S H``, so the rendered geometry is unchanged
    up to a similarity.  :func:`similarity_fit_residual` demonstrates this
    numerically in notebook 03.
    """
    if np.isscalar(scale_px):
        fx = fy = float(scale_px)
    else:
        fx, fy = (float(v) for v in scale_px)
    if principal_point is None:
        if canvas is None:
            raise ValueError("give either canvas=(W, H) or principal_point=(cx, cy)")
        W, H = (int(canvas[0]), int(canvas[1]))
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    else:
        cx, cy = (float(principal_point[0]), float(principal_point[1]))
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], float)


def look_at_rotation(camera_center: Sequence[float],
                     target: Sequence[float] = (0.0, 0.0, 0.0),
                     world_down: Sequence[float] = (0.0, 1.0, 0.0)
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """World->camera rotation and translation for a zero-roll look-at.

    Returns ``(R_wc, t)`` such that a world point ``X`` sits at ``R_wc X + t``
    in the camera frame.  See the module docstring for the derivation.
    """
    Cc = np.asarray(camera_center, float).reshape(3)
    O = np.asarray(target, float).reshape(3)
    down = np.asarray(world_down, float).reshape(3)

    v = O - Cc
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("camera centre coincides with the target")
    z_c = v / n

    xw = np.cross(down, z_c)
    nx = np.linalg.norm(xw)
    if nx < 1e-9:
        raise ValueError("world_down is parallel to the optical axis; the roll "
                         "is undefined for this pose")
    x_c = xw / nx
    y_c = np.cross(z_c, x_c)

    R_wc = np.vstack([x_c, y_c, z_c])
    t = -R_wc @ Cc
    return R_wc, t


def camera_pose_for_angle(theta_deg: float,
                          radius_cm: float = C.CAMERA_DISTANCE_CM,
                          world_down: Sequence[float] = (0.0, 1.0, 0.0)
                          ) -> Dict[str, np.ndarray]:
    """Orbit pose at ``theta_deg`` (positive = to the viewer's right)."""
    th = np.radians(float(theta_deg))
    Cc = np.array([radius_cm * np.sin(th), 0.0, -radius_cm * np.cos(th)], float)
    R, t = look_at_rotation(Cc, (0.0, 0.0, 0.0), world_down)
    return {"C": Cc, "R": R, "t": t, "theta_deg": float(theta_deg)}


def plane_to_image_homography(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``K [r1 r2 t]``: plane (Z=0) coordinates -> image pixels."""
    R = np.asarray(R, float).reshape(3, 3)
    t = np.asarray(t, float).reshape(3)
    M = np.column_stack([R[:, 0], R[:, 1], t])
    return np.asarray(K, float) @ M


def raster_to_plane_homography(corners: Sequence[Sequence[float]],
                               side_cm: float = C.OUTER_SIDE_CM) -> np.ndarray:
    """``G``: raster pixels -> plane centimetres.

    The four detected outer corners (canonical order, marker corner first then
    clockwise) are sent to ``(-h,-h), (h,-h), (h,h), (-h,h)`` with
    ``h = side_cm / 2``.  This is where the *given* 40 x 40 cm geometry enters,
    and it is the reason Model A never has to invent a metric size for the
    drawing.
    """
    src = np.asarray(corners, np.float32).reshape(4, 2)
    h = 0.5 * float(side_cm)
    dst = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def plane_induced_homography(K: np.ndarray,
                             R_rel: np.ndarray,
                             t_rel: Sequence[float],
                             n: Sequence[float] = (0.0, 0.0, 1.0),
                             d: float = C.CAMERA_DISTANCE_CM) -> np.ndarray:
    """``K (R_rel + t_rel n^T / d) K^-1`` -- PLUS sign, convention ``n^T X = d``."""
    K = np.asarray(K, float).reshape(3, 3)
    R_rel = np.asarray(R_rel, float).reshape(3, 3)
    t_rel = np.asarray(t_rel, float).reshape(3, 1)
    n = np.asarray(n, float).reshape(1, 3)
    return K @ (R_rel + (t_rel @ n) / float(d)) @ np.linalg.inv(K)


def normalize_homography(H: np.ndarray) -> np.ndarray:
    """Scale- and sign-normalise a homography so two of them can be compared."""
    H = np.asarray(H, float).reshape(3, 3)
    nrm = np.linalg.norm(H)
    if nrm < 1e-300:                                    # pragma: no cover
        return H.copy()
    Hn = H / nrm
    k = int(np.argmax(np.abs(Hn)))
    if Hn.ravel()[k] < 0:
        Hn = -Hn
    return Hn


def homography_distance(H1: np.ndarray, H2: np.ndarray) -> float:
    """Scale-normalised Frobenius distance between two homographies."""
    return float(np.linalg.norm(normalize_homography(H1) - normalize_homography(H2)))


def homography_sign_check(K: np.ndarray,
                          theta_deg: float,
                          radius_cm: float = C.CAMERA_DISTANCE_CM
                          ) -> Dict[str, object]:
    """Verify the PLUS sign numerically instead of asserting it.

    Compares the analytic ``K (R + t_rel n^T/d) K^-1`` against the
    independently composed ``K[r1 r2 t](theta) (K[r1 r2 t](0))^-1``, and also
    reports the distance for the *minus* sign so the difference is visible.
    """
    p0 = camera_pose_for_angle(0.0, radius_cm)
    pt = camera_pose_for_angle(theta_deg, radius_cm)
    t_rel = pt["t"] - pt["R"] @ p0["t"]

    H_plus = plane_induced_homography(K, pt["R"], t_rel, (0, 0, 1), radius_cm)
    H_minus = plane_induced_homography(K, pt["R"], -t_rel, (0, 0, 1), radius_cm)

    M_t = plane_to_image_homography(K, pt["R"], pt["t"])
    M_0 = plane_to_image_homography(K, p0["R"], p0["t"])
    H_comp = M_t @ np.linalg.inv(M_0)

    return {
        "theta_deg": float(theta_deg),
        "t_rel": t_rel,
        "H_analytic_plus": H_plus,
        "H_analytic_minus": H_minus,
        "H_composed": H_comp,
        "distance_plus": homography_distance(H_plus, H_comp),
        "distance_minus": homography_distance(H_minus, H_comp),
        "plus_wins": bool(homography_distance(H_plus, H_comp)
                          < homography_distance(H_minus, H_comp)),
    }


def homography_for_camera_pose(theta_deg: float,
                               geom: PortGeometry,
                               model: str = "metric",
                               K: Optional[np.ndarray] = None
                               ) -> Dict[str, object]:
    """Raster-pixel -> output-pixel homography for one pose, for either model.

    Returns a dict with ``H`` plus the pieces it was built from, so a notebook
    can re-compose it and check the arithmetic.
    """
    pose = camera_pose_for_angle(theta_deg, geom.distance_cm)

    if model == "metric":
        if K is None:
            K = estimate_camera_intrinsics(geom.metric_focal_px,
                                           principal_point=(0.0, 0.0))
        G = raster_to_plane_homography(geom.raster_corners, geom.side_cm)
        M = plane_to_image_homography(K, pose["R"], pose["t"])
        H = M @ G
        return {"H": H, "K": K, "G": G, "M": M, "pose": pose, "model": model}

    if model == "image_faithful":
        if K is None:
            K = estimate_camera_intrinsics(geom.image_faithful_focal_px,
                                           principal_point=geom.raster_center)
        p0 = camera_pose_for_angle(0.0, geom.distance_cm)
        t_rel = pose["t"] - pose["R"] @ p0["t"]
        H = plane_induced_homography(K, pose["R"], t_rel, (0, 0, 1), geom.distance_cm)
        return {"H": H, "K": K, "G": None, "M": None, "pose": pose,
                "t_rel": t_rel, "model": model}

    raise ValueError("model must be 'metric' or 'image_faithful', got {!r}".format(model))


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def project_points(H: np.ndarray, pts: Sequence[Sequence[float]]) -> np.ndarray:
    """Apply a homography, rejecting points on its projective horizon.

    Multiplying H by any finite nonzero scale must leave the answer unchanged.
    A zero denominator describes a point at infinity, not a very distant pixel.
    """
    matrix = np.asarray(H, float).reshape(3, 3)
    p = np.asarray(pts, float).reshape(-1, 2)
    if not np.isfinite(matrix).all() or not np.isfinite(p).all():
        raise ValueError("homography and points must be finite")
    scale = float(np.max(np.abs(matrix)))
    if scale == 0:
        raise ValueError("homography must be nonzero")
    matrix = matrix / scale
    ph = np.hstack([p, np.ones((len(p), 1))])
    q = (matrix @ ph.T).T
    w = q[:, 2:3]
    bound = np.abs(ph) @ np.abs(matrix[2])
    if np.any(np.abs(w[:, 0]) <= 8 * np.finfo(float).eps * bound):
        raise ValueError("cannot project a point on the homography horizon")
    return q[:, :2] / w


def similarity_fit_residual(src: Sequence[Sequence[float]],
                            dst: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Best similarity (scale + rotation + translation) from ``src`` to ``dst``.

    Closed-form Umeyama.  Used to show that changing ``f`` or the principal
    point moves the rendered corners **only** by a similarity: the residual
    after this fit is what matters, not the raw displacement.
    """
    A = np.asarray(src, float).reshape(-1, 2)
    B = np.asarray(dst, float).reshape(-1, 2)
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    Sigma = (B0.T @ A0) / len(A)
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    Rm = U @ S @ Vt
    var_a = float((A0 ** 2).sum() / len(A))
    scale = float(np.trace(np.diag(D) @ S) / var_a) if var_a > 1e-12 else 1.0
    tm = mb - scale * (Rm @ ma)
    pred = (scale * (Rm @ A.T)).T + tm
    res = np.linalg.norm(pred - B, axis=1)
    return {"scale": scale,
            "rotation_deg": float(np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))),
            "tx": float(tm[0]), "ty": float(tm[1]),
            "max_residual_px": float(res.max()),
            "rms_residual_px": float(np.sqrt((res ** 2).mean()))}


# ---------------------------------------------------------------------------
# Closed-form physical predictions (independent of the rendering code)
# ---------------------------------------------------------------------------

def predicted_edge_depths_cm(theta_deg: float,
                             side_cm: float = C.OUTER_SIDE_CM,
                             distance_cm: float = C.CAMERA_DISTANCE_CM
                             ) -> Dict[str, float]:
    """Depth (along the optical axis) of the port's two vertical edges.

    For ``P = (+-side/2, y, 0)`` and ``C = (R sin th, 0, -R cos th)`` with
    ``z_c = (-sin th, 0, cos th)``::

        (P - C) . z_c = R -+ (side/2) sin th

    so the ``+X`` (right) edge is the **near** one for ``theta > 0``.
    """
    s = np.sin(np.radians(float(theta_deg)))
    h = 0.5 * float(side_cm)
    return {"right_near_cm": float(distance_cm - h * s),
            "left_far_cm": float(distance_cm + h * s)}


def predicted_edge_height_ratio(theta_deg: float,
                                side_cm: float = C.OUTER_SIDE_CM,
                                distance_cm: float = C.CAMERA_DISTANCE_CM) -> float:
    """Projected height of the near (right) edge over the far (left) edge.

    Both vertical edges lie at *constant* depth, so their projected heights are
    exactly ``f * side / depth`` and the ratio is depth-only -- independent of
    ``f`` and of the principal point.
    """
    d = predicted_edge_depths_cm(theta_deg, side_cm, distance_cm)
    return float(d["left_far_cm"] / d["right_near_cm"])


def measure_edge_height_ratio(quad: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Measure the near/far edge-height ratio on a detected quad.

    ``quad`` is in canonical order (marker corner first, then clockwise), i.e.
    ``[TL, TR, BR, BL]`` in port-local terms, so the right edge is ``TR->BR``
    and the left edge is ``BL->TL``.
    """
    q = np.asarray(quad, float).reshape(4, 2)
    right = float(np.linalg.norm(q[2] - q[1]))
    left = float(np.linalg.norm(q[0] - q[3]))
    return {"right_px": right, "left_px": left,
            "ratio": right / left if left > 0 else float("nan"),
            "right_is_taller": bool(right > left)}


def arc_and_chord_cm(dtheta_deg: float,
                     radius_cm: float = C.CAMERA_DISTANCE_CM) -> Dict[str, float]:
    """Arc length and straight-line chord for one angular step, plus the
    mid-point radius of the *chord* (which is why the path must be an arc)."""
    dt = np.radians(float(dtheta_deg))
    return {"arc_cm": float(radius_cm * dt),
            "chord_cm": float(2.0 * radius_cm * np.sin(dt / 2.0)),
            "chord_midpoint_radius_cm": float(radius_cm * np.cos(dt / 2.0))}


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

@dataclass
class ViewRenderer:
    """A fixed canvas plus a model, able to render any pose in a theta list."""

    geom: PortGeometry
    model: str
    K: np.ndarray
    canvas: Tuple[int, int]             #: (W, H)
    offset: np.ndarray                  #: 3x3, applied *after* the model homography
    thetas: Tuple[float, ...]
    G: Optional[np.ndarray] = None
    info: Dict[str, object] = field(default_factory=dict)

    # -- homographies -----------------------------------------------------
    def model_homography(self, theta_deg: float) -> np.ndarray:
        """The raw model homography, with no canvas translation applied.

        For ``model='image_faithful'`` this is exactly ``I`` at ``theta = 0``.
        """
        return homography_for_camera_pose(theta_deg, self.geom, self.model, self.K)["H"]

    def homography(self, theta_deg: float) -> np.ndarray:
        """Raster pixels -> canvas pixels (canvas translation included)."""
        H = self.offset @ self.model_homography(theta_deg)
        return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H

    def project_port_corners(self, theta_deg: float) -> np.ndarray:
        """The port's four outer corners in canvas pixels, canonical order."""
        return project_points(self.homography(theta_deg), self.geom.raster_corners)

    def project_port_center(self, theta_deg: float) -> np.ndarray:
        return project_points(self.homography(theta_deg),
                              self.geom.raster_center.reshape(1, 2))[0]

    def render(self, image: np.ndarray, theta_deg: float,
               interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
        return render_port_view(image, self.homography(theta_deg), self.canvas,
                                interpolation=interpolation)


def make_renderer(geom: PortGeometry,
                  thetas: Sequence[float],
                  model: str = "metric",
                  focal_px=None,
                  margin_frac: Optional[float] = None) -> ViewRenderer:
    """Build a renderer whose canvas is fixed across the whole ``thetas`` sweep.

    The canvas is sized from the union of the projected quads over every angle
    that will be rendered, plus a margin, so Part C's single frame and Part D's
    sequence share one coordinate system and can be diffed frame by frame.
    """
    if margin_frac is None:
        margin_frac = C.CANVAS_MARGIN_FRAC
    thetas = tuple(float(t) for t in thetas)

    if model == "metric":
        f = geom.metric_focal_px if focal_px is None else focal_px
        K0 = estimate_camera_intrinsics(f, principal_point=(0.0, 0.0))
        G = raster_to_plane_homography(geom.raster_corners, geom.side_cm)
        probe = geom.raster_corners
        def _H(th, Kx):
            pose = camera_pose_for_angle(th, geom.distance_cm)
            return plane_to_image_homography(Kx, pose["R"], pose["t"]) @ G
    elif model == "image_faithful":
        f = geom.image_faithful_focal_px if focal_px is None else focal_px
        K0 = estimate_camera_intrinsics(f, principal_point=geom.raster_center)
        G = None
        H_img, W_img = geom.raster_shape
        probe = np.array([[0, 0], [W_img - 1, 0], [W_img - 1, H_img - 1], [0, H_img - 1]], float)
        def _H(th, Kx):
            return homography_for_camera_pose(th, geom, "image_faithful", Kx)["H"]
    else:
        raise ValueError("unknown model {!r}".format(model))

    pts = np.vstack([project_points(_H(th, K0), probe) for th in thetas])
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    m = margin_frac * span

    if model == "metric":
        # Fold the canvas placement into the principal point: K keeps square
        # pixels and the homography stays a single clean K [r1 r2 t] G product.
        W = int(np.ceil(hi[0] - lo[0] + 2 * m))
        H = int(np.ceil(hi[1] - lo[1] + 2 * m))
        K = estimate_camera_intrinsics(f, principal_point=(-lo[0] + m, -lo[1] + m))
        offset = np.eye(3)
    else:
        # K is pinned by the H_B(0) = I requirement, so the canvas placement is
        # a separate pure translation applied afterwards.  That translation is
        # rounded to whole pixels on purpose: a fractional shift would resample
        # every frame through the interpolator and destroy Model B's defining
        # property, that theta = 0 returns the supplied raster *exactly*.  The
        # canvas is then sized from the shifted bounding box so rounding can
        # never push content off the edge.
        K = K0
        tx = float(np.ceil(m - lo[0]))
        ty = float(np.ceil(m - lo[1]))
        W = int(np.ceil(hi[0] + tx + m))
        H = int(np.ceil(hi[1] + ty + m))
        offset = np.array([[1.0, 0.0, tx],
                           [0.0, 1.0, ty],
                           [0.0, 0.0, 1.0]], float)

    return ViewRenderer(geom=geom, model=model, K=K, canvas=(W, H), offset=offset,
                        thetas=thetas, G=G,
                        info={"focal_px": f, "margin_frac": margin_frac,
                              "bbox_min": lo, "bbox_max": hi})


def render_port_view(image: np.ndarray,
                     H: np.ndarray,
                     canvas: Sequence[int],
                     background: Sequence[int] = C.RENDER_BACKGROUND,
                     interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Warp ``image`` through ``H`` onto a ``(W, H)`` canvas."""
    W, Hc = int(canvas[0]), int(canvas[1])
    bg = tuple(float(v) for v in background)
    if image.ndim == 2:
        bg = bg[0]
    return cv2.warpPerspective(image, np.asarray(H, float), (W, Hc),
                               flags=interpolation,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=bg)


# ---------------------------------------------------------------------------
# Part D -- the return sequence
# ---------------------------------------------------------------------------

def generate_return_sequence(theta0_deg: float = C.PART_C_ANGLE_DEG,
                             dtheta_deg: float = C.PART_D_STEP_DEG,
                             radius_cm: float = C.CAMERA_DISTANCE_CM) -> np.ndarray:
    """Angles from ``theta0`` back to exactly 0, in steps of at most ``dtheta``.

    Every entry is an *absolute* pose on the 100 cm sphere; nothing is
    integrated from the previous frame, so ``||C|| = 100`` holds exactly at
    every step and there is no drift.
    """
    theta0 = float(theta0_deg)
    dt = abs(float(dtheta_deg))
    if not np.isfinite(theta0) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("theta0_deg must be finite and dtheta_deg finite and nonzero")
    n = int(np.ceil(abs(theta0) / dt))
    sgn = np.sign(theta0) if theta0 != 0 else 1.0
    seq = [theta0 - sgn * dt * k for k in range(n)]
    seq.append(0.0)
    return np.array(seq, float)


def return_sequence_table(thetas: Sequence[float],
                          radius_cm: float = C.CAMERA_DISTANCE_CM) -> List[Dict[str, float]]:
    """Per-step table: pose, radius, arc and chord."""
    rows: List[Dict[str, float]] = []
    prev = None
    for i, th in enumerate(thetas):
        pose = camera_pose_for_angle(float(th), radius_cm)
        Cc = pose["C"]
        if prev is None:
            arc = chord = 0.0
        else:
            dth = abs(float(th) - float(prev))
            ac = arc_and_chord_cm(dth, radius_cm)
            arc, chord = ac["arc_cm"], ac["chord_cm"]
        rows.append({"step": i, "theta_deg": float(th),
                     "Cx_cm": float(Cc[0]), "Cy_cm": float(Cc[1]), "Cz_cm": float(Cc[2]),
                     "radius_cm": float(np.linalg.norm(Cc)),
                     "arc_cm": float(arc), "chord_cm": float(chord)})
        prev = th
    return rows


# ---------------------------------------------------------------------------
# Independent validation: PnP on re-detected corners
# ---------------------------------------------------------------------------

def recover_pose_pnp(image_points: Sequence[Sequence[float]],
                     K: np.ndarray,
                     side_cm: float = C.OUTER_SIDE_CM,
                     dist_coeffs: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Recover the camera pose from four image points and the *physical* square.

    ``image_points`` must come from re-running the detector on the rendered
    frame -- **not** from projecting anything with the same camera model, which
    would validate the model against itself.  ``object_points`` are the true
    40 x 40 cm corners.

    ``SOLVEPNP_IPPE`` returns two solutions for a planar target; the one with
    the lower reprojection error is kept and both are reported.
    """
    h = 0.5 * float(side_cm)
    obj = np.array([[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]], np.float32)
    img = np.asarray(image_points, np.float32).reshape(4, 1, 2)
    K = np.asarray(K, np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1))

    retval, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj, img, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
    if not retval or len(rvecs) == 0:                   # pragma: no cover
        raise RuntimeError("solvePnP failed on the detected corners")

    sols = []
    for rv, tv in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rv)
        proj, _ = cv2.projectPoints(obj, rv, tv, K, dist_coeffs)
        err = float(np.sqrt(((proj.reshape(4, 2) - img.reshape(4, 2)) ** 2).sum(axis=1).mean()))
        Cc = (-R.T @ np.asarray(tv, float).reshape(3))
        sols.append({"R": R, "t": np.asarray(tv, float).reshape(3), "C": Cc,
                     "reprojection_rmse_px": err,
                     "distance_cm": float(np.linalg.norm(Cc)),
                     "theta_deg": float(np.degrees(np.arctan2(Cc[0], -Cc[2])))})

    sols.sort(key=lambda s: s["reprojection_rmse_px"])
    best = dict(sols[0])
    best["all_solutions"] = sols
    best["n_solutions"] = len(sols)
    best["object_points_cm"] = obj
    return best
