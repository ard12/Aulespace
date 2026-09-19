"""Part A -- recovering an arbitrary in-plane rotation angle.

Sign convention
---------------
``rotate_image(img, a)`` is implemented with ``cv2.getRotationMatrix2D(c, a, 1)``,
whose matrix in image axes (x right, y **down**) is

    M = [[ cos a,  sin a, ...],
         [-sin a,  cos a, ...]]

A point at polar angle ``theta = atan2(y - cy, x - cx)`` therefore moves to

    atan2(-s*x + c*y, c*x + s*y) = theta - a,

so **image-coordinate polar angles decrease by ``a``**, which is what makes a
positive ``a`` look counter-clockwise on screen (screen y points down).

Hence, with ``theta0`` the reference marker angle and ``theta1`` the rotated one:

    a = theta0 - theta1   (mod 360)

The same argument applied to the quad's side directions gives the refinement
cue, which is only defined modulo 90 deg because a square maps onto itself
under a quarter turn.  ``tests/test_rotation.py`` verifies this sign against
``cv2.getRotationMatrix2D`` directly rather than trusting the derivation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config as C
from .detect import PortDetection, detect_port

__all__ = [
    "rotate_image",
    "wrap180",
    "wrap360",
    "angular_error",
    "RotationEstimate",
    "estimate_rotation",
    "estimate_rotation_from_detections",
]


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------

def wrap360(a: float) -> float:
    return float(np.mod(a, 360.0))


def wrap180(a: float) -> float:
    return float((np.mod(a + 180.0, 360.0)) - 180.0)


def angular_error(estimate: float, truth: float) -> float:
    """Signed difference ``estimate - truth`` wrapped into (-180, 180]."""
    return wrap180(estimate - truth)


# ---------------------------------------------------------------------------
# Image rotation
# ---------------------------------------------------------------------------

def rotate_image(image: np.ndarray,
                 angle_deg: float,
                 background: Optional[Tuple[int, int, int]] = None,
                 expand: bool = True,
                 interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Rotate ``image`` by ``angle_deg`` (positive = counter-clockwise on screen).

    When ``expand`` is true the canvas grows so that nothing is clipped, and the
    freshly exposed area is filled with the page background colour sampled from
    the image border (so the detector's background estimate still works).
    """
    h, w = image.shape[:2]
    if background is None:
        r = max(1, int(round(C.BG_RING_FRAC * min(h, w))))
        if image.ndim == 2:
            ring = np.concatenate([image[:r, :].ravel(), image[-r:, :].ravel(),
                                   image[:, :r].ravel(), image[:, -r:].ravel()])
            background = (float(np.median(ring)),) * 3
        else:
            ring = np.concatenate([image[:r, :].reshape(-1, image.shape[2]),
                                   image[-r:, :].reshape(-1, image.shape[2]),
                                   image[:, :r].reshape(-1, image.shape[2]),
                                   image[:, -r:].reshape(-1, image.shape[2])])
            background = tuple(float(v) for v in np.median(ring, axis=0))

    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), float(angle_deg), 1.0)
    if expand:
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw = int(np.ceil(h * sin + w * cos))
        nh = int(np.ceil(h * cos + w * sin))
        M[0, 2] += (nw - 1) / 2.0 - cx
        M[1, 2] += (nh - 1) / 2.0 - cy
    else:
        nw, nh = w, h
    return cv2.warpAffine(image, M, (nw, nh), flags=interpolation,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=background)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

@dataclass
class RotationEstimate:
    angle_deg: float                  #: fused estimate in [0, 360)
    marker_angle_deg: float           #: coarse, full-range estimate from the marker
    edge_angle_mod90_deg: float       #: refinement cue, only defined mod 90
    refined: bool                     #: was the mod-90 refinement accepted?
    residual_deg: float               #: |marker estimate - refined estimate|
    confidence: float
    status: str                       #: 'ok' | 'low_confidence' | 'marker_missing'
    long_edge_experiment_deg: Optional[float] = None  #: mod-180 cue, EXPERIMENT ONLY
    notes: List[str] = field(default_factory=list)
    detection: Optional[PortDetection] = None
    extras: Dict[str, float] = field(default_factory=dict)


def estimate_rotation_from_detections(rotated: PortDetection,
                                      reference: PortDetection) -> RotationEstimate:
    """Fuse the full-range marker cue with the square's mod-90 edge cue."""
    notes: List[str] = []

    if not (rotated.marker_found and reference.marker_found):
        return RotationEstimate(
            angle_deg=float("nan"), marker_angle_deg=float("nan"),
            edge_angle_mod90_deg=float("nan"), refined=False,
            residual_deg=float("nan"), confidence=0.0, status="marker_missing",
            notes=["the circular marker was not localised; the square alone "
                   "only determines the angle modulo 90 deg"],
            detection=rotated)

    theta0 = reference.marker_angle_from_center_deg
    theta1 = rotated.marker_angle_from_center_deg
    coarse = wrap360(theta0 - theta1)

    phi0 = reference.edge_angle_mod90_deg()
    phi1 = rotated.edge_angle_mod90_deg()
    base = float(np.mod(phi0 - phi1, 90.0))

    cands = np.array([base + 90.0 * k for k in range(4)], float)
    diffs = np.array([wrap180(c - coarse) for c in cands])
    j = int(np.argmin(np.abs(diffs)))
    residual = float(abs(diffs[j]))

    if residual <= C.ROTATION_REFINE_TOL_DEG:
        angle = wrap360(float(cands[j]))
        refined = True
    else:
        angle = coarse
        refined = False
        notes.append("mod-90 edge cue disagrees with the marker by {:.2f} deg "
                     "(> {:.1f} deg tolerance); keeping the marker-only estimate"
                     .format(residual, C.ROTATION_REFINE_TOL_DEG))

    # Experiment only -- see PortDetection.long_edge_angle_mod180_deg.
    try:
        psi0 = reference.long_edge_angle_mod180_deg()
        psi1 = rotated.long_edge_angle_mod180_deg()
        long_edge = float(np.mod(psi0 - psi1, 180.0))
    except Exception:                                   # pragma: no cover
        long_edge = None

    conf = float(min(rotated.confidence, reference.confidence))
    status = "ok" if conf >= C.ROTATION_MIN_CONFIDENCE else "low_confidence"

    return RotationEstimate(
        angle_deg=angle,
        marker_angle_deg=coarse,
        edge_angle_mod90_deg=base,
        refined=refined,
        residual_deg=residual,
        confidence=conf,
        status=status,
        long_edge_experiment_deg=long_edge,
        notes=notes,
        detection=rotated,
        extras={"reference_marker_polar_deg": theta0,
                "rotated_marker_polar_deg": theta1,
                "reference_edge_mod90_deg": phi0,
                "rotated_edge_mod90_deg": phi1},
    )


def estimate_rotation(rotated_image: np.ndarray,
                      reference: PortDetection) -> RotationEstimate:
    """Detect the port in ``rotated_image`` and estimate its rotation angle."""
    det = detect_port(rotated_image)
    return estimate_rotation_from_detections(det, reference)
