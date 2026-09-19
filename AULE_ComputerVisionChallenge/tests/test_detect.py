"""Detector tests.

The assertions are all *relative* (fractions of the detected outer side, of the
measured marker radius, ...) so that nothing here encodes an answer for one
particular image.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from aulecv import config as C, detect, rotation


# ---------------------------------------------------------------------------
# Reference image
# ---------------------------------------------------------------------------

def test_reference_image_is_the_embedded_raster(reference_image):
    assert reference_image.shape == (558, 604, 3), (
        "assets/reference.png must be the 604x558 raster extracted from PDF "
        "xref 9, not a screenshot or a page render")


def test_outer_quad_is_convex_and_covers_most_of_the_image(reference_detection,
                                                           reference_image):
    q = np.asarray(reference_detection.quad_image_order, np.float32)
    assert cv2.isContourConvex(q.reshape(-1, 1, 2))
    area = cv2.contourArea(q)
    H, W = reference_image.shape[:2]
    assert 0.6 * H * W < area < 1.0 * H * W


def test_opposite_sides_match(reference_detection):
    s = reference_detection.side_lengths
    assert min(s[0], s[2]) / max(s[0], s[2]) > 0.98
    assert min(s[1], s[3]) / max(s[1], s[3]) > 0.98


def test_centre_is_the_diagonal_intersection(reference_detection):
    q = np.asarray(reference_detection.quad, float)
    c = np.asarray(reference_detection.center, float)
    # c must be collinear with both diagonals.
    for a, b in ((q[0], q[2]), (q[1], q[3])):
        d = b - a
        u = d / np.linalg.norm(d)
        v = c - a
        cross = abs(u[0] * v[1] - u[1] * v[0])
        assert cross < 1e-6 * reference_detection.outer_side_px


def test_border_thickness_is_a_small_fraction_of_the_side(reference_detection):
    t = reference_detection.border_thickness_px
    s = reference_detection.outer_side_px
    assert C.BORDER_THICKNESS_MIN_FRAC * s <= t <= C.BORDER_THICKNESS_MAX_FRAC * s


# ---------------------------------------------------------------------------
# Marker
# ---------------------------------------------------------------------------

def test_marker_is_found_and_round(reference_detection):
    d = reference_detection
    assert d.marker_found
    assert d.marker_axis_ratio >= C.MARKER_AXIS_RATIO_MIN
    assert d.marker_circularity >= C.MARKER_CIRCULARITY_MIN


def test_marker_circularity_is_below_one_as_measured(reference_detection):
    """The marker touches both square borders and the raster is stair-stepped,
    so circularity is well under 1.  This pins the *reason* the threshold in
    config is lenient, so nobody 'tightens' it later and breaks the detector."""
    assert 0.75 < reference_detection.marker_circularity < 0.95


def test_marker_cross_checks_agree(reference_detection):
    d = reference_detection
    off = np.linalg.norm(d.marker_center - d.marker_center_dt)
    assert off < C.MARKER_CENTER_AGREE_FRAC * d.marker_radius_px


def test_marker_sits_in_one_quadrant_not_at_the_centre(reference_detection):
    d = reference_detection
    v = d.marker_vector
    assert np.linalg.norm(v) > 0.15 * d.outer_side_px, (
        "the marker must be off-centre or it carries no orientation information")


def test_confidence_is_high_on_the_clean_reference(reference_detection):
    assert reference_detection.confidence >= C.ROTATION_MIN_CONFIDENCE
    assert reference_detection.ok


# ---------------------------------------------------------------------------
# Canonical ordering (this is what makes the detector rotation-usable)
# ---------------------------------------------------------------------------

def test_canonical_order_starts_at_the_marker_corner(reference_detection):
    d = reference_detection
    dists = np.linalg.norm(np.asarray(d.quad, float) - d.marker_center[None, :], axis=1)
    assert int(np.argmin(dists)) == 0


def test_canonical_order_is_clockwise_on_screen(reference_detection):
    """Clockwise *on screen* means a positive shoelace sum in y-down axes.

    Image y points down, so the usual "positive shoelace = counter-clockwise"
    rule is mirrored: the sequence TL -> TR -> BR -> BL that looks clockwise on
    a monitor gives a positive sum here.
    """
    q = np.asarray(reference_detection.quad, float)
    area2 = sum(q[i, 0] * q[(i + 1) % 4, 1] - q[(i + 1) % 4, 0] * q[i, 1] for i in range(4))
    assert area2 > 0


@pytest.mark.parametrize("angle", [0.0, 37.0, 90.0, 168.0, 265.0, 331.0])
def test_canonical_order_follows_the_port_through_rotation(reference_image,
                                                           reference_detection,
                                                           angle):
    """Corner k of the canonical quad must stay corner k of the *port*.

    Checked by rotating the canonical corners with the same matrix that rotated
    the image and comparing against the fresh detection.
    """
    rotated = rotation.rotate_image(reference_image, angle)
    d2 = detect.detect_port(rotated)
    assert d2.marker_found

    h, w = reference_image.shape[:2]
    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(np.ceil(h * sin + w * cos))
    nh = int(np.ceil(h * cos + w * sin))
    M[0, 2] += (nw - 1) / 2.0 - (w - 1) / 2.0
    M[1, 2] += (nh - 1) / 2.0 - (h - 1) / 2.0

    q = np.asarray(reference_detection.quad, float)
    expect = (M[:, :2] @ q.T).T + M[:, 2]
    err = np.linalg.norm(expect - np.asarray(d2.quad, float), axis=1)
    assert err.max() < 0.02 * reference_detection.outer_side_px


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [0.5, 0.75, 1.25, 1.5])
def test_detector_survives_rescaling(reference_image, reference_detection, scale):
    small = cv2.resize(reference_image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    d = detect.detect_port(small)
    assert d.marker_found
    # Marker position relative to the port must be scale invariant.
    ref_rel = reference_detection.marker_vector / reference_detection.outer_side_px
    new_rel = d.marker_vector / d.outer_side_px
    assert np.linalg.norm(ref_rel - new_rel) < 0.02


@pytest.mark.parametrize("sigma", [2.0, 6.0, 12.0])
def test_detector_survives_gaussian_noise(reference_image, reference_detection, sigma):
    rng = np.random.default_rng(C.RANDOM_SEED)
    noisy = np.clip(reference_image.astype(np.float32)
                    + rng.normal(0, sigma, reference_image.shape), 0, 255).astype(np.uint8)
    d = detect.detect_port(noisy)
    assert d.marker_found
    off = np.linalg.norm(d.marker_center - reference_detection.marker_center)
    assert off < 0.10 * reference_detection.marker_radius_px


def test_detector_survives_jpeg(reference_image, reference_detection):
    ok, buf = cv2.imencode(".jpg", reference_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    assert ok
    d = detect.detect_port(cv2.imdecode(buf, cv2.IMREAD_COLOR))
    assert d.marker_found
    off = np.linalg.norm(d.marker_center - reference_detection.marker_center)
    assert off < 0.10 * reference_detection.marker_radius_px


def test_detector_survives_blur(reference_image, reference_detection):
    blurred = cv2.GaussianBlur(reference_image, (0, 0), 2.0)
    d = detect.detect_port(blurred)
    assert d.marker_found
    off = np.linalg.norm(d.marker_center - reference_detection.marker_center)
    assert off < 0.10 * reference_detection.marker_radius_px


# ---------------------------------------------------------------------------
# Low-level building blocks
# ---------------------------------------------------------------------------

def test_order_quad_clockwise_is_permutation_invariant():
    quad = np.array([[10.0, 10.0], [90.0, 12.0], [92.0, 88.0], [8.0, 86.0]])
    base = detect.order_quad_clockwise(quad)
    for k in range(4):
        rolled = detect.order_quad_clockwise(np.roll(quad, k, axis=0))
        assert np.allclose(base, rolled)


def test_dark_mask_isolates_the_ink_not_the_grey_fill(reference_image, reference_detection):
    from aulecv.io import to_gray
    gray = to_gray(reference_image)
    region = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(region, np.round(reference_detection.quad).astype(np.int32), 255)
    mask, thr = detect.dark_mask(gray, region)
    # The ink is a thin skeleton plus one disk: a small fraction of the port.
    frac = float((mask > 0).sum()) / float((region > 0).sum())
    assert 0.02 < frac < 0.25
    assert thr < 128, "the mid-grey fill must not be swallowed into the ink mask"


def test_fold_handles_float_wraparound():
    assert detect._fold(-1e-15, 180.0) == 0.0
    assert detect._fold(180.0, 180.0) == 0.0
    assert detect._fold(90.0, 180.0) == pytest.approx(90.0)
