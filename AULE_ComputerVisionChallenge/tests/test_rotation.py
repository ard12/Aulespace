"""Part A tests: sign convention, accuracy, refinement and honest failure."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from aulecv import config as C, detect, rotation


# ---------------------------------------------------------------------------
# The sign, verified against OpenCV rather than asserted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle", [10.0, 45.0, 123.0, 250.0])
def test_image_polar_angles_decrease_under_positive_rotation(angle):
    """``getRotationMatrix2D(c, a)`` sends polar angle ``theta`` to ``theta - a``.

    This is the whole basis of ``a = theta_ref - theta_rotated``; verifying it
    against OpenCV directly means the estimator's sign can never silently flip
    with a library change.
    """
    centre = np.array([137.0, 61.0])
    M = cv2.getRotationMatrix2D(tuple(centre), angle, 1.0)
    for r, base in ((80.0, 0.0), (55.0, 117.0), (140.0, -40.0)):
        p = centre + r * np.array([np.cos(np.radians(base)), np.sin(np.radians(base))])
        q = M @ np.array([p[0], p[1], 1.0])
        th0 = np.degrees(np.arctan2(p[1] - centre[1], p[0] - centre[0]))
        th1 = np.degrees(np.arctan2(q[1] - centre[1], q[0] - centre[0]))
        assert rotation.wrap180(th0 - th1 - angle) == pytest.approx(0.0, abs=1e-6)


def test_rotate_image_expands_without_clipping(reference_image):
    out = rotation.rotate_image(reference_image, 45.0)
    h, w = reference_image.shape[:2]
    assert out.shape[0] >= h and out.shape[1] >= w
    d = detect.detect_port(out)
    assert d.marker_found


def test_rotate_image_fills_with_the_page_background(reference_image):
    out = rotation.rotate_image(reference_image, 45.0)
    # A corner of the expanded canvas is new area and must look like the page.
    assert out[2, 2].mean() > 200


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

REQUESTED = [0.0, 7.5, 23.0, 45.0, 90.0, 137.25, 180.0, 222.5, 270.0, 313.7, 359.0]


@pytest.mark.parametrize("angle", REQUESTED)
def test_rotation_is_recovered_on_clean_images(reference_image, reference_detection, angle):
    est = rotation.estimate_rotation(rotation.rotate_image(reference_image, angle),
                                     reference_detection)
    assert est.status == "ok"
    assert abs(rotation.angular_error(est.angle_deg, angle)) < 0.3


def test_full_range_is_covered_not_just_mod_90(reference_image, reference_detection):
    """The four quarter-turn-equivalent angles must come back distinct.

    A square-only estimator cannot tell these apart; the circular marker is
    exactly what breaks the ambiguity, and this is the test that proves the
    fused estimator uses it.
    """
    got = []
    for a in (31.0, 121.0, 211.0, 301.0):
        est = rotation.estimate_rotation(rotation.rotate_image(reference_image, a),
                                         reference_detection)
        got.append(est.angle_deg)
        assert abs(rotation.angular_error(est.angle_deg, a)) < 0.3
    spread = [rotation.wrap360(g) for g in got]
    assert len(set(round(s / 10) for s in spread)) == 4


def test_random_angles_meet_the_accuracy_target(reference_image, reference_detection):
    """Target (stated in the README as a target): MAE < 0.3 deg on clean images."""
    rng = np.random.default_rng(C.RANDOM_SEED)
    errs = []
    for a in rng.uniform(0.0, 360.0, 40):
        est = rotation.estimate_rotation(rotation.rotate_image(reference_image, float(a)),
                                         reference_detection)
        errs.append(abs(rotation.angular_error(est.angle_deg, float(a))))
    errs = np.asarray(errs)
    assert errs.mean() < 0.3
    assert errs.max() < 1.0


@pytest.mark.parametrize("interp,tol", [(cv2.INTER_NEAREST, 1.0),
                                        (cv2.INTER_LINEAR, 0.3),
                                        (cv2.INTER_CUBIC, 0.3)])
def test_interpolation_mode_robustness(reference_image, reference_detection, interp, tol):
    a = 41.0
    est = rotation.estimate_rotation(
        rotation.rotate_image(reference_image, a, interpolation=interp), reference_detection)
    assert abs(rotation.angular_error(est.angle_deg, a)) < tol


def test_noise_robustness(reference_image, reference_detection):
    rng = np.random.default_rng(C.RANDOM_SEED)
    a = 68.0
    img = rotation.rotate_image(reference_image, a)
    noisy = np.clip(img.astype(np.float32) + rng.normal(0, 8.0, img.shape), 0, 255).astype(np.uint8)
    est = rotation.estimate_rotation(noisy, reference_detection)
    assert abs(rotation.angular_error(est.angle_deg, a)) < 1.0


# ---------------------------------------------------------------------------
# Structure of the estimate
# ---------------------------------------------------------------------------

def test_edge_cue_is_only_defined_mod_90(reference_image, reference_detection):
    for a in (17.0, 107.0, 197.0, 287.0):
        d = detect.detect_port(rotation.rotate_image(reference_image, a))
        est = rotation.estimate_rotation_from_detections(d, reference_detection)
        assert 0.0 <= est.edge_angle_mod90_deg < 90.0
        # the raw mod-90 cue must agree with the answer modulo 90
        assert abs(rotation.wrap180(
            (est.angle_deg - est.edge_angle_mod90_deg) % 90.0)) < 0.5 or \
            abs(90.0 - ((est.angle_deg - est.edge_angle_mod90_deg) % 90.0)) < 0.5


def test_refinement_improves_on_the_marker_alone(reference_image, reference_detection):
    rng = np.random.default_rng(C.RANDOM_SEED + 1)
    coarse, fused = [], []
    for a in rng.uniform(0.0, 360.0, 25):
        est = rotation.estimate_rotation(rotation.rotate_image(reference_image, float(a)),
                                         reference_detection)
        coarse.append(abs(rotation.angular_error(est.marker_angle_deg, float(a))))
        fused.append(abs(rotation.angular_error(est.angle_deg, float(a))))
    assert np.mean(fused) <= np.mean(coarse)


def test_long_edge_cue_is_reported_as_an_experiment_only(reference_image, reference_detection):
    est = rotation.estimate_rotation(rotation.rotate_image(reference_image, 20.0),
                                     reference_detection)
    assert est.long_edge_experiment_deg is not None
    # It must not be what the estimator returns: the fused answer comes from the
    # marker + mod-90 edge cue, which is defined for every angle in [0, 360).
    assert est.angle_deg == pytest.approx(est.edge_angle_mod90_deg % 90.0
                                          + 90.0 * round((est.angle_deg - est.edge_angle_mod90_deg) / 90.0),
                                          abs=1e-6)


def test_missing_marker_is_reported_not_guessed(reference_image, reference_detection):
    """Paint the marker out; the estimator must refuse, not fall back to mod 90."""
    img = reference_image.copy()
    d = reference_detection
    cv2.circle(img, tuple(np.round(d.marker_center).astype(int)),
               int(round(d.marker_radius_px * 1.6)), (190, 190, 190), -1)
    est = rotation.estimate_rotation(img, d)
    assert est.status == "marker_missing"
    assert np.isnan(est.angle_deg)


def test_angular_error_wraps(reference_detection):
    assert rotation.angular_error(359.0, 1.0) == pytest.approx(-2.0)
    assert rotation.angular_error(1.0, 359.0) == pytest.approx(2.0)
    assert rotation.wrap360(-10.0) == pytest.approx(350.0)
