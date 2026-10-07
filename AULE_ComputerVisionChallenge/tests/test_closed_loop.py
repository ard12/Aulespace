"""Follow-up: closed-loop Parts B and D never trust their own motion."""

import numpy as np
import pytest

from aulecv import camera, closed_loop as cl, config as C, detect, navigation as nav
from aulecv import io as aio

VIEW = (200, 170)
START = (500.0, 465.0)        # centre of the original Part B crop at (400, 380)
PAD = 300


@pytest.fixture(scope="module")
def reference():
    return aio.load_reference()


@pytest.fixture(scope="module")
def det(reference):
    return detect.detect_port(reference)


@pytest.fixture(scope="module")
def ppc(det):
    return nav.pixels_per_cm(det)


def bad_motion(seed=7):
    return cl.MotionError(gain=(0.75, 1.2), bias_px=(0.5, -0.5), noise_px=1.0, seed=seed)


def truly_visible(camera_, det):
    return nav.marker_visible(camera_.true_window(), det, "auto")


# --------------------------------------------------------------------------- simulator

def test_motion_error_is_applied_and_hidden(reference, ppc):
    c = cl.PanCamera(reference, START, VIEW, ppc, cl.MotionError(gain=(0.5, 1.0)))
    c.move_cm(-10.0 / ppc[0], 0.0)                    # command 10 px left
    assert c.true_center[0] == pytest.approx(START[0] - 5.0)   # only 5 px happened


def test_zoomed_view_has_native_size_and_is_located(reference, ppc, det):
    c = cl.PanCamera(reference, START, VIEW, ppc, pad=PAD, max_zoom=3.0)
    c.set_zoom(2.0)
    view = c.capture()
    assert view.shape[:2] == (VIEW[1], VIEW[0])
    fix = cl.Localizer(reference, VIEW, pad=PAD).locate(view, 2.0)
    assert fix.status == "ok"
    assert np.linalg.norm(fix.center - c.true_center) <= 3.0


def test_original_route_hits_views_that_cannot_be_located(reference, det):
    """The problem behind the follow-up: re-localising along the original plan."""
    loc = nav.localize_crop(nav.extract_crop(reference, 400, 380, *VIEW), reference)
    plan = nav.plan_crop_movements(loc, det, reference.shape)
    statuses = [nav.localize_crop(nav.extract_crop(reference, *c.window_after), reference).status
                for c in plan.commands]
    assert statuses.count("ambiguous") > 20


# --------------------------------------------------------------------------- Part B

def test_trusting_motion_fails_and_does_not_know_it(reference, ppc, det):
    c = cl.PanCamera(reference, START, VIEW, ppc, bad_motion())
    r = cl.navigate_trusting_motion(c, cl.Localizer(reference, VIEW), det, reference.shape)
    assert r.status == "visible"                      # it believes it arrived...
    assert not truly_visible(c, det)                  # ...but it did not


def test_with_zoom_arrives_and_knows_where_it_is(reference, ppc, det):
    c = cl.PanCamera(reference, START, VIEW, ppc, bad_motion(), pad=PAD, max_zoom=3.0)
    r = cl.navigate_with_zoom(c, cl.Localizer(reference, VIEW, pad=PAD), det, reference.shape,
                              axis_first=True)
    assert r.status == "visible" and truly_visible(c, det)
    assert r.zoom_changes >= 2                        # zoomed out when lost, back in at the end
    assert r.final_error_px(c) < 2.0
    assert c.true_zoom == 1.0


def test_without_zoom_map_route_arrives(reference, ppc, det):
    loc = cl.Localizer(reference, VIEW)
    lmap = cl.LocalizabilityMap(reference, loc, det, stride=30)
    assert 0.5 < lmap.fraction_ok < 1.0
    c = cl.PanCamera(reference, START, VIEW, ppc, bad_motion())
    r = cl.navigate_without_zoom(c, loc, lmap, det)
    assert r.status == "visible" and truly_visible(c, det)
    assert r.final_error_px(c) < 2.0


def test_unlocatable_start_is_searched_out_of(reference, ppc, det):
    start = (390 + 100.0, 160 + 85.0)                  # only straight edges: slides along them
    view = nav.extract_crop(reference, 390, 160, *VIEW)
    assert cl.Localizer(reference, VIEW).locate(view).status != "ok"
    c = cl.PanCamera(reference, start, VIEW, ppc, bad_motion(), pad=PAD, max_zoom=3.0)
    r = cl.navigate_with_zoom(c, cl.Localizer(reference, VIEW, pad=PAD), det, reference.shape)
    assert r.status == "visible" and truly_visible(c, det)


# --------------------------------------------------------------------------- Part D

@pytest.fixture(scope="module")
def renderer(reference, det):
    geom = camera.PortGeometry.from_detection(det, reference.shape)
    thetas = camera.generate_return_sequence(C.PART_C_ANGLE_DEG, C.PART_D_STEP_DEG)
    return camera.make_renderer(geom, thetas, "metric")


def test_pose_from_exact_homography_is_exact(renderer):
    pose = camera.camera_pose_for_angle(13.0)
    H = camera.plane_to_image_homography(renderer.K, pose["R"], pose["t"]) @ renderer.G
    R, t = cl.pose_from_raster_homography(3.7 * H, renderer.K, renderer.G)   # any scale
    assert np.allclose(R, pose["R"], atol=1e-9)
    assert np.allclose(t, pose["t"], atol=1e-7)


def test_dense_refinement_fixes_the_near_front_angle(reference, renderer):
    est = cl.PoseEstimator(reference, renderer.K, renderer.G)
    cam_ = cl.OrbitCamera(reference, renderer, start_theta_deg=0.7,
                          error=cl.PoseError(sensor_noise=2.0))
    e = est.estimate(cam_.capture())
    assert e.method == "dense"
    assert abs(e.theta_deg - 0.7) < 0.05                 # every pixel
    assert abs(e.corner_theta_deg - 0.7) > 0.5           # four corners are not enough here


def test_closed_loop_return_converges_to_the_front(reference, renderer):
    est = cl.PoseEstimator(reference, renderer.K, renderer.G)
    err = cl.PoseError(gain=0.85, noise_cm=0.05, rot_noise_deg=0.05, sensor_noise=2.0, seed=11)
    cam_ = cl.OrbitCamera(reference, renderer, 22.5, 100.0, err)
    res = cl.return_to_front_closed_loop(cam_, est)
    assert res.status == "converged"
    assert abs(cam_.true_theta_deg) < 0.15
    assert abs(cam_.true_distance_cm - 100.0) < 0.15
    last = res.log[-1]
    assert abs(last["est_theta_deg"] - last["true_theta_deg"]) < 0.05
