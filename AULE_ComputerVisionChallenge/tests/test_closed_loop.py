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
    assert abs(last["est_theta_deg"] - cam_.observations[-1]["true_theta_deg"]) < 0.05
    assert cam_.true_pointing_error_deg < 0.15
    assert np.linalg.norm(cam_.true_center - [0, 0, -100]) < 0.15


class ImageOnlyPan:
    """An actual capability boundary: no access to simulator state or truth."""
    def __init__(self, sim):
        self.capture = sim.capture
        self.move_cm = sim.move_cm
        self.set_zoom = sim.set_zoom
        self.max_zoom = sim.max_zoom


class ImageOnlyOrbit:
    def __init__(self, sim):
        self.capture = sim.capture
        self.move = sim.move


@pytest.mark.parametrize("zoom", [False, True])
def test_pan_controller_only_needs_images_and_commands(reference, ppc, det, zoom):
    c = cl.PanCamera(reference, START, VIEW, ppc, bad_motion(),
                     pad=PAD if zoom else 0, max_zoom=3 if zoom else 1)
    lc = cl.Localizer(reference, VIEW, pad=PAD if zoom else 0)
    if zoom:
        r = cl.navigate_with_zoom(ImageOnlyPan(c), lc, det, reference.shape)
    else:
        mp = cl.LocalizabilityMap(reference, lc, det, stride=30)
        r = cl.navigate_without_zoom(ImageOnlyPan(c), lc, mp, det)
    assert r.status == "visible" and truly_visible(c, det)
    assert len(r.log) == len(c.observations)
    assert all(not any(key.startswith("true_") for key in row) for row in r.log)
    # Evaluation pairs a fix with truth at CAPTURE, not after the subsequent move.
    for row, truth in zip(r.log, c.observations):
        if row["status"] == "ok" and row["zoom"] == 1:
            assert np.linalg.norm([row["est_x"] - truth["true_x"],
                                   row["est_y"] - truth["true_y"]]) < 1


def test_zoom_confirmation_cannot_loop_forever(reference, det):
    class Camera:
        max_zoom = 2.0
        def capture(self):
            return None
        def set_zoom(self, z):
            pass
        def move_cm(self, x, y):
            pass  # deliberately stuck actuator
    class Locator:
        w, h = VIEW
        calls = 0
        def locate(self, image, zoom):
            self.calls += 1
            assert self.calls <= 20, "unbounded zoom-in/out loop"
            return cl.ViewFix("ok" if zoom > 1 else "ambiguous",
                              np.asarray(det.marker_center), zoom, 1.0, 1.0)
    r = cl.navigate_with_zoom(Camera(), Locator(), det, reference.shape,
                               zoom_levels=(1, 2), max_moves=2)
    assert r.status == "exhausted" and r.moves == 2


@pytest.mark.parametrize("levels", [(), (2,), (1, 1), (1, 0.5), (1, float("nan"))])
def test_zoom_schedule_is_validated(reference, det, ppc, levels):
    c = cl.PanCamera(reference, START, VIEW, ppc)
    with pytest.raises(ValueError, match="zoom_levels"):
        cl.navigate_with_zoom(c, cl.Localizer(reference, VIEW), det, reference.shape, zoom_levels=levels)


def test_zoom_does_not_teleport_camera(reference, ppc):
    c = cl.PanCamera(reference, (100, 85), VIEW, ppc, max_zoom=2)
    before = c.true_center
    with pytest.raises(ValueError, match="padding"):
        c.set_zoom(2)
    assert np.array_equal(c.true_center, before) and c.true_zoom == 1


def test_padded_visibility_does_not_invent_pixels(reference, det, ppc):
    c = cl.PanCamera(reference, (0, 0), VIEW, ppc, pad=PAD)
    x, y, w, h = c.true_window()
    assert (x, y) == (-100, -85)
    assert cl._native_window(c.true_center, w, h, reference.shape) == c.true_window()
    assert not cl._visible(c.true_center, w, h, det)


@pytest.mark.parametrize("stride", [0, -1, 2.5])
def test_map_rejects_invalid_stride(reference, det, stride):
    with pytest.raises(ValueError, match="stride"):
        cl.LocalizabilityMap(reference, cl.Localizer(reference, VIEW), det, stride=stride)


@pytest.mark.parametrize("theta", [-15.0, 0.7, 22.5])
def test_dense_pnp_handles_both_sides_and_photometric_change(reference, renderer, theta):
    import cv2
    c = cl.OrbitCamera(reference, renderer, theta, 103.0, cl.PoseError(sensor_noise=2, seed=102))
    # Include vertical displacement and roll, outside the original orbit-only study.
    rotation, _ = cv2.Rodrigues(np.radians(np.array([0.2, -0.1, 1.0])))
    c.move([0, 1.5, 0], rotation)
    view = np.clip(c.capture().astype(float) * 0.8 + 20, 0, 255).astype(np.uint8)
    e = cl.PoseEstimator(reference, renderer.K, renderer.G, dense_solver="pnp").estimate(view)
    assert e.method == "dense"
    assert np.linalg.norm(e.C - c.true_center) < 0.1


def test_orbit_controller_only_needs_images_and_commands(reference, renderer):
    c = cl.OrbitCamera(reference, renderer, -12.0, 103.0,
                       cl.PoseError(gain=0.8, sensor_noise=2, seed=88))
    e = cl.PoseEstimator(reference, renderer.K, renderer.G, dense_solver="pnp")
    r = cl.return_to_front_closed_loop(ImageOnlyOrbit(c), e)
    assert r.status == "converged"
    assert np.linalg.norm(c.true_center - [0, 0, -100]) < 0.15
    assert c.true_pointing_error_deg < 0.15
    assert len(c.observations) == len(r.log)
    assert all(not any(key.startswith("true_") for key in row) for row in r.log)


class CountingCamera:
    def __init__(self):
        self.moves = 0
    def capture(self):
        return None
    def move(self, *args):
        self.moves += 1


def fixed_estimator(center, R=None):
    if R is None:
        R = np.eye(3)
    class Estimator:
        def estimate(self, image, refine=True):
            return cl.PoseEstimator._readout(R, -R @ np.asarray(center), "dense", 1.0, 0.0, 100.0)
    return Estimator()


@pytest.mark.parametrize("mode", ["vertical", "pointing", "roll"])
def test_frontal_azimuth_and_radius_are_not_enough(mode):
    import cv2
    center, R = np.array([0., 0., -100.]), np.eye(3)
    if mode == "vertical":
        center = np.array([0., 10., -np.sqrt(10000 - 100)])
    else:
        rv = [0., 0.1, 0.] if mode == "pointing" else [0., 0., 0.1]
        R, _ = cv2.Rodrigues(np.array(rv))
    c = CountingCamera()
    r = cl.return_to_front_closed_loop(c, fixed_estimator(center, R), max_iterations=2)
    assert r.status == "not_converged" and c.moves == 1


def test_confirmation_does_not_move_a_settled_camera():
    c = CountingCamera()
    r = cl.return_to_front_closed_loop(c, fixed_estimator([0, 0, -100]))
    assert r.status == "converged" and r.iterations == 2 and c.moves == 0


def test_last_move_is_always_observed():
    c = CountingCamera()
    r = cl.return_to_front_closed_loop(c, fixed_estimator([10, 0, -100]), max_iterations=1)
    assert r.status == "not_converged" and c.moves == 0 and len(r.log) == 1


def test_detection_failure_stops_instead_of_moving():
    class FailedEstimator:
        def estimate(self, image, refine=True):
            raise ValueError("target not found")
    c = CountingCamera()
    r = cl.return_to_front_closed_loop(c, FailedEstimator())
    assert r.status == "localization_failed" and c.moves == 0


@pytest.mark.parametrize("kwargs", [{"step_deg": 0}, {"tol_deg": -1},
                                    {"radius_cm": float("nan")}, {"max_iterations": -1}])
def test_return_parameters_are_validated(kwargs):
    with pytest.raises(ValueError):
        cl.return_to_front_closed_loop(CountingCamera(), fixed_estimator([0, 0, -100]), **kwargs)


def test_zoom_out_blocked_by_scene_edge_does_not_crash(reference, det, ppc):
    """Without background padding a zoom-out near the edge is refused; the
    navigator must treat it as unavailable and keep exploring."""
    class Lost(cl.Localizer):
        def locate(self, view, zoom):
            return cl.ViewFix("ambiguous", None, zoom, 0.0, float("inf"))

    c = cl.PanCamera(reference, (100.0, 85.0), VIEW, ppc, max_zoom=3.0)
    r = cl.navigate_with_zoom(c, Lost(reference, VIEW), det, reference.shape, max_moves=5)
    assert r.status == "exhausted" and r.moves == 5
    assert any(e["event"] == "zoom unavailable" for e in r.log)
