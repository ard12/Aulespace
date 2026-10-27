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
    assert any(e.get("zoom_unavailable") for e in r.log)
    assert len(r.log) == len(c.observations)
    assert [e["capture"] for e in r.log] == list(range(len(c.observations)))


@pytest.mark.parametrize("vertical", [False, True])
def test_partial_axis_uses_the_entire_score_ridge(vertical):
    ref = np.full((120, 200), 220, np.uint8)
    ref[45:49, :] = 20
    crop = ref[30:65, 55:95].copy()
    if vertical:
        ref, crop = ref.T.copy(), crop.T.copy()
    fix = cl.Localizer(ref, (crop.shape[1], crop.shape[0])).locate(crop)
    axis = 0 if vertical else 1
    assert fix.status == ("x_only" if vertical else "y_only")
    assert fix.center is None and fix.coordinate(1 - axis) is None
    assert abs(fix.coordinate(axis) - 47.5) <= 2
    assert fix.axis_bounds[axis, 0] <= 47.5 <= fix.axis_bounds[axis, 1]
    assert np.diff(fix.axis_bounds[1 - axis])[0] > 100


def test_two_identical_lines_do_not_give_a_false_y_fix():
    ref = np.full((140, 200), 220, np.uint8)
    ref[35:39, :] = 20
    ref[95:99, :] = 20
    crop = ref[25:50, 55:95].copy()
    fix = cl.Localizer(ref, (40, 25)).locate(crop)
    assert fix.status == "ambiguous" and fix.observed_axis is None
    assert fix.coordinate(0) is None and fix.coordinate(1) is None
    assert fix.axis_bounds[1, 1] - fix.axis_bounds[1, 0] >= 60


def test_blank_view_is_not_promoted_to_partial():
    ref = np.full((100, 150), 200, np.uint8)
    fix = cl.Localizer(ref, (40, 30)).locate(ref[20:50, 40:80])
    assert fix.status == "featureless" and fix.observed_axis is None


def test_weak_match_is_not_promoted_to_partial(reference, monkeypatch):
    weak = nav.CropLocalization("ambiguous", x=188, y=380, score=0.8,
                               candidate_bounds_px=(100, 380, 300, 381))
    monkeypatch.setattr(cl.nav, "localize_crop", lambda *args: weak)
    fix = cl.Localizer(reference, VIEW).locate(reference[380:550, 188:388])
    assert fix.status == "ambiguous" and fix.observed_axis is None


def test_candidate_bounds_handle_explicit_negative_score_threshold():
    crop = (np.arange(400).reshape(20, 20) % 256).astype(np.uint8)
    result = nav.localize_crop(crop, 255 - crop, min_score=-1.0)
    assert result.candidate_bounds_px == (0, 0, 0, 0)


def test_partial_bounds_are_in_reference_units_at_zoom():
    ref = np.full((200, 300, 3), 220, np.uint8)
    ref[80:84, :, :] = 20
    c = cl.PanCamera(ref, (150, 90), (60, 40), (1, 1), pad=80, max_zoom=2)
    c.set_zoom(2)
    fix = cl.Localizer(ref, (60, 40), pad=80).locate(c.capture(), zoom=2)
    assert fix.status == "y_only" and fix.center is None
    assert abs(fix.coordinate(1) - 90) <= 2
    assert np.all(c.true_center >= fix.axis_bounds[:, 0])
    assert np.all(c.true_center <= fix.axis_bounds[:, 1])


@pytest.mark.parametrize("zoom", [1.5, 2.0, 3.0])
def test_partial_bounds_cover_fractional_zoom_and_odd_dimensions(zoom):
    ref = np.full((199, 301, 3), 220, np.uint8)
    ref[90:94, :, :] = 20
    rng = np.random.default_rng(16)
    lc = cl.Localizer(ref, (61, 41), pad=128)
    partial_count = 0
    for _ in range(16):
        centre = (rng.uniform(100, 200), rng.uniform(85, 110))
        c = cl.PanCamera(ref, centre, (61, 41), (1, 1), pad=128, max_zoom=3)
        c.set_zoom(zoom)
        fix = lc.locate(c.capture(), zoom)
        if fix.observed_axis is not None:
            partial_count += 1
            assert np.all(c.true_center >= fix.axis_bounds[:, 0])
            assert np.all(c.true_center <= fix.axis_bounds[:, 1])
    assert partial_count > 0


def test_partial_mode_can_be_disabled_for_an_ablation(reference):
    crop = nav.extract_crop(reference, 188, 380, *VIEW)
    old = cl.Localizer(reference, VIEW, partial_axes=False).locate(crop)
    new = cl.Localizer(reference, VIEW).locate(crop)
    assert old.status == "ambiguous" and new.status == "y_only"
    assert new.center is None and abs(new.coordinate(1) - 465) <= 2
    assert new.axis_bounds[0, 0] <= 288 <= new.axis_bounds[0, 1]


def test_partial_command_corrects_only_the_measured_axis(det):
    fix = cl.ViewFix("y_only", None, 1, 1, float("inf"),
                     axis_bounds=np.array([[190., 390.], [463., 467.]]))
    vec, anchor = cl._partial_command(fix, None, det, 10)
    assert anchor == (1, 465) and vec[1] == 0
    shifted = cl.ViewFix("y_only", None, 1, 1, float("inf"),
                         axis_bounds=np.array([[190., 390.], [468., 472.]]))
    correction, same_anchor = cl._partial_command(shifted, anchor, det, 10)
    assert correction[1] < 0 and same_anchor == anchor
    assert np.linalg.norm(correction) <= 10 + 1e-10
    assert shifted.center is None  # no midpoint/command-integrated x position


@pytest.mark.parametrize("controller", ["zoom_disabled", "map"])
def test_partial_crossing_reacquires_without_truth_access(reference, det, ppc, controller):
    lc = cl.Localizer(reference, VIEW)
    c = cl.PanCamera(reference, (288, 465), VIEW, ppc,
                     cl.MotionError(gain=(0.7, 1.2), bias_px=(0.5, 2), noise_px=0.5, seed=93))
    if controller == "map":
        mp = cl.LocalizabilityMap(reference, lc, det, stride=20)
        assert mp.fraction_partial > 0 and mp.y_only.any() and mp.x_only.any()
        r = cl.navigate_without_zoom(ImageOnlyPan(c), lc, mp, det)
    else:
        r = cl.navigate_with_zoom(ImageOnlyPan(c), lc, det, reference.shape, zoom_levels=(1,))
    assert r.status == "visible" and c.true_marker_visible(det)
    assert r.partial_views > 0
    assert len(r.log) == len(c.observations)
    partial = [e for e in r.log if e["status"] == "y_only"]
    assert partial and all(e["est_x"] is None for e in partial)
    for e, truth in zip(r.log, c.observations):
        if e["status"] == "y_only":
            assert e["x_range"][0] <= truth["true_x"] <= e["x_range"][1]
            assert abs(e["est_y"] - truth["true_y"]) <= 3
    assert any(e["command_cm"][1] < 0 for e in partial if e["command_cm"])
    prefix_y = []
    for e, truth in zip(r.log, c.observations):
        if e["status"] != "y_only":
            break
        prefix_y.append(truth["true_y"])
    assert len(prefix_y) >= 3
    assert max(abs(y - prefix_y[0]) for y in prefix_y) < 5


def test_route_prefers_partial_cells_to_fully_blind_cells():
    mp = object.__new__(cl.LocalizabilityMap)
    mp.ok = np.zeros((3, 5), bool)
    mp.ok[1, [0, 4]] = True
    mp.x_only = np.zeros_like(mp.ok)
    mp.y_only = np.zeros_like(mp.ok)
    mp.y_only[0, 1:4] = True
    mp.blocked = np.zeros_like(mp.ok)
    mp.clearance = np.ones(mp.ok.shape)
    path = mp.route((1, 0), (1, 4))
    assert all(mp.partial[cell] for cell in path[1:-1])


def test_partial_fix_cannot_claim_arrival_even_at_marker(reference, det):
    class Stuck:
        max_zoom = 1
        def capture(self): return None
        def move_cm(self, x, y): pass
        def set_zoom(self, z): pass
    class Partial:
        w, h = VIEW
        def locate(self, view, zoom):
            x, y = det.marker_center
            return cl.ViewFix("y_only", None, 1, 1, float("inf"),
                              axis_bounds=np.array([[x - 100, x + 100], [y, y]]))
    r = cl.navigate_with_zoom(Stuck(), Partial(), det, reference.shape, max_moves=3)
    assert r.status == "exhausted" and r.moves == 3 and r.estimate is None
