"""Part B tests: localisation, honest refusal, and the movement convention."""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from aulecv import config as C, navigation


# ---------------------------------------------------------------------------
# Localisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x,y,w,h", [(400, 380, 140, 120),
                                     (120, 110, 150, 150),
                                     (300, 200, 200, 160),
                                     (330, 250, 160, 150)])
def test_exact_crops_are_localised_to_the_pixel(reference_image, x, y, w, h):
    crop = navigation.extract_crop(reference_image, x, y, w, h)
    loc = navigation.localize_crop(crop, reference_image)
    assert loc.status == "ok", loc.reasons
    assert abs(loc.x - x) <= 1 and abs(loc.y - y) <= 1


def test_many_random_crops_localise(reference_image):
    rng = np.random.default_rng(C.RANDOM_SEED)
    H, W = reference_image.shape[:2]
    ok = err = 0
    flagged = 0
    for _ in range(60):
        w = int(rng.integers(120, 220)); h = int(rng.integers(120, 220))
        x = int(rng.integers(0, W - w)); y = int(rng.integers(0, H - h))
        loc = navigation.localize_crop(navigation.extract_crop(reference_image, x, y, w, h),
                                       reference_image)
        if loc.status == "ok":
            ok += 1
            err = max(err, max(abs(loc.x - x), abs(loc.y - y)))
        else:
            flagged += 1
    assert ok > 0
    assert err <= 1, "every crop reported 'ok' must be exact to within 1 px"
    assert ok + flagged == 60


def test_featureless_crop_is_flagged(reference_image):
    # A patch of the uniform page margin.
    loc = navigation.localize_crop(navigation.extract_crop(reference_image, 4, 4, 18, 18),
                                   reference_image)
    assert loc.status == "featureless"


def test_aperture_crop_is_flagged_ambiguous(reference_image):
    """A thin strip along a straight border slides freely -- one axis is free."""
    loc = navigation.localize_crop(navigation.extract_crop(reference_image, 200, 20, 120, 16),
                                   reference_image)
    assert loc.status == "ambiguous"
    ex, ey = loc.plateau_extent_px
    assert max(ex, ey) > 1


def test_near_tied_positions_inside_nms_are_not_reported_as_unique(reference_image,
                                                                  reference_detection):
    # Independent random-crop audit found a 9 px error on this exact crop.
    # The 22 px plateau fitted the old crop-relative limit, and NMS hid every
    # competing candidate, so a wrong location was previously marked 'ok'.
    crop = reference_image[234:524, 249:366].copy()
    loc = navigation.localize_crop(crop, reference_image)
    assert loc.status == "ambiguous"
    assert any("indistinguishable" in reason for reason in loc.reasons)
    assert not navigation.plan_crop_movements(loc, reference_detection, reference_image.shape).commands


def test_noise_does_not_break_localisation(reference_image):
    rng = np.random.default_rng(C.RANDOM_SEED)
    x, y, w, h = 330, 250, 160, 150
    crop = navigation.extract_crop(reference_image, x, y, w, h).astype(np.float32)
    crop = np.clip(crop + rng.normal(0, 6.0, crop.shape), 0, 255).astype(np.uint8)
    loc = navigation.localize_crop(crop, reference_image)
    assert loc.status == "ok"
    assert abs(loc.x - x) <= 1 and abs(loc.y - y) <= 1


def test_oversized_crop_is_refused(reference_image):
    big = np.zeros((reference_image.shape[0] + 10, reference_image.shape[1] + 10, 3), np.uint8)
    loc = navigation.localize_crop(big, reference_image)
    assert loc.status != "ok"


# ---------------------------------------------------------------------------
# Refusal to guess  (acceptance item 9)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("win", [(4, 4, 18, 18), (200, 20, 120, 16)])
def test_non_ok_localisation_produces_no_commands(reference_image, reference_detection, win):
    loc = navigation.localize_crop(navigation.extract_crop(reference_image, *win),
                                   reference_image)
    assert loc.status != "ok"
    plan = navigation.plan_crop_movements(loc, reference_detection, reference_image.shape)
    assert plan.status == loc.status
    assert plan.commands == []
    assert plan.n_steps == 0


def test_plan_carries_no_pan_or_tilt_fields():
    """The core output is translation-only, by construction."""
    fields = set(navigation.MovementCommand.__dataclass_fields__)
    assert not any(k in fields for k in ("pan", "tilt", "yaw", "pitch", "pan_deg", "tilt_deg"))
    assert set(("direction", "dx_px", "dy_px", "distance_px", "distance_cm")) <= fields


# ---------------------------------------------------------------------------
# Planning and the movement convention
# ---------------------------------------------------------------------------

def _plan_from(reference_image, reference_detection, win):
    loc = navigation.localize_crop(navigation.extract_crop(reference_image, *win),
                                   reference_image)
    assert loc.status == "ok", loc.reasons
    return loc, navigation.plan_crop_movements(loc, reference_detection,
                                               reference_image.shape)


def test_plan_reaches_a_window_where_the_marker_is_visible(reference_image,
                                                           reference_detection):
    loc, plan = _plan_from(reference_image, reference_detection, (400, 380, 200, 170))
    assert plan.status == "ok"
    assert plan.n_steps > 0
    final = plan.commands[-1].window_after
    assert navigation.marker_visible(final, reference_detection, plan.goal)


def test_marker_already_visible_produces_an_empty_plan(reference_image, reference_detection):
    d = reference_detection
    r = int(round(d.marker_radius_px * 1.6))
    cx, cy = np.round(d.marker_center).astype(int)
    win = (max(0, cx - r), max(0, cy - r), 2 * r, 2 * r)
    loc = navigation.localize_crop(navigation.extract_crop(reference_image, *win),
                                   reference_image)
    if loc.status != "ok":
        pytest.skip("this window is not uniquely localisable; covered elsewhere")
    plan = navigation.plan_crop_movements(loc, d, reference_image.shape)
    assert plan.status == "already_visible"
    assert plan.n_steps == 0


def test_right_means_plus_x_and_down_means_plus_y(reference_image, reference_detection):
    """The single convention: RIGHT = +x_img, DOWN = +y_img, for the window."""
    _, plan = _plan_from(reference_image, reference_detection, (400, 380, 200, 170))
    for cmd in plan.commands:
        if cmd.direction == "RIGHT":
            assert cmd.dx_px > 0 and cmd.dy_px == 0
        elif cmd.direction == "LEFT":
            assert cmd.dx_px < 0 and cmd.dy_px == 0
        elif cmd.direction == "DOWN":
            assert cmd.dy_px > 0 and cmd.dx_px == 0
        elif cmd.direction == "UP":
            assert cmd.dy_px < 0 and cmd.dx_px == 0
        else:                                           # pragma: no cover
            pytest.fail("unknown direction {}".format(cmd.direction))


def test_moves_are_4_connected_and_dominant_axis_first(reference_image, reference_detection):
    _, plan = _plan_from(reference_image, reference_detection, (400, 380, 200, 170))
    for cmd in plan.commands:
        assert (cmd.dx_px == 0) != (cmd.dy_px == 0), "4-connected: one axis per step"
    axes = [("x" if c.dx_px != 0 else "y") for c in plan.commands]
    # at most one switch between axes
    assert sum(1 for a, b in zip(axes, axes[1:]) if a != b) <= 1
    dx, dy = plan.total_px
    if abs(dx) > abs(dy):
        assert axes[0] == "x"
    elif abs(dy) > abs(dx):
        assert axes[0] == "y"


def test_step_size_is_relative_to_the_crop(reference_image, reference_detection):
    for win in [(400, 380, 200, 170), (350, 330, 120, 200)]:
        _, plan = _plan_from(reference_image, reference_detection, win)
        expected = max(1.0, C.NAV_STEP_FRAC * min(win[2], win[3]))
        assert plan.step_px == pytest.approx(expected)


def test_step_count_matches_the_optimal_count(reference_image, reference_detection):
    _, plan = _plan_from(reference_image, reference_detection, (400, 380, 200, 170))
    assert plan.n_steps == plan.optimal_steps


@pytest.mark.parametrize("bad_step", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_step_size_is_rejected(reference_image, reference_detection, bad_step):
    loc = navigation.localize_crop(
        navigation.extract_crop(reference_image, 400, 380, 140, 120),
        reference_image)
    assert loc.status == "ok"
    with pytest.raises(ValueError, match="finite positive"):
        navigation.plan_crop_movements(
            loc, reference_detection, reference_image.shape, step_px=bad_step)


@pytest.mark.parametrize("step", [0.1, np.nextafter(0.0, 1.0)])
def test_step_cap_returns_unreachable_without_a_partial_route(reference_image,
                                                               reference_detection, step):
    loc = navigation.localize_crop(
        navigation.extract_crop(reference_image, 400, 380, 140, 120),
        reference_image)
    assert loc.status == "ok"
    plan = navigation.plan_crop_movements(
        loc, reference_detection, reference_image.shape, step_px=step)
    assert plan.status == "unreachable"
    assert plan.commands == []
    assert plan.optimal_steps > C.NAV_MAX_STEPS
    assert "no partial route" in plan.reasons[0]


def test_windows_stay_inside_the_image(reference_image, reference_detection):
    H, W = reference_image.shape[:2]
    _, plan = _plan_from(reference_image, reference_detection, (400, 380, 200, 170))
    for cmd in plan.commands:
        x, y, w, h = cmd.window_after
        assert 0 <= x <= W - w and 0 <= y <= H - h


def test_cm_conversion_uses_the_measured_raster_scale(reference_detection):
    sx, sy = navigation.pixels_per_cm(reference_detection)
    q = np.asarray(reference_detection.quad, float)
    width = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[3] - q[2]))
    assert sx == pytest.approx(width / C.OUTER_SIDE_CM)
    assert 10.0 < sx < 20.0 and 10.0 < sy < 20.0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def test_simulation_re_cuts_every_frame_from_the_reference(reference_image,
                                                           reference_detection):
    _, plan = _plan_from(reference_image, reference_detection, (400, 380, 200, 170))
    frames = navigation.simulate_crop_navigation(reference_image, plan)
    assert len(frames) == plan.n_steps + 1
    for fr in frames:
        x, y, w, h = fr["window"]
        assert np.array_equal(fr["image"],
                              reference_image[y:y + h, x:x + w]), \
            "frames must be cut fresh from the reference, never warped forward"
    last = frames[-1]["window"]
    assert navigation.marker_visible(last, reference_detection, plan.goal)


# ---------------------------------------------------------------------------
# Optional extension
# ---------------------------------------------------------------------------

def test_exploration_uses_feedback_not_the_true_start_window(reference_image,
                                                             reference_detection):
    """The search sees only pixels and a relative move-and-capture callback."""
    H, W = reference_image.shape[:2]
    start = (500, 250, 90, 90)
    state = {"x": start[0], "y": start[1], "calls": 0,
             "positions": [start[:2]]}

    def move_and_capture(dx, dy):
        state["x"] = int(np.clip(round(state["x"] + dx), 0, W - start[2]))
        state["y"] = int(np.clip(round(state["y"] + dy), 0, H - start[3]))
        state["calls"] += 1
        state["positions"].append((state["x"], state["y"]))
        return navigation.extract_crop(reference_image, state["x"], state["y"],
                                       start[2], start[3])

    initial = navigation.extract_crop(reference_image, *start)
    assert navigation.localize_crop(initial, reference_image).status != "ok"
    out = navigation.explore_until_localized(
        initial, reference_image, reference_detection, move_and_capture)
    assert out["status"] == "localized"
    assert state["calls"] == out["steps_explored"]
    assert len(out["commands"]) == out["steps_explored"]
    assert out["trail"][0] == (0, 0)
    assert out["plan"].status in ("ok", "already_visible")
    assert out["window"][:2] == (state["x"], state["y"])
    assert "explore" in navigation.explore_until_localized.__name__
    assert "OPTIONAL" in navigation.explore_until_localized.__doc__


def test_explicit_full_disk_goal_is_never_downgraded(reference_image, reference_detection):
    loc = navigation.CropLocalization(status="ok", x=400, y=380, w=30, h=30)
    plan = navigation.plan_crop_movements(loc, reference_detection, reference_image.shape,
                                         goal="full_disk")
    assert plan.status == "unreachable"
    assert plan.goal == "full_disk" and not plan.commands
    auto = navigation.plan_crop_movements(loc, reference_detection, reference_image.shape)
    assert auto.status == "ok" and auto.goal == "marker_centre"
    assert navigation.marker_visible(auto.target_window, reference_detection, auto.goal)


def test_marker_on_excluded_crop_boundary_requires_a_move(reference_image, reference_detection):
    det = replace(reference_detection, marker_center=np.array([100.0, 100.0]))
    win = (0, 0, 100, 100)
    assert not navigation.marker_visible(win, det, "marker_centre")
    target, goal = navigation.marker_visibility_target(win, det, reference_image.shape,
                                                     "marker_centre")
    assert target == (1, 1)
    assert navigation.marker_visible((*target, 100, 100), det, goal)


@pytest.mark.parametrize("win", [(10, 10, 0, 30), (10, 10, -5, 30),
                                  (10, 10, 30, -5), (-1, 10, 30, 30),
                                  (10.5, 10, 30, 30)])
def test_invalid_crop_windows_raise(reference_image, win):
    with pytest.raises(ValueError):
        navigation.extract_crop(reference_image, *win)


@pytest.mark.parametrize("bad_crop", [np.zeros((0, 10), np.uint8),
                                       np.zeros((10, 10, 0), np.uint8),
                                       np.full((10, 10), np.nan, np.float32)])
def test_malformed_crops_raise_before_matching(reference_image, bad_crop):
    with pytest.raises(ValueError):
        navigation.localize_crop(bad_crop, reference_image)


def test_bad_goal_and_step_are_rejected_even_if_already_visible(reference_image,
                                                               reference_detection):
    loc = navigation.localize_crop(reference_image, reference_image)
    with pytest.raises(ValueError, match="goal"):
        navigation.plan_crop_movements(loc, reference_detection, reference_image.shape,
                                       goal="ful_disk")
    with pytest.raises(ValueError, match="finite positive"):
        navigation.plan_crop_movements(loc, reference_detection, reference_image.shape, step_px=0)


@pytest.mark.parametrize("step", [1.1, 3.0, 7.3, 50.0])
def test_fractional_plans_match_their_budget_and_reach_goal(reference_image,
                                                          reference_detection, step):
    loc = navigation.localize_crop(navigation.extract_crop(reference_image, 400, 380, 140, 120),
                                   reference_image)
    plan = navigation.plan_crop_movements(loc, reference_detection, reference_image.shape, step)
    assert plan.status == "ok"
    assert plan.n_steps == plan.optimal_steps <= C.NAV_MAX_STEPS
    displacement = np.sum([[cmd.dx_px, cmd.dy_px] for cmd in plan.commands], axis=0)
    assert np.allclose(displacement + [loc.x, loc.y], plan.target_window[:2], atol=1e-8)
    assert navigation.marker_visible(plan.commands[-1].window_after, reference_detection, plan.goal)


def _feedback_camera(reference, start, gain=1.0):
    """Ground truth belongs only to the test's camera, never the controller."""
    x, y, w, h = start
    state = {"x": x, "y": y, "calls": 0}
    def capture(dx, dy):
        state["x"] = int(np.clip(round(state["x"] + gain * dx), 0, reference.shape[1] - w))
        state["y"] = int(np.clip(round(state["y"] + gain * dy), 0, reference.shape[0] - h))
        state["calls"] += 1
        return navigation.extract_crop(reference, state["x"], state["y"], w, h)
    return navigation.extract_crop(reference, *start), capture, state


@pytest.mark.parametrize("gain", [1.0, 0.6])
def test_feedback_navigation_reaches_marker_despite_motion_error(reference_image,
                                                                reference_detection, gain):
    initial, capture, state = _feedback_camera(reference_image, (500, 250, 90, 90), gain)
    out = navigation.navigate_until_visible(initial, reference_image, reference_detection, capture)
    assert out["status"] == "visible", out["reason"]
    assert out["steps_taken"] == state["calls"] <= C.NAV_MAX_STEPS
    actual = (state["x"], state["y"], 90, 90)
    assert out["window"] == actual
    assert navigation.marker_visible(actual, reference_detection, out["goal"])
    assert np.array_equal(out["last_view"], navigation.extract_crop(reference_image, *actual))


def test_feedback_navigation_detects_a_stalled_camera(reference_image, reference_detection):
    initial, capture, state = _feedback_camera(reference_image, (400, 380, 140, 120), gain=0)
    out = navigation.navigate_until_visible(initial, reference_image, reference_detection, capture)
    assert out["status"] == "stalled"
    assert state["calls"] == out["steps_taken"] == 1


def test_feedback_navigation_obeys_one_total_budget(reference_image, reference_detection):
    initial, capture, state = _feedback_camera(reference_image, (500, 250, 90, 90))
    out = navigation.navigate_until_visible(initial, reference_image, reference_detection,
                                           capture, max_steps=20)
    assert out["status"] == "exhausted"
    assert state["calls"] == out["steps_taken"] == 20
    assert 0 < out["steps_explored"] < out["steps_taken"]


def test_zero_budget_never_calls_the_camera(reference_image, reference_detection):
    def forbidden(dx, dy):
        pytest.fail("zero budget must not move the camera")
    out = navigation.navigate_until_visible(reference_image, reference_image, reference_detection,
                                           forbidden, max_steps=0)
    assert out["status"] == "visible" and out["steps_taken"] == 0


def test_impossible_full_disk_goal_never_moves_the_camera(reference_image, reference_detection):
    def forbidden(dx, dy):
        pytest.fail("an impossible goal must be rejected before moving")
    initial = navigation.extract_crop(reference_image, 200, 20, 120, 16)
    out = navigation.navigate_until_visible(initial, reference_image, reference_detection,
                                           forbidden, goal="full_disk")
    assert out["status"] == "unreachable" and out["steps_taken"] == 0


def test_exploration_rejects_bad_feedback(reference_image, reference_detection):
    initial = navigation.extract_crop(reference_image, 500, 250, 90, 90)
    with pytest.raises(ValueError, match="same height and width"):
        navigation.explore_until_localized(initial, reference_image, reference_detection,
                                           lambda dx, dy: np.zeros((90, 90, 0), np.uint8))
