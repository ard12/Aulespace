"""Part B tests: localisation, honest refusal, and the movement convention."""

from __future__ import annotations

import numpy as np
import pytest

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

def test_exploration_is_optional_and_separate(reference_image, reference_detection):
    """The spiral search exists, works, and is *not* on the core path."""
    out = navigation.explore_until_localized(reference_image, reference_detection,
                                             (200, 20, 120, 16))
    assert out["status"] in ("localized", "exhausted")
    if out["status"] == "localized":
        assert out["plan"].status in ("ok", "already_visible")
    assert "explore" in navigation.explore_until_localized.__name__
    assert "OPTIONAL" in navigation.explore_until_localized.__doc__
