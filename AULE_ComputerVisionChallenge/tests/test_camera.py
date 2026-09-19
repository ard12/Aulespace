"""Parts C and D tests: frames, homography signs, physical predictions, PnP."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from aulecv import camera, config as C, detect


# ---------------------------------------------------------------------------
# Look-at and pose  (acceptance item 3)
# ---------------------------------------------------------------------------

def test_look_at_is_identity_at_theta_zero():
    R, t = camera.look_at_rotation((0.0, 0.0, -C.CAMERA_DISTANCE_CM),
                                   (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert np.allclose(R, np.eye(3), atol=1e-12)
    assert np.allclose(t, [0.0, 0.0, C.CAMERA_DISTANCE_CM], atol=1e-12)


@pytest.mark.parametrize("theta", [-40.0, -22.5, 0.0, 5.0, 22.5, 60.0, 89.0])
def test_pose_is_a_proper_rotation_on_the_100cm_sphere(theta):
    p = camera.camera_pose_for_angle(theta)
    R, t, Cc = p["R"], p["t"], p["C"]
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.norm(Cc) == pytest.approx(C.CAMERA_DISTANCE_CM,
                                               abs=C.POSE_RADIUS_TOL_CM)
    # the optical axis passes through the origin
    z_c = R[2, :]
    assert np.allclose(np.cross(z_c, -Cc), 0.0, atol=1e-9)
    assert float(z_c @ (-Cc)) > 0, "the port must be in front of the camera"
    # t is (0, 0, 100) for every theta: always 100 cm along the optical axis
    assert np.allclose(t, [0.0, 0.0, C.CAMERA_DISTANCE_CM], atol=1e-9)


def test_positive_theta_moves_the_camera_to_the_viewers_right():
    assert camera.camera_pose_for_angle(22.5)["C"][0] > 0


def test_world_down_stays_down_zero_roll():
    """Zero roll: the world-down direction must project to the camera's +y."""
    for th in (0.0, 22.5, 45.0):
        R = camera.camera_pose_for_angle(th)["R"]
        assert np.allclose(R @ np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_look_at_refuses_a_degenerate_roll():
    with pytest.raises(ValueError):
        camera.look_at_rotation((0.0, -100.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))


# ---------------------------------------------------------------------------
# Model A geometry  (acceptance item 2)
# ---------------------------------------------------------------------------

def test_G_maps_the_detected_corners_onto_the_physical_square(geometry):
    G = camera.raster_to_plane_homography(geometry.raster_corners, geometry.side_cm)
    got = camera.project_points(G, geometry.raster_corners)
    assert np.allclose(got, geometry.plane_corners_cm, atol=1e-6)


def test_metric_model_uses_square_pixels(renderer_metric):
    K = renderer_metric.K
    assert K[0, 0] == pytest.approx(K[1, 1])
    assert K[0, 1] == 0.0 and K[1, 0] == 0.0


def test_metric_model_does_not_inherit_the_rasters_rectangularity(geometry):
    """The rejected shortcut (back-projecting the raster corners with fx = fy)
    would imply a ~41 x 39 cm port.  Model A must not do that."""
    aspect = geometry.width_px / geometry.height_px
    assert aspect > 1.03, "the source raster really is non-square (schematic drawing)"
    G = camera.raster_to_plane_homography(geometry.raster_corners, geometry.side_cm)
    plane = camera.project_points(G, geometry.raster_corners)
    w_cm = 0.5 * (np.linalg.norm(plane[1] - plane[0]) + np.linalg.norm(plane[2] - plane[3]))
    h_cm = 0.5 * (np.linalg.norm(plane[2] - plane[1]) + np.linalg.norm(plane[3] - plane[0]))
    assert w_cm == pytest.approx(C.OUTER_SIDE_CM, abs=1e-6)
    assert h_cm == pytest.approx(C.OUTER_SIDE_CM, abs=1e-6)


@pytest.mark.parametrize("theta", [0.0, 5.0, 12.5, 22.5])
def test_port_centre_projects_to_the_principal_point(renderer_metric, geometry, theta):
    got = renderer_metric.project_port_center(theta)
    assert np.allclose(got, renderer_metric.K[:2, 2], atol=1e-6)


@pytest.mark.parametrize("theta", [0.0, 7.5, 15.0, 22.5])
def test_projected_quad_stays_convex(renderer_metric, theta):
    q = np.asarray(renderer_metric.project_port_corners(theta), np.float32)
    assert cv2.isContourConvex(q.reshape(-1, 1, 2))


def test_metric_theta_zero_is_the_rectified_square(renderer_metric, geometry):
    q = renderer_metric.project_port_corners(0.0)
    lens = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    assert max(lens) / min(lens) == pytest.approx(1.0, abs=1e-6)
    expected = geometry.metric_focal_px * C.OUTER_SIDE_CM / C.CAMERA_DISTANCE_CM
    assert np.mean(lens) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Plane-induced homography and the PLUS sign  (acceptance item 4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theta", [5.0, 22.5, 40.0, -22.5])
def test_plus_sign_matches_the_independently_composed_homography(theta):
    K = camera.estimate_camera_intrinsics(1234.5, canvas=(640, 480))
    out = camera.homography_sign_check(K, theta)
    assert out["distance_plus"] < C.HOMOGRAPHY_EQUAL_TOL
    assert out["distance_minus"] > 1e-3
    assert out["plus_wins"]


@pytest.mark.parametrize("theta", [5.0, 22.5])
def test_analytic_homography_relates_the_two_metric_poses(renderer_metric, geometry, theta):
    """H(theta) must equal  plane_induced(theta) @ H(0)  for Model A too."""
    K = renderer_metric.K
    H_t = renderer_metric.model_homography(theta)
    H_0 = renderer_metric.model_homography(0.0)
    p0 = camera.camera_pose_for_angle(0.0, geometry.distance_cm)
    pt = camera.camera_pose_for_angle(theta, geometry.distance_cm)
    t_rel = pt["t"] - pt["R"] @ p0["t"]
    H_rel = camera.plane_induced_homography(K, pt["R"], t_rel, (0, 0, 1),
                                            geometry.distance_cm)
    assert camera.homography_distance(H_rel @ H_0, H_t) < C.HOMOGRAPHY_EQUAL_TOL


def test_homography_distance_is_scale_invariant():
    H = np.array([[2.0, 0.1, 3.0], [0.0, 1.7, -4.0], [1e-4, 2e-4, 1.0]])
    assert camera.homography_distance(H, 7.3 * H) < 1e-12
    assert camera.homography_distance(H, -H) < 1e-12


# ---------------------------------------------------------------------------
# Model B  (acceptance item 6)
# ---------------------------------------------------------------------------

def test_image_faithful_homography_is_identity_at_theta_zero(renderer_image_faithful):
    H = renderer_image_faithful.model_homography(0.0)
    H = H / H[2, 2]
    assert np.linalg.norm(H - np.eye(3)) < 1e-12


def test_image_faithful_uses_non_square_pixels(geometry, renderer_image_faithful):
    fx, fy = geometry.image_faithful_focal_px
    assert fx != pytest.approx(fy)
    assert renderer_image_faithful.K[0, 0] == pytest.approx(fx)
    assert renderer_image_faithful.K[1, 1] == pytest.approx(fy)
    assert renderer_image_faithful.K[0, 0] == pytest.approx(C.FOCAL_SCALE * geometry.width_px)


def test_model_a_and_b_differ_measurably_and_the_gap_is_reported(geometry, return_thetas,
                                                                 renderer_metric,
                                                                 renderer_image_faithful):
    """A and B are the same scene and pose; they differ by the output sensor.

    If the raster's outer region were an exact rectangle they would differ by a
    pure affine resampling.  It is not (the drawing is schematic), so they do
    not -- and that residual is exactly what the notebook figure reports.
    """
    qa = renderer_metric.project_port_corners(C.PART_C_ANGLE_DEG)
    qb = renderer_image_faithful.project_port_corners(C.PART_C_ANGLE_DEG)
    fit = camera.similarity_fit_residual(qa, qb)
    assert fit["max_residual_px"] > 0.0
    assert fit["max_residual_px"] < 0.10 * geometry.width_px, (
        "the two models must stay close enough to be recognisably the same view")


# ---------------------------------------------------------------------------
# Intrinsics invariance  (acceptance item 6)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("f,pp", [(700.0, (0.0, 0.0)),
                                  (2600.0, (0.0, 0.0)),
                                  (1334.0, (211.0, -97.0)),
                                  (450.0, (1000.0, 1000.0))])
def test_changing_f_or_the_principal_point_is_only_a_similarity(geometry, f, pp):
    theta = C.PART_C_ANGLE_DEG
    base_K = camera.estimate_camera_intrinsics(geometry.metric_focal_px,
                                               principal_point=(0.0, 0.0))
    alt_K = camera.estimate_camera_intrinsics(f, principal_point=pp)
    q0 = camera.project_points(
        camera.homography_for_camera_pose(theta, geometry, "metric", base_K)["H"],
        geometry.raster_corners)
    q1 = camera.project_points(
        camera.homography_for_camera_pose(theta, geometry, "metric", alt_K)["H"],
        geometry.raster_corners)
    fit = camera.similarity_fit_residual(q0, q1)
    assert fit["max_residual_px"] < 1e-6, (
        "f and the principal point may only zoom/translate the output; the "
        "perspective geometry is fixed by the pose and the 40 cm / 100 cm scene")


def test_edge_ratio_is_independent_of_f(geometry):
    ratios = []
    for f in (500.0, 1334.0, 3000.0):
        K = camera.estimate_camera_intrinsics(f, principal_point=(0.0, 0.0))
        H = camera.homography_for_camera_pose(C.PART_C_ANGLE_DEG, geometry, "metric", K)["H"]
        ratios.append(camera.measure_edge_height_ratio(
            camera.project_points(H, geometry.raster_corners))["ratio"])
    assert max(ratios) - min(ratios) < 1e-9


# ---------------------------------------------------------------------------
# Closed-form physical predictions  (acceptance item 5)
# ---------------------------------------------------------------------------

def test_predicted_depths_are_the_derived_expressions():
    d = camera.predicted_edge_depths_cm(22.5)
    s = np.sin(np.radians(22.5))
    assert d["right_near_cm"] == pytest.approx(100.0 - 20.0 * s)
    assert d["left_far_cm"] == pytest.approx(100.0 + 20.0 * s)
    assert d["right_near_cm"] < d["left_far_cm"]


def test_predicted_ratio_at_22_5_degrees():
    assert camera.predicted_edge_height_ratio(22.5) == pytest.approx(1.1657, abs=5e-4)


def test_prediction_matches_the_projection(renderer_metric, geometry):
    pred = camera.predicted_edge_height_ratio(C.PART_C_ANGLE_DEG)
    got = camera.measure_edge_height_ratio(
        renderer_metric.project_port_corners(C.PART_C_ANGLE_DEG))
    assert got["ratio"] == pytest.approx(pred, rel=1e-9)
    assert got["right_is_taller"]


def test_prediction_matches_a_fresh_detection_on_the_rendered_image(side_view_detection):
    """(b) of the two-way check: measured on the *pixels*, not on the model."""
    pred = camera.predicted_edge_height_ratio(C.PART_C_ANGLE_DEG)
    got = camera.measure_edge_height_ratio(side_view_detection.quad)
    assert got["right_is_taller"], "the near (right) side must be the taller one"
    assert got["ratio"] == pytest.approx(pred, rel=0.01)


def test_rendered_frame_is_a_perspective_view_not_a_flat_crop(side_view_detection):
    q = np.asarray(side_view_detection.quad, float)
    top = np.linalg.norm(q[1] - q[0])
    right = np.linalg.norm(q[2] - q[1])
    left = np.linalg.norm(q[0] - q[3])
    assert abs(right - left) > 0.05 * top, "no vertical foreshortening: not a perspective view"


# ---------------------------------------------------------------------------
# Part D  (acceptance item 7)
# ---------------------------------------------------------------------------

def test_return_sequence_starts_at_22_5_and_ends_exactly_at_zero(return_thetas):
    assert return_thetas[0] == pytest.approx(C.PART_C_ANGLE_DEG)
    assert return_thetas[-1] == 0.0
    assert len(return_thetas) == 10
    assert np.all(np.diff(return_thetas) < 0)


def test_every_step_keeps_the_camera_at_100_cm(return_thetas):
    for row in camera.return_sequence_table(return_thetas):
        assert row["radius_cm"] == pytest.approx(C.CAMERA_DISTANCE_CM,
                                                 abs=C.POSE_RADIUS_TOL_CM)


def test_arc_and_chord_lengths():
    ac = camera.arc_and_chord_cm(C.PART_D_STEP_DEG)
    assert ac["arc_cm"] == pytest.approx(4.363, abs=1e-3)
    assert ac["chord_cm"] == pytest.approx(4.363, abs=1e-3)
    assert ac["arc_cm"] > ac["chord_cm"]
    assert abs(ac["arc_cm"] - ac["chord_cm"]) < 1e-3


def test_a_straight_chord_would_violate_the_100_cm_constraint():
    """Why the path must be an arc: the 22.5 -> 0 chord dips to 98.08 cm."""
    ac = camera.arc_and_chord_cm(C.PART_C_ANGLE_DEG)
    assert ac["chord_midpoint_radius_cm"] == pytest.approx(98.078, abs=1e-3)
    assert ac["chord_midpoint_radius_cm"] < C.CAMERA_DISTANCE_CM - 1.0


def test_every_frame_is_rendered_from_the_original_raster(reference_image,
                                                          renderer_metric, return_thetas):
    """No frame may be warped from its predecessor: rendering the last angle
    directly must be bit-identical to rendering it inside the sweep."""
    direct = renderer_metric.render(reference_image, float(return_thetas[-1]))
    seq = [renderer_metric.render(reference_image, float(t)) for t in return_thetas]
    assert np.array_equal(seq[-1], direct)
    assert seq[0].shape == seq[-1].shape, "the canvas is fixed across the sweep"


def test_final_frame_equals_the_rectified_reference(reference_image, renderer_metric):
    """Consistency check (not an independent validation): because every frame is
    rendered afresh from the raster, the theta = 0 frame *is* the rectified
    reference, so a non-zero difference would mean drift had crept in."""
    final = renderer_metric.render(reference_image, 0.0)
    rect = camera.render_port_view(reference_image, renderer_metric.homography(0.0),
                                   renderer_metric.canvas)
    assert np.array_equal(final, rect)


def test_round_trip_through_the_22_5_view_is_faithful(reference_image, renderer_metric,
                                                      side_view):
    """A genuine pixel test: warp the rendered side view back and compare."""
    H = renderer_metric.homography(C.PART_C_ANGLE_DEG)
    H0 = renderer_metric.homography(0.0)
    back = cv2.warpPerspective(side_view, H0 @ np.linalg.inv(H),
                               renderer_metric.canvas, flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    front = renderer_metric.render(reference_image, 0.0)
    a = back.astype(np.float64)
    b = front.astype(np.float64)
    mse = float(((a - b) ** 2).mean())
    psnr = 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-12))
    assert psnr > 20.0, "round trip PSNR {:.2f} dB is too low".format(psnr)


def test_image_faithful_final_frame_reproduces_the_raw_reference(reference_image,
                                                                 renderer_image_faithful):
    H = renderer_image_faithful.model_homography(0.0)
    out = camera.render_port_view(reference_image, H / H[2, 2],
                                  (reference_image.shape[1], reference_image.shape[0]))
    assert np.array_equal(out, reference_image)


# ---------------------------------------------------------------------------
# PnP  (acceptance item 8)
# ---------------------------------------------------------------------------

def test_pnp_uses_physical_object_points(side_view_detection, renderer_metric):
    out = camera.recover_pose_pnp(side_view_detection.quad, renderer_metric.K)
    obj = out["object_points_cm"]
    h = 0.5 * C.OUTER_SIDE_CM
    assert np.allclose(np.abs(obj[:, :2]), h)
    assert np.allclose(obj[:, 2], 0.0)


def test_pnp_recovers_the_commanded_pose_from_redetected_corners(side_view_detection,
                                                                 renderer_metric):
    """Corners come from re-running the detector on the rendered pixels, never
    from projecting anything with the model under test."""
    assert side_view_detection.marker_found
    out = camera.recover_pose_pnp(side_view_detection.quad, renderer_metric.K)
    assert out["n_solutions"] >= 1
    assert abs(out["theta_deg"] - C.PART_C_ANGLE_DEG) < C.PNP_TARGET_ANGLE_ERR_DEG
    assert abs(out["distance_cm"] - C.CAMERA_DISTANCE_CM) < C.PNP_TARGET_DISTANCE_ERR_CM
    assert out["reprojection_rmse_px"] < 1.0
    assert out["C"][0] > 0, "the recovered camera must be on the viewer's right"


def test_pnp_picks_the_lower_reprojection_error_solution(side_view_detection,
                                                         renderer_metric):
    out = camera.recover_pose_pnp(side_view_detection.quad, renderer_metric.K)
    errs = [s["reprojection_rmse_px"] for s in out["all_solutions"]]
    assert out["reprojection_rmse_px"] == pytest.approx(min(errs))


@pytest.mark.parametrize("theta", [0.0, 2.5, 7.5, 15.0, 22.5, -22.5])
def test_pnp_is_exact_on_exactly_projected_corners(renderer_metric, geometry, theta):
    """Separates the model from the estimator.

    Given *exact* corners, PnP must return the commanded pose to machine
    precision at every angle -- including frontal, where the same solve on
    *detected* corners is ill-conditioned.  If this ever fails, the convention,
    the corner ordering or a sign is wrong; if only the detected-corner version
    degrades, that is conditioning, which notebook 04 section 5 quantifies.
    """
    proj = renderer_metric.project_port_corners(theta)
    out = camera.recover_pose_pnp(proj, renderer_metric.K, geometry.side_cm)
    assert abs(out["theta_deg"] - theta) < 1e-3
    assert abs(out["distance_cm"] - C.CAMERA_DISTANCE_CM) < 1e-3
    # The reprojection budget is 0.05 px on a ~534 px port (1 part in 10^4).
    # It is not machine epsilon because IPPE's two solutions coalesce as the
    # view becomes frontal, so the solver itself is numerically softer there --
    # which is the same conditioning effect the next test measures.
    assert out["reprojection_rmse_px"] < 0.05


def test_near_frontal_pnp_is_ill_conditioned(renderer_metric, geometry):
    """The conditioning claim itself is a test, so it cannot rot silently.

    Perturbing the exact corners by the same sub-pixel noise must produce a much
    wider spread of recovered angles near frontal than at 22.5 deg.
    """
    rng = np.random.default_rng(C.RANDOM_SEED)
    spreads = {}
    for theta in (22.5, 0.0):
        proj = renderer_metric.project_port_corners(theta)
        vals = [camera.recover_pose_pnp(proj + rng.normal(0, 0.3, proj.shape),
                                        renderer_metric.K, geometry.side_cm)["theta_deg"]
                for _ in range(60)]
        spreads[theta] = float(np.std(vals))
    assert spreads[0.0] > 4.0 * spreads[22.5], (
        "expected the frontal angle to be far less determined than 22.5 deg; "
        "got {:.3f} vs {:.3f} deg".format(spreads[0.0], spreads[22.5]))


@pytest.mark.parametrize("theta", [7.5, 15.0, 22.5])
def test_pnp_across_the_return_sequence(reference_image, renderer_metric, theta):
    view = renderer_metric.render(reference_image, theta)
    det = detect.detect_port(view)
    assert det.marker_found
    out = camera.recover_pose_pnp(det.quad, renderer_metric.K)
    assert abs(out["theta_deg"] - theta) < C.PNP_TARGET_ANGLE_ERR_DEG
    assert abs(out["distance_cm"] - C.CAMERA_DISTANCE_CM) < C.PNP_TARGET_DISTANCE_ERR_CM


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def test_intrinsics_helper_defaults_to_the_canvas_centre():
    K = camera.estimate_camera_intrinsics(100.0, canvas=(641, 481))
    assert K[0, 2] == pytest.approx(320.0)
    assert K[1, 2] == pytest.approx(240.0)


def test_unknown_model_is_rejected(geometry):
    with pytest.raises(ValueError):
        camera.homography_for_camera_pose(0.0, geometry, "not_a_model")


def test_similarity_fit_is_exact_for_a_similarity():
    src = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 7.0], [0.0, 7.0]])
    ang = np.radians(33.0)
    Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = (2.5 * (Rm @ src.T)).T + np.array([11.0, -4.0])
    fit = camera.similarity_fit_residual(src, dst)
    assert fit["max_residual_px"] < 1e-9
    assert fit["scale"] == pytest.approx(2.5)
