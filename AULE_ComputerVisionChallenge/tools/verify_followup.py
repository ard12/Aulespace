"""Independent, reproducible hold-out checks for the follow-up controllers.

Run from the project: python tools/verify_followup.py
Truth is used only here for scoring, never passed to a controller.
"""
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aulecv import camera, closed_loop as cl, detect, io, navigation


class PanInterface:
    def __init__(self, sim):
        self.capture, self.move_cm, self.set_zoom = sim.capture, sim.move_cm, sim.set_zoom
        self.max_zoom = sim.max_zoom


class OrbitInterface:
    def __init__(self, sim):
        self.capture, self.move = sim.capture, sim.move


def main():
    t0 = time.perf_counter()
    reference = io.load_reference()
    det = detect.detect_port(reference)
    ppc = navigation.pixels_per_cm(det)
    geom = camera.PortGeometry.from_detection(det, reference.shape)
    renderer = camera.make_renderer(geom, [22.5, 0], "metric")
    rng = np.random.default_rng(719031)  # independent of the demonstration seed
    out = ROOT / "outputs" / "verification"
    out.mkdir(parents=True, exist_ok=True)

    # Compare pose readouts on the SAME held-out images, including 6-DOF
    # perturbations, varying radius, sensor noise, and brightness/contrast.
    rows = []
    estimators = {s: cl.PoseEstimator(reference, renderer.K, renderer.G, dense_solver=s)
                  for s in ("homography", "pnp")}
    for k in range(18):
        theta, radius = rng.uniform(-25, 25), rng.uniform(96, 105)
        c = cl.OrbitCamera(reference, renderer, theta, radius,
                           cl.PoseError(sensor_noise=3, seed=k + 500))
        rv = np.radians(rng.uniform([-0.7, -0.7, -2], [0.7, 0.7, 2]))
        c.move([0, rng.uniform(-2, 2), 0], cv2.Rodrigues(rv)[0])
        view = np.clip(c.capture().astype(float) * rng.uniform(0.65, 1.05)
                       + rng.uniform(0, 20), 0, 255).astype(np.uint8)
        for solver, estimator in estimators.items():
            e = estimator.estimate(view)
            rows.append(dict(run=k, solver=solver, method=e.method,
                             position_error_cm=np.linalg.norm(e.C - c.true_center),
                             angle_error_deg=abs(e.theta_deg - c.true_theta_deg),
                             distance_error_cm=abs(e.distance_cm - c.true_distance_cm)))
    poses = pd.DataFrame(rows)
    poses.to_csv(out / "pose_solver_holdout.csv", index=False)
    print("POSE HOLDOUT", flush=True)
    print(poses.groupby("solver")[["position_error_cm", "angle_error_deg", "distance_error_cm"]]
          .agg(["mean", "max"]).round(5).to_string(), flush=True)

    rows = []
    for view in [(200, 170), (120, 100)]:
        native = cl.Localizer(reference, view)
        zoomed = cl.Localizer(reference, view, pad=300)
        mp = cl.LocalizabilityMap(reference, native, det, stride=20)
        for k in range(16):
            start = rng.uniform(np.array(view) / 2,
                                np.array(reference.shape[1::-1]) - np.array(view) / 2)
            err = cl.MotionError(gain=tuple(rng.uniform(0.4, 1.6, 2)),
                                 bias_px=tuple(rng.uniform(-2, 2, 2)), noise_px=2, seed=k + 700)
            for mode in ("zoom", "map_probe"):
                c = cl.PanCamera(reference, start, view, ppc, err,
                                 pad=300 if mode == "zoom" else 0,
                                 max_zoom=3 if mode == "zoom" else 1)
                interface = PanInterface(c)
                r = (cl.navigate_with_zoom(interface, zoomed, det, reference.shape)
                     if mode == "zoom" else cl.navigate_without_zoom(interface, native, mp, det))
                # Independent bounds check using the true captured native window.
                x, y, w, h = c.true_window()
                x0, y0, x1, y1 = navigation._marker_box(det)
                visible = (x <= x0 and y <= y0 and x1 < x + w and y1 < y + h)
                if w < x1 - x0 or h < y1 - y0:
                    mx, my = det.marker_center
                    visible = x <= mx < x + w and y <= my < y + h
                rows.append(dict(view=str(view), run=k, method=mode, status=r.status,
                                 visible=visible, false_success=r.status == "visible" and not visible,
                                 moves=r.moves, lost=r.lost_views, error_px=r.final_error_px(c)))
        print("B finished", view, flush=True)
    b = pd.DataFrame(rows)
    b.to_csv(out / "navigation_holdout.csv", index=False)
    print(b.groupby(["view", "method"]).agg(runs=("visible", "size"),
          reached=("visible", "sum"), false_success=("false_success", "sum"),
          median_moves=("moves", "median"), worst_error_px=("error_px", "max")).to_string(), flush=True)

    rows = []
    for k in range(8):
        theta = float(rng.choice([-1, 1]) * rng.uniform(5, 25))
        radius = float(rng.uniform(97, 104))
        error = cl.PoseError(gain=float(rng.uniform(0.65, 1.25)), noise_cm=0.05,
                             rot_noise_deg=0.05, sensor_noise=3, seed=k + 900)
        for solver, estimator in estimators.items():
            c = cl.OrbitCamera(reference, renderer, theta, radius, error)
            c.move([0, 1, 0], cv2.Rodrigues(np.radians(np.array([0.4, 0.2, 1.0])))[0])
            r = cl.return_to_front_closed_loop(OrbitInterface(c), estimator)
            pos_error = np.linalg.norm(c.true_center - [0, 0, -100])
            rows.append(dict(run=k, solver=solver, status=r.status, iterations=r.iterations,
                             position_error_cm=pos_error, pointing_error_deg=c.true_pointing_error_deg,
                             angle_error_deg=abs(c.true_theta_deg),
                             max_radius_deviation_cm=max(abs(o["true_distance_cm"] - 100)
                                                         for o in c.observations),
                             false_success=r.status == "converged" and
                             (pos_error > 0.15 or c.true_pointing_error_deg > 0.15)))
        print("D finished", k, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "return_holdout.csv", index=False)
    print(d.groupby("solver").agg(converged=("status", lambda s: (s == "converged").sum()),
          false_success=("false_success", "sum"), worst_position_cm=("position_error_cm", "max"),
          worst_pointing_deg=("pointing_error_deg", "max"), median_iterations=("iterations", "median"))
          .round(5).to_string(), flush=True)
    print(f"Elapsed: {time.perf_counter() - t0:.1f} seconds", flush=True)
    if b.false_success.any() or d.false_success.any():
        raise AssertionError("False success in held-out verification")


if __name__ == "__main__":
    main()
