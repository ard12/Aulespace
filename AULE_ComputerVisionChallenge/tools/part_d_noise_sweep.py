"""Part D: how the stop tolerance relates to the camera's per-move error.

Noise makes landing inside tight tolerances less likely within a finite budget.
The controller stops on estimates; report true errors separately at both the
requested tolerance and an explicitly looser 2x evaluation envelope.
Writes outputs/verification/return_noise_sweep.csv.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aulecv import camera, closed_loop as cl, detect, io

reference = io.load_reference()
det = detect.detect_port(reference)
geom = camera.PortGeometry.from_detection(det, reference.shape)
renderer = camera.make_renderer(geom, [22.5, 0], "metric")
est = cl.PoseEstimator(reference, renderer.K, renderer.G)
rng = np.random.default_rng(4242)
rows, t0 = [], time.perf_counter()
for noise in (0.05, 0.07, 0.10):
    for k in range(6):
        theta = float(rng.choice([-1, 1]) * rng.uniform(5, 25))
        gain = float(rng.uniform(0.7, 1.2))
        for tol in (0.1, 0.25):
            err = cl.PoseError(gain=gain, noise_cm=noise, rot_noise_deg=noise, sensor_noise=2.0, seed=k + 300)
            c = cl.OrbitCamera(reference, renderer, theta, 100.0, err)
            r = cl.return_to_front_closed_loop(c, est, tol_deg=tol, tol_cm=tol)
            rows.append(dict(noise=noise, tol=tol, run=k, status=r.status, iterations=r.iterations,
                             pos_err_cm=float(np.linalg.norm(c.true_center - [0, 0, -100])),
                             pointing_deg=c.true_pointing_error_deg, theta_deg=abs(c.true_theta_deg),
                             orientation_deg=c.true_orientation_error_deg))
    print("noise", noise, "done", round(time.perf_counter() - t0), flush=True)
d = pd.DataFrame(rows)
for scale, column in ((1, "outside_true_tolerance"), (2, "outside_2x_envelope")):
    d[column] = (d.status == "converged") & ((d.pos_err_cm > scale * d.tol)
        | (d.pointing_deg > scale * d.tol) | (d.orientation_deg > scale * d.tol)
        | (d.theta_deg > scale * d.tol))
d.to_csv(ROOT / "outputs" / "verification" / "return_noise_sweep.csv", index=False)
print(d.groupby(["noise", "tol"]).agg(converged=("status", lambda s: (s == "converged").sum()),
      runs=("status", "size"), median_it=("iterations", "median"),
      worst_pos=("pos_err_cm", "max"), worst_point=("pointing_deg", "max"),
      outside_true_tolerance=("outside_true_tolerance", "sum"),
      outside_2x_envelope=("outside_2x_envelope", "sum")).round(3).to_string())
