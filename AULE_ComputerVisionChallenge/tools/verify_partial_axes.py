"""Validate partial-axis coverage and compare with the previous all-or-nothing method.

Run: python tools/verify_partial_axes.py
The control interfaces expose only images and commands. Truth is scored afterwards.
"""
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aulecv import closed_loop as cl, detect, io, navigation as nav
from verify_followup import PanInterface


def main():
    started = time.perf_counter()
    ref = io.load_reference()
    det = detect.detect_port(ref)
    ppc = nav.pixels_per_cm(det)
    rng = np.random.default_rng(170320)
    out = ROOT / "outputs" / "verification"
    out.mkdir(parents=True, exist_ok=True)
    coverage, runs = [], []
    for view in ((200, 170), (120, 100)):
        new = cl.Localizer(ref, view)
        old = cl.Localizer(ref, view, partial_axes=False)
        maps = {"partial axes": cl.LocalizabilityMap(ref, new, det, stride=20),
                "all or nothing": cl.LocalizabilityMap(ref, old, det, stride=20)}
        mp = maps["partial axes"]
        starts = {0: [], 1: []}
        for j, y in enumerate(mp.ys):
            for i, x in enumerate(mp.xs):
                truth = np.array([x + view[0] / 2, y + view[1] / 2])
                fix = new.locate(nav.extract_crop(ref, int(x), int(y), *view))
                if fix.observed_axis is None:
                    continue
                axis = fix.observed_axis
                inside = bool(np.all(truth >= fix.axis_bounds[:, 0])
                              and np.all(truth <= fix.axis_bounds[:, 1]))
                coverage.append(dict(view=str(view), x=x, y=y, status=fix.status,
                                     known_axis_error_px=abs(fix.coordinate(axis) - truth[axis]),
                                     bounds_contain_truth=inside))
                starts[axis].append(truth)
        for k in range(12):
            axis = k % 2
            start = starts[axis][int(rng.integers(len(starts[axis])))]
            err = cl.MotionError(gain=tuple(rng.uniform(0.6, 1.4, 2)),
                                 bias_px=tuple(rng.uniform(-2, 2, 2)), noise_px=1.5, seed=600 + k)
            for method, lc in (("all or nothing", old), ("partial axes", new)):
                c = cl.PanCamera(ref, start, view, ppc, err)
                r = cl.navigate_without_zoom(PanInterface(c), lc, maps[method], det)
                visible = c.true_marker_visible(det)
                assert len(r.log) == len(c.observations)
                partial_errors = []
                for entry, truth in zip(r.log, c.observations):
                    if entry["status"] in ("x_only", "y_only"):
                        observed = "x" if entry["status"] == "x_only" else "y"
                        partial_errors.append(abs(entry["est_" + observed] - truth["true_" + observed]))
                        for key in ("x", "y"):
                            lo, hi = entry[key + "_range"]
                            assert lo <= truth["true_" + key] <= hi, "Partial range excluded the actual position"
                runs.append(dict(view=str(view), run=k, start_axis=axis, method=method, status=r.status,
                                 reached=visible, false_arrival=r.status == "visible" and not visible,
                                 moves=r.moves, blind_views=r.lost_views, partial_views=r.partial_views,
                                 worst_observed_axis_error_px=max(partial_errors, default=0),
                                 final_error_px=r.final_error_px(c)))
        print("finished", view, "partial-map fraction", round(mp.fraction_partial, 3), flush=True)
    cov, df = pd.DataFrame(coverage), pd.DataFrame(runs)
    cov.to_csv(out / "partial_axis_coverage.csv", index=False)
    df.to_csv(out / "partial_axis_ablation.csv", index=False)
    summary = df.groupby(["view", "method"]).agg(runs=("reached", "size"), reached=("reached", "sum"),
        false_arrivals=("false_arrival", "sum"), median_moves=("moves", "median"),
        blind_views=("blind_views", "sum"), partial_views=("partial_views", "sum"),
        worst_observed_axis_error_px=("worst_observed_axis_error_px", "max"))
    summary.to_csv(out / "partial_axis_summary.csv")
    print(summary.round(3).to_string(), flush=True)
    print("Grid partial fixes:", len(cov), "bounds violations:", int((~cov.bounds_contain_truth).sum()),
          "worst observed-axis error:", cov.known_axis_error_px.max(), flush=True)
    assert cov.bounds_contain_truth.all() and not df.false_arrival.any()
    assert df.loc[df.method == "partial axes", "status"].eq("visible").all()
    print("Elapsed", round(time.perf_counter() - started, 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
