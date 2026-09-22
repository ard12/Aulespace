# AULE Computer Vision Challenge

Parts A–D of the satellite-port challenge: rotation estimation, crop localisation
and camera navigation, a 22.5° perspective render, and an incremental return to
the front view.

**Every number in this README was produced by running the code in this
repository.** Where the plan set a *target* before anything was run, the target
is labelled as such and the measured value is printed next to it.

---

## Quick start

### Locally

```bash
pip install -r requirements.txt
python -m pytest tests -q                 # full regression suite
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

`conftest.py` at the repository root puts `src/` on `sys.path`, so there is no
install step. Notebooks locate the project root by walking up from the working
directory, so they run from anywhere inside the tree.

### In Colab

Upload the folder to Drive, open any notebook and run the first cell. It mounts
Drive, finds `PROJECT_ROOT`, and puts `src/` on the path. `numpy`, `opencv`,
`scipy`, `matplotlib`, `pandas` and `imageio` are already present in Colab, so
nothing normally needs installing. On a headless machine use
`opencv-python-headless` — the code never calls a GUI function.

Read the notebooks in order; `00` builds the detector that the other four use.

The committed reference image is sufficient to run everything. To repeat PDF
extraction, place `Aule_ComputerVisionChallenge.pdf` in the project root or
`assets/`, or set `AULECV_ASSIGNMENT_PDF` to its path. No personal download path
is required. In Colab, the folder must be at
`/content/drive/MyDrive/AULE_ComputerVisionChallenge` with `src/` and `assets/`
intact; opening a notebook alone does not upload those dependencies.

---

## Layout

```
README.md  requirements.txt  conftest.py
assets/reference.png                 604x558, extracted byte-identically from the PDF
src/aulecv/
    config.py      every threshold, all relative; no pixel coordinates, no answer angles
    io.py          paths, image I/O, lossless PDF extraction
    detect.py      outer quad (sub-pixel), port centre, circular marker, confidence
    rotation.py    Part A
    navigation.py  Part B
    camera.py      Parts C and D: poses, homographies, rendering, PnP
    viz.py         figures only (computes no geometry)
notebooks/         00 detection, 01 Part A, 02 Part B, 03 Part C, 04 Part D  (executed)
tests/             test_detect, test_rotation, test_navigation, test_camera
outputs/           figures, CSVs, side_view_22_5.png, camera_return.gif, ...
```

---

## The finding that drives everything: the drawing is schematic

Notebook 00 measures the three nested regions from a scanline through the port
centre rather than trusting the caption:

| region | measured (px) | aspect w/h | side ÷ outer, measured | side ÷ outer, from the text |
|---|---|---|---|---|
| outer  | 549 × 517 | 1.062 | 1.000 | 1.000 (40 cm) |
| middle | 385 × 322 | 1.196 | 0.701 | 0.875 (35 cm) |
| inner  | 148 × 164 | 0.902 | 0.270 | 0.750 (30 cm) |

A metric rendering would show 1.000 in the aspect column and matching ratios in
the last two. It does not. **The picture is an illustration of the port, not a
calibrated photograph of it.**

This is not a detail — it decides the Part C model (below), and it is why the
"outer square is 6% wider than tall" orientation cue is reported as an
experiment and never used by the estimator.

Is that 6% a non-square-pixel sensor? No: a sensor with `fx/fy = 1.062` would
stretch the circular marker by the same factor. The marker measures **0.964**
(bounding box) / **0.967** (fitted ellipse). It is round. The rectangularity is
drawing inaccuracy.

---

## Part A — rotation from an arbitrary angle

**Approach.** A square is invariant under quarter turns, so the square alone only
gives the angle mod 90°. The circular marker — which the assignment calls the
*reference angle* — breaks that ambiguity. So: coarse, unambiguous estimate from
the centre→marker vector over the full 360°, then snap to the nearest
90°-equivalent of the much more precise estimate from the quad's side directions
(hundreds of sub-pixel edge samples per side). If the two disagree by more than
25° the refinement is refused and the marker-only answer is returned with a note.

**Sign.** `getRotationMatrix2D(c, a)` sends an image-space polar angle `θ` to
`θ − a`, so `a = θ_ref − θ_rotated`. Notebook 01 and `test_rotation.py` verify
this against OpenCV directly rather than trusting the derivation.

**Measured accuracy** (target set beforehand: MAE < 0.3°, max < 1°):

| experiment | n | fused MAE | fused max | marker-only MAE |
|---|---|---|---|---|
| 1° sweep, 0–359° | 360 | **0.00103°** | **0.01044°** | 0.01565° |
| seeded random angles | 200 | **0.00108°** | **0.01585°** | 0.01557° |

Over the full robustness grid (3 interpolation modes × noise σ ∈ {4, 10, 16},
blur σ ∈ {1.5, 3}, JPEG q ∈ {70, 35}, scale ∈ {0.5 … 1.5}): **worst MAE 0.0231°,
worst single error 0.0458°**, with nothing undetected.

**Honest failure.** Paint the marker out and the estimator returns
`status='marker_missing'` and `nan` — it does not fall back to the mod-90 cue and
present a 1-in-4 guess.

---

## Part B — localise a crop, then drive to the circle

**One movement convention, used everywhere:** a command is a **translation of the
camera parallel to the port plane**, `RIGHT = +x_img`, `DOWN = +y_img`.

Translating the camera centre by `delta_C = (tx, ty, 0)` at depth
`d` induces `H = K(I - delta_C nᵀ/d)K⁻¹`. The relative extrinsic translation
is `t_rel = -delta_C`, consistent with the PLUS form in Part C. Notebook 02
evaluates this numerically and
finds to be an **exact pure image translation** (linear part − I = 0.0, bottom row
− [0 0 1] = 0.0). So "the field of view slides over the reference image" is
physically correct for this motion. A pan/tilt would give `H = K R K⁻¹`, a real
projective warp — so **no pan/tilt angles are emitted**.

**Localisation** is `matchTemplate` (`TM_CCOEFF_NORMED`), but the part that
matters is deciding whether the argmax means anything. Three tests, all scaled to
the crop's own size: crop texture (`featureless`); the extent of the score
plateau near the peak (the aperture problem); and a competing second peak after
NMS. A further check rejects indistinguishable scores at distinct positions
before suppression, including nearby alternatives that NMS would hide.
**A non-`ok` status produces no deterministic movement commands at all.**
A separate optional feedback search can request a relative move, receive the
next camera frame, and try localisation again; it is never given the crop's
hidden ground-truth position.

**Measured over 300 seeded random crops:**

| | |
|---|---|
| `ok` | 201 (67.0%) |
| `ambiguous` | 91 (30.3%) |
| `featureless` | 8 (2.7%) |
| localisation error among `ok` | max **0 px**, mean **0 px**, exactly 0 px in **100%** |
| within 1 px (target: within 1 px when `ok`) | **100%** |
| plans that reached the goal | **201 / 201** |
| steps issued = optimal ⌈d/step⌉ | **true for every plan** |
| crops with status ≠ `ok` that produced commands | **0** |

Localisation stays exact up to additive noise of **σ = 30** grey levels.

The step-by-step closed-loop exploration is a **clearly labelled optional
section** (notebook 02, section 7), not part of the core answer. Its interface is
`initial_crop + move_and_capture(dx, dy)`: the simulator owns the true window,
while the search algorithm sees only returned pixels and relative commands. It
is bounded and reports `exhausted` rather than pretending that exploration is a
guarantee.

`navigate_until_visible(initial_crop, reference, detection, move_and_capture)`
executes the search and movement plan using fresh camera frames. It replans
after each move and searches through ambiguous intermediate views until the
marker is visible in a confidently localised crop. Search and navigation share
one budget; exhaustion and stalls are explicit outcomes. Notebook 02 verifies
arrival from an ambiguous start both with accurate movement and with only 60%
of the commanded motion. This remains a simulation under the same-scale crop
assumption; it is not a guarantee for arbitrary hardware or unobservable views.

Movement planning also fails closed: a step size must be finite and positive,
and a route requiring more than the 400-step safety cap returns
`status='unreachable'` with **no partial command list**.
An explicit `goal='full_disk'` is never reduced to a centre-only goal; a crop
too small to contain it is unreachable. Only `goal='auto'` permits that fallback.

---

## Part C — the 22.5° view

### The model, and the one that was rejected

The tempting shortcut is to back-project the raster's detected corners at 100 cm
with `fx = fy`. It buys `H(0) = I`, which *looks* like a strong check. Notebook 03
computes what object it implies: **41.22 cm × 38.82 cm**. It contradicts the given
40 × 40 cm geometry by +1.22 / −1.18 cm. Rejected.

**Model A (primary, `model='metric'`).** The port *is* the given 40 × 40 cm square.
`G` maps raster pixels to plane centimetres by sending the four detected outer
corners to (±20, ±20) — max error **8.2e-07 cm**. The output camera has genuinely
square pixels, `f = 2.5·√(w·h) = 1334.401 px`. Then
`H_A(θ) = K [r₁ r₂ t](θ) G` — one warp of the original raster per frame.
At θ = 0 this renders the **rectified** square port.

**Model B (secondary, `model='image_faithful'`).** Takes the raster *as* the θ = 0
camera image and warps it with the plane-induced homography using `fx = 2.5w`,
`fy = 2.5h`. Then **‖H_B(0) − I‖ = 1.11e-16**, and the θ = 0 frame *as rendered and
shipped* contains the input raster byte for byte (max pixel difference **0**).
That second half is not automatic: the frames also carry a canvas translation,
and a fractional one would put every Model B frame through the interpolator and
leave the θ = 0 output merely *resembling* the raster. The Model B canvas offset
is therefore rounded to whole pixels, and a test asserts the property on the
delivered render path rather than on the homography alone. The price of the
model is that the implied object is not exactly square.

Both are rendered. At 22.5° their corners differ by **11.31 px after the best
similarity fit — 2.06% of the port width**. If the raster's outer region were an
exact rectangle that residual would be zero; it is not zero precisely because the
drawing is schematic. That number *is* the cost of the modelling choice, stated
openly rather than hidden.

### The plane-induced homography sign, derived

Writing the plane as `nᵀX₀ = d` with `n = (0,0,1)`, `d = +100`, and
`t_rel = t − R t₀`, the plane-induced homography is

> **H = K (R + t_rel nᵀ / d) K⁻¹** — a **PLUS** sign.

(The familiar minus sign belongs to the other convention, `nᵀX + d = 0`.)
Verified numerically against the independently composed
`K[r₁ r₂ t](θ)·(K[r₁ r₂ t](0))⁻¹`: **PLUS agrees to 1.55e-14**, MINUS is off by
**1.404**.

### The key defensible point: the missing intrinsics do not matter

We were never told `f` or the principal point. Replacing `K` by `SK` with `S` a
similarity replaces `H` by `SH`, so the output changes by a similarity only.
Measured over `f` from 400 to 5000 px and principal points up to (1000, 1000):
raw corner shifts up to **2495 px**, but the residual **after fitting a similarity
is at most 2.27e-13 px**, and the near/far edge ratio is identical to **4.4e-16**.
The unknown intrinsics fix the output's scale and framing, never its perspective
geometry.

### Validation (all 13 checks pass)

| check | measured | expected |
|---|---|---|
| ‖C‖ | 100.000000 cm | 100 |
| det R | 1.000000 | 1 |
| optical axis through the port centre | 0.0 | 0 |
| port centre → principal point | 5.55e-06 px | 0 |
| R(0) = I | 0.0 | 0 |
| analytic H (PLUS) vs composed H | 1.55e-14 | 0 |
| projected quad convex | True | True |
| near/far ratio, **closed form** | **1.165760** | — |
| near/far ratio, from the projection | 1.165760 | = closed form |
| near/far ratio, **re-detected on the rendered pixels** | **1.166214** | ≈ closed form |
| intrinsics-invariance residual | 2.27e-13 px | 0 |
| ‖H_B(0) − I‖ | 1.11e-16 | 0 |
| Model A θ=0 max/min side | 1.000000 | 1 |

The near/far ratio is checked **two independent ways** — from the closed form
`(100 + 20 sin θ)/(100 − 20 sin θ)`, and by re-running the detector on the output
image and measuring the two vertical edges. The **right (near) side is the taller
one** in every non-zero-θ frame, as the geometry requires.

Output: `outputs/side_view_22_5.png` (Model A),
`outputs/side_view_22_5_image_faithful.png` (Model B).

---

## Part D — back to the front view

**Why an arc, not a chord.** A straight interpolation of the camera position from
22.5° to 0° passes **98.078528 cm** from the port centre at its midpoint — it
breaks the stated 100 cm constraint by **1.92 cm**. So the path interpolates the
*angle* and recomputes the position on the sphere.

10 poses, 2.5° apart. Per step: arc **4.363323 cm**, chord **4.362977 cm** (they
separate only in the 4th decimal), yaw **−2.500000°** about the camera's own Y
axis — the sign read off `R_rel`, not chosen by hand. **Max |‖C‖ − 100| over the
whole sequence: 1.4e-14 cm.**

**Every frame is rendered afresh from the original raster**, never warped from its
predecessor, so nothing accumulates. Verified: rendering the last angle standalone
is bit-identical to the last frame of the sweep.

Output: `outputs/camera_return.gif`, plus the ten frames and a grid.

### PnP validation — and what it does and does not prove

`solvePnP` with `SOLVEPNP_IPPE`, using the **true physical (±20, ±20, 0) cm object
points** and corners obtained by **re-running the detector on each rendered
frame** — never corners back-computed from the camera model under test.

Targets set beforehand: θ error < 0.5°, distance error < 1 cm.

| measured over 10 frames | |
|---|---|
| distance error | mean 0.0920 cm, **max 0.1118 cm** — target met on every frame |
| θ error | mean 0.3106°, max 1.5467° — target met on **8 of 10** frames |
| reprojection RMSE | mean 0.2864 px, max 1.4118 px |

**The two misses are both near the front view, and notebook 04 §5 shows why.**
Feeding PnP the *exactly projected* corners returns the commanded angle to
**1.7e-05°** at every angle — so the pose model, corner ordering, object points
and every sign are exactly right. The residual error is **conditioning**:

* the detector's own corner accuracy, measured against the projected truth, is
  **0.319 px RMS**;
* a Monte-Carlo at exactly that noise level gives a recovered-angle σ of
  **0.132° at θ = 22.5°** rising to **2.046° at θ = 0** — a factor of 15.4 with no
  code change in between;
* every frame's actual error is within **1.53σ** of its own predicted
  uncertainty, i.e. the estimator is performing *at* the conditioning limit;
* where the target is achievable at all (predicted σ < 0.5°, which is θ ≥ 7.5°),
  it is met on **7 of 7** frames.

The reason is geometric: a square viewed head-on projects to a rectangle for any
small θ, and the only signal is the near/far height ratio, whose derivative at
θ = 0 is 0.4 per radian — just **3.73 px** of edge-height difference per degree
over a 534 px port, the same order as the corner noise itself.

**What PnP proves:** the rendered pixels are consistent with a pinhole at the
commanded pose viewing a 40 cm square, verified end to end through an independent
image-processing path.
**What it does not prove:** that `K` is the true camera (`K` is shared, so the
intrinsics are circular — §6 of notebook 03 is the answer to that); that the
40 cm / 100 cm premise is true (it was given); or that the interior content is
faithful beyond the outer quad.

### Arrival checks

* Final frame vs the rectified reference: identical, max abs diff **0**. This is
  0 *by construction* — it asserts that no drift crept in, and is labelled in the
  notebook as a consistency check, not independent evidence.
* **Pixel round trip** (a real test): warping the 22.5° *frame* back to frontal
  through two bicubic resamplings gives **PSNR 48.64 dB**, max abs diff 17.
* Model B's θ = 0 frame reproduces the raw reference byte for byte (max abs
  diff **0**), canvas translation included — see Part C for why that offset is
  rounded to whole pixels.

---

## Testing

```
188 passed
```

`tests/` covers: lossless extraction and image identity; quad convexity, sub-pixel
corners and diagonal-intersection centre; marker shape and cross-check agreement;
canonical ordering following the port through rotation; detector robustness to
rescaling, noise, JPEG and blur; the Part A sign against `getRotationMatrix2D`,
accuracy, mod-90 refinement and honest failure; Part B exactness, aperture and
featureless refusal, the no-commands-when-not-`ok` rule, the absence of any
pan/tilt field, 4-connected dominant-axis-first moves, safe step-cap handling,
feedback-only exploration, verified arrival with imperfect motion, recovery
from ambiguous intermediate frames, stall/budget handling and fresh-cut frames;
look-at properties, the PLUS sign, Model A squareness, Model B's `H_B(0) = I`,
intrinsics invariance, the closed-form predictions, arc/chord lengths, no
frame-to-frame warping, PnP on both exact and re-detected corners, and the
near-frontal conditioning claim itself. Projection tests also cover homography
scale invariance and points at infinity. Model B is checked on the *delivered*
render path, canvas offset included, not only on its homography. Export tests
reopen the GIF and check frame delays, colours, and playback order.

---

## Assumptions and limitations

* **The drawing is schematic.** Measured above. The raster is treated as a texture
  on the *given* 40 × 40 cm square; Model B keeps the raster instead, and the gap
  between the two is measured (11.31 px, 2.06% of the port width).
* **The intrinsics are assumed** — but measured to be irrelevant to the geometry
  (residual after a similarity fit ≤ 2.3e-13 px).
* **Roll is assumed zero.** "Looking directly at it" pins the optical axis, not
  the roll about it; a level camera is a choice, stated once and used everywhere.
* **Ambiguous crops cannot be localised.** A strip of one straight border or a
  patch of uniform fill does not determine a position. The pipeline says so
  (30.3% + 2.7% of random crops) instead of guessing.
* **Part B is translation-only** and same-scale: `matchTemplate` is not scale
  invariant, and pan/tilt would break the crop model.
* **Part B centimetres are approximate:** one mean pixel/cm scale per axis is
  inferred from the schematic outer edges. Unlike the projective mapping in
  Parts C/D, these constants cannot describe local scale changes in the drawing.
* **Near-frontal angles are not recoverable to 0.5° from four corners** — measured
  and explained above. The *rendering* is unaffected; it is the *inverse* problem
  that is ill-conditioned.
* **Only the outer quad is validated.** Nothing here measures the rendered
  position of the inner squares or the marker.
* **A pinhole with no lens distortion**, one port per image, and no mirroring.
* **Confidence is a heuristic**, not a calibrated probability — the minimum of
  four normalised agreement terms. Note that its `quad_opposite_sides` term is a
  *squareness* prior, so it legitimately drops on the foreshortened Part C/D
  renders; that is the near/far ratio, not a detection failure.

---

## Reproducibility

All randomness is seeded from `config.RANDOM_SEED = 20240517`. Thresholds live in
`config.py` and are all relative — fractions of the detected outer side length, of
the measured border thickness, of the contour perimeter or of the crop size.
There are no pixel coordinates and no answer angles anywhere in the estimators.
