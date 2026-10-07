# AULE Computer Vision Challenge

Solutions to Parts A-D: rotation estimation, crop localisation and camera
navigation, perspective rendering at 22.5 degrees, and a return to the frontal
view on a nominal 100 cm orbit. The follow-up explicitly measures deviations
caused by unknown motion errors.

## Setup

### Google Colab

1. Make the folder visible in your Google Drive. If you received a shared link,
   open it, right-click the `AULE_ComputerVisionChallenge` folder and choose
   **Organize > Add shortcut > My Drive**. Otherwise upload the folder to My Drive.
2. Open any notebook in `notebooks/` with Google Colaboratory and choose
   **Runtime > Run all**. Notebook `00` is the natural starting point, but each
   notebook runs on its own.

The first cell mounts Drive and finds the folder whether it is in My Drive, in a
subfolder, behind a shortcut or in a shared drive. A view-only folder is copied
to `/content` so the notebook can save its outputs, and any missing package is
installed. If the folder cannot be found, the cell explains what to do; setting
`PROJECT_DIR` in that cell also works.

### Local

Run these commands from this folder:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

`requirements.txt` lists what the notebooks need; `requirements-dev.txt` adds the
test runner and notebook tooling. The notebooks and tests load the package
directly from `src/`, so no install step is needed for the code itself.

Tested configurations (all 188 original tests pass and the original five notebooks execute without
errors, with identical results):

| Python | NumPy | OpenCV | matplotlib | pandas |
|---|---|---|---|---|
| 3.10.11 | 1.23.5 | 4.6.0 | 3.7.0 | 1.5.3 |
| 3.11.9 | 2.0.2 | 4.12.0 | 3.10.0 | 2.2.2 |
| 3.14.3 | 2.4.3 | 4.13.0 | 3.10.8 | 3.0.1 |

The included reference image is sufficient to run everything. To repeat its
extraction from the PDF, install `pymupdf` (listed in `requirements-dev.txt`),
place `Aule_ComputerVisionChallenge.pdf` in the project root or `assets/`, or
set `AULECV_ASSIGNMENT_PDF` to its path.

## Files

| Location | Contents |
|---|---|
| `assets/reference.png` | Reference raster extracted from the challenge PDF |
| `notebooks/00_reference_and_detection.ipynb` | Reference measurements and port detector |
| `notebooks/01_rotation_detection.ipynb` | Part A: rotation estimation |
| `notebooks/02_crop_navigation.ipynb` | Part B: localisation, movement plans and feedback navigation |
| `notebooks/03_homography_22_5deg.ipynb` | Part C: perspective rendering and geometric validation |
| `notebooks/04_camera_return.ipynb` | Part D: camera trajectory, animation and pose validation |
| `notebooks/05_closed_loop_crop_navigation.ipynb` | Follow-up: Part B without trusting the motion (with and without zoom) |
| `notebooks/06_closed_loop_camera_return.ipynb` | Follow-up: Part D with the pose found from the image every step |
| `src/aulecv/` | Shared implementation |
| `tests/` | Regression tests |
| `tools/verify_followup.py` | Independent held-out navigation and pose experiments |
| `outputs/` | Rendered images, animations and numerical results |

## Approach

**Part A.** Detect the outer quadrilateral and circular marker. The vector from
the port centre to the marker resolves the square's 90-degree ambiguity; the
quad's edge directions refine the angle. A missing marker produces
`marker_missing`. Positive rotation follows OpenCV's counter-clockwise convention.

**Part B.** Match the crop against the reference using normalised correlation.
Texture, peak spread, competing peaks and near-ties determine whether the match
is identifiable. For accepted matches, generate camera translations parallel to
the port plane: RIGHT is positive image x, and DOWN is positive image y.
Uncertain matches produce no deterministic route.

The feedback runner accepts an initial frame and a `move_and_capture(dx, dy)`
callback. It searches, re-localises and replans using returned frames, with one
movement budget. Stalls and exhaustion are reported. `full_disk` requires the
whole marker; only `auto` may use the marker-centre criterion for smaller crops.

**Part C.** Map the detected outer quad onto the specified 40 x 40 cm plane.
Place the camera 100 cm from its centre, 22.5 degrees to the right, with its
optical axis directed at the centre. Render using the plane homography
`H = K [r1 r2 t] G`. A second model preserves the input raster at the frontal
pose for comparison. Its canvas offset is an integer translation, preserving
the reference pixels at zero degrees.

**Part D.** Decrease the viewing angle from 22.5 to 0 degrees in 2.5-degree
increments. Recompute each pose on the 100 cm orbit and render each frame from
the original raster. Validate the geometry using projected corners and PnP on
corners detected in the rendered images.

## Results

The executed notebooks contain the experiments, derivations and full result tables.

- Rotation: mean absolute error 0.00103 degrees over a 360-angle sweep.
- Crop localisation: 201 of 300 sampled crops accepted, all with zero pixel
  localisation error and successful movement plans. The other 99 were ambiguous
  or featureless.
- Feedback navigation: verified arrival with accurate movement and with only
  60% of each commanded movement.
- Return trajectory: maximum distance deviation from 100 cm was 1.4e-14 cm.
- PnP on rendered frames: maximum distance error 0.1118 cm; the 0.5-degree angle
  target was met on 8 of 10 frames. The two misses occurred near the frontal view.
- Test suite: 226 tests passed (188 original, 38 follow-up cases).

Main visual outputs are `outputs/side_view_22_5.png`,
`outputs/side_view_22_5_image_faithful.png` and `outputs/camera_return.gif`.

## Follow-up: Parts B and D without trusting the motion

As a follow-up, Parts B and D were redone so that the camera never trusts its
own moves. The true camera position is hidden inside a simulator, every move is
executed with errors the navigator is not told about, and after every move the
position is worked out again from the image. Code: `src/aulecv/closed_loop.py`;
notebooks `05` and `06`; tests `tests/test_closed_loop.py`.

**The problem.** Re-localising at every step of the original Part B route shows that
28 of its 65 views cannot be located on their own: they contain only horizontal
lines, so they fix the height but not the left-right position. Trusting the moves
hid this, and with motion errors it fails: in 40 random runs with actual motion
equal to 50-150% of the commanded distance, trusting the motion reached the marker in only 22% of runs, and in 31
runs it reported success while the marker was not in view.

**Part B, subtask A (with zoom).** After every move the view is template-matched at
the current zoom. A view that cannot be placed triggers a zoom out (1, 1.5, 2, 3x);
the camera travels at the zoom that gave a fix, then zooms back in and confirms
arrival from a native-zoom fix. If a wider view would leave the scene, that zoom
level is treated as unavailable there and the camera explores instead.

**Part B, subtask B (no zoom).** A localisability map is computed once: for every
grid position, can a view there be located? The route is the cheapest path over
that map. It prefers views far from unlocatable ones, and only crosses unlocatable
views where separate areas must be joined. After every move the view is
re-localised and the route re-planned. Across gaps it repeats bounded probes toward
an informative region, without updating position from commanded movement. If a
promised view turns out unplaceable, a reverse move is attempted and that planned
cell is penalised. Reversing is not assumed to restore the old position.

**Diagonal moves.** Each command is the straight-line vector from the estimated
position to the goal, so x and y move together. `axis_first=True` moves along the
larger offset first, as the original Part B route did; the comparison table below uses
it. On 12 runs with the same starts and hidden motion errors, both reached the marker
in 12 of 12; diagonal took a median of 15 moves, one axis at a time 16.5. The
no-zoom route also allows diagonal steps, because it searches over all eight
neighbours. Each move is re-localised either way, so a diagonal step that lands wrong
is corrected on the next one.

| 40 runs, 200 x 170 view | Reached the marker | Wrongly claimed success | Worst final error |
|---|---|---|---|
| Trust the movement | 22% | 31 | 149 px |
| Re-localise only | 100% | 0 | 0.57 px |
| A: with zoom | 100% | 0 | 0.57 px |
| B: no zoom, planned route | 100% | 0 | 0.62 px |

With a smaller 120 x 100 view (only 45% of positions locatable), every
re-localising method still reached the marker in 30 of 30 runs. Re-localising alone
needed 355 views it could not place, zoom needed 45, and the planned route needed 223
(mostly planned crossings) with 2 undos. Zoom used the fewest moves (median 22).

**Part D.** The pose is estimated from each image: four-corner PnP for a first guess,
then dense alignment of the whole image to the reference (ECC). That gives the
reference-to-image homography. A final PnP fit to a 5 x 5 grid of ECC-aligned plane
points constrains the result to a physical camera pose. These correlated points
are not independent feature detections. The controller then
aims for the orbit point up to 2.5 degrees closer to the front, at 100 cm, looking
at the centre, and sends the move needed from the estimated pose. Moves are executed
with 70-120% of the commanded translation, translation and rotation noise, and image
noise.

- The dense estimate stays within 0.011 degrees of the truth at every angle tested.
  Corner PnP is off by up to 2.6 degrees near the front.
- In 8 of 8 runs the dense loop converged (median 13.5 iterations), ending within
  0.077 degrees of the front and 0.085 cm of 100 cm. Worst full 3-D position error
  was 0.139 cm; worst pointing error was 0.099 degrees. The largest observed
  radial deviation during these runs was 0.147 cm.
- With corner PnP alone it converged in 0 of 8 runs: near the front its estimate
  is too noisy, so it keeps chasing errors that are not there.

The stop check now covers full 3-D position, orientation, pointing and azimuth,
not just distance and horizontal angle. It requires two confirming frames without
an intervening move. Detection failure stops control, and the last commanded move
is always observed. Controller logs contain estimates only; simulator truth is
joined afterwards by the evaluator using the same captured frame. Tests run both
controllers through interfaces that expose no simulator state.

**Precision limit.** The loop cannot stop closer to the front than its own moves
allow. The runs above use 0.05 cm and 0.05 degrees of random error per move per axis.
`tools/part_d_noise_sweep.py` varies that error, 6 runs per setting:

| Error per move | Converged, 0.1 tolerance | Converged, 0.25 tolerance |
|---|---|---|
| 0.05 | 6 of 6 | 6 of 6 |
| 0.07 | 6 of 6 | 6 of 6 |
| 0.10 | 2 of 6 | 6 of 6 |

With 0.10 per move and a 0.1 tolerance, a single move rarely lands inside the
tolerance, so 4 runs used all 30 iterations. They still ended near the front (worst
0.20 cm, 0.30 degrees) and reported "not converged" rather than claiming success.
None of the 36 runs claimed convergence falsely. The stop test is applied to
estimates, so the true final error can sit slightly above it: with the 0.25
tolerance the worst true position error was 0.26 cm. The tolerance should therefore be set
above the camera's real per-move precision (about 2-3 times its per-axis error).

### Independent verification

Run `python tools/verify_followup.py` (and `python tools/part_d_noise_sweep.py` for the
precision limit above). Results are saved in `outputs/verification/`.
The held-out seed differs from the demonstration seed:

- Both B methods reached the marker in all 32 additional cases each, across two
  view sizes, motion gains 0.4-1.6, bias and noise. No false arrival claims;
  worst final localisation error was 0.676 px.
- Across 18 perturbed views (both sides, different radii, vertical offset, roll,
  sensor noise and global brightness/contrast changes), ECC-to-PnP reduced worst
  pose-position error from 0.0262 cm to 0.0139 cm versus homography decomposition.
- Both pose solvers converged in all 8 additional return cases. For ECC-to-PnP,
  worst final 3-D position error was 0.122 cm and pointing error 0.094 degrees.
  The hold-out evaluator uses 0.15 cm / 0.15 degree final position/pointing bounds;
  those are evaluation bounds, not a guarantee of the controller's 0.1 thresholds.
- All seven project notebooks execute without errors on Python 3.14.3.

**Limits of the follow-up.** The camera and scene are simulated with the same flat
target model the estimators use, so dense alignment sees near-ideal images apart from
the added noise. ECC tolerates global brightness and contrast changes, but local
shadows, reflections, occlusion and lens distortion remain unvalidated. The lens
must be calibrated first. In Part
B the zoom level is assumed to be read back from the lens, and moves are translations
parallel to the target. A featureless view cannot provide a unique position;
bounded search may exhaust its budget. Exactly 100 cm at every instant cannot be
guaranteed under unknown motion errors and intermittent imaging. The simulator
tests sampled poses, not continuous orbital dynamics. Stop thresholds apply to
estimates: measured true position can be slightly outside the 0.1 cm threshold.

## Assumptions and limitations

The supplied drawing is schematic: its measured proportions differ from the
stated dimensions. The primary rendering model treats it as a texture on the
specified 40 cm square. Only the outer quad is geometrically validated.

The camera model assumes no lens distortion and zero roll. Focal length sets
the display scale; under the chosen square-pixel model, changing it or the
principal point changes the rendering by a similarity transform.

Part B assumes same-scale crops and translation parallel to the port plane.
Pixel-to-centimetre conversion uses approximate mean scales from the schematic
drawing. Featureless or repeated views may remain unlocalisable, and the bounded
feedback search is not guaranteed to succeed on every crop.

Near-frontal pose recovery from four corners is sensitive to detection noise.
Notebook 04 measures this limitation separately from the accuracy of the forward
rendering model. Confidence scores are heuristics, not calibrated probabilities.

Random experiments use `config.RANDOM_SEED = 20240517`; thresholds and camera
parameters are in `src/aulecv/config.py`.
