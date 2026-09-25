# AULE Computer Vision Challenge

Solutions to Parts A-D: rotation estimation, crop localisation and camera
navigation, perspective rendering at 22.5 degrees, and a return to the frontal
view while remaining 100 cm from the port centre.

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

Tested configurations (all 188 tests pass and all five notebooks execute without
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
| `src/aulecv/` | Shared implementation |
| `tests/` | Regression tests |
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
- Test suite: 188 tests passed.

Main visual outputs are `outputs/side_view_22_5.png`,
`outputs/side_view_22_5_image_faithful.png` and `outputs/camera_return.gif`.

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
