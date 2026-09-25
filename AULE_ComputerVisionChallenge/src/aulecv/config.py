"""Central configuration for the AULE computer-vision challenge.

Every threshold in this file is either

* a *relative* quantity (a fraction of the detected outer-square side length,
  of the measured border thickness, of the contour perimeter, ...), or
* a physical constant taken straight from the assignment text.

Nothing here is a pixel coordinate or an answer angle tuned to one particular
test image.  The detectors in :mod:`aulecv.detect`, :mod:`aulecv.rotation` and
:mod:`aulecv.navigation` read their thresholds from here so that the numbers
are auditable in one place.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Physical geometry (given in the assignment)
# ---------------------------------------------------------------------------

OUTER_SIDE_CM = 40.0      #: outermost square side length
INNER_SIDE_CM = 30.0      #: innermost square side length
SQUARE_SPACING_CM = 2.5   #: spacing between consecutive squares
MIDDLE_SIDE_CM = OUTER_SIDE_CM - 2 * SQUARE_SPACING_CM  #: 35.0 cm
CAMERA_DISTANCE_CM = 100.0  #: "the view shown is from a camera which is 1 m away"

#: Part C viewing angle, degrees, measured about the world "down" axis,
#: positive = camera moves to the viewer's right.
PART_C_ANGLE_DEG = 22.5

#: Part D angular step size, degrees.
PART_D_STEP_DEG = 2.5

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = 20240517

# ---------------------------------------------------------------------------
# Foreground / background separation
# ---------------------------------------------------------------------------

#: Width of the image-border ring used to estimate the page background colour,
#: as a fraction of min(H, W).
BG_RING_FRAC = 0.02

#: A pixel counts as foreground when its grey level differs from the estimated
#: background level by more than this many grey levels.  Deliberately generous:
#: the reference drawing uses 191 / 128 / 255 / 0, so any sane value separates
#: the port from the page.
FG_TOLERANCE = 18

# ---------------------------------------------------------------------------
# Outer-quad extraction
# ---------------------------------------------------------------------------

#: Size of the morphological closing applied to the foreground mask before the
#: outer contour is traced, as a fraction of min(H, W).  It only has to bridge
#: single-pixel holes left by anti-aliasing, so it is deliberately tiny.
FG_CLOSE_FRAC = 0.004

#: ``approxPolyDP`` epsilon, as a fraction of the contour perimeter.  Searched
#: over this list until a 4-gon comes out, so the detector is not sensitive to
#: the exact value.
APPROX_EPS_FRACTIONS = (0.02, 0.01, 0.03, 0.005, 0.04, 0.05)

#: Fraction trimmed from *each* end of a polygon side before ``fitLine``.
#: Corner rounding (anti-aliasing, the JPEG-style stair-stepping in the source
#: raster) lives in these end regions.
SIDE_TRIM_FRAC = 0.15

#: A side must retain at least this many contour samples after trimming for the
#: line fit to be used; otherwise the ``approxPolyDP`` vertex is kept.
SIDE_MIN_SAMPLES = 8

#: Maximum allowed distance between a refined (line-intersection) corner and the
#: corresponding coarse ``approxPolyDP`` vertex, as a fraction of the outer side
#: length.  Guards against near-parallel side fits blowing up.
CORNER_REFINE_MAX_SHIFT_FRAC = 0.05

# ---------------------------------------------------------------------------
# Border thickness (drives every morphological kernel)
# ---------------------------------------------------------------------------

#: Fallback border thickness, as a fraction of the outer side length, used only
#: if the distance-transform ridge estimate fails.
BORDER_THICKNESS_FALLBACK_FRAC = 0.01

#: Clamp the measured border thickness into this range (fractions of the outer
#: side) so a pathological input cannot produce a degenerate kernel.
BORDER_THICKNESS_MIN_FRAC = 0.002
BORDER_THICKNESS_MAX_FRAC = 0.05

#: Minimum distance-transform value counted as part of the medial-axis ridge.
#: A stroke exactly one pixel wide has dt = 0.5 there, so this is the smallest
#: meaningful value rather than a tuned threshold.
RIDGE_MIN_DT = 0.5

#: Minimum number of ridge samples before the thickness estimate is trusted.
RIDGE_MIN_SAMPLES = 8

# ---------------------------------------------------------------------------
# Marker (circle) detection
# ---------------------------------------------------------------------------

#: Diameter of the disk structuring element used to open the dark mask,
#: expressed as a multiple of the measured border thickness.  It must be large
#: enough to erase the square borders (thickness t) and small enough to leave
#: the marker (diameter ~0.2 x outer side) standing.
MARKER_OPEN_DIAM_FACTOR = 3.0

#: Absolute floor / ceiling on that kernel, as a fraction of the outer side.
MARKER_OPEN_DIAM_MIN_FRAC = 0.010
MARKER_OPEN_DIAM_MAX_FRAC = 0.060

#: Candidate blob area bounds, as fractions of the outer quad area.
MARKER_AREA_MIN_FRAC = 0.005
MARKER_AREA_MAX_FRAC = 0.200

#: Circularity 4*pi*A/P^2 of the *true* marker measures 0.827 on the supplied
#: raster -- the outline is stair-stepped and the disk bulges where it touches
#: the two square borders.  The threshold is therefore deliberately lenient and
#: circularity is only one of several cues, never the sole discriminator.
MARKER_CIRCULARITY_MIN = 0.70

#: Minimum minor/major axis ratio of the fitted ellipse (0.975 on the raster).
MARKER_AXIS_RATIO_MIN = 0.70

#: The distance-transform peak and the fitted-ellipse centre must agree to
#: within this fraction of the marker radius for full confidence.  These are
#: two measurements of the *same* blob, so the budget is tight.
MARKER_CENTER_AGREE_FRAC = 0.30

#: HoughCircles votes into an integer accumulator over a coarse radius band on
#: the raw gradient image, so its centre is only a qualitative confirmation
#: that a circle is there.  It gets a much looser budget and is *reported*
#: rather than folded into the strict confidence score.
MARKER_HOUGH_AGREE_FRAC = 0.50

#: HoughCircles search band, as a fraction of the *expected* marker radius
#: (which is itself derived from the opened blob, not hard-coded).
HOUGH_RADIUS_BAND = 0.35
HOUGH_PARAM1 = 100.0
HOUGH_PARAM2 = 20.0

#: Floor on HoughCircles' ``minDist``, as a fraction of the expected marker
#: radius.  There is only ever one marker, so this only has to stop the
#: accumulator returning a cluster of near-duplicate circles.
HOUGH_MIN_DIST_FRAC = 1.0

# ---------------------------------------------------------------------------
# Part A -- rotation estimation
# ---------------------------------------------------------------------------

#: The square-edge cue only determines the angle modulo 90 deg.  It is used to
#: refine the marker estimate when the two agree to within this many degrees.
ROTATION_REFINE_TOL_DEG = 25.0

#: Minimum detector confidence for the fused rotation estimate to be reported
#: as reliable.
ROTATION_MIN_CONFIDENCE = 0.5

# ---------------------------------------------------------------------------
# Part B -- crop localisation and navigation
# ---------------------------------------------------------------------------

#: A crop whose grey-level standard deviation is below this is declared
#: ``featureless`` -- there is nothing in it to match.
CROP_MIN_STD = 3.0

#: Template-match peak score below this => not confidently localised.
MATCH_MIN_SCORE = 0.60

#: Distinct positions whose correlation differs by less than numerical/texture
#: resolution cannot identify a unique crop, even inside the NMS neighbourhood.
MATCH_TIE_TOL = 1e-5

#: Scores within this much of the peak count as "part of the peak plateau".
MATCH_PLATEAU_TOL = 0.02

#: The plateau may not span more than this fraction of the *crop*'s own size
#: along either axis before the localisation is declared ambiguous.  This is
#: the aperture problem: a crop showing only one straight border slides freely
#: along that border.
MATCH_PLATEAU_MAX_FRAC = 0.25

#: Radius of the non-maximum-suppression window around the peak, as a fraction
#: of the crop size, used before looking for a competing second peak.
MATCH_NMS_FRAC = 0.5

#: A second peak this close to the best one (ratio of scores) makes the
#: localisation ambiguous.
MATCH_SECOND_PEAK_RATIO = 0.92

#: Camera step size, as a fraction of the crop's *smaller* side.
NAV_STEP_FRAC = 0.05

#: Hard cap on the number of navigation steps, so a pathological case
#: terminates.
NAV_MAX_STEPS = 400

#: OPTIONAL exploration only: the square-spiral *search* stride, as a fraction
#: of the view's larger side.  A search wants consecutive stops that barely
#: overlap; reusing the (much finer) tracking stride above would make a small
#: view re-examine nearly the same pixels thousands of times.
EXPLORE_STEP_FRAC = 0.5

#: Margin, as a fraction of the marker radius, required between the marker disk
#: and the window edge before the marker counts as "fully visible".
NAV_MARKER_MARGIN_FRAC = 0.10

# ---------------------------------------------------------------------------
# Parts C / D -- rendering
# ---------------------------------------------------------------------------

#: Focal length used by the metric (Model A) camera, as a multiple of the
#: geometric-mean raster size of the port:  f = FOCAL_SCALE * sqrt(w*h).
#: FOCAL_SCALE = CAMERA_DISTANCE_CM / OUTER_SIDE_CM = 2.5 makes the theta = 0
#: render come out at the raster's own geometric-mean size.  f is a pure
#: display scale: see the intrinsics-invariance demonstration in notebook 03.
FOCAL_SCALE = CAMERA_DISTANCE_CM / OUTER_SIDE_CM  # 2.5 px per (px/cm)

#: Extra canvas margin around the union of all projected quads, as a fraction
#: of the largest projected extent.
CANVAS_MARGIN_FRAC = 0.08

#: Background (page) colour written into the un-covered part of the canvas.
RENDER_BACKGROUND = (255, 255, 255)

#: Tolerance on the scale-normalised Frobenius distance between two
#: homographies that are supposed to be identical up to scale.
HOMOGRAPHY_EQUAL_TOL = 1e-8

#: Tolerance (cm) on || C || == 100 for every rendered pose.
POSE_RADIUS_TOL_CM = 1e-6

#: PnP evaluation targets (labelled as *targets*; the measured values are
#: reported next to them in notebook 04 and in the README).
PNP_TARGET_ANGLE_ERR_DEG = 0.5
PNP_TARGET_DISTANCE_ERR_CM = 1.0
