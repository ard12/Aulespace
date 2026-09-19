"""Matplotlib / OpenCV visualisation helpers.

Nothing here computes geometry -- these functions only draw what the other
modules measured, so a figure can never disagree with the numbers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .io import to_rgb

__all__ = [
    "draw_detection",
    "draw_quad",
    "show",
    "grid",
    "movement_convention_figure",
    "draw_window",
    "navigation_figure",
    "frame_strip",
    "save_gif",
    "annotate",
]

# A small, colour-blind-safe palette (BGR, because these draw on cv2 images).
_BGR = {
    "quad": (0, 150, 255),        # orange
    "corner": (0, 0, 255),        # red
    "center": (255, 80, 0),       # blue
    "marker": (0, 200, 0),        # green
    "axis": (200, 0, 200),        # magenta
    "window": (0, 0, 255),
    "target": (0, 170, 0),
}


def _thick(img: np.ndarray, base: float = 0.004) -> int:
    return max(1, int(round(base * min(img.shape[:2]))))


def draw_quad(canvas: np.ndarray,
              quad: Sequence[Sequence[float]],
              color: Tuple[int, int, int] = _BGR["quad"],
              thickness: Optional[int] = None,
              label_corners: bool = False) -> np.ndarray:
    q = np.asarray(quad, float).reshape(-1, 2)
    t = thickness or _thick(canvas)
    cv2.polylines(canvas, [np.round(q).astype(np.int32)], True, color, t, cv2.LINE_AA)
    if label_corners:
        for i, p in enumerate(q):
            cv2.circle(canvas, tuple(np.round(p).astype(int)), 2 * t, _BGR["corner"], -1, cv2.LINE_AA)
            cv2.putText(canvas, str(i), tuple(np.round(p + np.array([6 * t, -3 * t])).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 * t / 2.0, _BGR["corner"],
                        max(1, t // 2), cv2.LINE_AA)
    return canvas


def draw_detection(image: np.ndarray, detection, label_corners: bool = True) -> np.ndarray:
    """Overlay the detected quad, centre, marker ellipse and marker vector."""
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    t = _thick(canvas)

    draw_quad(canvas, detection.quad, label_corners=label_corners)
    c = np.round(np.asarray(detection.center, float)).astype(int)
    cv2.drawMarker(canvas, tuple(c), _BGR["center"], cv2.MARKER_CROSS, 12 * t, t, cv2.LINE_AA)

    if detection.marker_found:
        mc = np.asarray(detection.marker_center, float)
        major, minor = detection.marker_axes
        cv2.ellipse(canvas, tuple(np.round(mc).astype(int)),
                    (int(round(major / 2)), int(round(minor / 2))),
                    float(detection.marker_angle_deg), 0, 360, _BGR["marker"], t, cv2.LINE_AA)
        cv2.drawMarker(canvas, tuple(np.round(mc).astype(int)), _BGR["marker"],
                       cv2.MARKER_TILTED_CROSS, 8 * t, t, cv2.LINE_AA)
        cv2.arrowedLine(canvas, tuple(c), tuple(np.round(mc).astype(int)),
                        _BGR["axis"], t, cv2.LINE_AA, tipLength=0.08)
        if detection.marker_center_dt is not None:
            cv2.drawMarker(canvas, tuple(np.round(detection.marker_center_dt).astype(int)),
                           (0, 255, 255), cv2.MARKER_SQUARE, 6 * t, max(1, t // 2), cv2.LINE_AA)
    return canvas


def annotate(image: np.ndarray, text: str, org: Tuple[int, int] = (10, 28),
             color: Tuple[int, int, int] = (0, 0, 0),
             scale: float = 0.6) -> np.ndarray:
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Matplotlib
# ---------------------------------------------------------------------------

def show(image: np.ndarray, title: str = "", ax=None, figsize=(6, 6)):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(to_rgb(image))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def grid(images: Sequence[np.ndarray], titles: Optional[Sequence[str]] = None,
         cols: int = 3, figsize_per: float = 3.4, suptitle: str = ""):
    import matplotlib.pyplot as plt
    n = len(images)
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * figsize_per, rows * figsize_per))
    axes = np.atleast_1d(np.asarray(axes)).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            show(images[i], titles[i] if titles else "", ax=ax)
        else:
            ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


def draw_window(image: np.ndarray, window: Sequence[int],
                color: Tuple[int, int, int] = _BGR["window"],
                thickness: Optional[int] = None,
                label: str = "") -> np.ndarray:
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    x, y, w, h = [int(v) for v in window]
    t = thickness or _thick(out)
    cv2.rectangle(out, (x, y), (x + w, y + h), color, t, cv2.LINE_AA)
    if label:
        out = annotate(out, label, (x + 4, max(16, y - 6)), color)
    return out


def movement_convention_figure(reference: np.ndarray, detection,
                               window: Sequence[int], shift_px: int = 90):
    """Three-panel illustration of the single movement convention.

    Left: the field-of-view window on the reference image.  Middle: what the
    camera sees.  Right: the same after a RIGHT command -- the window slides to
    +x and the *content* slides the opposite way.
    """
    import matplotlib.pyplot as plt
    from .navigation import extract_crop

    x, y, w, h = [int(v) for v in window]
    H, W = reference.shape[:2]
    x2 = int(np.clip(x + shift_px, 0, W - w))

    board = draw_window(reference, (x, y, w, h), _BGR["window"], label="before")
    board = draw_window(board, (x2, y, w, h), _BGR["target"], label="after RIGHT")
    cv2.arrowedLine(board, (x + w // 2, y + h // 2), (x2 + w // 2, y + h // 2),
                    (0, 0, 0), _thick(board), cv2.LINE_AA, tipLength=0.15)

    before = extract_crop(reference, x, y, w, h)
    after = extract_crop(reference, x2, y, w, h)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    show(board, "camera translation parallel to the port plane\n"
                "RIGHT = +x_img,  DOWN = +y_img", ax=axes[0])
    show(before, "view before", ax=axes[1])
    show(after, "view after 'move camera RIGHT {} px'\n"
                "(window moved +x, content moved -x)".format(x2 - x), ax=axes[2])
    fig.tight_layout()
    return fig


def navigation_figure(reference: np.ndarray, detection, plan, max_panels: int = 8):
    """Overview board + a strip of the simulated views along the plan."""
    import matplotlib.pyplot as plt
    from .navigation import simulate_crop_navigation

    board = reference.copy()
    if plan.start_window:
        board = draw_window(board, plan.start_window, _BGR["window"], label="start")
    if plan.target_window:
        board = draw_window(board, plan.target_window, _BGR["target"], label="goal")
    for cmd in plan.commands:
        x, y, w, h = cmd.window_after
        cv2.circle(board, (x + w // 2, y + h // 2), max(2, _thick(board)),
                   (120, 120, 120), -1, cv2.LINE_AA)
    if detection.marker_found:
        mc = np.round(np.asarray(detection.marker_center)).astype(int)
        cv2.circle(board, tuple(mc), int(round(detection.marker_radius_px)),
                   _BGR["marker"], _thick(board), cv2.LINE_AA)

    frames = simulate_crop_navigation(reference, plan)
    if len(frames) > max_panels:
        idx = np.unique(np.linspace(0, len(frames) - 1, max_panels).round().astype(int))
        frames = [frames[i] for i in idx]

    fig = plt.figure(figsize=(14, 7.5))
    ax0 = fig.add_subplot(2, 1, 1)
    show(board, "plan on the reference image (start -> goal, camera translations)", ax=ax0)
    for i, fr in enumerate(frames):
        ax = fig.add_subplot(2, len(frames), len(frames) + i + 1)
        show(fr["image"], "step {} {}".format(fr["step"], fr["direction"]), ax=ax)
    fig.tight_layout()
    return fig


def frame_strip(frames: Sequence[np.ndarray], titles: Optional[Sequence[str]] = None,
                cols: int = 5, figsize_per: float = 2.9, suptitle: str = ""):
    return grid(list(frames), titles, cols=cols, figsize_per=figsize_per, suptitle=suptitle)


def save_gif(path: str, frames: Sequence[np.ndarray], fps: float = 6.0,
             loop: int = 0, boomerang: bool = False) -> str:
    """Write a GIF from BGR frames (converted to RGB on the way out)."""
    import os
    import imageio.v2 as imageio

    seq = [to_rgb(f) for f in frames]
    if boomerang and len(seq) > 2:
        seq = seq + seq[-2:0:-1]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimsave(path, seq, duration=1.0 / float(fps), loop=loop)
    return path
