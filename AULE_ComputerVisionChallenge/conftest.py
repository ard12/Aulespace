"""Pytest bootstrap: put ``src/`` on the path and share expensive fixtures.

Keeping this at the repository root means ``python -m pytest tests -q`` works
from a clean checkout with no install step and no ``PYTHONPATH`` juggling --
which is also how the Colab notebooks bootstrap themselves.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from aulecv import camera, config as C, detect, io as aio  # noqa: E402


@pytest.fixture(scope="session")
def reference_image():
    return aio.load_reference()


@pytest.fixture(scope="session")
def reference_detection(reference_image):
    return detect.detect_port(reference_image)


@pytest.fixture(scope="session")
def geometry(reference_detection, reference_image):
    return camera.PortGeometry.from_detection(reference_detection, reference_image.shape)


@pytest.fixture(scope="session")
def return_thetas():
    return camera.generate_return_sequence(C.PART_C_ANGLE_DEG, C.PART_D_STEP_DEG)


@pytest.fixture(scope="session")
def renderer_metric(geometry, return_thetas):
    return camera.make_renderer(geometry, return_thetas, "metric")


@pytest.fixture(scope="session")
def renderer_image_faithful(geometry, return_thetas):
    return camera.make_renderer(geometry, return_thetas, "image_faithful")


@pytest.fixture(scope="session")
def side_view(reference_image, renderer_metric):
    """The Part C frame, rendered once and reused by several tests."""
    return renderer_metric.render(reference_image, C.PART_C_ANGLE_DEG)


@pytest.fixture(scope="session")
def side_view_detection(side_view):
    """Corners *re-detected* on the rendered frame -- never back-projected."""
    return detect.detect_port(side_view)


@pytest.fixture(autouse=True)
def _fixed_seed():
    np.random.seed(C.RANDOM_SEED)
