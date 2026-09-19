"""AULE computer-vision challenge: detection, rotation, navigation and rendering."""

from __future__ import annotations

__version__ = "1.0.0"

from . import config, detect, io, navigation, rotation, camera, viz  # noqa: F401

__all__ = ["config", "detect", "io", "navigation", "rotation", "camera", "viz"]
