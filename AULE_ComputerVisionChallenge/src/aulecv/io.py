"""Project paths, image loading/saving and lossless extraction from the PDF."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "project_root",
    "asset_path",
    "output_path",
    "ensure_output_dirs",
    "load_reference",
    "load_image",
    "save_image",
    "extract_reference_from_pdf",
    "to_gray",
    "to_rgb",
]


def project_root() -> str:
    """Absolute path of the repository root (the folder holding ``README.md``)."""
    env = os.environ.get("AULECV_PROJECT_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))       # .../src/aulecv
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


def asset_path(*parts: str) -> str:
    return os.path.join(project_root(), "assets", *parts)


def output_path(*parts: str) -> str:
    return os.path.join(project_root(), "outputs", *parts)


def ensure_output_dirs() -> str:
    """Create ``outputs/`` and its sub-folders; return the outputs root."""
    root = output_path()
    for sub in ("", "detection", "rotation_tests", "crop_navigation",
                "homography", "camera_return"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    return root


def load_image(path: str) -> np.ndarray:
    """Load a BGR uint8 image, raising a clear error if the path is wrong."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("could not read image: {}".format(path))
    return img


def load_reference() -> np.ndarray:
    """Load ``assets/reference.png`` (the 604x558 raster embedded in the PDF)."""
    return load_image(asset_path("reference.png"))


def save_image(path: str, img: np.ndarray) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not cv2.imwrite(path, img):
        raise IOError("could not write image: {}".format(path))
    return path


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_rgb(img: np.ndarray) -> np.ndarray:
    """BGR (OpenCV) -> RGB, for matplotlib."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_reference_from_pdf(pdf_path: str,
                               out_path: Optional[str] = None,
                               xref: int = 9) -> Tuple[str, dict]:
    """Extract the embedded reference raster *losslessly* from the PDF.

    The image is pulled out of the PDF's object stream with PyMuPDF.  When the
    embedded stream is already a PNG (it is -- ``xref`` 9 of the supplied PDF),
    the bytes are written out verbatim, so ``assets/reference.png`` is
    byte-identical to what is inside the PDF.  Nothing is ever screenshotted or
    re-rendered at a page DPI.

    Returns ``(path, info)`` where ``info`` carries the reported width, height,
    extension and whether the write was byte-verbatim.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import pymupdf as fitz            # PyMuPDF >= 1.24 preferred name
        except ImportError:                    # pragma: no cover
            import fitz                        # noqa: F401

    doc = fitz.open(pdf_path)
    try:
        raw_info = doc.extract_image(xref)
    finally:
        doc.close()

    data = raw_info["image"]
    ext = raw_info["ext"]
    if out_path is None:
        out_path = asset_path("reference.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    verbatim = (ext == "png")
    if verbatim:
        with open(out_path, "wb") as fh:
            fh.write(data)
    else:                                       # pragma: no cover
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(out_path, arr)

    info = {
        "xref": xref,
        "width": raw_info["width"],
        "height": raw_info["height"],
        "ext": ext,
        "bytes": len(data),
        "byte_verbatim": verbatim,
    }
    return out_path, info
