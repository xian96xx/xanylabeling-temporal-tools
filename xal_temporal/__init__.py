"""Interval smoothing and SAM2 gap-filling for X-AnyLabeling video boxes.

Pure-logic modules (smoothing, tracking transactions, geometry, cache) import
without Qt, torch or a GPU. Dialogs live in ``xal_temporal.qt``. The SAM2
worker is optional.
"""

__version__ = "0.1.0"

from .smoothing import refine_boxes, smooth_boxes

__all__ = ["refine_boxes", "smooth_boxes", "__version__"]
