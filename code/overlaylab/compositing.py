"""Stage 3 Composite: occlusion handling and compositing.

Implement the functions marked ``YOUR CODE HERE``. Motion-compensated
differencing is the required occlusion signal; ``mode="colour"`` is an
optional ablation. Permitted library calls: ``cv2.morphologyEx``,
``cv2.connectedComponentsWithStats``, ``cv2.distanceTransform`` (feathering),
and your Lab 1 ``hysteresis``.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .config import OcclusionConfig, CompositeConfig
from .supplied import to_gray, to_float
from .geometry import warp_artwork
from . import metrics


def hysteresis(strong: np.ndarray, weak: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Keep weak pixels reachable from strong pixels (reuse your Lab 1 implementation).

    ``strong`` and ``weak`` are Boolean masks of the same shape (weak includes
    strong). Traverse with the 4- or 8-connected neighbourhood from every
    strong pixel, through chains of weak pixels, and return the Boolean mask
    of reached pixels. Validate shapes and the connectivity value.
    """
    # YOUR CODE HERE
    raise NotImplementedError("YOUR CODE HERE")


def compensated_difference(frame: np.ndarray, warped_reference: np.ndarray, coverage: np.ndarray,
                           config: OcclusionConfig) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Per-pixel difference after a global gain/offset fit, zero outside ``coverage``.

    Fit I_frame ~ a * I_ref + b on the ``config.gain_quantile`` fraction of
    covered pixels whose raw difference is smallest (so that illumination
    change alone does not become an occluder), then return the mean absolute
    difference over the colour channels between the frame and the compensated
    reference, together with (a, b). Use colour, not grayscale alone: a
    coloured occluder over a coloured board can have almost the same luminance
    (the ball over the blue board differs by only ~0.13 in grayscale but ~0.3
    in mean channel difference), and the default thresholds assume colour.
    """
    # YOUR CODE HERE
    raise NotImplementedError("YOUR CODE HERE")


def occlusion_mask(frame: np.ndarray, H: np.ndarray, reference: np.ndarray, config: OcclusionConfig
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Foreground mask inside the board by motion-compensated differencing.

    1. Warp ``reference`` into the frame with ``H`` (``warp_artwork``) and
       compute the ``compensated_difference`` inside the coverage mask.
    2. Threshold at ``config.strong`` and ``config.weak`` and apply
       ``hysteresis`` with ``config.connectivity``.
    3. Open with a disc of radius ``config.open_radius`` (``cv2.morphologyEx``:
       erosion then dilation, which deletes anything thinner than 2r+1 pixels;
       0 disables) and keep connected components with at least
       ``config.min_area`` pixels.
    4. Return (mask bool H x W, difference map float H x W, component stats).

    ``component stats`` is an array with one row per kept component:
    [label, x, y, width, height, area]. With ``config.mode == "colour"`` an
    appearance model may be used instead (optional ablation).
    """
    # YOUR CODE HERE
    raise NotImplementedError("YOUR CODE HERE")


def select_difference_thresholds(frames: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                 masks: Sequence[np.ndarray], config: OcclusionConfig
                                 ) -> Tuple[OcclusionConfig, List[Dict]]:
    """Choose strong/weak thresholds and the minimum area on validation masks only.

    ``frames`` is a sequence of (frame, H, reference) triples and ``masks`` the
    matching ground-truth occlusion masks. Grid-search the thresholds and the
    minimum area, score each setting by mean ``metrics.mask_iou`` over the
    validation frames, and return (frozen config, search log) where the log
    holds one dict per setting with its parameters and score.

    Several settings usually tie at the best score: the few validation masks
    do not separate the strong threshold. Break ties towards the middle of the
    tied range of ``strong`` values (not the first or the largest grid point),
    so that the frozen value sits inside the plateau rather than on its edge,
    and state the rule in the report.
    """
    # YOUR CODE HERE
    raise NotImplementedError("YOUR CODE HERE")


def composite(frame: np.ndarray, warped: np.ndarray, coverage: np.ndarray, occlusion: np.ndarray,
              config: CompositeConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Alpha-composite the warped artwork into the frame.

    alpha = C * (1 - O) * F, where C is the coverage mask, O the occlusion
    mask and F an optional feathering term that softens the quadrilateral and
    occluder boundaries over ``config.feather`` pixels (0 disables).
    I_out = alpha * W + (1 - alpha) * I. Optional brightness matching of the
    artwork to the board is controlled by ``config.match_gain``.
    Returns (composited RGB float32 frame, alpha map float32 in [0, 1]).
    """
    # YOUR CODE HERE
    raise NotImplementedError("YOUR CODE HERE")


__all__ = ["hysteresis", "compensated_difference", "occlusion_mask", "select_difference_thresholds", "composite"]
