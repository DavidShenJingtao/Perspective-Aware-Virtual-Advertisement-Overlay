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
    from collections import deque

    if strong.shape != weak.shape:
        raise ValueError("strong and weak should have the same shape")
    
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    height, width = strong.shape

    # Convert inputs to Boolean masks without modifying the caller's arrays.
    strong = np.asarray(strong, dtype=bool)
    weak = np.asarray(weak, dtype=bool)

    # Seed the result and a deque with every strong-pixel coordinate.
    result = strong.copy()
    queue = deque(map(tuple, np.argwhere(strong)))

    # Define the neighbourhood
    # Four-connectivity uses vertical/horizontal offsets.
    if connectivity == 4:
        neighbors = (
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1)    # right
        )

    # Eight-connectivity additionally includes all diagonal offsets.
    else:
        neighbors = (
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1),   # right
            (-1, -1), # up-left
            (-1, 1),  # up-right
            (1, -1),  # down-left
            (1, 1)    # down-right
        )

    # Traverse the full connected component
    while queue:
        # Pop a coordinate
        y, x = queue.popleft()
        # Check in-bounds allowed neighbors
        for dy, dx in neighbors:
            next_y = y + dy
            next_x = x + dx
            is_in_bounds = (0 <= next_y < height) and (0 <= next_x < width)
            if is_in_bounds and weak[next_y, next_x] and not result[next_y, next_x]:
                # Accept every unvisited weak pixel connected to a seed
                result[next_y, next_x] = True
                # Enqueue newly accepted pixels so multi-pixel weak chains are retained.
                queue.append((next_y, next_x))

    # Return the final Boolean mask after the queue is exhausted.
    return result


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
    observed = np.asarray(to_float(frame), dtype=np.float64)
    reference = np.asarray(to_float(warped_reference), dtype=np.float64)
    coverage = np.asarray(coverage, dtype=bool)

    if observed.ndim == 2:
        observed = observed[..., None]

    if reference.ndim == 2:
        reference = reference[..., None]

    if observed.shape != reference.shape:
        raise ValueError("frame and warped reference must have the same shape")

    if observed.shape[:2] != coverage.shape:
        raise ValueError("coverage must match the image dimensions")

    if not 0 < config.gain_quantile <= 1:
        raise ValueError("gain_quantile must lie in (0, 1]")

    difference = np.zeros(coverage.shape, dtype=np.float32)
    if not coverage.any():
        return difference, (1.0, 0.0)

    frame_pixels = observed[coverage]
    reference_pixels = reference[coverage]
    raw_difference = np.mean(np.abs(frame_pixels - reference_pixels), axis=1)

    # Select pixels, then use all their colour channels in a single fit.
    count = max(1, int(np.ceil(config.gain_quantile * len(raw_difference))))
    selected = np.argsort(raw_difference, kind="stable")[:count]
    x = reference_pixels[selected].reshape(-1)
    y = frame_pixels[selected].reshape(-1)

    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_centered = x - x_mean
    denominator = float(x_centered @ x_centered)

    if denominator > 1e-12 * max(len(x), 1):
        gain = float((x_centered @ (y - y_mean)) / denominator)
        offset = y_mean - gain * x_mean
    else:
        gain = 1.0
        offset = y_mean - x_mean

    compensated = gain * reference_pixels + offset
    difference[coverage] = np.mean(
        np.abs(frame_pixels - compensated), axis=1
    ).astype(np.float32)

    return difference, (gain, offset)


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
    # Validate the required mode and mask-processing settings.
    if config.mode == "colour":
        raise NotImplementedError("Colour-based occlusion detection is not implemented")

    if config.mode != "difference":
        raise ValueError("Invalid mode: must be 'difference' or 'colour'")

    if config.connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    if not 0 <= config.weak <= config.strong:
        raise ValueError("thresholds must satisfy 0 <= weak <= strong")

    if config.open_radius < 0 or config.min_area < 0:
        raise ValueError("open_radius and min_area must be non-negative")

    # Warp the reference image into the frame using the provided homography.
    warped_reference, coverage = warp_artwork(reference, H, frame.shape)
    difference, _ = compensated_difference(frame, warped_reference, coverage, config)

    # Threshold at strong and weak and apply hysteresis
    strong = coverage & (difference >= config.strong)
    weak = coverage & (difference >= config.weak)
    mask = hysteresis(strong, weak, config.connectivity)

    # Morphological opening
    radius = int(config.open_radius)
    if radius > 0:
        yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
        kernel = ((xx**2 + yy**2) <= radius**2).astype(np.uint8)
        mask = cv2.morphologyEx(
            mask.astype(np.uint8),
            cv2.MORPH_OPEN,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0
        ).astype(bool)

    # Opening must never create a foreground pixel outside the board.
    mask &= coverage
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(mask, dtype=np.uint8),
        connectivity=config.connectivity,
    )

    keep = np.zeros(n_labels, dtype=bool)
    rows = []
    for label in range(1, n_labels):
        x, y, width, height, area = stats[label]
        if area >= config.min_area:
            keep[label] = True
            rows.append([label, x, y, width, height, area])

    mask = keep[labels]
    component_stats = np.array(rows, dtype=np.int32).reshape(-1, 6)

    return mask, difference, component_stats


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
    if config.mode != "difference":
        raise ValueError("Threshold selection is only implemented for mode='difference'")

    frames = list(frames)
    masks = [np.asarray(mask, dtype=bool) for mask in masks]

    if not frames or len(frames) != len(masks):
        raise ValueError("frames and masks must be non-empty and of the same length")

    for (frame, H, reference), truth in zip(frames, masks):
        if H is None:
            raise ValueError("Homography H cannot be None")

        if truth.shape != frame.shape[:2]:
            raise ValueError("Ground-truth mask shape must match the frame shape")


    # A fixed grid of strong thresholds and a few weak thresholds per strong threshold.
    strong_grid = sorted({0.15, 0.20, 0.25, 0.30, 0.35, float(config.strong)})
    weak_grid = sorted({0.06, 0.09, 0.12, 0.15, float(config.weak)})
    area_grid = sorted({
        max(1, int(round(config.min_area * multiplier)))
        for multiplier in [0.5, 1.0, 2.0]
    })

    search_log = []
    for strong in strong_grid:
        for weak in weak_grid:
            if weak >= strong:
                continue

            for min_area in area_grid:
                candidate = replace(config, strong=float(strong), weak=float(weak), min_area=int(min_area))
                scores = []
                for (frame, H, reference), truth in zip(frames, masks):
                    prediction, _, _ = occlusion_mask(frame, H, reference, candidate)
                    score = metrics.mask_iou(prediction, truth)

                    if not prediction.any() and not truth.any():
                        score = 1.0

                    scores.append(float(score))
                
                search_log.append({
                    "strong": float(strong),
                    "weak": float(weak),
                    "min_area": int(min_area),
                    "mean_iou": float(np.mean(scores))
                })

    # Find all settings tied for the highest validation mean IoU.
    best_score = max(row["mean_iou"] for row in search_log)
    tied = [
        row for row in search_log
        if np.isclose(
            row["mean_iou"], best_score, rtol=0.0, atol=1e-12
        )
    ]

    # Favor the middle of the tied strong range
    midpoint = (
        min(row["strong"] for row in tied) + max(row["strong"] for row in tied)
    ) / 2.0

    # Remaining ties favor the incoming weak/area settings
    chosen = min(
        tied,
        key=lambda row: (
            abs(row["strong"] - midpoint),
            abs(row["weak"] - config.weak),
            abs(row["min_area"] - config.min_area),
            row["strong"], row["weak"], row["min_area"]
        )
    )

    frozen_config = replace(
        config, strong=chosen["strong"], weak=chosen["weak"], min_area=chosen["min_area"]
    )

    return frozen_config, search_log


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
    observed = np.asarray(to_float(frame), dtype=np.float32)
    artwork = np.asarray(to_float(warped), dtype=np.float32)

    if observed.ndim == 2:
        observed = np.repeat(observed[..., None], 3, axis=2)

    if artwork.ndim == 2:
        artwork = np.repeat(artwork[..., None], 3, axis=2)

    if observed.ndim == 3 and observed.shape[2] == 1:
        observed = np.repeat(observed, 3, axis=2)

    if artwork.ndim == 3 and artwork.shape[2] == 1:
        artwork = np.repeat(artwork, 3, axis=2)

    if observed.shape != artwork.shape or observed.shape[2] != 3:
        raise ValueError("frame and warped must have the same shape and be RGB")

    coverage = np.asarray(coverage, dtype=bool)
    occlusion = np.asarray(occlusion, dtype=bool)
    if coverage.shape != observed.shape[:2] or occlusion.shape != observed.shape[:2]:
        raise ValueError("coverage and occlusion must match the image dimensions")

    if config.feather < 0:
        raise ValueError("feather must be nonnegative")

    paint = coverage & ~occlusion
    alpha = paint.astype(np.float32)

    if config.feather > 0 and paint.any():
        padded = np.pad(paint.astype(np.uint8), 1, mode="constant")
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        alpha *= np.clip(distance / float(config.feather), 0.0, 1.0)

    if config.match_gain and paint.any():
        source_mean = float(to_gray(artwork)[paint].mean())
        target_mean = float(to_gray(observed)[paint].mean())
        if source_mean > 1e-8:
            gain = np.clip(target_mean / source_mean, 0.5, 2.0)
            artwork = np.clip(artwork * gain, 0.0, 1.0)

    a = alpha[..., None]
    output = a * artwork + (1.0 - a) * observed
    return output.astype(np.float32), alpha.astype(np.float32)


__all__ = ["hysteresis", "compensated_difference", "occlusion_mask", "select_difference_thresholds", "composite"]
