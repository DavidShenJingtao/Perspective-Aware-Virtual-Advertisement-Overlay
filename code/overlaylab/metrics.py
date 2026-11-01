"""Supplied evaluation metrics. All reported numbers must come from here.

Corner errors are normalised by the longer diagonal of the annotated
quadrilateral so that they are comparable across resolutions and board sizes
(equation (7) of the handout). Masks are Boolean arrays; "painted" means the
alpha map used by ``composite`` (or a Boolean coverage mask when no occlusion
handling is applied).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .supplied import longer_diagonal, quad_mask, to_gray, to_float

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def corner_error(pred_corners: np.ndarray, gt_corners: np.ndarray) -> float:
    """Normalised corner-transfer error: mean corner distance / longer GT diagonal."""
    p = np.asarray(pred_corners, dtype=np.float64).reshape(4, 2)
    g = np.asarray(gt_corners, dtype=np.float64).reshape(4, 2)
    if not np.all(np.isfinite(p)):
        return float("inf")
    d = longer_diagonal(g)
    return float(np.mean(np.linalg.norm(p - g, axis=1)) / max(d, 1e-9))


def corner_errors(pred: Dict[int, np.ndarray], gt: Dict[int, np.ndarray]) -> Dict[int, float]:
    """Per-frame normalised corner error for every annotated frame present in ``pred``."""
    return {f: corner_error(pred[f], gt[f]) if f in pred and pred[f] is not None else float("inf")
            for f in gt}


def success_rate(errors: Iterable[float], threshold: float) -> float:
    e = np.asarray(list(errors), dtype=np.float64)
    if e.size == 0:
        return float("nan")
    return float(np.mean(e <= threshold))


def summarise_errors(errors: Iterable[float], threshold: float) -> Dict[str, float]:
    e = np.asarray(list(errors), dtype=np.float64)
    finite = e[np.isfinite(e)]
    return {
        "n": int(e.size),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "p90": float(np.percentile(finite, 90)) if finite.size else float("nan"),
        "success": success_rate(e, threshold),
        "failed": int(np.sum(~np.isfinite(e))),
    }


def interpolate_corners(gt_frames: Sequence[int], gt_corners: Sequence[np.ndarray],
                        query_frames: Sequence[int]) -> np.ndarray:
    """Linear interpolation of annotated corners to arbitrary frames (n x 4 x 2)."""
    gf = np.asarray(gt_frames, dtype=np.float64)
    order = np.argsort(gf)
    gf = gf[order]
    gc = np.asarray(gt_corners, dtype=np.float64)[order]  # k x 4 x 2
    q = np.asarray(query_frames, dtype=np.float64)
    out = np.zeros((len(q), 4, 2))
    for k in range(4):
        for j in range(2):
            out[:, k, j] = np.interp(q, gf, gc[:, k, j])
    return out


def jitter(pred_corners: Dict[int, np.ndarray], gt_frames: Sequence[int],
           gt_corners: Sequence[np.ndarray]) -> float:
    """Median frame-to-frame corner displacement after removing interpolated GT motion (px).

    ``pred_corners`` maps frame index -> 4 x 2 corners for consecutive frames.
    Frames outside the annotated range or with missing predictions are skipped.
    """
    frames = sorted(f for f in pred_corners if pred_corners[f] is not None
                    and np.all(np.isfinite(pred_corners[f])))
    if len(frames) < 2:
        return float("nan")
    lo, hi = min(gt_frames), max(gt_frames)
    frames = [f for f in frames if lo <= f <= hi]
    gt_interp = interpolate_corners(gt_frames, gt_corners, frames)
    disp = []
    for i in range(1, len(frames)):
        if frames[i] != frames[i - 1] + 1:
            continue
        dp = np.asarray(pred_corners[frames[i]]) - np.asarray(pred_corners[frames[i - 1]])
        dg = gt_interp[i] - gt_interp[i - 1]
        disp.append(np.mean(np.linalg.norm(dp - dg, axis=1)))
    return float(np.median(disp)) if disp else float("nan")


def drift(errors: Dict[int, float], reinit_frames: Sequence[int]) -> List[Tuple[int, float]]:
    """(frames since last reinitialisation, corner error) pairs for annotated frames."""
    r = np.asarray(sorted(set(reinit_frames)), dtype=np.int64)
    out = []
    for f in sorted(errors):
        prev = r[r <= f]
        since = int(f - prev[-1]) if prev.size else int(f)
        out.append((since, float(errors[f])))
    return out


def recovery_time(errors_by_frame: Dict[int, float], event_end: int, threshold: float,
                  max_wait: int = 60) -> Optional[int]:
    """Frames after ``event_end`` until the error first drops under ``threshold``."""
    for k in range(0, max_wait + 1):
        f = event_end + k
        if f in errors_by_frame and errors_by_frame[f] <= threshold:
            return k
    return None


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred, dtype=bool); g = np.asarray(gt, dtype=bool)
    union = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / union) if union else float("nan")


def mask_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred, dtype=bool); g = np.asarray(gt, dtype=bool)
    tp = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return float(2 * tp / denom) if denom else float("nan")


def leakage(painted: np.ndarray, gt_foreground: np.ndarray) -> float:
    """Fraction of annotated foreground pixels that received artwork.

    ``painted`` is the alpha map from ``composite`` (float in [0, 1]) or a
    Boolean mask; values above 0.5 count as painted.
    """
    a = np.asarray(painted)
    painted_b = a > 0.5 if a.dtype != bool else a
    g = np.asarray(gt_foreground, dtype=bool)
    n = g.sum()
    return float(np.logical_and(painted_b, g).sum() / n) if n else float("nan")


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    """F1 between mask boundaries, allowing ``tolerance`` px of slack."""
    def boundary(m):
        m8 = np.asarray(m, dtype=np.uint8)
        er = cv2.erode(m8, np.ones((3, 3), np.uint8))
        return (m8 - er).astype(bool)
    pb, gb = boundary(pred), boundary(gt)
    k = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    pd = cv2.dilate(pb.astype(np.uint8), k).astype(bool)
    gd = cv2.dilate(gb.astype(np.uint8), k).astype(bool)
    prec = np.logical_and(pb, gd).sum() / max(pb.sum(), 1)
    rec = np.logical_and(gb, pd).sum() / max(gb.sum(), 1)
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


# ---------------------------------------------------------------------------
# Content and side effects (Stage 4, generative comparison)
# ---------------------------------------------------------------------------

def outside_board_change(frame_a: np.ndarray, frame_b: np.ndarray, gt_corners: np.ndarray,
                         margin: int = 4) -> float:
    """Mean absolute difference outside the (dilated) ground-truth quadrilateral, [0, 1] scale."""
    a = to_float(frame_a); b = to_float(frame_b)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    inside = quad_mask(a.shape[:2], gt_corners).astype(np.uint8)
    if margin > 0:
        inside = cv2.dilate(inside, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    outside = inside == 0
    if not outside.any():
        return float("nan")
    return float(np.mean(np.abs(a - b)[outside]))


def rectify_board(frame: np.ndarray, gt_corners: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Warp the board quadrilateral of ``frame`` to a (width, height) rectangle."""
    w, h = size
    src = np.asarray(gt_corners, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(to_float(frame), M, (w, h), flags=cv2.INTER_LINEAR)


def rectified_content_fidelity(frame: np.ndarray, gt_corners: np.ndarray, artwork: np.ndarray,
                               size: Tuple[int, int] = (400, 100)) -> Dict[str, float]:
    """Compare the rectified board region with the artwork.

    Returns normalised cross-correlation of grayscale (1 = identical up to
    gain/offset), mean absolute colour difference, and a structural score
    (correlation of Sobel magnitudes) that ignores global tone changes.
    """
    rect = rectify_board(frame, gt_corners, size)
    art = cv2.resize(to_float(artwork), size, interpolation=cv2.INTER_AREA)
    g1, g2 = to_gray(rect), to_gray(art)
    def ncc(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
    def grad_mag(g):
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
        return np.sqrt(gx * gx + gy * gy)
    return {
        "ncc": ncc(g1, g2),
        "mae": float(np.mean(np.abs(rect - art))),
        "structure": ncc(grad_mag(g1), grad_mag(g2)),
    }


def temporal_identity(frames: Sequence[np.ndarray], gt_corners: Sequence[np.ndarray],
                      size: Tuple[int, int] = (400, 100)) -> Dict[str, float]:
    """Flicker of the inserted content: mean |difference| between consecutive rectified boards."""
    rects = [rectify_board(f, c, size) for f, c in zip(frames, gt_corners)]
    if len(rects) < 2:
        return {"flicker": float("nan")}
    diffs = [float(np.mean(np.abs(rects[i] - rects[i - 1]))) for i in range(1, len(rects))]
    return {"flicker": float(np.mean(diffs)), "flicker_max": float(np.max(diffs))}


__all__ = [
    "corner_error", "corner_errors", "success_rate", "summarise_errors", "interpolate_corners",
    "jitter", "drift", "recovery_time", "mask_iou", "mask_f1", "leakage", "boundary_f1",
    "outside_board_change", "rectify_board", "rectified_content_fidelity", "temporal_identity",
]
