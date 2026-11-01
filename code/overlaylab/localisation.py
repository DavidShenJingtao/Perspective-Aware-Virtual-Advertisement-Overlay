"""Stage 1 Localise: board localisation from the reference image and single-frame overlay.

Implement the functions marked ``YOUR CODE HERE`` using your keypoint and
geometry branches. Optional feature families and alternative settings must be
controlled only through ``LocalisationConfig``.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .config import LocalisationConfig
from .supplied import to_gray, to_float, rect_corners, gaussian_smooth
from .keypoints import detect_corners, describe_keypoints, match_descriptors
from .geometry import ransac_homography, transfer_points, warp_artwork, rectangle_to_rectangle
from . import metrics


def localise_board(reference: np.ndarray, frame: np.ndarray, config: LocalisationConfig
                   ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Find the board in ``frame`` from the rectified ``reference`` image alone.

    Detect and describe keypoints in both images, match with the ratio test,
    estimate the reference-to-frame homography with RANSAC, and project the
    four reference corners (TL, TR, BR, BL) into the frame.

    Returns (H, corners 4 x 2, matches M x 2 (reference index, frame index),
    inlier_mask M, diagnostics). ``diagnostics`` must contain exactly these keys
    (the notebooks and the grading harness read them): ``n_keypoints_ref``,
    ``n_keypoints_frame``, ``n_matches``, ``n_inliers``, ``inlier_ratio``
    (inliers / ratio-test matches), ``median_residual``, ``iterations``,
    ``status``, ``residuals`` (the per-match symmetric transfer errors e_i from
    RANSAC, length M), ``keypoints_ref`` and ``keypoints_frame`` (the arrays the
    match indices refer to). ``median_residual`` is the median over inliers of
    sqrt(e_i / 2), i.e. the RMS one-way transfer error in pixels.
    When localisation fails, ``H`` is None, ``corners`` is NaN and ``status``
    is False.
    """
    # YOUR CODE HERE
    det, des = config.detector, config.descriptor
    kp_ref, desc_ref = _reference_features(reference, config)
    g_frame = to_gray(frame)
    kp_f, _ = detect_corners(g_frame, det)
    desc_f, keep_f = describe_keypoints(g_frame, kp_f, des)
    kp_f = kp_f[keep_f]

    pairs, _, _ = match_descriptors(desc_ref, desc_f, config.matcher)
    m = len(pairs)
    H, inliers, residuals, iters, status = None, np.zeros(m, dtype=bool), np.full(m, np.inf), 0, False
    if m >= 4:
        H, inliers, residuals, iters, status = ransac_homography(kp_ref[pairs[:, 0]], kp_f[pairs[:, 1]], config.ransac)

    ref_h, ref_w = reference.shape[:2]
    corners = np.full((4, 2), np.nan)
    if status and H is not None:
        corners = transfer_points(H, rect_corners(ref_w, ref_h))
        if not _plausible_quad(corners):
            status = False            # a folded or degenerate quadrilateral is not a board
    if not status:
        H = None
        corners = np.full((4, 2), np.nan)

    n_in = int(inliers.sum())
    diagnostics = {
        "n_keypoints_ref": int(len(kp_ref)),
        "n_keypoints_frame": int(len(kp_f)),
        "n_matches": int(m),
        "n_inliers": n_in,
        "inlier_ratio": float(n_in / m) if m else 0.0,
        "median_residual": float(np.median(np.sqrt(residuals[inliers] / 2.0))) if n_in else float("nan"),
        "iterations": int(iters),
        "status": bool(status),
        "residuals": np.asarray(residuals, dtype=np.float64),
        "keypoints_ref": kp_ref,
        "keypoints_frame": kp_f,
    }
    return H, corners, pairs, inliers, diagnostics


def _plausible_quad(c: np.ndarray) -> bool:
    """Finite, strictly convex quadrilateral with the reference's orientation (no reflection)."""
    if not np.all(np.isfinite(c)):
        return False
    d = np.roll(c, -1, axis=0) - c
    cross = d[:, 0] * np.roll(d, -1, axis=0)[:, 1] - d[:, 1] * np.roll(d, -1, axis=0)[:, 0]
    return bool(np.all(cross > 0))   # TL, TR, BR, BL is clockwise on screen (y down) -> positive cross products


# Reference features depend only on the reference image and the keypoint settings; caching them
# avoids re-describing the same reference on every frame of a clip (the independent variant).
_REF_CACHE: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}


def _reference_features(reference: np.ndarray, config: LocalisationConfig) -> Tuple[np.ndarray, np.ndarray]:
    import hashlib
    ref = np.ascontiguousarray(reference)
    key = (ref.shape, str(ref.dtype), hashlib.blake2b(ref.tobytes(), digest_size=16).hexdigest(),
           repr(config.detector), repr(config.descriptor))
    hit = _REF_CACHE.get(key)
    if hit is None:
        g = to_gray(ref)
        kp, _ = detect_corners(g, config.detector)
        desc, keep = describe_keypoints(g, kp, config.descriptor)
        hit = (kp[keep], desc)
        if len(_REF_CACHE) >= 8:
            _REF_CACHE.clear()
        _REF_CACHE[key] = hit
    return hit[0].copy(), hit[1]


def artwork_to_frame(artwork: np.ndarray, H: np.ndarray, config: LocalisationConfig) -> np.ndarray:
    """Compose the artwork rectangle with the board-to-frame homography.

    The artwork is placed by mapping its own pixel rectangle onto the
    reference board rectangle (``config.warp.reference_shape``; the artwork's
    own shape when unset) and then applying ``H``. Document in your report how
    the two transformations compose.
    """
    # YOUR CODE HERE
    ah, aw = artwork.shape[:2]
    ref_shape = config.warp.reference_shape or (ah, aw)
    rh, rw = int(ref_shape[0]), int(ref_shape[1])
    S = rectangle_to_rectangle((aw, ah), (rw, rh))       # artwork pixels -> reference board pixels
    Ha = np.asarray(H, dtype=np.float64) @ S              # then reference board -> frame
    if abs(Ha[2, 2]) > 1e-12:
        Ha = Ha / Ha[2, 2]
    return Ha


def overlay_frame(frame: np.ndarray, artwork: np.ndarray, H: np.ndarray,
                  config: LocalisationConfig) -> np.ndarray:
    """Single overlaid frame without occlusion handling.

    Warp the artwork with the composed homography (``artwork_to_frame``) and
    paint it where the coverage mask is set. Returns an RGB float32 frame.
    """
    # YOUR CODE HERE
    out = to_float(frame).copy()
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=-1)
    art = to_float(artwork)
    if art.ndim == 2:
        art = np.repeat(art[..., None], 3, axis=-1)
    art = art[..., :3]
    Ha = artwork_to_frame(art, H, config)
    warped, cov = warp_artwork(art, Ha, out.shape)
    if not cov.any():
        return out.astype(np.float32)
    if config.warp.match_gain:
        # optional: scale the artwork so that its mean brightness matches the board it replaces
        gain = float(np.clip(to_gray(out)[cov].mean() / max(to_gray(warped)[cov].mean(), 1e-6), 0.5, 2.0))
        warped = np.clip(warped * gain, 0.0, 1.0)
    alpha = cov.astype(np.float32)
    if config.warp.feather > 0:
        # soft edge: the alpha ramps from 0 at the quadrilateral edge to 1 about ``feather`` px inside
        alpha = np.clip(2.0 * gaussian_smooth(alpha, config.warp.feather / 2.0) - 1.0, 0.0, 1.0) * cov
    a = alpha[..., None]
    out = a * warped + (1.0 - a) * out
    return out.astype(np.float32)


def evaluate_audit_frames(reference: np.ndarray, frames: Dict[int, np.ndarray],
                          gt_corners: Dict[int, np.ndarray], config: LocalisationConfig) -> Dict:
    """Run ``localise_board`` on annotated frames and summarise with ``metrics``.

    ``frames`` and ``gt_corners`` map frame index -> image / 4 x 2 corners.
    Returns a dict with ``per_frame`` (index -> dict of error, n_matches,
    n_inliers, inlier_ratio, median_residual, status, corners) and ``summary``
    from ``metrics.summarise_errors`` using ``config.success_threshold``.
    """
    # YOUR CODE HERE
    per_frame = {}
    for f in sorted(gt_corners):
        if f not in frames:
            continue
        H, corners, _, _, diag = localise_board(reference, frames[f], config)
        ok = bool(diag["status"]) and H is not None
        err = metrics.corner_error(corners, gt_corners[f]) if ok else float("inf")
        per_frame[int(f)] = {
            "error": float(err),
            "n_matches": diag["n_matches"],
            "n_inliers": diag["n_inliers"],
            "inlier_ratio": diag["inlier_ratio"],
            "median_residual": diag["median_residual"],
            "status": ok,
            "corners": corners,
        }
    summary = metrics.summarise_errors([r["error"] for r in per_frame.values()], config.success_threshold)
    return {"per_frame": per_frame, "summary": summary}


__all__ = ["localise_board", "artwork_to_frame", "overlay_frame", "evaluate_audit_frames"]
