"""Stage 1 Localise, geometry branch.

Implement the functions marked ``YOUR CODE HERE``. Allowed: numpy (including
``numpy.linalg.svd`` and ``numpy.linalg.inv``) and your Lab 1
``bilinear_sample`` (pasted into this file). ``cv2.findHomography`` and ``cv2.warpPerspective``
may appear only in the labelled sanity-check and comparison cells of Notebook 1.

Conventions: points are N x 2 arrays of (x, y) with x indexing columns; a
homography H maps source homogeneous coordinates to destination ones,
x' ~ H x, and is scaled so that H[2, 2] = 1 whenever that entry is not close
to zero. Pixel centres sit at integer coordinates.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import RansacConfig


def bilinear_sample(image: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sample a 2-D or H x W x C image at floating-point image coordinates (reuse Lab 1).

    Same signature as Lab 1 (``y`` first, then ``x``); extend it so that a
    colour image returns an array with a trailing channel axis. Coordinates
    are clamped to the valid closed interval, and an exact integer coordinate
    reproduces the corresponding pixel. ``y`` and ``x`` may be arrays of any
    matching shape so that a whole grid can be sampled at once.
    """
    # YOUR CODE HERE
    img = np.asarray(image)
    h, w = img.shape[:2]
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, h - 1)
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, w - 1)
    # integer neighbours; the upper one is clamped so that y = h-1 (or x = w-1) is exact
    y0 = np.floor(y).astype(np.int64); x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1); x1 = np.minimum(x0 + 1, w - 1)
    wy = y - y0; wx = x - x0
    if img.ndim == 3:                       # broadcast the weights over the channel axis
        wy = wy[..., None]; wx = wx[..., None]
    f = img.astype(np.float64, copy=False)
    top = (1.0 - wx) * f[y0, x0] + wx * f[y0, x1]
    bottom = (1.0 - wx) * f[y1, x0] + wx * f[y1, x1]
    out = (1.0 - wy) * top + wy * bottom
    return out.astype(np.float32)


def rectangle_to_rectangle(src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> np.ndarray:
    """Supplied helper: affine scaling that maps a (w, h) pixel rectangle onto another.

    Corner (w-1, h-1) of the source maps to corner (w'-1, h'-1) of the
    destination (pixel-centre convention). Used to compose the artwork
    rectangle with the board-to-frame homography.
    """
    sw, sh = src_size
    dw, dh = dst_size
    return np.array([[(dw - 1) / max(sw - 1, 1), 0, 0],
                     [0, (dh - 1) / max(sh - 1, 1), 0],
                     [0, 0, 1.0]], dtype=np.float64)


def normalise_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation.

    Returns (T, normalised) where T is the 3 x 3 similarity that translates the
    points to zero centroid and scales them so that the mean distance from the
    origin is sqrt(2), and ``normalised`` are the transformed N x 2 points.
    """
    # YOUR CODE HERE
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    centroid = p.mean(axis=0)
    mean_dist = np.linalg.norm(p - centroid, axis=1).mean()
    s = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0   # coincident points: translate only
    T = np.array([[s, 0.0, -s * centroid[0]],
                  [0.0, s, -s * centroid[1]],
                  [0.0, 0.0, 1.0]])
    normalised = (p - centroid) * s
    return T, normalised


def dlt_homography(src: np.ndarray, dst: np.ndarray
                   ) -> Tuple[Optional[np.ndarray], np.ndarray, bool]:
    """Normalised direct linear transform.

    Builds the 2n x 9 system A h = 0 from normalised correspondences, solves
    for h as the right singular vector with the smallest singular value,
    denormalises H = T'^{-1} H_hat T, and scales so that H[2, 2] = 1 when that
    entry is not close to zero.

    Raises ``ValueError`` for fewer than four correspondences. Returns
    (H, singular_values, status); when A is rank deficient (e.g. collinear
    source points) ``status`` is False and ``H`` is None rather than a
    numerically meaningless matrix. ``singular_values`` are those of A, so
    that conditioning can be inspected.
    """
    # YOUR CODE HERE
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) != len(dst):
        raise ValueError("src and dst must have the same number of points")
    n = len(src)
    if n < 4:
        raise ValueError(f"a homography needs at least four correspondences, got {n}")
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        return None, np.full(min(2 * n, 9), np.nan), False

    T, s_n = normalise_points(src)
    Tp, d_n = normalise_points(dst)
    x, y = s_n[:, 0], s_n[:, 1]
    xp, yp = d_n[:, 0], d_n[:, 1]
    zeros, ones = np.zeros(n), np.ones(n)
    # two rows per correspondence of x' x H x = 0:
    #   [-x -y -1  0  0  0  x'x x'y x'] h = 0
    #   [ 0  0  0 -x -y -1  y'x y'y y'] h = 0
    A = np.empty((2 * n, 9))
    A[0::2] = np.stack([-x, -y, -ones, zeros, zeros, zeros, xp * x, xp * y, xp], axis=1)
    A[1::2] = np.stack([zeros, zeros, zeros, -x, -y, -ones, yp * x, yp * y, yp], axis=1)

    # full_matrices=True so that Vt is 9 x 9 even for the minimal 8 x 9 system
    _, svals, vt = np.linalg.svd(A, full_matrices=True)
    # A must have rank 8 for a unique (up to scale) solution: the 8th singular value must not vanish
    rank_tol = 1e-8
    if svals[0] <= 0 or svals[7] / svals[0] < rank_tol:
        return None, svals, False

    H_hat = vt[-1].reshape(3, 3)
    # a (near-)singular H_hat maps the plane onto a line: not a usable homography
    sv_h = np.linalg.svd(H_hat, compute_uv=False)
    if sv_h[-1] / sv_h[0] < 1e-10:
        return None, svals, False

    H = np.linalg.inv(Tp) @ H_hat @ T            # denormalise: H = T'^-1 H_hat T
    if abs(H[2, 2]) > 1e-12 * np.abs(H).max():
        H = H / H[2, 2]
    else:
        H = H / np.abs(H).max()
    if not np.all(np.isfinite(H)):
        return None, svals, False
    return H, svals, True


def transfer_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to N x 2 points and dehomogenise: pi(H x)."""
    # YOUR CODE HERE
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    Hm = np.asarray(H, dtype=np.float64)
    hom = p @ Hm[:, :2].T + Hm[:, 2]          # (H [x y 1]^T)^T for every row
    with np.errstate(divide="ignore", invalid="ignore"):
        return hom[:, :2] / hom[:, 2:3]


def symmetric_transfer_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-correspondence e_i = |x' - pi(H x)|^2 + |x - pi(H^-1 x')|^2."""
    # YOUR CODE HERE
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    try:
        H_inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    except np.linalg.LinAlgError:
        return np.full(len(src), np.inf)
    forward = np.sum((dst - transfer_points(H, src)) ** 2, axis=1)
    backward = np.sum((src - transfer_points(H_inv, dst)) ** 2, axis=1)
    e = forward + backward
    e[~np.isfinite(e)] = np.inf                 # points mapped to infinity are never inliers
    return e


def _three_collinear(p: np.ndarray, tol: float = 1e-6) -> bool:
    """Supplied helper: True if any three of the four points are (nearly) collinear."""
    from itertools import combinations
    scale = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), 1.0)
    for i, j, k in combinations(range(len(p)), 3):
        a, b, c = p[i], p[j], p[k]
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if area2 < tol * scale * scale:
            return True
    return False


def ransac_homography(src: np.ndarray, dst: np.ndarray, config: RansacConfig
                      ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, int, bool]:
    """Robust homography with adaptive RANSAC and a final refit on all inliers.

    Each iteration draws four correspondences with ``config.rng()``, rejects
    the sample if any three source points are collinear, fits
    ``dlt_homography``, and scores every correspondence by the symmetric
    transfer error; correspondences under ``config.threshold`` are inliers.
    Keep the model with the most inliers (ties: smaller sum of inlier
    residuals); update N = log(1-p) / log(1-(1-eps)^4) from the current inlier
    fraction, capped by ``config.max_iterations``; refit on all inliers.

    Returns (H, inlier_mask, residuals, iterations_used, status) where status
    is False when no model reached ``config.min_inliers`` (H may then be None).
    """
    # YOUR CODE HERE
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    n = len(src)
    if n < 4:
        return None, np.zeros(n, dtype=bool), np.full(n, np.inf), 0, False

    rng = config.rng()
    thr = float(config.threshold)
    log_fail = np.log(max(1.0 - float(config.confidence), 1e-300))
    max_iter = max(1, int(config.max_iterations))

    best_H, best_resid = None, np.full(n, np.inf)
    best_count, best_cost = -1, np.inf
    n_required = max_iter
    it = 0
    while it < min(n_required, max_iter):
        it += 1
        sample = rng.choice(n, size=4, replace=False)
        if _three_collinear(src[sample]):
            continue                                   # degenerate minimal sample
        H, _, ok = dlt_homography(src[sample], dst[sample])
        if not ok:
            continue
        resid = symmetric_transfer_error(H, src, dst)
        inl = resid < thr
        count = int(inl.sum())
        cost = float(resid[inl].sum())
        if count > best_count or (count == best_count and cost < best_cost):
            best_H, best_resid, best_count, best_cost = H, resid, count, cost
            # adaptive termination from the inlier fraction of the best model so far
            w4 = (count / n) ** 4                      # (1 - eps)^4
            if w4 >= 1.0 - 1e-12:
                n_required = it                        # every point is an inlier: stop now
            elif w4 > 0:
                n_required = int(np.ceil(log_fail / np.log(1.0 - w4)))

    if best_H is None:
        return None, np.zeros(n, dtype=bool), best_resid, it, False

    # final least-squares refit on all inliers, repeated while the inlier set grows (consensus refinement)
    H, resid = best_H, best_resid
    mask = resid < thr
    for _ in range(5):
        if mask.sum() < 4:
            break
        H_new, _, ok = dlt_homography(src[mask], dst[mask])
        if not ok:
            break
        resid_new = symmetric_transfer_error(H_new, src, dst)
        mask_new = resid_new < thr
        if mask_new.sum() < mask.sum():
            break                                      # the refit lost support: keep the previous model
        converged = np.array_equal(mask_new, mask)
        H, resid, mask = H_new, resid_new, mask_new
        if converged:
            break

    status = bool(mask.sum() >= config.min_inliers)
    return H, mask, resid, it, status


def warp_artwork(artwork: np.ndarray, H: np.ndarray, frame_shape: Tuple[int, ...]
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Destination-to-source warp of ``artwork`` into a frame of shape ``frame_shape``.

    ``H`` maps artwork pixel coordinates to frame pixel coordinates. For every
    frame pixel inside the projected artwork quadrilateral, map it back through
    H^{-1} and sample the artwork bilinearly. Returns (warped H x W x C float32,
    coverage H x W bool); pixels that map outside the artwork receive no
    coverage and are zero in ``warped``.
    """
    # YOUR CODE HERE
    art = np.asarray(artwork)
    if art.dtype == np.uint8:
        art = art.astype(np.float32) / 255.0
    if art.ndim == 2:
        art = art[..., None]
    ah, aw, ch = art.shape
    fh, fw = int(frame_shape[0]), int(frame_shape[1])
    warped = np.zeros((fh, fw, ch), dtype=np.float32)
    coverage = np.zeros((fh, fw), dtype=bool)
    H = np.asarray(H, dtype=np.float64)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return warped, coverage

    # only frame pixels inside the bounding box of the projected quadrilateral can be covered
    corners = np.array([[0, 0], [aw - 1, 0], [aw - 1, ah - 1], [0, ah - 1]], dtype=np.float64)
    w_c = corners @ H[2, :2] + H[2, 2]
    if np.all(w_c > 0):
        proj = transfer_points(H, corners)
        x_lo = max(0, int(np.floor(proj[:, 0].min())) - 1); x_hi = min(fw - 1, int(np.ceil(proj[:, 0].max())) + 1)
        y_lo = max(0, int(np.floor(proj[:, 1].min())) - 1); y_hi = min(fh - 1, int(np.ceil(proj[:, 1].max())) + 1)
    else:                                               # board crosses the camera plane: scan the whole frame
        x_lo, x_hi, y_lo, y_hi = 0, fw - 1, 0, fh - 1
    if x_lo > x_hi or y_lo > y_hi:
        return warped, coverage

    ys, xs = np.mgrid[y_lo:y_hi + 1, x_lo:x_hi + 1].astype(np.float64)
    # destination -> source: (u, v, w)^T = H^-1 (x, y, 1)^T
    u = H_inv[0, 0] * xs + H_inv[0, 1] * ys + H_inv[0, 2]
    v = H_inv[1, 0] * xs + H_inv[1, 1] * ys + H_inv[1, 2]
    w = H_inv[2, 0] * xs + H_inv[2, 1] * ys + H_inv[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = u / w; sy = v / w
    tol = 1e-6                                          # absorb round-off at the exact artwork border
    inside = (w > 0) & (sx >= -tol) & (sx <= aw - 1 + tol) & (sy >= -tol) & (sy <= ah - 1 + tol)
    if not inside.any():
        return warped, coverage
    samples = bilinear_sample(art, sy[inside], sx[inside])          # clamps the tolerance band
    box = warped[y_lo:y_hi + 1, x_lo:x_hi + 1]
    box[inside] = samples.reshape(-1, ch)
    coverage[y_lo:y_hi + 1, x_lo:x_hi + 1] = inside
    return warped, coverage


__all__ = ["bilinear_sample", "rectangle_to_rectangle", "normalise_points", "dlt_homography", "transfer_points",
           "symmetric_transfer_error", "ransac_homography", "warp_artwork"]
