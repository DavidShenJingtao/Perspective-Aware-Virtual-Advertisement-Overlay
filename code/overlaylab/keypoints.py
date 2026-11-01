"""Stage 1 Localise, keypoint branch.

Implement the four functions marked ``YOUR CODE HERE``. Do not change the
signatures or return types; the tests and the grading harness call them.
Use only numpy and the helpers in ``overlaylab.supplied`` (Sobel gradients,
Gaussian smoothing, bilinear sampling). OpenCV feature functions
(``cv2.cornerHarris``, ``cv2.SIFT`` and friends) are not allowed here; the
SIFT baseline lives only in the labelled ablation cell of Notebook 1.

Conventions: grayscale images are float32 in [0, 1]; ``x`` indexes columns and
``y`` rows; keypoints are N x 2 arrays of (x, y); descriptors are N x 128.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import DetectorConfig, DescriptorConfig, MatchConfig
from .supplied import sobel_gradients, gaussian_smooth
from .geometry import bilinear_sample


def harris_response(gray: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """Harris response R = det(M) - kappa * tr(M)^2 at every pixel.

    M is the Gaussian-weighted (scale ``config.sigma_w``) second-moment matrix
    built from the supplied normalised Sobel gradients. Returns a float32 map
    with the same shape as ``gray``.
    """
    # YOUR CODE HERE
    gx, gy, _, _ = sobel_gradients(gray)
    gx = gx.astype(np.float64); gy = gy.astype(np.float64)
    # Gaussian-weighted second-moment matrix M = w * [[Ix^2, IxIy], [IxIy, Iy^2]]
    sxx = gaussian_smooth(gx * gx, config.sigma_w).astype(np.float64)
    syy = gaussian_smooth(gy * gy, config.sigma_w).astype(np.float64)
    sxy = gaussian_smooth(gx * gy, config.sigma_w).astype(np.float64)
    det = sxx * syy - sxy * sxy
    trace = sxx + syy
    return (det - config.kappa * trace * trace).astype(np.float32)


def _max_filter(a: np.ndarray, r: int) -> np.ndarray:
    """(2r+1) x (2r+1) maximum filter (separable), -inf padding outside the image."""
    if r <= 0:
        return a.copy()
    from numpy.lib.stride_tricks import sliding_window_view
    p = np.pad(a, ((0, 0), (r, r)), constant_values=-np.inf)
    rows = sliding_window_view(p, 2 * r + 1, axis=1).max(axis=-1)
    p = np.pad(rows, ((r, r), (0, 0)), constant_values=-np.inf)
    return sliding_window_view(p, 2 * r + 1, axis=0).max(axis=-1)


def detect_corners(gray: np.ndarray, config: DetectorConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Harris corners with non-maximum suppression.

    Steps: compute ``harris_response``; discard responses below
    ``config.response_fraction * max(R)``; apply NMS in mode ``config.nms_mode``
    with radius ``config.nms_radius`` ("window": keep a pixel only if it is the
    maximum of its (2r+1)x(2r+1) neighbourhood; "radius": sort survivors by R
    and greedily accept the strongest while discarding weaker candidates within
    distance r); drop corners within ``config.border`` pixels of the image edge.

    Returns (keypoints, responses): an N x 2 float array of (x, y) ordered by
    decreasing response (N <= config.max_corners) and the matching responses.
    """
    # YOUR CODE HERE
    if config.nms_mode not in ("window", "radius"):
        raise ValueError(f"unknown nms_mode {config.nms_mode!r}; use 'window' or 'radius'")
    empty = (np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float32))
    R = harris_response(gray, config).astype(np.float64)
    h, w = R.shape
    r_max = float(R.max()) if R.size else 0.0
    if r_max <= 0:
        return empty
    # corners only: positive responses above the relative threshold
    cand = (R > 0) & (R >= config.response_fraction * r_max)
    # the border is excluded before suppression so that a response next to the image edge
    # (where gradients see the reflected padding) cannot suppress an interior corner
    b = int(config.border)
    if b > 0:
        cand[:b, :] = False; cand[h - b:, :] = False
        cand[:, :b] = False; cand[:, w - b:] = False
    r = int(config.nms_radius)

    if config.nms_mode == "window":
        Rc = np.where(cand, R, -np.inf)
        keep = cand & (Rc >= _max_filter(Rc, r))
        ys, xs = np.nonzero(keep)
        resp = R[ys, xs]
        order = np.argsort(-resp, kind="stable")[:config.max_corners]
        ys, xs, resp = ys[order], xs[order], resp[order]
    else:
        ys, xs = np.nonzero(cand)
        resp = R[ys, xs]
        order = np.argsort(-resp, kind="stable")
        ys, xs, resp = ys[order], xs[order], resp[order]
        # greedy acceptance: an occupancy map marks the disc of radius r around every accepted corner
        suppressed = np.zeros((h, w), dtype=bool)
        dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
        disc = (dx * dx + dy * dy) <= r * r
        acc = []
        for k in range(len(ys)):
            y, x = ys[k], xs[k]
            if suppressed[y, x]:
                continue
            acc.append(k)
            if len(acc) >= config.max_corners:
                break
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            suppressed[y0:y1, x0:x1] |= disc[y0 - (y - r):y1 - (y - r), x0 - (x - r):x1 - (x - r)]
        acc = np.asarray(acc, dtype=np.int64)
        ys, xs, resp = ys[acc], xs[acc], resp[acc]

    if len(ys) == 0:
        return empty
    kps = np.stack([xs, ys], axis=1).astype(np.float64)
    return kps, resp.astype(np.float32)


# 16 x 16 sample grid in units of the sample spacing, centred on the keypoint: -7.5 ... 7.5
_GRID = np.arange(16, dtype=np.float64) - 7.5


def describe_keypoints(gray: np.ndarray, keypoints: np.ndarray,
                       config: DescriptorConfig) -> Tuple[np.ndarray, np.ndarray]:
    """128-D gradient-histogram descriptors with dominant-orientation normalisation.

    For each keypoint take a 16 x 16 grid of samples spaced ``config.patch_scale``
    pixels apart (so the window is 16 * patch_scale pixels wide),
    build a ``config.orientation_bins``-bin orientation histogram weighted by
    gradient magnitude, assign the dominant orientation, rotate the sampling
    grid so that the first histogram bin is aligned to it, and accumulate a
    ``cells x cells`` grid of ``cell_bins``-bin histograms (4 x 4 x 8 = 128).
    Apply Gaussian spatial weighting when ``config.gaussian_weighting`` is set,
    normalise to unit L2 norm, clamp to ``config.clamp``, and renormalise.

    Keypoints whose (rotated) window leaves the image are dropped. Returns
    (descriptors N' x 128, kept_indices N') where kept_indices indexes the input.
    """
    # YOUR CODE HERE
    cells, cbins, obins = int(config.cells), int(config.cell_bins), int(config.orientation_bins)
    dim = cells * cells * cbins
    kps = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
    if len(kps) == 0:
        return np.zeros((0, dim), dtype=np.float32), np.zeros(0, dtype=np.int64)
    g = np.asarray(gray, dtype=np.float32)
    h, w = g.shape
    s = float(config.patch_scale)
    if s > 1.0:                                    # coarser sampling: pre-smooth against aliasing
        g = gaussian_smooth(g, 0.5 * s)
    gx, gy, _, _ = sobel_gradients(g)
    half = 8.0 * s                                 # half-width of the 16 * s window
    n = len(kps)

    u, v = np.meshgrid(_GRID, _GRID)               # u along x (columns), v along y (rows), grid units
    u = u.ravel(); v = v.ravel()                   # 256 samples
    weight = (np.exp(-(u * u + v * v) / (2.0 * 8.0 ** 2)) if config.gaussian_weighting
              else np.ones_like(u))                # sigma = half the window width

    def inside(points_x, points_y):
        return ((points_x >= 0) & (points_x <= w - 1) & (points_y >= 0) & (points_y <= h - 1)).all(axis=1)

    # 1. dominant orientation from the axis-aligned window
    ok = inside(kps[:, :1] + np.array([-half, half]), kps[:, 1:] + np.array([-half, half]))
    px = kps[:, :1] + s * u; py = kps[:, 1:] + s * v                               # n x 256
    sgx = bilinear_sample(gx, py, px).astype(np.float64)
    sgy = bilinear_sample(gy, py, px).astype(np.float64)
    mag = np.hypot(sgx, sgy) * weight
    ang = np.mod(np.arctan2(sgy, sgx), 2 * np.pi)
    ob = ang / (2 * np.pi) * obins                                                   # soft binning
    b0 = np.floor(ob).astype(np.int64) % obins; f = ob - np.floor(ob)
    rows = np.repeat(np.arange(n), u.size)
    hist = np.zeros((n, obins))
    np.add.at(hist, (rows, b0.ravel()), (mag * (1 - f)).ravel())
    np.add.at(hist, (rows, ((b0 + 1) % obins).ravel()), (mag * f).ravel())
    hist = (np.roll(hist, 1, axis=1) + 2 * hist + np.roll(hist, -1, axis=1)) / 4.0   # circular smoothing
    peak = np.argmax(hist, axis=1)
    left = hist[np.arange(n), (peak - 1) % obins]; centre = hist[np.arange(n), peak]
    right = hist[np.arange(n), (peak + 1) % obins]
    denom = left - 2 * centre + right
    offset = np.where(np.abs(denom) > 1e-12, 0.5 * (left - right) / np.where(denom == 0, 1, denom), 0.0)
    theta = (peak + offset) * (2 * np.pi / obins)          # soft binning puts angle k*2pi/obins on bin k
    c, sn = np.cos(theta)[:, None], np.sin(theta)[:, None]

    # 2. rotated grid: u along the dominant direction, v perpendicular (proper rotation, y down)
    rx = kps[:, :1] + s * (c * u - sn * v)
    ry = kps[:, 1:] + s * (sn * u + c * v)
    corner_u = np.array([-half, half, half, -half]); corner_v = np.array([-half, -half, half, half])
    ok &= inside(kps[:, :1] + c * corner_u - sn * corner_v, kps[:, 1:] + sn * corner_u + c * corner_v)
    rgx = bilinear_sample(gx, ry, rx).astype(np.float64)
    rgy = bilinear_sample(gy, ry, rx).astype(np.float64)
    rmag = np.hypot(rgx, rgy) * weight
    rang = np.mod(np.arctan2(rgy, rgx) - theta[:, None], 2 * np.pi)    # orientation relative to theta

    # 3. trilinear accumulation into cells x cells x cell_bins
    cell = 16.0 / cells
    cu = (u + 7.5 + 0.5) / cell - 0.5; cv = (v + 7.5 + 0.5) / cell - 0.5          # continuous cell coords
    cu0 = np.floor(cu).astype(np.int64); cv0 = np.floor(cv).astype(np.int64)
    fu = cu - cu0; fv = cv - cv0
    oo = rang / (2 * np.pi) * cbins
    o0 = np.floor(oo).astype(np.int64) % cbins; fo = oo - np.floor(oo)
    desc = np.zeros((n, cells, cells, cbins))
    for dv_, wv in ((0, 1 - fv), (1, fv)):
        iv = cv0 + dv_
        for du_, wu in ((0, 1 - fu), (1, fu)):
            iu = cu0 + du_
            valid = (iu >= 0) & (iu < cells) & (iv >= 0) & (iv < cells)
            if not valid.any():
                continue
            wsp = (wu * wv * valid)[None, :]
            ivc = np.clip(iv, 0, cells - 1); iuc = np.clip(iu, 0, cells - 1)
            for do_, wo in ((0, 1 - fo), (1, fo)):
                ib = (o0 + do_) % cbins
                np.add.at(desc, (rows, np.tile(ivc, n), np.tile(iuc, n), ib.ravel()),
                          (rmag * wsp * wo).ravel())
    desc = desc.reshape(n, dim)

    # 4. normalise, clamp, renormalise
    norm = np.linalg.norm(desc, axis=1)
    ok &= norm > 1e-12                              # textureless window: nothing to describe
    desc = desc[ok] / norm[ok, None]
    desc = np.minimum(desc, config.clamp)
    desc /= np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-12)
    return desc.astype(np.float32), np.nonzero(ok)[0]


def match_descriptors(desc_a: np.ndarray, desc_b: np.ndarray,
                      config: MatchConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour matching with Lowe's ratio test (optionally mutual).

    For every descriptor in A find the nearest and second-nearest descriptor in
    B by Euclidean distance; accept the match when d1 / d2 < ``config.ratio``.
    With ``config.mutual`` the match must also be the nearest neighbour of the B
    descriptor in A. Returns (pairs M x 2 of (index_a, index_b), distances M,
    ratios M).
    """
    # YOUR CODE HERE
    A = np.asarray(desc_a, dtype=np.float64); B = np.asarray(desc_b, dtype=np.float64)
    empty = (np.zeros((0, 2), dtype=np.int64), np.zeros(0), np.zeros(0))
    if A.ndim != 2 or B.ndim != 2 or len(A) == 0 or len(B) == 0:
        return empty
    # squared distances |a|^2 + |b|^2 - 2 a.b, clipped against round-off
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    D = np.sqrt(np.maximum(d2, 0.0))
    ia = np.arange(len(A))
    if len(B) >= 2:
        two = np.argpartition(D, 1, axis=1)[:, :2]
        da, db = D[ia, two[:, 0]], D[ia, two[:, 1]]
        swap = db < da
        nn = np.where(swap, two[:, 1], two[:, 0])
        d1 = np.minimum(da, db); d2nd = np.maximum(da, db)
    else:                                           # a single candidate: no second neighbour to compare
        nn = np.zeros(len(A), dtype=np.int64); d1 = D[:, 0]; d2nd = np.full(len(A), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d2nd > 0, d1 / d2nd, np.where(d1 > 0, np.inf, 1.0))
    accept = ratio < config.ratio
    if config.mutual:
        back = np.argmin(D, axis=0)                 # nearest A descriptor of every B descriptor
        accept &= back[nn] == ia
    sel = np.nonzero(accept)[0]
    pairs = np.stack([sel, nn[sel]], axis=1).astype(np.int64)
    return pairs, d1[sel], ratio[sel]


__all__ = ["harris_response", "detect_corners", "describe_keypoints", "match_descriptors"]
