"""Stage 2 Track: optical flow, temporal variants, and failure monitoring.

Implement the functions marked ``YOUR CODE HERE``. ``cv2.calcOpticalFlowPyrLK``
and ``cv2.goodFeaturesToTrack`` may appear only in the labelled comparison
cell of Notebook 2. Grayscale images are float32 in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import FlowConfig, MonitorConfig, TrackConfig, LocalisationConfig
from .supplied import to_gray, sobel_gradients, gaussian_pyramid, quad_mask, rect_corners
from .geometry import ransac_homography, transfer_points, bilinear_sample
from .localisation import localise_board

OK, REINIT, HIDE = "OK", "REINIT", "HIDE"


def lk_flow(prev_gray: np.ndarray, next_gray: np.ndarray, points: np.ndarray, config: FlowConfig
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Iterative single-level Lucas-Kanade at sparse points.

    For each point use a (2w+1) x (2w+1) window (``config.window``), the
    supplied Sobel gradients of ``prev_gray``, bilinear sampling of the warped
    window in ``next_gray`` at every iteration, and at most
    ``config.iterations`` updates (stop early when the update is below
    ``config.convergence``). The structure tensor is normalised by the number
    of window pixels so that ``config.min_eigenvalue`` is scale-free.

    Returns (displacements N x 2, eigenvalues N x 2 with lambda_1 >= lambda_2,
    residuals N (mean absolute brightness residual), valid N bool). A point is
    invalid when lambda_2 < config.min_eigenvalue, the residual exceeds
    config.max_residual, the update failed to converge, or the point leaves
    the image. Coarse-to-fine estimation over ``config.pyramid_levels`` is an
    optional extension.
    """
    prev = np.asarray(prev_gray, dtype=np.float32)
    nxt = np.asarray(next_gray, dtype=np.float32)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if prev.ndim != 2 or nxt.ndim != 2 or prev.shape != nxt.shape:
        raise ValueError("prev_gray and next_gray must be same-shape 2-D images")

    n = len(pts)
    displacements = np.zeros((n, 2), dtype=np.float64)
    eigenvalues = np.zeros((n, 2), dtype=np.float64)
    residuals = np.full(n, np.inf, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    if n == 0:
        return displacements, eigenvalues, residuals, valid

    gx, gy, _, _ = sobel_gradients(prev)
    w = int(config.window)
    offsets_y, offsets_x = np.mgrid[-w:w + 1, -w:w + 1]
    offsets_x = offsets_x.astype(np.float64)
    offsets_y = offsets_y.astype(np.float64)
    height, width = prev.shape

    def window_inside(x_grid: np.ndarray, y_grid: np.ndarray) -> bool:
        return bool(x_grid.min() >= 0.0 and x_grid.max() <= width - 1 and
                    y_grid.min() >= 0.0 and y_grid.max() <= height - 1)

    for i, (x, y) in enumerate(pts):
        sample_x = x + offsets_x
        sample_y = y + offsets_y
        if not np.isfinite(x + y) or not window_inside(sample_x, sample_y):
            continue

        prev_patch = bilinear_sample(prev, sample_y, sample_x).astype(np.float64)
        ix = bilinear_sample(gx, sample_y, sample_x).astype(np.float64)
        iy = bilinear_sample(gy, sample_y, sample_x).astype(np.float64)

        # The eigenvalue threshold is defined for the mean (normalised)
        # second-moment matrix, while the unnormalised matrix is convenient
        # for solving the normal equations.
        G = np.array([[np.sum(ix * ix), np.sum(ix * iy)],
                      [np.sum(ix * iy), np.sum(iy * iy)]], dtype=np.float64)
        eig = np.linalg.eigvalsh(G / ix.size)[::-1]
        eigenvalues[i] = eig

        d = np.zeros(2, dtype=np.float64)
        converged = False
        stayed_inside = True
        if eig[1] >= float(config.min_eigenvalue):
            for _ in range(int(config.iterations)):
                warped_x = sample_x + d[0]
                warped_y = sample_y + d[1]
                if not window_inside(warped_x, warped_y):
                    stayed_inside = False
                    break

                warped = bilinear_sample(nxt, warped_y, warped_x).astype(np.float64)
                brightness_error = warped - prev_patch
                rhs = -np.array([np.sum(ix * brightness_error),
                                 np.sum(iy * brightness_error)])
                try:
                    update = np.linalg.solve(G, rhs)
                except np.linalg.LinAlgError:
                    stayed_inside = False
                    break
                if not np.all(np.isfinite(update)):
                    stayed_inside = False
                    break

                d += update
                if np.linalg.norm(update) < float(config.convergence):
                    converged = True
                    break

        displacements[i] = d
        final_x = sample_x + d[0]
        final_y = sample_y + d[1]
        if window_inside(final_x, final_y):
            final_patch = bilinear_sample(nxt, final_y, final_x)
            residuals[i] = float(np.mean(np.abs(final_patch - prev_patch)))

        valid[i] = bool(eig[1] >= float(config.min_eigenvalue) and
                        stayed_inside and converged and
                        residuals[i] <= float(config.max_residual))

    return displacements, eigenvalues, residuals, valid


@dataclass
class TrackInit:
    """What ``track_board`` needs to (re)initialise from the reference image."""
    reference: np.ndarray
    localisation: LocalisationConfig


@dataclass
class TrackResult:
    homographies: List[Optional[np.ndarray]]
    corners: List[np.ndarray]                 # 4 x 2 per frame (NaN when lost)
    status: List[str]                          # "ok" or "lost"
    stats: List[Dict]                          # consumed by failure_monitor
    reinit_frames: List[int] = field(default_factory=list)
    flow: List[Dict] = field(default_factory=list)  # per-frame vectors, eigenvalues, validity


def _frame_stats(prev_corners, corners, gray_prev, gray, inlier_count, inlier_ratio,
                 median_residual, mean_flow, n_valid, status, n_matches=None):
    """Supplied helper: assemble the per-frame statistics dictionary (``n_matches`` on (re)initialised frames)."""
    disp = float(np.mean(np.linalg.norm(corners - prev_corners, axis=1))) if (
        prev_corners is not None and np.all(np.isfinite(corners)) and np.all(np.isfinite(prev_corners))) else float("nan")
    if gray_prev is not None and np.all(np.isfinite(corners)):
        m = quad_mask(gray.shape, corners)
        bright = float(np.mean(np.abs(gray[m] - gray_prev[m]))) if m.any() else float("nan")
    else:
        bright = float("nan")
    return {"inlier_count": int(inlier_count), "inlier_ratio": float(inlier_ratio),
            "median_residual": float(median_residual), "corner_displacement": disp,
            "mean_flow": float(mean_flow), "brightness_change": bright,
            "n_valid_points": int(n_valid), "status": status,
            "n_matches": (int(n_matches) if n_matches is not None else None)}


def track_board(frames: Sequence[np.ndarray], init: TrackInit, config: TrackConfig) -> TrackResult:
    """Track the board through a sequence with one of two variants.

    ``config.variant == "independent"``: run ``localise_board`` on every frame
    from the reference image.

    ``config.variant == "flow_assisted"``: localise frame 0, then propagate the
    inlier correspondences (or the corners, ``config.propagate``) to the next
    frame with ``lk_flow``, drop invalid points, re-estimate the homography
    with RANSAC between the reference points and the propagated points, and
    reinitialise from the reference image when fewer than
    ``config.reinit_min_points`` valid points remain, every
    ``config.reinit_every`` frames (0 disables), or when RANSAC fails.

    Optional corner propagation has four points only: use reinit_min_points <= 4,
    RANSAC min_inliers = 4 and monitor min_inlier_count <= 4 if selecting it.

    Both variants return a ``TrackResult`` with a homography, projected
    corners, a status flag and the statistics the failure monitor consumes
    for every frame (build them with the supplied ``_frame_stats``; use
    ``inlier_ratio`` = inliers / valid propagated points for the flow variant
    and ``median_residual`` = median over inliers of sqrt(e_i / 2) px). The
    flow-assisted variant also stores, for every frame after the first, a dict
    in ``TrackResult.flow[t]`` with keys ``points`` (tracked points N x 2),
    ``vectors`` (displacements N x 2), ``eigenvalues`` (N x 2), ``residuals``
    (N) and ``valid`` (N bool); frames that were (re)initialised store {}.
    """
    if config.variant not in {"independent", "flow_assisted"}:
        raise ValueError(f"unknown tracking variant: {config.variant!r}")
    if config.propagate not in {"inliers", "corners"}:
        raise ValueError(f"unknown propagation mode: {config.propagate!r}")

    homographies: List[Optional[np.ndarray]] = []
    corners_out: List[np.ndarray] = []
    statuses: List[str] = []
    stats: List[Dict] = []
    reinit_frames: List[int] = []
    flow_records: List[Dict] = []

    ref_h, ref_w = init.reference.shape[:2]
    reference_corners = rect_corners(ref_w, ref_h)
    gray_prev: Optional[np.ndarray] = None
    previous_corners: Optional[np.ndarray] = None
    reference_points: Optional[np.ndarray] = None
    frame_points: Optional[np.ndarray] = None

    def initialise(frame: np.ndarray, gray: np.ndarray, t: int) -> Tuple[
            Optional[np.ndarray], np.ndarray, str, Dict,
            Optional[np.ndarray], Optional[np.ndarray]]:
        H, corners, pairs, inliers, diag = localise_board(
            init.reference, frame, init.localisation)
        ok = bool(diag["status"]) and H is not None
        status = "ok" if ok else "lost"

        new_reference_points: Optional[np.ndarray] = None
        new_frame_points: Optional[np.ndarray] = None
        if ok:
            if config.propagate == "corners":
                new_reference_points = reference_corners.copy()
                new_frame_points = corners.copy()
            else:
                matched = np.asarray(pairs)[np.asarray(inliers, dtype=bool)]
                new_reference_points = diag["keypoints_ref"][matched[:, 0]].copy()
                new_frame_points = diag["keypoints_frame"][matched[:, 1]].copy()
            reinit_frames.append(t)

        frame_stats = _frame_stats(
            previous_corners, corners, gray_prev, gray,
            diag["n_inliers"], diag["inlier_ratio"], diag["median_residual"],
            float("nan"), diag["n_inliers"], status,
            n_matches=diag["n_matches"])
        return H, corners, status, frame_stats, new_reference_points, new_frame_points

    for t, frame in enumerate(frames):
        gray = to_gray(frame)

        if config.variant == "independent":
            H, corners, status, frame_stats, _, _ = initialise(frame, gray, t)
            homographies.append(H)
            corners_out.append(corners)
            statuses.append(status)
            stats.append(frame_stats)
            flow_records.append({})
            previous_corners = corners
            gray_prev = gray
            continue

        scheduled = bool(config.reinit_every > 0 and t > 0 and
                         t % config.reinit_every == 0)
        need_initialisation = bool(
            t == 0 or gray_prev is None or reference_points is None or
            frame_points is None or len(frame_points) < config.reinit_min_points or
            scheduled)

        if need_initialisation:
            H, corners, status, frame_stats, reference_points, frame_points = initialise(
                frame, gray, t)
            homographies.append(H)
            corners_out.append(corners)
            statuses.append(status)
            stats.append(frame_stats)
            flow_records.append({})
            previous_corners = corners
            gray_prev = gray
            continue

        # Propagate the current frame positions from t-1 to t.  Keep all LK
        # decisions in the diagnostic record, then use only valid points for
        # homography estimation.
        vectors, eigenvalues, residuals_lk, valid_lk = lk_flow(
            gray_prev, gray, frame_points, config.flow)
        flow_record = {
            "points": frame_points.copy(),
            "vectors": vectors,
            "eigenvalues": eigenvalues,
            "residuals": residuals_lk,
            "valid": valid_lk,
        }
        n_valid = int(valid_lk.sum())

        if n_valid < config.reinit_min_points:
            H, corners, status, frame_stats, reference_points, frame_points = initialise(
                frame, gray, t)
            flow_record = {}  # the public contract marks reinitialised frames this way
        else:
            reference_valid = reference_points[valid_lk]
            tracked_valid = frame_points[valid_lk] + vectors[valid_lk]
            H, inliers, ransac_residuals, _, ok = ransac_homography(
                reference_valid, tracked_valid, init.localisation.ransac)

            if not ok or H is None:
                H, corners, status, frame_stats, reference_points, frame_points = initialise(
                    frame, gray, t)
                flow_record = {}
            else:
                corners = transfer_points(H, reference_corners)
                if not np.all(np.isfinite(corners)):
                    H, corners, status, frame_stats, reference_points, frame_points = initialise(
                        frame, gray, t)
                    flow_record = {}
                else:
                    status = "ok"
                    n_inliers = int(inliers.sum())
                    inlier_ratio = n_inliers / max(n_valid, 1)
                    median_residual = (float(np.median(np.sqrt(
                        ransac_residuals[inliers] / 2.0))) if n_inliers else float("nan"))
                    mean_flow = float(np.mean(np.linalg.norm(vectors[valid_lk], axis=1)))
                    frame_stats = _frame_stats(
                        previous_corners, corners, gray_prev, gray,
                        n_inliers, inlier_ratio, median_residual, mean_flow,
                        n_valid, status)

                    # Only geometrically consistent tracks survive into the
                    # next frame; LK-valid RANSAC outliers are discarded.
                    reference_points = reference_valid[inliers].copy()
                    frame_points = tracked_valid[inliers].copy()

        homographies.append(H)
        corners_out.append(corners)
        statuses.append(status)
        stats.append(frame_stats)
        flow_records.append(flow_record)
        previous_corners = corners
        gray_prev = gray

    return TrackResult(homographies=homographies, corners=corners_out,
                       status=statuses, stats=stats,
                       reinit_frames=reinit_frames, flow=flow_records)


def failure_monitor(stats: Sequence[Dict], config: MonitorConfig) -> List[Tuple[str, str]]:
    """Decide OK / REINIT / HIDE for every frame from the per-frame statistics.

    ``stats`` is the list produced by ``track_board``: dictionaries with
    ``inlier_count``, ``inlier_ratio``, ``median_residual``,
    ``corner_displacement``, ``mean_flow``, ``brightness_change`` and
    ``status``. Returns a list of (flag, trigger) where ``trigger`` names the
    statistic (or "none") responsible for the decision. Because a homography
    is defined only up to scale, compare projected-corner displacement and
    reprojection error, never homography matrices.

    Required policy (the tests check it; the thresholds are yours to set):
    1. ``status != "ok"`` -> HIDE with trigger "no_model";
    2. ``inlier_count`` < min_inlier_count or ``inlier_ratio`` < min_inlier_ratio
       -> REINIT with trigger "inlier_count" / "inlier_ratio", escalating to
       HIDE once this has happened on ``hide_after_consecutive`` consecutive
       frames (frames under rule 1 count towards the run);
    3. otherwise the first of ``median_residual``, ``corner_displacement``,
       ``mean_flow``, ``brightness_change`` that exceeds its maximum -> REINIT
       with that statistic as trigger (NaN statistics never fire);
    4. otherwise OK with trigger "none"; an OK frame resets the run.
    """
    decisions: List[Tuple[str, str]] = []
    consecutive_bad = 0

    for frame_stats in stats:
        if frame_stats.get("status") != "ok":
            consecutive_bad += 1
            decisions.append((HIDE, "no_model"))
            continue

        if frame_stats["inlier_count"] < config.min_inlier_count:
            consecutive_bad += 1
            flag = HIDE if consecutive_bad >= config.hide_after_consecutive else REINIT
            decisions.append((flag, "inlier_count"))
            continue

        if frame_stats["inlier_ratio"] < config.min_inlier_ratio:
            consecutive_bad += 1
            flag = HIDE if consecutive_bad >= config.hide_after_consecutive else REINIT
            decisions.append((flag, "inlier_ratio"))
            continue

        # This frame breaks a consecutive no-model/low-inlier run.  The
        # remaining diagnostics may request reinitialisation, but do not form
        # part of the low-inlier escalation policy.
        consecutive_bad = 0
        checks = (
            ("median_residual", config.max_median_residual),
            ("corner_displacement", config.max_corner_displacement),
            ("mean_flow", config.max_mean_flow),
            ("brightness_change", config.max_brightness_change),
        )
        trigger = "none"
        for name, maximum in checks:
            value = float(frame_stats[name])
            if np.isfinite(value) and value > maximum:
                trigger = name
                break

        decisions.append((REINIT, trigger) if trigger != "none" else (OK, "none"))

    return decisions


__all__ = ["OK", "REINIT", "HIDE", "lk_flow", "TrackInit", "TrackResult", "track_board", "failure_monitor"]
