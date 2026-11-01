"""Supplied evaluation of a finished overlay video (classical or generative).

Everything here is computed from the output video, the original clip, the
ground-truth corners and masks, and the artwork; no student code is called,
so the same report is produced for a classical pipeline output, a
generative-AI output, or a staff reference. The report follows the
structure-preservation / fidelity / consistency decomposition used to
evaluate appearance transfer in ReStyle3D (Zhu et al., SIGGRAPH 2025), with
classical analogues of each axis:

* placement     -- the artwork is re-localised in the *output* frames with
                   SIFT + RANSAC (independent of the pipeline that made the
                   video); recovered corners are compared with the ground
                   truth (median / p90 error, success rate, detection rate)
* fidelity      -- rectified board region vs. artwork: NCC, SSIM, MAE and a
                   gradient-structure correlation (the "style fidelity" axis)
* temporal      -- flicker of the rectified content, flow-warping error of
                   the board between consecutive frames using the ORIGINAL
                   clip's motion, and jitter of the recovered corners
                   (the "multi-view consistency" axis)
* preservation  -- PSNR / SSIM / mean |diff| outside the ground-truth board
                   and background homography drift (the "structure" axis)
* occlusion     -- on the mask frames: leakage and foreground PSNR (frames whose
                   ground-truth mask is non-empty), the fraction of the *visible*
                   board that was painted (``coverage_visible``) and, on frames
                   whose mask is empty (nothing in front of the board), the
                   false-occlusion fraction ``1 - coverage_visible``. Coverage and
                   false occlusion are only computed on frames where the artwork is
                   actually detected in the output (``overlay_absent`` otherwise);
                   leakage is computed on every non-empty mask frame (a hidden
                   overlay has zero leakage, which is correct)

Two input modes share every per-frame computation:

* ``--video``  a finished video (classical pipeline output or a video model)
* ``--frames`` a set of edited still frames named ``frame_XXXXXX.png`` whose
               index is the clip frame number (the generative comparison of
               Stage 4); temporal terms are computed on every supplied pair
               ``(f, f + stride)``

Command line::

    python -m overlaylab.evaluate --video outputs/videos/common_clip_flow_assisted.mp4
    python -m overlaylab.evaluate --video outputs/videos/coke_blur_flow_assisted.mp4 --clip coke_blur
    python -m overlaylab.evaluate --frames outputs/genai/frame_*.png --clip <personal id> --artwork <name>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import data
from . import supplied as sp
from . import metrics as M
from .supplied import to_gray, to_float, quad_mask


# ---------------------------------------------------------------------------
# image-quality helpers
# ---------------------------------------------------------------------------

def psnr(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    a = to_float(a); b = to_float(b)
    d = (a - b) ** 2
    if mask is not None:
        m = np.asarray(mask, bool)
        if not m.any():
            return float("nan")
        d = d[m]
    mse = float(np.mean(d))
    return float(10 * np.log10(1.0 / mse)) if mse > 1e-12 else 99.0


def ssim(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None, sigma: float = 1.5) -> float:
    """Grayscale SSIM with a Gaussian window (Wang et al. 2004), optionally averaged over a mask."""
    x = to_gray(a).astype(np.float64); y = to_gray(b).astype(np.float64)
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    blur = lambda im: cv2.GaussianBlur(im, (0, 0), sigma)
    mx, my = blur(x), blur(y)
    sxx = blur(x * x) - mx * mx; syy = blur(y * y) - my * my; sxy = blur(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    if mask is not None:
        m = np.asarray(mask, bool)
        return float(s[m].mean()) if m.any() else float("nan")
    return float(s.mean())


# ---------------------------------------------------------------------------
# artwork re-localisation in the output (pipeline-independent)
# ---------------------------------------------------------------------------

class ArtworkLocaliser:
    """Find the artwork in an output frame with SIFT + RANSAC (OpenCV, staff tool)."""

    def __init__(self, artwork: np.ndarray, work_width: int = 800, ratio: float = 0.75, min_inliers: int = 15):
        self.sift = cv2.SIFT_create(nfeatures=3000)
        art = sp.to_uint8(artwork)
        self.scale = work_width / art.shape[1]
        self.art = cv2.resize(art, (work_width, int(round(art.shape[0] * self.scale))))
        self.kp, self.desc = self.sift.detectAndCompute(cv2.cvtColor(self.art, cv2.COLOR_RGB2GRAY), None)
        self.ratio = ratio; self.min_inliers = min_inliers
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        h, w = self.art.shape[:2]
        self.corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32).reshape(-1, 1, 2)

    def __call__(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        g = cv2.cvtColor(sp.to_uint8(frame), cv2.COLOR_RGB2GRAY)
        kp, desc = self.sift.detectAndCompute(g, None)
        if desc is None or self.desc is None or len(kp) < 8:
            return None, 0
        pairs = self.matcher.knnMatch(self.desc, desc, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < self.ratio * n.distance]
        if len(good) < self.min_inliers:
            return None, len(good)
        src = np.float32([self.kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        n_in = int(inl.sum()) if inl is not None else 0
        if H is None or n_in < self.min_inliers:
            return None, n_in
        c = cv2.perspectiveTransform(self.corners, H).reshape(4, 2)
        if not np.all(np.isfinite(c)) or not cv2.isContourConvex(c.astype(np.float32)):
            return None, n_in
        return c.astype(np.float64), n_in


def background_drift(original: np.ndarray, output: np.ndarray, gt_corners: np.ndarray, margin: int = 12) -> float:
    """Mean displacement (px) of the frame corners under the homography that best maps
    original -> output outside the board. Zero when the background is untouched."""
    sift = cv2.SIFT_create(nfeatures=2000)
    g0 = cv2.cvtColor(sp.to_uint8(original), cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(sp.to_uint8(output), cv2.COLOR_RGB2GRAY)
    inside = quad_mask(g0.shape, gt_corners).astype(np.uint8)
    inside = cv2.dilate(inside, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    outside = (1 - inside).astype(np.uint8) * 255
    k0, d0 = sift.detectAndCompute(g0, outside); k1, d1 = sift.detectAndCompute(g1, outside)
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return float("nan")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0, d1, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return float("nan")
    src = np.float32([k0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return float("nan")
    h, w = g0.shape
    fc = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(fc, H).reshape(4, 2)
    return float(np.mean(np.linalg.norm(moved - fc.reshape(4, 2), axis=1)))


def warping_error(prev_orig: np.ndarray, next_orig: np.ndarray, prev_out: np.ndarray, next_out: np.ndarray,
                  board_mask: np.ndarray) -> float:
    """Temporal warping error inside the board: warp prev_out with the ORIGINAL clip's
    dense flow and compare with next_out (lower = more temporally consistent)."""
    g0 = cv2.cvtColor(sp.to_uint8(prev_orig), cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(sp.to_uint8(next_orig), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g0, None, 0.5, 3, 21, 3, 5, 1.2, 0)  # next -> prev
    h, w = g0.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    mapx = xs + flow[..., 0]; mapy = ys + flow[..., 1]
    warped = cv2.remap(to_float(prev_out), mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m = np.asarray(board_mask, bool)
    return float(np.mean(np.abs(warped - to_float(next_out))[m])) if m.any() else float("nan")


# ---------------------------------------------------------------------------
# main evaluation
# ---------------------------------------------------------------------------

FRAME_NAME = re.compile(r"frame_(\d+)\.(png|jpg|jpeg)$", re.I)


def _read_frames(path, indices: Sequence[int], offset: int = 0, size=None) -> Dict[int, np.ndarray]:
    """Frames of ``path`` at clip indices ``indices``; ``offset`` is the clip index of the video's frame 0."""
    wanted = {i - offset for i in indices}
    out = {}
    for j, frame in sp.iter_video(path, step=1, stop=max(wanted) + 1 if wanted else 0):
        if j in wanted:
            if size is not None and (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            out[j + offset] = frame
    return out


def _resize_to(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an RGB frame to ``size`` = (width, height); area filter when shrinking, linear otherwise."""
    if (frame.shape[1], frame.shape[0]) == tuple(size):
        return frame
    shrink = frame.shape[1] > size[0] or frame.shape[0] > size[1]
    return cv2.resize(frame, tuple(size), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


def frame_index_from_name(path) -> int:
    """Clip frame index encoded in a file name ``frame_XXXXXX.png``."""
    m = FRAME_NAME.search(Path(path).name)
    if m is None:
        raise ValueError(f"{path}: expected a file name of the form frame_XXXXXX.png (XXXXXX = clip frame index)")
    return int(m.group(1))


def _corners_at(frame: int, corner_dict: Dict[int, np.ndarray], gt_frames: Sequence[int],
                gt_list: Sequence[np.ndarray]) -> np.ndarray:
    """Exact ground-truth corners when the frame is annotated, otherwise linear interpolation."""
    if frame in corner_dict:
        return np.asarray(corner_dict[frame], np.float64)
    return M.interpolate_corners(gt_frames, gt_list, [frame])[0]


# --- per-frame computations shared by the video and the frames mode ------------------------------

def placement_frame(recovered: Optional[np.ndarray], out: np.ndarray, orig: np.ndarray, corners: np.ndarray,
                    artwork: np.ndarray, rect_size: Tuple[int, int]) -> Tuple[float, Dict, Dict]:
    """Placement (``recovered`` = artwork corners re-localised in ``out`` by ``ArtworkLocaliser``, or None, vs the
    ground-truth ``corners``), fidelity of the rectified board, and preservation outside the ground-truth quad."""
    err = M.corner_error(recovered, corners) if recovered is not None else float("inf")
    fid = M.rectified_content_fidelity(out, corners, artwork, size=rect_size)
    rect_o = M.rectify_board(out, corners, rect_size)
    rect_a = cv2.resize(to_float(artwork), rect_size, interpolation=cv2.INTER_AREA)
    fid["ssim"] = ssim(rect_o, rect_a)
    inside = quad_mask(orig.shape[:2], corners)
    pres = {"psnr_outside": psnr(out, orig, ~inside), "ssim_outside": ssim(out, orig, ~inside),
            "mad_outside": M.outside_board_change(orig, out, corners), "background_drift_px": background_drift(orig, out, corners)}
    return err, fid, pres


def temporal_pair(rca: Optional[np.ndarray], rcb: Optional[np.ndarray], out_a: np.ndarray, out_b: np.ndarray,
                  orig_a: np.ndarray, orig_b: np.ndarray, corners_a: np.ndarray, corners_b: np.ndarray,
                  rect_size: Tuple[int, int]) -> Tuple[float, float, Optional[float]]:
    """Flicker of the rectified content, flow-warping error under the ORIGINAL clip's motion, and jitter of the
    recovered corners ``rca`` / ``rcb`` (None when the artwork is not found in both frames) for one (a, b) pair."""
    ra = M.rectify_board(out_a, corners_a, rect_size); rb = M.rectify_board(out_b, corners_b, rect_size)
    flick = float(np.mean(np.abs(ra - rb)))
    werr = warping_error(orig_a, orig_b, out_a, out_b, quad_mask(orig_b.shape[:2], corners_b))
    jit = float(np.mean(np.linalg.norm((rcb - rca) - (corners_b - corners_a), axis=1))) if rca is not None and rcb is not None else None
    return flick, werr, jit


def occlusion_frame(out: np.ndarray, orig: np.ndarray, gt_mask: np.ndarray, corners: Optional[np.ndarray],
                    leakage_tolerance: float = 0.08, overlay_present: bool = True) -> Dict:
    """Occlusion terms of one mask frame.

    ``painted`` = pixels whose |output - original| exceeds the tolerance. ``leakage`` is the fraction of the
    ground-truth foreground that was painted (nan when the mask is empty; computed on every frame, a hidden
    overlay has zero leakage). ``coverage_visible`` is the fraction of the pixels inside the ground-truth quad
    and outside the mask (the visible board) that were painted, and ``false_occlusion = 1 - coverage_visible``
    on frames whose mask is empty: board pixels left unpainted although nothing occludes the board. Both are
    only meaningful when the overlay is present in the frame: with ``overlay_present=False`` (the artwork was
    not detected in the output) they are nan and ``overlay_absent`` is True.
    """
    m = np.asarray(gt_mask, bool)
    painted = np.mean(np.abs(to_float(out) - to_float(orig)), axis=2) > leakage_tolerance
    rec = {"leakage": M.leakage(painted, m), "psnr_foreground": psnr(out, orig, m), "mask_empty": bool(not m.any()),
           "foreground_pixels": int(m.sum()), "overlay_absent": bool(not overlay_present),
           "coverage_visible": float("nan"), "false_occlusion": float("nan")}
    if overlay_present and corners is not None:
        visible = quad_mask(orig.shape[:2], corners) & ~m
        rec["coverage_visible"] = float(painted[visible].mean()) if visible.any() else float("nan")
        if rec["mask_empty"] and np.isfinite(rec["coverage_visible"]):
            rec["false_occlusion"] = float(1.0 - rec["coverage_visible"])
    return rec


def _nanmean(values) -> float:
    vals = [v for v in values if v is not None]
    return float(np.nanmean(vals)) if vals and np.any(np.isfinite(vals)) else float("nan")


def _evaluate_frame_sets(out_frames: Dict[int, np.ndarray], orig_frames: Dict[int, np.ndarray], artwork: np.ndarray,
                         gt_corners: Dict[int, np.ndarray], gt_masks: Dict[int, np.ndarray], frames_c: Sequence[int],
                         frames_m: Sequence[int], temporal_pairs: Sequence[Tuple[int, int]], extra_corners: Optional[Dict[int, np.ndarray]],
                         success_threshold: float, leakage_tolerance: float, temporal_stride: int) -> Dict:
    """The report body shared by ``evaluate_video`` and ``evaluate_frames``.

    ``frames_c`` are the placement / fidelity / preservation frames (must have ground-truth corners), ``frames_m``
    the occlusion frames (must have ground-truth masks), ``temporal_pairs`` the (a, b) pairs for the temporal terms.
    ``extra_corners`` are exact corners of frames that are *not* placement frames (e.g. the combined clip's mask frames);
    they are merged into the corner dictionary used for interpolation and quad masks only.
    """
    corner_dict = dict(gt_corners); corner_dict.update(extra_corners or {})
    gt_frames = sorted(corner_dict); gt_list = [corner_dict[f] for f in gt_frames]
    loc = ArtworkLocaliser(artwork)
    cache: Dict[int, Optional[np.ndarray]] = {}

    def recovered(f: int) -> Optional[np.ndarray]:
        """Artwork corners re-localised in output frame ``f`` (SIFT + RANSAC), computed once per frame."""
        if f not in cache:
            cache[f] = loc(out_frames[f])[0]
        return cache[f]
    # rectified board size for the fidelity / flicker terms: 400 px wide in the artwork's own aspect ratio
    rect_size = (400, max(1, int(round(400 * artwork.shape[0] / artwork.shape[1]))))
    report: Dict = {}

    # placement + fidelity + preservation on GT corner frames
    rec_err, fidel, pres = {}, [], []
    for f in frames_c:
        if f not in out_frames or f not in orig_frames:
            rec_err[f] = float("inf")
            continue
        err, fid, pr = placement_frame(recovered(f), out_frames[f], orig_frames[f], gt_corners[f], artwork, rect_size)
        rec_err[f] = err; fidel.append(fid); pres.append({"frame": f, **pr})
    errs = list(rec_err.values())
    report["placement"] = {
        **M.summarise_errors(errs, success_threshold),
        "detection_rate": float(np.mean([np.isfinite(e) for e in errs])) if errs else float("nan"),
        "threshold": success_threshold,
        "per_frame": {int(f): (None if not np.isfinite(e) else float(e)) for f, e in rec_err.items()},
    }
    report["fidelity"] = {k: float(np.nanmean([d[k] for d in fidel])) for k in ("ncc", "ssim", "mae", "structure")} if fidel else {}
    report["preservation"] = {k: float(np.nanmean([d[k] for d in pres])) for k in
                              ("psnr_outside", "ssim_outside", "mad_outside", "background_drift_px")} if pres else {}
    report["preservation_per_frame"] = pres

    # temporal: flicker of rectified content + warping error + recovered-corner jitter on (a, b) pairs
    flick, werr, jit, used = [], [], [], []
    for a, b in temporal_pairs:
        if any(f not in out_frames or f not in orig_frames for f in (a, b)):
            continue
        ca = _corners_at(a, corner_dict, gt_frames, gt_list); cb = _corners_at(b, corner_dict, gt_frames, gt_list)
        fl, we, ji = temporal_pair(recovered(a), recovered(b), out_frames[a], out_frames[b], orig_frames[a], orig_frames[b], ca, cb, rect_size)
        flick.append(fl); werr.append(we); used.append([int(a), int(b)])
        if ji is not None:
            jit.append(ji)
    report["temporal"] = {"stride": temporal_stride, "flicker": float(np.mean(flick)) if flick else float("nan"),
                          "warping_error": float(np.nanmean(werr)) if werr else float("nan"),
                          "recovered_corner_jitter_px": float(np.median(jit)) if jit else float("nan"),
                          "pairs": len(flick), "pair_frames": used}

    # occlusion on mask frames: leakage / foreground PSNR (non-empty masks, every frame); coverage of the visible
    # board and false occlusion (empty masks) only where the artwork is detected in the output (overlay present)
    occ = []
    for f in frames_m:
        if f not in out_frames or f not in orig_frames:
            continue
        corners = _corners_at(f, corner_dict, gt_frames, gt_list) if gt_frames else None
        occ.append({"frame": int(f), **occlusion_frame(out_frames[f], orig_frames[f], gt_masks[f], corners, leakage_tolerance,
                                                       overlay_present=recovered(f) is not None)})
    nonempty = [d for d in occ if not d["mask_empty"]]
    present = [d for d in occ if not d["overlay_absent"]]
    empty_present = [d for d in present if d["mask_empty"]]
    report["occlusion"] = {"leakage": _nanmean([d["leakage"] for d in nonempty]),
                           "psnr_foreground": _nanmean([d["psnr_foreground"] for d in nonempty]),
                           "coverage_visible": _nanmean([d["coverage_visible"] for d in present]),
                           "false_occlusion": _nanmean([d["false_occlusion"] for d in empty_present]),
                           "mask_frames": len(occ), "empty_mask_frames": sum(d["mask_empty"] for d in occ),
                           "n_coverage_frames": len(present), "n_false_occlusion_frames": len(empty_present),
                           "overlay_absent_frames": [d["frame"] for d in occ if d["overlay_absent"]], "per_frame": occ}
    return report


def evaluate_video(output_video, original_video=None, artwork=None, gt_corners: Optional[Dict[int, np.ndarray]] = None,
                   gt_masks: Optional[Dict[int, np.ndarray]] = None, interval: Optional[Tuple[int, int]] = None,
                   offset: int = 0, success_threshold: float = 0.02, leakage_tolerance: float = 0.08,
                   temporal_stride: int = 1, verbose: bool = True,
                   extra_corners: Optional[Dict[int, np.ndarray]] = None) -> Dict:
    """Evaluate ``output_video`` on the annotated frames inside ``interval``.

    ``offset`` is the clip index of the output video's first frame (0 for a
    full-clip output). The output is resized to the original resolution if the
    two differ. ``extra_corners`` are exact corners of mask frames that are not
    placement frames (see ``_evaluate_frame_sets``). Returns a nested report dict.
    """
    original_video = original_video or data.COMMON_CLIP
    artwork = data.load_artwork("ad") if artwork is None else artwork
    gt_corners = gt_corners if gt_corners is not None else data.load_gt_corners()
    gt_masks = gt_masks if gt_masks is not None else data.load_gt_masks()
    info = sp.video_info(original_video); size = (info["width"], info["height"])
    out_info = sp.video_info(output_video)
    if temporal_stride < 1 or offset < 0:
        raise ValueError("stride must be positive and offset must be non-negative")
    if not np.isclose(out_info["fps"], info["fps"], rtol=1e-4, atol=0.01):
        raise ValueError("--video requires the source frame rate; use --frames for generated videos with different timing")
    lo, hi = interval if interval else (offset, info["frames"] - 1)
    if not 0 <= lo <= hi < info["frames"] or lo < offset:
        raise ValueError("interval must lie inside the original clip and covered output indices")
    if hi - offset >= out_info["frames"]:
        raise ValueError(f"incomplete output: interval requires source frame {hi}, but output ends at {offset + out_info['frames'] - 1}")
    frames_c = [f for f in sorted(gt_corners) if lo <= f <= hi]
    frames_m = [f for f in sorted(gt_masks) if lo <= f <= hi]

    # frames needed: GT corner frames, mask frames, and (f, f+stride) pairs for temporal terms
    temporal_pairs = [(f, f + temporal_stride) for f in frames_c if f + temporal_stride <= hi]
    need = sorted(set(frames_c) | set(frames_m) | {g for p in temporal_pairs for g in p})
    out_frames = _read_frames(output_video, need, offset=offset, size=size)
    orig_frames = _read_frames(original_video, need)
    missing = [f for f in need if f not in out_frames]

    report: Dict = {"mode": "video", "video": str(output_video), "interval": [lo, hi], "offset": offset,
                    "output_resolution": [out_info["width"], out_info["height"]],
                    "original_resolution": list(size), "resized": [out_info["width"], out_info["height"]] != list(size),
                    "missing_frames": missing}
    report.update(_evaluate_frame_sets(out_frames, orig_frames, artwork, gt_corners, gt_masks, frames_c, frames_m,
                                       temporal_pairs, extra_corners, success_threshold, leakage_tolerance, temporal_stride))
    if verbose:
        print(format_report(report))
    return report


def evaluate_frames(frame_paths: Sequence, original_video=None, artwork=None, gt_corners: Optional[Dict[int, np.ndarray]] = None,
                    gt_masks: Optional[Dict[int, np.ndarray]] = None, success_threshold: float = 0.02,
                    leakage_tolerance: float = 0.08, temporal_stride: int = 1, verbose: bool = True,
                    extra_corners: Optional[Dict[int, np.ndarray]] = None) -> Dict:
    """Evaluate a set of edited still frames (``frame_XXXXXX.png``, index = clip frame number).

    Placement / fidelity / preservation are computed on the supplied frames that have ground-truth corners,
    occlusion on the supplied frames that have ground-truth masks, and the temporal terms on every pair
    ``(f, f + temporal_stride)`` of which both frames were supplied (corners interpolated from the ground truth
    where needed). Frames whose resolution differs from the original clip are resized to it (``resized`` flag,
    ``output_resolution`` per frame). Same per-frame code paths as ``evaluate_video``.
    """
    original_video = original_video or data.COMMON_CLIP
    artwork = data.load_artwork("ad") if artwork is None else artwork
    gt_corners = gt_corners if gt_corners is not None else data.load_gt_corners()
    gt_masks = gt_masks if gt_masks is not None else data.load_gt_masks()
    info = sp.video_info(original_video); size = (info["width"], info["height"])
    paths = {frame_index_from_name(p): Path(p) for p in frame_paths}
    if len(paths) != len(frame_paths):
        raise ValueError("duplicate source frame indices in --frames")
    if temporal_stride < 1 or any(f < 0 or f >= info["frames"] for f in paths):
        raise ValueError("frame indices must lie inside the original clip and stride must be positive")
    if not paths:
        raise ValueError("no frames supplied")
    out_frames, resolutions, resized = {}, {}, {}
    for f, p in sorted(paths.items()):
        img = sp.read_image(p)
        resolutions[int(f)] = [int(img.shape[1]), int(img.shape[0])]
        resized[int(f)] = (img.shape[1], img.shape[0]) != size
        out_frames[f] = _resize_to(img, size)
    supplied = sorted(out_frames)
    frames_c = [f for f in supplied if f in gt_corners]
    frames_m = [f for f in supplied if f in gt_masks]
    temporal_pairs = [(f, f + temporal_stride) for f in supplied if f + temporal_stride in out_frames]
    orig_frames = _read_frames(original_video, supplied)
    missing_in_original = [f for f in supplied if f not in orig_frames]
    if missing_in_original:
        raise ValueError(f"cannot read original frames {missing_in_original}")

    folder = paths[supplied[0]].resolve().parent
    report: Dict = {"mode": "frames", "video": str(folder), "frames": {int(f): str(paths[f]) for f in supplied},
                    "interval": [int(supplied[0]), int(supplied[-1])], "offset": 0,
                    "output_resolution": resolutions, "original_resolution": list(size),
                    "resized": any(resized.values()), "resized_frames": [f for f in supplied if resized[f]],
                    "frames_with_corners": [int(f) for f in frames_c], "frames_with_masks": [int(f) for f in frames_m],
                    "missing_frames": missing_in_original}
    report.update(_evaluate_frame_sets(out_frames, orig_frames, artwork, gt_corners, gt_masks, frames_c, frames_m,
                                       temporal_pairs, extra_corners, success_threshold, leakage_tolerance, temporal_stride))
    if verbose:
        print(format_report(report))
    return report


def format_report(r: Dict) -> str:
    p, f, t, s, o = r["placement"], r.get("fidelity", {}), r["temporal"], r.get("preservation", {}), r["occlusion"]
    g = lambda d, k, fmt="{:.3f}": (fmt.format(d[k]) if k in d and d[k] is not None and np.isfinite(d[k]) else "n/a")
    what = f"{len(r.get('frames', {}))} frames in {r['video']}" if r.get("mode") == "frames" else r["video"]
    n_mask = o.get("mask_frames", len(o.get("per_frame", []))); n_empty = o.get("empty_mask_frames", 0)
    n_cov = o.get("n_coverage_frames", n_mask); n_fo = o.get("n_false_occlusion_frames", n_empty); n_abs = len(o.get("overlay_absent_frames", []))
    lines = [
        f"Evaluation of {what}  (clip frames {r['interval'][0]}-{r['interval'][1]})",
        f"  placement    : median {g(p,'median','{:.2%}')}  p90 {g(p,'p90','{:.2%}')}  success {g(p,'success','{:.0%}')}  detected {g(p,'detection_rate','{:.0%}')}  (n={p['n']})",
        f"  fidelity     : NCC {g(f,'ncc')}  SSIM {g(f,'ssim')}  structure {g(f,'structure')}  MAE {g(f,'mae')}",
        f"  temporal     : flicker {g(t,'flicker','{:.4f}')}  warping error {g(t,'warping_error','{:.4f}')}  jitter {g(t,'recovered_corner_jitter_px','{:.2f} px')}  (stride {t['stride']}, {t['pairs']} pairs)",
        f"  preservation : PSNR {g(s,'psnr_outside','{:.1f} dB')}  SSIM {g(s,'ssim_outside')}  mean|diff| {g(s,'mad_outside','{:.4f}')}  bg drift {g(s,'background_drift_px','{:.2f} px')}",
        f"  occlusion    : leakage {g(o,'leakage','{:.1%}')}  foreground PSNR {g(o,'psnr_foreground','{:.1f} dB')}  "
        f"visible-board coverage {g(o,'coverage_visible','{:.1%}')} over {n_cov} frames  "
        f"false occlusion {g(o,'false_occlusion','{:.1%}')} over {n_fo} frames, overlay absent on {n_abs}  "
        f"({n_mask} mask frames, {n_empty} with an empty mask)",
    ]
    if r.get("resized"):
        lines.append(f"  note: output resized to the original resolution {r['original_resolution']}")
    if r.get("missing_frames"):
        src = "the original clip" if r.get("mode") == "frames" else "the output video"
        lines.append(f"  WARNING: {len(r['missing_frames'])} required frames missing from {src}")
    return "\n".join(lines)


def ground_truth_for_clip(clip_id: str):
    """(corners, masks, extra corners for mask frames) for ``common_clip``, a stress clip or a personal clip."""
    if clip_id == "common_clip":
        return data.load_gt_corners(), data.load_gt_masks(), {}
    if clip_id in data.diagnostic_clip_ids():
        gt_c = data.load_diagnostic_corners(clip_id)
        gt_m = data.load_diagnostic_masks(clip_id)
        # exact POT corners of the mask frames (the combined clip's mask frames are not stride-16 frames): used for
        # the quad masks and the interpolation only, never as placement frames
        extra = {f: c for f, c in data.load_diagnostic_mask_corners(clip_id).items() if f not in gt_c}
        return gt_c, gt_m, extra
    gt_c = data.load_audit_corners(clip_id); gt_m = data.load_audit_masks(clip_id)
    if not gt_c:
        raise SystemExit(f"no corners for clip {clip_id!r} in manifests/diagnostic_corners.csv or audit_corners.csv")
    return gt_c, gt_m, {}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a finished overlay video, or a set of edited frames, against the ground truth.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="finished output video")
    src.add_argument("--frames", nargs="+", metavar="PNG",
                     help="edited still frames named frame_XXXXXX.png (XXXXXX = clip frame index), e.g. outputs/genai/frame_*.png")
    ap.add_argument("--clip", default="common_clip",
                    help="common_clip, a stress clip id (data/diagnostic/<clip_id>/, manifests/diagnostic_corners.csv + "
                         "diagnostic_masks.csv), or a personal clip id (data/personal/<clip_id>/ and the audit manifests)")
    ap.add_argument("--original", default=None, help="original clip (default: from --clip)")
    ap.add_argument("--artwork", default="ad")
    ap.add_argument("--interval", type=int, nargs=2, default=None, help="clip frame range to evaluate (inclusive; --video only)")
    ap.add_argument("--offset", type=int, default=0, help="clip index of the output video's first frame (--video only)")
    ap.add_argument("--stride", type=int, default=1, help="frame gap for the temporal terms")
    ap.add_argument("--threshold", type=float, default=None, help="override the frozen success threshold")
    ap.add_argument("--thresholds", default=str(data.OUTPUT_DIR / "thresholds.json"), help="frozen configuration, when present")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args(argv)
    from .config import PipelineConfig
    frozen = Path(args.thresholds)
    cfg = PipelineConfig.from_json(frozen.read_text()) if frozen.exists() else PipelineConfig()
    threshold = args.threshold if args.threshold is not None else cfg.localisation.success_threshold
    original = args.original
    if args.clip != "common_clip":
        original = original or data.clip_paths(args.clip)[0]
    gt_c, gt_m, extra = ground_truth_for_clip(args.clip)
    artwork = data.load_artwork(args.artwork)
    if args.video:
        rep = evaluate_video(args.video, original, artwork, gt_corners=gt_c, gt_masks=gt_m,
                             interval=tuple(args.interval) if args.interval else None, offset=args.offset,
                             success_threshold=threshold, temporal_stride=args.stride, extra_corners=extra)
        default_out = data.OUTPUT_DIR / "evaluation" / (Path(args.video).stem + "_report.json")
    else:
        rep = evaluate_frames(args.frames, original, artwork, gt_corners=gt_c, gt_masks=gt_m,
                              success_threshold=threshold, temporal_stride=args.stride, extra_corners=extra)
        default_out = data.OUTPUT_DIR / "evaluation" / (Path(args.frames[0]).resolve().parent.name + "_frames_report.json")
    out = Path(args.out) if args.out else default_out
    sp.write_json(out, rep)
    print("report written to", out)


if __name__ == "__main__":
    main()
