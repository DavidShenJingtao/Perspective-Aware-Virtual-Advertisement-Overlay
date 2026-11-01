"""Supplied orchestration: run the full overlay on a clip and write the video.

Pass 1 streams the clip through ``track_board`` (student code) and keeps only
the per-frame homographies and statistics. Pass 2 streams the clip again and,
for every frame, computes the occlusion mask, warps the artwork, composites,
honours the failure monitor (HIDE frames receive no artwork), and writes the
result. This keeps memory flat for full-resolution clips.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import supplied as sp
from .config import PipelineConfig
from .geometry import warp_artwork
from .localisation import artwork_to_frame
from .temporal import TrackInit, track_board, failure_monitor, HIDE
from .compositing import occlusion_mask, composite


def _frames(video_path, width: Optional[int], max_frames: Optional[int]) -> Iterable[np.ndarray]:
    for idx, frame in sp.iter_video(video_path, step=1, width=width, stop=max_frames or None):
        yield frame


def _scale_reference(reference: np.ndarray, config: PipelineConfig, frame_width_full: int) -> np.ndarray:
    """Resize the reference by the same factor as the frames.

    The descriptor is not scale invariant, so the reference must stay at a
    scale comparable to the board's on-screen size: processing at
    ``config.resize_width`` shrinks the frames, and the reference must shrink
    with them.
    """
    if not config.resize_width or config.resize_width == frame_width_full:
        return reference
    factor = config.resize_width / frame_width_full
    return sp.resize_width(reference, max(8, int(round(reference.shape[1] * factor))))


def run_clip(video_path, reference: np.ndarray, artwork: np.ndarray, config: PipelineConfig,
             out_video, out_stats=None, max_frames: Optional[int] = None,
             occlusion: bool = True, verbose: bool = True, webm: bool = False,
             save_masks: Optional[Sequence[int]] = None) -> Dict:
    """Run tracking + compositing on ``video_path`` and write ``out_video``.

    Returns a dict with per-frame corners, flags, statistics and timings; the
    same dict is written to ``out_stats`` (JSON) when given. ``webm=True`` also
    writes a VP9 ``.webm`` copy of the output that VS Code's preview can play.
    ``save_masks`` lists clip frame indices (the ground-truth mask frames of the
    clip) whose occlusion mask is written to ``<out_video stem>_masks/mask_XXXXXX.png``
    at the processing resolution, so that ``overlaylab.report`` can score mask IoU
    and F1; frames on which the overlay is hidden get no file (``mask_frames_hidden``).
    """
    config = deepcopy(config)  # do not rescale the caller's frozen settings in place
    frozen_config = config.thresholds()
    info = sp.video_info(video_path)
    width = config.resize_width
    reference = _scale_reference(reference, config, info["width"])
    config.localisation.warp.reference_shape = tuple(reference.shape[:2])
    if width and width != info["width"]:   # min_area is defined at full resolution
        config.occlusion.min_area = max(1, int(round(config.occlusion.min_area * (width / info["width"]) ** 2)))
    t0 = time.time()
    init = TrackInit(reference=reference, localisation=config.localisation)
    result = track_board(_frames(video_path, width, max_frames), init, config.track)
    t_track = time.time() - t0
    flags = failure_monitor(result.stats, config.track.monitor)
    if verbose:
        n_ok = sum(1 for s in result.status if s == "ok")
        print(f"[pass 1] tracked {len(result.status)} frames ({n_ok} ok) in {t_track:.1f}s; "
              f"{len(result.reinit_frames)} (re)initialisations")
    t1 = time.time()
    out_w = width or info["width"]
    out_h = int(round(info["height"] * out_w / info["width"]))
    painted_fraction = []
    wanted = set(int(f) for f in (save_masks or []))
    mask_dir = Path(out_video).with_name(Path(out_video).stem + "_masks") if wanted else None
    masks_written, masks_hidden = [], []
    if mask_dir:
        mask_dir.mkdir(parents=True, exist_ok=True)
        for stale in mask_dir.glob("mask_*.png"):
            stale.unlink()  # this run replaces its generated masks

    with sp.VideoWriter(out_video, info["fps"], (out_w, out_h), webm=webm) as vw:
        for t, frame in enumerate(_frames(video_path, width, max_frames)):
            H = result.homographies[t]
            flag, _ = flags[t]
            if H is None or flag == HIDE:
                vw.write(frame); painted_fraction.append(0.0)
                if t in wanted:
                    masks_hidden.append(t)
                continue
            Ha = artwork_to_frame(artwork, H, config.localisation)
            warped, cov = warp_artwork(artwork, Ha, frame.shape)
            if occlusion:
                occ, _, _ = occlusion_mask(frame, H, reference, config.occlusion)
            else:
                occ = np.zeros(cov.shape, bool)
            out, alpha = composite(frame, warped, cov, occ, config.composite)
            painted_fraction.append(float(alpha.mean()))
            vw.write(out)
            if t in wanted:
                sp.write_mask(mask_dir / f"mask_{t:06d}.png", occ)
                masks_written.append(t)
    t_comp = time.time() - t1
    if verbose:
        print(f"[pass 2] composited {len(painted_fraction)} frames in {t_comp:.1f}s -> {out_video}")
    summary = {
        "video": str(video_path), "output": str(out_video), "variant": config.track.variant,
        "resize_width": width, "fps": info["fps"], "n_frames": len(result.status),
        "timing": {"track_s": t_track, "composite_s": t_comp},
        "reinit_frames": result.reinit_frames,
        "corners": [c.tolist() for c in result.corners],
        "status": result.status,
        "flags": [list(f) for f in flags],
        "stats": result.stats,
        "painted_fraction": painted_fraction,
        "mask_dir": mask_dir.name if mask_dir else None,
        "frame_size": [out_w, out_h], "source_width": info["width"],
        "occlusion_enabled": occlusion, "frozen_config": frozen_config,
        "flow": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rec.items()}
                 for rec in result.flow],
        "mask_frames": masks_written, "mask_frames_hidden": masks_hidden,
        "config": json.loads(config.to_json()),
    }
    if out_stats:
        sp.write_json(out_stats, summary)
    return summary


def load_stats(path) -> Dict:
    d = sp.read_json(path)
    d["corners"] = [np.array(c) for c in d["corners"]]
    return d


__all__ = ["run_clip", "load_stats"]
