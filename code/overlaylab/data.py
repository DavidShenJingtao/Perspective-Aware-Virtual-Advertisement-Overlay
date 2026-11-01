"""Dataset layout and loaders.

Layout (``data/`` sits beside ``code/``)::

    data/common_clip/board_clip.mp4
    data/common_clip/board_reference.png
    data/common_clip/frames/frame_000000.png   # python -m overlaylab.extract_frames
    data/common_clip/masks/mask_000096.png     # six ground-truth occlusion masks
    data/artwork/ad.png, <your_artwork>.png
    data/diagnostic/<clip_id>/clip.mp4, reference.png   # the stress clips of the same poster (one factor each)
    data/diagnostic/<clip_id>/masks/mask_XXXXXX.png     # occlusion masks of the viewpoint / combined clips (manifests/diagnostic_masks.csv)
    data/personal/<clip_id>/clip.mp4, reference.png, frames/, masks/

Clips. ``common_clip`` is the only clip with validation frames: every threshold is chosen there and frozen.
The other clips are *stress clips*, one capture factor each, named by that factor (``clip_factor``):
``coke_perspective`` = viewpoint, ``coke_scale`` = scale, ``coke_blur`` = blur, ``coke_out_of_view`` = exit,
``coke_unconstrained`` = combined (``coke_rotation`` = rotation is supplementary and not in any table).
Each Stage reports on the common clip and on its two stress clips, ``stress_clips()`` ->
{1: {"viewpoint": ..., "scale": ...}, 2: {"blur": ..., "exit": ...}, 3: {"viewpoint": ..., "combined": ...}}
(from ``dataset_summary.json``); ``python -m overlaylab.report`` builds the three tables.
Ground-truth corners for the common clip are in ``manifests/common_clip_corners.csv``
(every 16th frame; even-indexed annotated frames are validation, odd-indexed are test).
Corners for the stress clips are in ``manifests/diagnostic_corners.csv`` (every 16th frame, all ``test``:
thresholds are never chosen on them; column ``factor``). ``manifests/diagnostic_masks.csv`` lists the
occlusion masks of the combined clip and the three *empty* masks of the viewpoint clip (occluder "none":
the false-occlusion test) together with the POT corners of those frames (``load_diagnostic_masks`` /
``load_diagnostic_mask_corners``). ``mask_frames_for_clip`` gives the mask frames of any clip id.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import supplied as sp

CODE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = CODE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
MANIFEST_DIR = CODE_DIR / "manifests"
OUTPUT_DIR = CODE_DIR / "outputs"

COMMON_CLIP = DATA_DIR / "common_clip" / "board_clip.mp4"
BOARD_REFERENCE = DATA_DIR / "common_clip" / "board_reference.png"
FRAMES_DIR = DATA_DIR / "common_clip" / "frames"
MASKS_DIR = DATA_DIR / "common_clip" / "masks"
ARTWORK_DIR = DATA_DIR / "artwork"
PERSONAL_DIR = DATA_DIR / "personal"
DIAGNOSTIC_DIR = DATA_DIR / "diagnostic"

CORNERS_MANIFEST = MANIFEST_DIR / "common_clip_corners.csv"
MASKS_MANIFEST = MANIFEST_DIR / "common_clip_masks.csv"
DIAGNOSTIC_CORNERS_MANIFEST = MANIFEST_DIR / "diagnostic_corners.csv"
DIAGNOSTIC_MASKS_MANIFEST = MANIFEST_DIR / "diagnostic_masks.csv"
GENAI_CONDITIONS = MANIFEST_DIR / "genai_conditions.json"
DATASET_SUMMARY = MANIFEST_DIR / "dataset_summary.json"

GT_STRIDE = 16
NOTEBOOK_STEP = 8          # divides GT_STRIDE, so every annotated frame is on the notebook subset
NOTEBOOK_WIDTH = 640


def split_of_annotation_index(k: int) -> str:
    """Even-indexed annotated frames are validation, odd-indexed are test (mirrors Lab 1)."""
    return "validation" if k % 2 == 0 else "test"


def frame_path(index: int, frames_dir: Path = FRAMES_DIR) -> Path:
    return Path(frames_dir) / f"frame_{index:06d}.png"


def mask_path(index: int, masks_dir: Path = MASKS_DIR) -> Path:
    return Path(masks_dir) / f"mask_{index:06d}.png"


def load_reference() -> np.ndarray:
    return sp.read_image(BOARD_REFERENCE)


def load_artwork(name: str = "ad") -> np.ndarray:
    p = Path(name)
    if not p.exists():
        p = ARTWORK_DIR / (name if name.endswith(".png") else f"{name}.png")
    return sp.read_image(p)


def load_frame(index: int, width: Optional[int] = None) -> np.ndarray:
    """Load an extracted full-resolution frame (run extract_frames first)."""
    p = frame_path(index)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python -m overlaylab.extract_frames ../data/common_clip`")
    img = sp.read_image(p)
    return sp.resize_width(img, width) if width else img


def load_gt_corners(split: Optional[str] = None, manifest: Path = CORNERS_MANIFEST,
                    scale: float = 1.0) -> Dict[int, np.ndarray]:
    """frame index -> 4 x 2 corners (optionally restricted to 'validation' or 'test')."""
    out = {}
    for row in sp.read_csv(manifest):
        if split is not None and row["split"] != split:
            continue
        out[int(row["frame"])] = sp.corners_from_row(row) * scale
    return out


def load_gt_masks(split: Optional[str] = None, manifest: Path = MASKS_MANIFEST,
                  width: Optional[int] = None) -> Dict[int, np.ndarray]:
    """frame index -> Boolean occlusion mask (resized to ``width`` if given)."""
    out = {}
    for row in sp.read_csv(manifest):
        if split is not None and row["split"] != split:
            continue
        m = sp.read_mask(sp.resolve(manifest, row["mask_path"]))
        if width is not None and m.shape[1] != width:
            m = sp.resize_width(m.astype(np.uint8) * 255, width) > 127
        out[int(row["frame"])] = m
    return out


def scale_corners(corners: Dict[int, np.ndarray], factor: float) -> Dict[int, np.ndarray]:
    return {k: np.asarray(v) * factor for k, v in corners.items()}


def load_notebook_subset(step: int = NOTEBOOK_STEP, width: int = NOTEBOOK_WIDTH,
                         cache: bool = True) -> Tuple[List[int], List[np.ndarray], float]:
    """Deterministic subset used by Notebooks 1-3: every ``step``-th frame at ``width`` px.

    Returns (frame indices, RGB float frames, scale factor relative to full resolution).
    Frames are cached under data/common_clip/subset_<width>/ after the first call.
    """
    info = sp.video_info(COMMON_CLIP)
    factor = width / info["width"]
    cache_dir = DATA_DIR / "common_clip" / f"subset_{width}"
    indices = list(range(0, info["frames"], step))
    if cache and all(frame_path(i, cache_dir).exists() for i in indices):
        return indices, [sp.read_image(frame_path(i, cache_dir)) for i in indices], factor
    frames = []
    for idx, frame in sp.iter_video(COMMON_CLIP, step=step, width=width):
        frames.append(frame)
        if cache:
            sp.write_image(frame_path(idx, cache_dir), frame)
    return indices[:len(frames)], frames, factor


def load_consecutive_frames(start: int, stop: int, width: int = NOTEBOOK_WIDTH) -> Tuple[List[int], List[np.ndarray], float]:
    """Consecutive frames [start, stop) at ``width`` px, for optical-flow experiments."""
    info = sp.video_info(COMMON_CLIP)
    factor = width / info["width"]
    idx, frames = [], []
    for i, f in sp.iter_video(COMMON_CLIP, step=1, width=width, start=start, stop=stop):
        idx.append(i); frames.append(f)
    return idx, frames, factor


def subset_gt_corners(indices: List[int], factor: float, split: Optional[str] = None) -> Dict[int, np.ndarray]:
    """Ground-truth corners that fall on the notebook subset, scaled to the subset resolution."""
    gt = load_gt_corners(split=split, scale=factor)
    return {i: gt[i] for i in indices if i in gt}


# ---------------------------------------------------------------------------
# Diagnostic clips (same poster, one difficulty per clip) and personal clips
# ---------------------------------------------------------------------------

def diagnostic_clip_dir(clip_id: str) -> Path:
    return DIAGNOSTIC_DIR / clip_id


def diagnostic_clip_ids(factor: Optional[str] = None) -> List[str]:
    """Stress / diagnostic clip ids in manifest order; ``factor`` keeps only the clip of one capture factor
    (``"viewpoint"``, ``"scale"``, ``"blur"``, ``"exit"``, ``"combined"``, ``"rotation"``)."""
    if not DIAGNOSTIC_CORNERS_MANIFEST.exists():
        return []
    seen: List[str] = []
    for row in sp.read_csv(DIAGNOSTIC_CORNERS_MANIFEST):
        if factor is not None and str(row.get("factor", "")) != str(factor):
            continue
        if row["clip_id"] not in seen:
            seen.append(row["clip_id"])
    return seen


def _summary() -> dict:
    return sp.read_json(DATASET_SUMMARY) if DATASET_SUMMARY.exists() else {}


def clip_factor(clip_id: str) -> Optional[str]:
    """The capture factor a clip isolates (``"common"`` for the common clip, ``"viewpoint"``, ``"scale"``, ``"blur"``,
    ``"exit"``, ``"combined"``, ``"rotation"``), or None for a personal clip."""
    if clip_id == "common_clip":
        return "common"
    factors = _summary().get("factors", {})
    if clip_id in factors:
        return factors[clip_id]
    for row in sp.read_csv(DIAGNOSTIC_CORNERS_MANIFEST) if DIAGNOSTIC_CORNERS_MANIFEST.exists() else []:
        if row["clip_id"] == clip_id:
            return row.get("factor")
    return None


def stress_clips(stage: Optional[int] = None):
    """The stress clips of every Stage, ``{1: {"viewpoint": "coke_perspective", "scale": "coke_scale"}, 2: {"blur": ...,
    "exit": ...}, 3: {"viewpoint": ..., "combined": ...}}`` from ``dataset_summary.json``; with ``stage`` only that
    Stage's ``{factor: clip_id}`` dictionary (insertion order = table order)."""
    raw = _summary().get("stress_clips", {})
    table = {int(k): dict(v) for k, v in raw.items()}
    return table[int(stage)] if stage is not None else table


def stress_clip_ids() -> List[str]:
    """Every clip id that appears in some Stage's stress table, in Stage order, without duplicates."""
    out: List[str] = []
    for st in sorted(stress_clips()):
        for cid in stress_clips(st).values():
            if cid not in out:
                out.append(cid)
    return out


def load_diagnostic_corners(clip_id: str, manifest: Path = DIAGNOSTIC_CORNERS_MANIFEST,
                            scale: float = 1.0) -> Dict[int, np.ndarray]:
    """frame index -> 4 x 2 corners for one stress clip (all frames are 'test')."""
    out = {}
    for row in sp.read_csv(manifest):
        if row["clip_id"] != clip_id:
            continue
        out[int(row["frame"])] = sp.corners_from_row(row) * scale
    return out


def _diagnostic_mask_rows(clip_id: str, manifest: Path) -> List[dict]:
    if not Path(manifest).exists():
        return []
    return [row for row in sp.read_csv(manifest) if row["clip_id"] == clip_id]


def load_diagnostic_masks(clip_id: str, manifest: Path = DIAGNOSTIC_MASKS_MANIFEST) -> Dict[int, np.ndarray]:
    """frame index -> Boolean occlusion mask for one stress clip (``manifests/diagnostic_masks.csv``).

    The viewpoint clip's rows carry an *empty* mask (occluder "none"); it is returned as an all-False array so
    that the false-occlusion test can run on those frames."""
    out = {}
    for row in _diagnostic_mask_rows(clip_id, manifest):
        out[int(row["frame"])] = sp.read_mask(sp.resolve(manifest, row["mask_path"]))
    return out


def load_diagnostic_mask_corners(clip_id: str, manifest: Path = DIAGNOSTIC_MASKS_MANIFEST,
                                 scale: float = 1.0) -> Dict[int, np.ndarray]:
    """frame index -> 4 x 2 POT corners of the mask frames of one stress clip (the combined clip's mask frames
    are not stride-16 frames, so their exact corners are stored in ``diagnostic_masks.csv``)."""
    out = {}
    for row in _diagnostic_mask_rows(clip_id, manifest):
        out[int(row["frame"])] = sp.corners_from_row(row) * scale
    return out


def clip_paths(clip_id: str) -> Tuple[Path, Path]:
    """(video, reference image) for ``common_clip``, a stress clip id, or a personal clip id."""
    if clip_id == "common_clip":
        return COMMON_CLIP, BOARD_REFERENCE
    d = diagnostic_clip_dir(clip_id)
    if (d / "clip.mp4").exists():
        return d / "clip.mp4", d / "reference.png"
    d = personal_clip_dir(clip_id)
    return d / "clip.mp4", d / "reference.png"


def mask_frames_for_clip(clip_id: str) -> List[int]:
    """Frames with a ground-truth occlusion mask: the common clip's six masks, a stress clip's manifest masks,
    or a personal clip's audit masks (empty list when the clip has none / the audit manifest does not exist)."""
    if clip_id == "common_clip":
        return sorted(int(r["frame"]) for r in sp.read_csv(MASKS_MANIFEST))
    if clip_id in diagnostic_clip_ids():
        return sorted(int(r["frame"]) for r in _diagnostic_mask_rows(clip_id, DIAGNOSTIC_MASKS_MANIFEST))
    manifest = MANIFEST_DIR / "audit_masks.csv"
    if not manifest.exists():
        return []
    return sorted(int(r["frame"]) for r in sp.read_csv(manifest) if r["clip_id"] == clip_id)


def gt_corners_for_clip(clip_id: str, scale: float = 1.0) -> Dict[int, np.ndarray]:
    """Ground-truth corners of any clip id (common clip: all annotated frames; stress clip: manifest; personal clip:
    audit corners)."""
    if clip_id == "common_clip":
        return load_gt_corners(scale=scale)
    if clip_id in diagnostic_clip_ids():
        return load_diagnostic_corners(clip_id, scale=scale)
    return {f: c * scale for f, c in load_audit_corners(clip_id).items()}


def gt_masks_for_clip(clip_id: str) -> Dict[int, np.ndarray]:
    """Ground-truth occlusion masks of any clip id (see ``mask_frames_for_clip``)."""
    if clip_id == "common_clip":
        return load_gt_masks()
    if clip_id in diagnostic_clip_ids():
        return load_diagnostic_masks(clip_id)
    manifest = MANIFEST_DIR / "audit_masks.csv"
    return load_audit_masks(clip_id) if manifest.exists() else {}


def personal_clip_dir(clip_id: str) -> Path:
    return PERSONAL_DIR / clip_id


def load_audit_corners(clip_id: str, manifest: Path = MANIFEST_DIR / "audit_corners.csv",
                       annotator: Optional[str] = None) -> Dict[int, np.ndarray]:
    out = {}
    for row in sp.read_csv(manifest):
        if row["clip_id"] != clip_id:
            continue
        if annotator is not None and row.get("annotator", "") != annotator:
            continue
        out[int(row["frame"])] = sp.corners_from_row(row)
    return out


def load_audit_masks(clip_id: str, manifest: Path = MANIFEST_DIR / "audit_masks.csv") -> Dict[int, np.ndarray]:
    out = {}
    for row in sp.read_csv(manifest):
        if row["clip_id"] != clip_id:
            continue
        out[int(row["frame"])] = sp.read_mask(sp.resolve(manifest, row["mask_path"]))
    return out


def load_genai_conditions() -> dict:
    return sp.read_json(GENAI_CONDITIONS)


__all__ = [
    "CODE_DIR", "ROOT_DIR", "DATA_DIR", "MANIFEST_DIR", "OUTPUT_DIR", "COMMON_CLIP",
    "BOARD_REFERENCE", "FRAMES_DIR", "MASKS_DIR", "ARTWORK_DIR", "PERSONAL_DIR", "DIAGNOSTIC_DIR",
    "CORNERS_MANIFEST", "MASKS_MANIFEST", "DIAGNOSTIC_CORNERS_MANIFEST", "DIAGNOSTIC_MASKS_MANIFEST",
    "GENAI_CONDITIONS", "DATASET_SUMMARY", "GT_STRIDE",
    "split_of_annotation_index", "frame_path", "mask_path", "load_reference", "load_artwork",
    "load_frame", "load_gt_corners", "load_gt_masks", "scale_corners", "load_notebook_subset", "load_consecutive_frames",
    "subset_gt_corners", "diagnostic_clip_dir", "diagnostic_clip_ids", "clip_factor", "stress_clips", "stress_clip_ids",
    "load_diagnostic_corners", "load_diagnostic_masks", "load_diagnostic_mask_corners", "clip_paths",
    "mask_frames_for_clip", "gt_corners_for_clip", "gt_masks_for_clip", "personal_clip_dir", "load_audit_corners", "load_audit_masks", "load_genai_conditions",
]
