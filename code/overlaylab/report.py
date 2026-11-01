"""Supplied: build the three stress tables of the report from the finished runs.

    python -m overlaylab.report [--artwork ad] [--thresholds outputs/thresholds.json] [--width W]

Every Stage is reported on the same shape of table: the common clip (validation and test frames) and the
Stage's two stress clips, named by the capture factor they isolate (``data.stress_clips()``). The tool knows
which clip belongs to which Stage, so nothing has to be looked up by hand: run
``python -m overlaylab.run_pipeline --all --thresholds outputs/thresholds.json`` first, then this command.
It writes ``outputs/stress_tables.md`` (paste it into the report) and ``outputs/stress_tables.json``.

* **Stage 1 Localise** -- ``localise_board`` is run here, with the frozen configuration, on every annotated
  frame of the common clip (validation / test) and of the two Stage 1 stress clips (viewpoint, scale):
  median and 90th-percentile corner error, success at the frozen threshold, median inlier count and ratio,
  and the corner error of a 6-parameter affine map fitted by least squares to the *same* RANSAC inliers
  (``affine_from_points``; large on the viewpoint clip, where only a homography fits).
* **Stage 2 Track** -- from the ``_stats.json`` written by ``run_pipeline`` (per-frame corners, status,
  monitor flags, reinitialisations): the same corner metrics on the annotated frames, jitter
  (``metrics.jitter``), drift (median corner error on annotated frames at least ``DRIFT_AFTER`` frames after
  the last (re)initialisation, ``metrics.drift``; n/a when the tracker never carries a track that long), the
  longest run of frames without a (re)initialisation, frames tracked, and the REINIT / HIDE counts. The common
  clip is reported for both temporal variants, the stress clips (blur, exit) for the flow-assisted variant.
* **Stage 3 Composite** -- mask IoU and F1 from the occlusion masks ``run_pipeline`` saves on the clip's
  ground-truth mask frames (``<video>_masks/``), leakage and the false-occlusion fraction from the evaluator
  report of the video (``overlaylab.evaluate``, computed here if ``outputs/evaluation/`` has none yet), and
  the leakage of the no-occlusion baseline on the common clip. Rows: common (validation / test masks), the
  combined clip (six masks) and the viewpoint clip (three empty masks: false occlusion only).

All numbers come from ``overlaylab.metrics`` or ``overlaylab.evaluate``; the ground truth is read only here,
never inside the pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import data
from . import supplied as sp
from . import metrics as M
from .config import PipelineConfig

DRIFT_AFTER = 16          # frames since the last (re)initialisation after which an annotated frame counts for drift (one GT stride)
STAGE_TITLES = {1: "Stage 1 Localise", 2: "Stage 2 Track", 3: "Stage 3 Composite"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def affine_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 6-parameter affine map src -> dst as a 3 x 3 matrix with last row (0, 0, 1)."""
    A = np.c_[np.asarray(src, float), np.ones(len(src))]
    Mx = np.linalg.lstsq(A, np.asarray(dst, float), rcond=None)[0]
    return np.vstack([Mx.T, [0.0, 0.0, 1.0]])


def row_label(clip_id: str, split: Optional[str] = None, variant: Optional[str] = None) -> str:
    factor = data.clip_factor(clip_id) or clip_id
    label = factor if clip_id == "common_clip" else f"{factor} ({clip_id})"
    if split:
        label += f" · {split}"
    if variant:
        label += f" · {variant.replace('_', '-')}"
    return label


def _fmt(v, kind: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if kind == "pct":
        return f"{100 * v:.2f}%"
    if kind == "pct0":
        return f"{100 * v:.0f}%"
    if kind == "px":
        return f"{v:.2f} px"
    if kind == "f3":
        return f"{v:.3f}"
    if kind == "int":
        return f"{int(v)}"
    return str(v)


def markdown_table(columns: Sequence[Tuple[str, str, str]], rows: Sequence[Dict]) -> str:
    """``columns`` = (key, header, kind); ``rows`` = dicts. Missing rows carry a ``note`` that spans the table."""
    head = "| row | " + " | ".join(h for _, h, _ in columns) + " |"
    sep = "|---|" + "|".join("---" for _ in columns) + "|"
    out = [head, sep]
    for r in rows:
        if r.get("note") and all(r.get(k) is None for k, _, _ in columns):
            out.append(f"| {r['label']} | {r['note']} |" + " |" * (len(columns) - 1))
            continue
        out.append(f"| {r['label']} | " + " | ".join(_fmt(r.get(k), kind) for k, _, kind in columns) + " |")
    return "\n".join(out)


def _config(thresholds: Optional[str], width: Optional[int]) -> PipelineConfig:
    cfg = PipelineConfig.from_json(Path(thresholds).read_text()) if thresholds else PipelineConfig()
    if width:
        cfg.resize_width = width
    return cfg


def _frames_at(video, indices: Sequence[int], width: Optional[int]) -> Dict[int, np.ndarray]:
    wanted = set(int(i) for i in indices)
    out = {}
    for i, f in sp.iter_video(video, step=1, width=width, stop=(max(wanted) + 1) if wanted else 0):
        if i in wanted:
            out[i] = f
    return out


def _errors_summary(errors: Dict[int, float], threshold: float) -> Dict:
    s = M.summarise_errors(list(errors.values()), threshold)
    return {"n": s["n"], "median_error": s["median"], "p90_error": s["p90"], "success": s["success"], "failed": s["failed"]}


# ---------------------------------------------------------------------------
# Stage 1: localise_board on the annotated frames
# ---------------------------------------------------------------------------

def stage1_row(clip_id: str, gt: Dict[int, np.ndarray], config: PipelineConfig, label: str, verbose: bool = True) -> Dict:
    from .localisation import localise_board          # student code
    from .geometry import transfer_points
    from .pipelines import _scale_reference
    video, ref_path = data.clip_paths(clip_id)
    info = sp.video_info(video)
    width = config.resize_width
    factor = (width / info["width"]) if width and width != info["width"] else 1.0
    reference = _scale_reference(sp.read_image(ref_path), config, info["width"])
    from copy import deepcopy
    cfg = deepcopy(config.localisation)
    cfg.warp.reference_shape = tuple(reference.shape[:2])
    ref_rect = sp.rect_corners(reference.shape[1], reference.shape[0])
    frames = _frames_at(video, sorted(gt), width)
    errs, aff, inl, rat, per_frame = {}, {}, [], [], {}
    for f in sorted(gt):
        if f not in frames:
            continue
        g = np.asarray(gt[f]) * factor
        H, corners, pairs, inliers, diag = localise_board(reference, frames[f], cfg)
        e = M.corner_error(corners, g) if H is not None and diag.get("status", False) else float("inf")
        errs[f] = e
        n_in = int(np.sum(inliers)) if inliers is not None and len(inliers) else 0
        inl.append(n_in); rat.append(float(diag.get("inlier_ratio", float("nan"))))
        if np.isfinite(e) and n_in >= 4:          # the affine comparison is made on the frames the localiser solved
            src = diag["keypoints_ref"][pairs[inliers, 0]]; dst = diag["keypoints_frame"][pairs[inliers, 1]]
            aff[f] = M.corner_error(transfer_points(affine_from_points(src, dst), ref_rect), g)
        else:
            aff[f] = float("inf")
        per_frame[int(f)] = {"error": e, "affine_error": aff[f], "inliers": n_in, "inlier_ratio": rat[-1]}
        if verbose:
            print(f"  [stage 1] {label}: frame {f:4d} error {e * 100:6.2f}%  affine {aff[f] * 100:6.2f}%  inliers {n_in:4d} ({rat[-1]:.2f})")
    row = {"label": label, "clip_id": clip_id, **_errors_summary(errs, cfg.success_threshold),
           "median_inliers": float(np.median(inl)) if inl else float("nan"),
           "median_inlier_ratio": float(np.nanmedian(rat)) if rat else float("nan"),
           "affine_median_error": float(np.median([v for v in aff.values() if np.isfinite(v)])) if any(np.isfinite(list(aff.values()))) else float("nan"),
           "affine_success": M.success_rate(list(aff.values()), cfg.success_threshold),
           "per_frame": per_frame}
    return row


STAGE1_COLUMNS = [("n", "frames", "int"), ("median_error", "median err", "pct"), ("p90_error", "p90 err", "pct"),
                  ("success", "success", "pct0"), ("median_inliers", "inliers (median)", "int"),
                  ("median_inlier_ratio", "inlier ratio", "f3"), ("affine_median_error", "affine median err", "pct"),
                  ("affine_success", "affine success", "pct0")]
# Both success columns use every eligible annotated frame; unsuccessful localisation is a failure for both.
# Median and p90 errors describe finite estimates, alongside the explicit success/failure counts.


def stage1_table(config: PipelineConfig, verbose: bool = True) -> List[Dict]:
    rows = []
    for split in ("validation", "test"):
        rows.append(stage1_row("common_clip", data.load_gt_corners(split=split), config, row_label("common_clip", split), verbose))
    for factor, cid in data.stress_clips(1).items():
        rows.append(stage1_row(cid, data.load_diagnostic_corners(cid), config, row_label(cid), verbose))
    return rows


# ---------------------------------------------------------------------------
# Stage 2: from the _stats.json of the runs
# ---------------------------------------------------------------------------

def stats_path(clip_id: str, variant: str, occlusion: bool = True) -> Path:
    suffix = variant + ("" if occlusion else "_no_occlusion")
    return data.OUTPUT_DIR / "videos" / f"{clip_id}_{suffix}_stats.json"


def summarise_tracking(stats: Dict, gt: Dict[int, np.ndarray], threshold: float, full_width: int,
                       drift_after: int = DRIFT_AFTER, motion_gt: Optional[Dict[int, np.ndarray]] = None) -> Dict:
    """Corner metrics, jitter, drift, frames tracked and monitor counts of one run on the annotated frames ``gt``
    (full-resolution corners; the run's corners are rescaled from ``stats['resize_width']``)."""
    scale = full_width / (stats.get("resize_width") or full_width)
    corners = [np.asarray(c, float) * scale for c in stats["corners"]]
    status = stats["status"]; flags = [tuple(f) for f in stats["flags"]]
    n = len(corners)
    hidden = {i for i, (fl, _) in enumerate(flags) if fl == "HIDE"}
    pred = {i: corners[i] for i in range(n) if status[i] == "ok" and np.all(np.isfinite(corners[i])) and i not in hidden}
    errs = {f: (M.corner_error(pred[f], gt[f]) if f in pred else float("inf")) for f in gt}
    motion = motion_gt if motion_gt is not None else gt
    gt_frames = sorted(motion); gt_list = [np.asarray(motion[f]) for f in gt_frames]
    jit = M.jitter(pred, gt_frames, gt_list) if len(gt_frames) >= 2 else float("nan")
    reinits = sorted(set(int(r) for r in stats.get("reinit_frames", [])))
    since = M.drift(errs, reinits)
    late = [e for s, e in since if s >= drift_after and np.isfinite(e)]
    runs = np.diff(reinits + [n]) if reinits else np.array([n])
    out = {**_errors_summary(errs, threshold), "jitter_px": jit,
           "drift_error": float(np.median(late)) if late else float("nan"), "drift_frames": len(late),
           "longest_run": int(runs.max()) if len(runs) else n,
           "frames_tracked": sum(1 for s in status if s == "ok"), "frames_total": n, "localisations": len(reinits),
           "reinit": sum(1 for fl, _ in flags if fl == "REINIT"), "hide": sum(1 for fl, _ in flags if fl == "HIDE"),
           "per_frame": {int(f): (None if not np.isfinite(e) else float(e)) for f, e in errs.items()}}
    return out


STAGE2_COLUMNS = [("n", "frames", "int"), ("median_error", "median err", "pct"), ("p90_error", "p90 err", "pct"),
                  ("success", "success", "pct0"), ("jitter_px", "jitter", "px"), ("drift_error", f"drift (≥{DRIFT_AFTER} fr)", "pct"),
                  ("longest_run", "longest run", "int"), ("frames_tracked", "tracked", "int"), ("localisations", "(re)localisations", "int"), ("reinit", "REINIT flags", "int"), ("hide", "HIDE flags", "int")]


def _validate_saved_run(stats: Dict, config: PipelineConfig, clip_id: str) -> None:
    """Keep all table rows tied to the same complete, frozen configuration."""
    from copy import deepcopy
    saved = deepcopy(stats.get("frozen_config"))
    expected = config.thresholds()
    if saved is None:
        raise ValueError("run metadata predates this release; regenerate videos with --all --force")
    saved["track"]["variant"] = expected["track"]["variant"]  # intended comparison factor
    if saved != expected or stats.get("resize_width") != config.resize_width:
        raise ValueError(f"{clip_id}: settings/resolution differ from this table; rerun with the frozen thresholds")
    required = sp.video_info(data.clip_paths(clip_id)[0])["frames"]
    if stats.get("n_frames") != required:
        raise ValueError(f"{clip_id}: incomplete run; expected {required} frames")


def stage2_table(config: PipelineConfig, verbose: bool = True) -> List[Dict]:
    thr = config.localisation.success_threshold
    summary = sp.read_json(data.DATASET_SUMMARY)
    variants = summary.get("stress_tables", {}).get("common_clip_variants", ["independent", "flow_assisted"])
    default_variant = summary.get("stress_tables", {}).get("variant", "flow_assisted")
    rows = []

    def add(clip_id, gt, variant, label):
        p = stats_path(clip_id, variant)
        if not p.exists():
            rows.append({"label": label, "clip_id": clip_id, "variant": variant,
                         "note": f"missing {p.name}: python -m overlaylab.run_pipeline --clip {clip_id} --variant {variant} --thresholds outputs/thresholds.json"})
            return
        st = sp.read_json(p)
        _validate_saved_run(st, config, clip_id)
        full_w = sp.video_info(data.clip_paths(clip_id)[0])["width"]
        rows.append({"label": label, "clip_id": clip_id, "variant": variant, **summarise_tracking(st, gt, thr, full_w, motion_gt=data.gt_corners_for_clip(clip_id))})
        if verbose:
            r = rows[-1]
            print(f"  [stage 2] {label}: median {r['median_error'] * 100:.2f}%  success {r['success']:.0%}  jitter {r['jitter_px']:.2f}px  tracked {r['frames_tracked']}/{r['frames_total']}  REINIT {r['reinit']}  HIDE {r['hide']}")

    for split in ("validation", "test"):
        gt = data.load_gt_corners(split=split)
        for v in variants:
            add("common_clip", gt, v, row_label("common_clip", split, v))
    for factor, cid in data.stress_clips(2).items():
        add(cid, data.load_diagnostic_corners(cid), default_variant, row_label(cid, None, default_variant))
    return rows


# ---------------------------------------------------------------------------
# Stage 3: saved masks + evaluator reports
# ---------------------------------------------------------------------------

def evaluation_report(clip_id: str, variant: str, artwork_name: str, occlusion: bool = True, verbose: bool = True, success_threshold: Optional[float] = None) -> Optional[Dict]:
    """Evaluate the current video and write a fresh report, avoiding stale cached scores."""
    from . import evaluate as E
    suffix = variant + ("" if occlusion else "_no_occlusion")
    video = data.OUTPUT_DIR / "videos" / f"{clip_id}_{suffix}.mp4"
    out = data.OUTPUT_DIR / "evaluation" / f"{video.stem}_report.json"
    # Recompute from the current video: a cached report can predate a rerun.

    if not video.exists():
        return None
    if verbose:
        print(f"  [evaluate] {video.name} (no report in outputs/evaluation/ yet)")
    gt_c, gt_m, extra = E.ground_truth_for_clip(clip_id)
    frozen = data.OUTPUT_DIR / "thresholds.json"
    threshold = success_threshold if success_threshold is not None else (PipelineConfig.from_json(frozen.read_text()).localisation.success_threshold if frozen.exists() else 0.02)
    rep = E.evaluate_video(video, data.clip_paths(clip_id)[0], data.load_artwork(artwork_name), success_threshold=threshold, gt_corners=gt_c,
                           gt_masks=gt_m, extra_corners=extra, verbose=False)
    sp.write_json(out, rep)
    return rep


def mask_scores(stats: Dict, gt_masks: Dict[int, np.ndarray], frames: Sequence[int]) -> Dict:
    """IoU / F1 of the saved occlusion masks against the ground truth on ``frames`` (empty GT masks are skipped)."""
    mask_dir = Path(stats["mask_dir"]) if stats.get("mask_dir") else None
    ious, f1s, hidden, missing = [], [], [], []
    for f in frames:
        g = gt_masks[f]
        if not np.any(g):
            continue
        p = mask_dir / f"mask_{int(f):06d}.png" if mask_dir else None
        if f in stats.get("mask_frames_hidden", []):
            hidden.append(int(f)); continue
        if p is None or not p.exists():
            missing.append(int(f)); continue
        m = sp.read_mask(p)
        if m.shape != g.shape:
            g = sp.resize_width(g.astype(np.uint8) * 255, m.shape[1]) > 127
        ious.append(M.mask_iou(m, g)); f1s.append(M.mask_f1(m, g))
    return {"iou": float(np.mean(ious)) if ious else float("nan"), "f1": float(np.mean(f1s)) if f1s else float("nan"),
            "mask_frames_scored": len(ious), "mask_frames_hidden": hidden, "mask_frames_missing": missing}


def _per_frame_mean(report: Optional[Dict], key: str, frames: Sequence[int]) -> float:
    if not report:
        return float("nan")
    vals = [d[key] for d in report["occlusion"].get("per_frame", []) if d["frame"] in set(int(f) for f in frames)
            and d.get(key) is not None and np.isfinite(d[key])]
    return float(np.mean(vals)) if vals else float("nan")


STAGE3_COLUMNS = [("mask_frames", "eligible masks", "int"), ("mask_frames_scored", "scored masks", "int"), ("hidden_count", "hidden masks", "int"), ("missing_count", "missing masks", "int"), ("absent_count", "overlay absent", "int"), ("iou", "IoU", "f3"), ("f1", "F1", "f3"),
                  ("leakage", "leakage", "pct"), ("baseline_leakage", "baseline leakage", "pct"),
                  ("false_occlusion", "false occlusion", "pct")]


def stage3_table(config: PipelineConfig, artwork_name: str, verbose: bool = True) -> List[Dict]:
    summary = sp.read_json(data.DATASET_SUMMARY)
    variant = summary.get("stress_tables", {}).get("variant", "flow_assisted")
    rows = []
    reports = {}  # validation/test rows share the same current-video evaluation

    def add(clip_id, frames, gt_masks, label, baseline: bool):
        p = stats_path(clip_id, variant)
        if not p.exists():
            rows.append({"label": label, "clip_id": clip_id, "note": f"missing {p.name}: python -m overlaylab.run_pipeline --all --thresholds outputs/thresholds.json"})
            return
        st = sp.read_json(p)
        _validate_saved_run(st, config, clip_id)
        st["mask_dir"] = str(p.with_name(p.stem.removesuffix("_stats") + "_masks"))
        if (clip_id, True) not in reports:
            reports[(clip_id, True)] = evaluation_report(clip_id, variant, artwork_name, True, verbose, config.localisation.success_threshold)
        rep = reports[(clip_id, True)]
        if baseline and (clip_id, False) not in reports:
            reports[(clip_id, False)] = evaluation_report(clip_id, variant, artwork_name, False, verbose, config.localisation.success_threshold)
        base = reports.get((clip_id, False)) if baseline else None
        scores = mask_scores(st, gt_masks, frames)
        row = {"label": label, "clip_id": clip_id, "mask_frames": len(frames), **scores,
               "leakage": _per_frame_mean(rep, "leakage", frames),
               "baseline_leakage": _per_frame_mean(base, "leakage", frames) if baseline else float("nan"),
               "false_occlusion": _per_frame_mean(rep, "false_occlusion", frames),
               "overlay_absent_frames": [f for f in (rep or {}).get("occlusion", {}).get("overlay_absent_frames", []) if f in set(frames)]}
        row.update(hidden_count=sum(f in st.get("mask_frames_hidden", []) for f in frames),
                   missing_count=sum(f not in st.get("mask_frames_hidden", []) and not (Path(st["mask_dir"]) / f"mask_{f:06d}.png").exists() for f in frames),
                   absent_count=len(row["overlay_absent_frames"]))
        rows.append(row)
        if verbose:
            print(f"  [stage 3] {label}: IoU {row['iou']:.3f}  F1 {row['f1']:.3f}  leakage {row['leakage'] * 100:.1f}%  "
                  f"false occlusion {row['false_occlusion'] * 100:.1f}%  (masks scored {scores['mask_frames_scored']}, hidden {scores['mask_frames_hidden']})")

    gt_all = data.load_gt_masks()
    for split in ("validation", "test"):
        frames = sorted(data.load_gt_masks(split=split))
        add("common_clip", frames, gt_all, row_label("common_clip", split), baseline=True)
    for factor, cid in reversed(list(data.stress_clips(3).items())):     # combined (real masks) before viewpoint (empty masks)
        gm = data.load_diagnostic_masks(cid)
        add(cid, sorted(gm), gm, row_label(cid), baseline=False)
    return rows


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def render(tables: Dict[int, List[Dict]], config: PipelineConfig, artwork_name: str) -> str:
    thr = config.localisation.success_threshold
    parts = ["# Stress tables", "",
             f"Generated by `python -m overlaylab.report` with the frozen thresholds (success threshold {thr:.2%}), artwork `{artwork_name}`"
             + (f", processing width {config.resize_width}" if config.resize_width else ", full resolution") + ".", ""]
    notes = {
        1: "`localise_board` on every annotated frame with the frozen configuration; *affine* = 6-parameter least-squares fit to the same RANSAC inliers.",
        2: f"From the `_stats.json` of each run. Jitter uses the full available corner trajectory for motion interpolation. Jitter, longest run, tracked and the counts are whole-clip statistics (identical in validation and test rows for a variant); (re)localisations counts actual localiser calls, whereas REINIT/HIDE count monitor flags; drift = median corner error on the row's annotated frames at least {DRIFT_AFTER} frames after the last (re)initialisation (n/a when no track is carried that long: the independent variant reinitialises every frame); longest run = most consecutive frames without a (re)initialisation.",
        3: "IoU / F1 from the occlusion masks saved by `run_pipeline` on the ground-truth mask frames (frames on which the overlay is hidden are not scored); leakage, baseline leakage and false occlusion from `overlaylab.evaluate` on the finished videos. The viewpoint clip has empty masks: only false occlusion applies.",
    }
    cols = {1: STAGE1_COLUMNS, 2: STAGE2_COLUMNS, 3: STAGE3_COLUMNS}
    for st in (1, 2, 3):
        if st not in tables:
            continue
        parts += [f"## {STAGE_TITLES[st]}", "", markdown_table(cols[st], tables[st]), "", notes[st], ""]
    return "\n".join(parts)


def build(artwork_name: str = "ad", thresholds: Optional[str] = None, width: Optional[int] = None,
          stages: Sequence[int] = (1, 2, 3), verbose: bool = True) -> Tuple[str, Dict]:
    config = _config(thresholds, width)
    tables: Dict[int, List[Dict]] = {}
    if 1 in stages:
        tables[1] = stage1_table(config, verbose)
    if 2 in stages:
        tables[2] = stage2_table(config, verbose)
    if 3 in stages:
        tables[3] = stage3_table(config, artwork_name, verbose)
    md = render(tables, config, artwork_name)
    return md, {"artwork": artwork_name, "thresholds": thresholds, "width": width,
                "success_threshold": config.localisation.success_threshold, "tables": {str(k): v for k, v in tables.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artwork", default="ad", help="artwork used for the runs (name under data/artwork or a path)")
    default_thr = data.OUTPUT_DIR / "thresholds.json"
    ap.add_argument("--thresholds", default=str(default_thr) if default_thr.exists() else None,
                    help="frozen configuration (default: outputs/thresholds.json when it exists)")
    ap.add_argument("--width", type=int, default=None, help="processing width of the runs (default: full resolution)")
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default=str(data.OUTPUT_DIR / "stress_tables.md"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    md, payload = build(args.artwork, args.thresholds, args.width, args.stages, verbose=not args.quiet)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    sp.write_json(out.with_suffix(".json"), payload)
    print(md)
    print("written to", out, "and", out.with_suffix(".json"))


__all__ = ["affine_from_points", "row_label", "markdown_table", "summarise_tracking", "mask_scores",
           "stage1_table", "stage2_table", "stage3_table", "render", "build", "DRIFT_AFTER"]

if __name__ == "__main__":
    main()
