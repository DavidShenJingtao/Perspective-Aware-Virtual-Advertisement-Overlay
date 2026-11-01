"""Command-line entry point that produces the submitted overlay videos.

Examples::

    python -m overlaylab.run_pipeline --all --thresholds outputs/thresholds.json
    python -m overlaylab.run_pipeline --clip common_clip --variant independent
    python -m overlaylab.run_pipeline --clip coke_blur --variant flow_assisted --width 960
    python -m overlaylab.run_pipeline --clip my_clip --artwork my_artwork --thresholds outputs/thresholds.json

``--all`` runs every video the stress tables need with the frozen thresholds: the
common clip with both temporal variants and the no-occlusion baseline, and each
stress clip (``data.stress_clip_ids()``) with the flow-assisted variant; videos
that already exist are skipped unless ``--force`` is given. Afterwards
``python -m overlaylab.report`` evaluates them and writes the tables.

``--clip common_clip`` uses data/common_clip/board_clip.mp4 and the supplied
reference; a stress clip id (data/diagnostic/<clip_id>/) or a personal clip id
(data/personal/<clip_id>/) uses that folder's ``clip.mp4`` and ``reference.png``. Outputs go to
outputs/videos/<clip_id>_<variant>.mp4 plus a <clip_id>_<variant>_stats.json
with the per-frame corners, statistics and monitor flags, the occlusion masks of the
clip's ground-truth mask frames under <clip_id>_<variant>_masks/, and a .webm copy of the
video (VP9) that VS Code's built-in preview can play; ``--no-webm`` skips it.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
from typing import List, Tuple

from . import data
from . import supplied as sp
from .config import PipelineConfig
from .pipelines import run_clip


def all_runs() -> List[Tuple[str, str, bool]]:
    """(clip_id, variant, occlusion) for every video of the stress tables, in the order they are run."""
    runs = [("common_clip", "independent", True), ("common_clip", "flow_assisted", True),
            ("common_clip", "flow_assisted", False)]
    runs += [(cid, "flow_assisted", True) for cid in data.stress_clip_ids()]
    return runs


def run_one(clip: str, variant: str, occlusion: bool, artwork_name: str, thresholds, width=None,
            max_frames=None, out=None, webm: bool = True, force: bool = True) -> Path:
    video, ref_path = data.clip_paths(clip)
    reference = sp.read_image(ref_path)
    artwork = data.load_artwork(artwork_name)
    config = PipelineConfig.from_json(Path(thresholds).read_text()) if thresholds else PipelineConfig()
    config.track.variant = variant
    if width:
        config.resize_width = width
    suffix = variant + ("" if occlusion else "_no_occlusion")
    out = Path(out) if out else data.OUTPUT_DIR / "videos" / f"{clip}_{suffix}.mp4"
    stats = out.with_name(out.stem + "_stats.json")
    fingerprint = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        fingerprint.update(source.name.encode())
        fingerprint.update(source.read_bytes())
    request = {"code_sha256": fingerprint.hexdigest(),
               "source_sha256": hashlib.sha256(Path(video).read_bytes()).hexdigest(),
               "reference_sha256": hashlib.sha256(np.ascontiguousarray(reference).tobytes()).hexdigest(),
               "config": config.thresholds(), "width": config.resize_width, "variant": variant,
               "occlusion": occlusion, "max_frames": max_frames,
               "artwork_sha256": hashlib.sha256(np.ascontiguousarray(artwork).tobytes()).hexdigest()}
    if not force and out.exists() and stats.exists():
        prior = json.loads(stats.read_text())
        expected = min(max_frames or sp.video_info(video)["frames"], sp.video_info(video)["frames"])
        if prior.get("request") == request and prior.get("n_frames") == expected and sp.video_info(out)["frames"] == expected:
            print(f"[skip] {out.name}: complete run with matching settings (use --force to redo)")
            return out
    print(f"[run] {clip} {suffix} -> {out}")
    run_clip(video, reference, artwork, config, out, stats, max_frames=max_frames, occlusion=occlusion,
             webm=webm and sp.ffmpeg_available(), save_masks=data.mask_frames_for_clip(clip))
    saved = json.loads(stats.read_text())
    saved["request"] = request
    sp.write_json(stats, saved)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="common_clip")
    ap.add_argument("--variant", choices=["independent", "flow_assisted"], default="flow_assisted")
    ap.add_argument("--all", action="store_true", help="run every video of the stress tables (see above)")
    ap.add_argument("--force", action="store_true", help="with --all: redo videos that already exist")
    ap.add_argument("--artwork", default="ad", help="name under data/artwork or a path")
    ap.add_argument("--width", type=int, default=None, help="process at this width (default: full resolution)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-occlusion", action="store_true", help="baseline without occlusion handling")
    ap.add_argument("--thresholds", default=None, help="JSON PipelineConfig frozen on the validation frames")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-webm", action="store_true",
                    help="skip the VP9 .webm copy (written next to the .mp4 so that VS Code can play it)")
    args = ap.parse_args(argv)

    if args.all:
        for clip, variant, occlusion in all_runs():
            run_one(clip, variant, occlusion, args.artwork, args.thresholds, args.width, args.max_frames,
                    None, not args.no_webm, force=args.force)
        print("done; next: python -m overlaylab.report --artwork", args.artwork)
        return
    run_one(args.clip, args.variant, not args.no_occlusion, args.artwork, args.thresholds, args.width,
            args.max_frames, args.out, not args.no_webm, force=True)


if __name__ == "__main__":
    main()
