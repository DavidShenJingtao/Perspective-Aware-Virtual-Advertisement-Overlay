"""Extract PNG frames from a clip with zero-padded indices.

    python -m overlaylab.extract_frames ../data/common_clip            # board_clip.mp4 -> frames/
    python -m overlaylab.extract_frames ../data/personal/primary_01    # clip.mp4 -> frames/
    python -m overlaylab.extract_frames ../data/common_clip --only 0 15 30   # selected frames only

Do not rename the frames: the manifests reference them by index.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import supplied as sp


def extract(clip_dir, step: int = 1, only=None, width=None) -> int:
    clip_dir = Path(clip_dir)
    video = clip_dir / "board_clip.mp4"
    if not video.exists():
        video = clip_dir / "clip.mp4"
    out = clip_dir / "frames"
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(only) if only else None
    n = 0
    for idx, frame in sp.iter_video(video, step=step, width=width):
        if wanted is not None and idx not in wanted:
            continue
        sp.write_image(out / f"frame_{idx:06d}.png", frame)
        n += 1
    return n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip_dir")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--only", type=int, nargs="*", default=None)
    ap.add_argument("--width", type=int, default=None)
    args = ap.parse_args(argv)
    n = extract(args.clip_dir, args.step, args.only, args.width)
    print(f"wrote {n} frames to {Path(args.clip_dir) / 'frames'}")


if __name__ == "__main__":
    main()
