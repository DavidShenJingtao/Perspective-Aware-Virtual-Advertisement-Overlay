"""Convert videos to VP9 WebM so that they play inside VS Code's built-in preview.

    python -m overlaylab.to_webm outputs/videos/*.mp4
    python -m overlaylab.to_webm data/common_clip/board_clip.mp4 --out-dir outputs/preview

VS Code's Electron build ships no H.264 / MPEG-4 decoder, so the .mp4 files written by
run_pipeline (mp4v) and the data bundle (H.264) show up as "cannot play" there. The .webm
copy is for viewing only; keep submitting and evaluating the .mp4.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import supplied as sp


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out-dir", default=None, help="write the .webm files here instead of next to the sources")
    ap.add_argument("--crf", type=int, default=32, help="VP9 quality (lower = better, larger); 32 is visually close to the source")
    ap.add_argument("--speed", type=int, default=4, help="libvpx-vp9 -cpu-used (0 slowest/best .. 8 fastest)")
    args = ap.parse_args(argv)
    if not sp.ffmpeg_available():
        raise SystemExit("ffmpeg not found on PATH")
    for v in args.videos:
        src = Path(v)
        out = Path(args.out_dir) / src.with_suffix(".webm").name if args.out_dir else None
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
        dst = sp.transcode_for_preview(src, out=out, crf=args.crf, speed=args.speed)
        print(f"{src} -> {dst}")


if __name__ == "__main__":
    main()
