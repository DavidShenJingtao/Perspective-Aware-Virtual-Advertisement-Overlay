"""Audit annotation tool for your own clip (Stage 4): board corners, occlusion masks, and a check.

    python -m overlaylab.annotate corners <clip_id> [--frames 0 40 80 ...] [--count 10] [--annotator <name>]
    python -m overlaylab.annotate masks   <clip_id> --frames 150 192 230 ... [--occluder "blue folder"] [--annotator <name>]
    python -m overlaylab.annotate check   <clip_id>

The clip is data/personal/<clip_id>/clip.mp4 with its rectified board data/personal/<clip_id>/reference.png.
``corners`` and ``masks`` open an OpenCV window (run them on your own machine, not on a remote server without a
display); ``check`` needs no window and writes preview images to outputs/annotation_check/<clip_id>/.

What is written
* data/personal/<clip_id>/frames/frame_XXXXXX.png    every annotated frame, extracted from the clip
* manifests/audit_corners.csv                        one row per (frame, annotator): x1,y1 .. x4,y4 in pixels of the
                                                     full-resolution frame, in the corner order of reference.png:
                                                     1 = top-left, 2 = top-right, 3 = bottom-right, 4 = bottom-left
* data/personal/<clip_id>/masks/mask_XXXXXX.png      binary mask (white = occluder in front of the board), frame size
* manifests/audit_masks.csv                          one row per mask frame
Re-annotating a frame replaces its row; rows of other clips and other annotators are kept.

Corners window: left click = place the next corner (a magnifier shows the area under the cursor), u = undo the last
corner, r = restart this frame, Enter / Space = accept (after four corners), n = skip this frame, q = save and quit.
Masks window: left click = add a polygon vertex around the occluder, Enter = close and fill the polygon, b = brush mode
(drag to paint), e = eraser mode, p = back to polygon mode, + / - = brush size, u = undo, r = clear,
s = save this mask and go to the next frame (an empty mask is allowed: it means nothing occludes the board),
n = skip, q = save and quit. If the frame has audit corners, the mask is clipped to the board quadrilateral, so
annotate the corners of your mask frames first.

Annotate before running any method on the clip (annotated_before_inference = yes), and do not use a promptable
segmenter for the audit masks.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import data
from . import supplied as sp

CORNER_FIELDS = ["clip_id", "frame", "frame_path", "annotator", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
                 "annotated_before_inference"]
MASK_FIELDS = ["clip_id", "frame", "frame_path", "mask_path", "annotator", "occluder", "annotated_before_inference"]
CORNER_NAMES = ["1 top-left", "2 top-right", "3 bottom-right", "4 bottom-left"]
COLOURS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255)]          # BGR, one per corner
MAX_W, MAX_H = 1400, 820                                                   # display size limit


# --------------------------------------------------------------------------------------------- files and manifests
def clip_video(clip_id: str) -> Path:
    d = data.personal_clip_dir(clip_id)
    for name in ("clip.mp4", "clip.mov", "clip.avi"):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"no clip.mp4 in {d}")


def read_frames(video: Path, frames: Sequence[int]) -> Dict[int, np.ndarray]:
    """BGR uint8 frames by index."""
    want, out = set(frames), {}
    cap = cv2.VideoCapture(str(video))
    i = 0
    while len(out) < len(want):
        ok, im = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = im
        i += 1
    cap.release()
    missing = sorted(want - set(out))
    if missing:
        raise ValueError(f"frames {missing} are beyond the end of {video} ({i} frames)")
    return out


def rel(path: Path) -> str:
    """Path as written in the manifests (relative to the manifest folder, like the supplied manifests)."""
    import os
    return Path(os.path.relpath(Path(path).absolute(), data.MANIFEST_DIR.absolute())).as_posix()


def read_rows(manifest: Path) -> List[dict]:
    return sp.read_csv(manifest) if manifest.exists() else []


def upsert_rows(manifest: Path, fields: Sequence[str], new: Sequence[dict]) -> None:
    """Replace rows with the same (clip_id, frame, annotator), keep every other row, sort by clip, frame, annotator."""
    keys = {(r["clip_id"], str(r["frame"]), r.get("annotator", "")) for r in new}
    rows = [r for r in read_rows(manifest) if (r["clip_id"], str(r["frame"]), r.get("annotator", "")) not in keys]
    rows += [{k: r.get(k, "") for k in fields} for r in new]
    rows.sort(key=lambda r: (r["clip_id"], int(r["frame"]), r.get("annotator", "")))
    sp.write_csv(manifest, rows, fields)


def save_frame(clip_id: str, f: int, im: np.ndarray) -> Path:
    p = data.personal_clip_dir(clip_id) / "frames" / f"frame_{f:06d}.png"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), im)
    return p


# --------------------------------------------------------------------------------------------- geometry checks
def signed_area(q: np.ndarray) -> float:
    x, y = q[:, 0], q[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def quad_problems(q: np.ndarray, shape: Tuple[int, int]) -> List[str]:
    """Reasons why a 4 x 2 corner array (image coordinates) is not a usable board annotation."""
    h, w = shape
    out = []
    if q.shape != (4, 2) or not np.all(np.isfinite(q)):
        return ["needs four finite corners"]
    if np.any(q[:, 0] < -0.5 * w) or np.any(q[:, 0] > 1.5 * w) or np.any(q[:, 1] < -0.5 * h) or np.any(q[:, 1] > 1.5 * h):
        out.append("corners far outside the frame")
    a = signed_area(q)
    if abs(a) < 100:
        out.append("degenerate quadrilateral (area < 100 px)")
    elif a < 0:
        out.append("corners go anticlockwise on screen: the order must be top-left, top-right, bottom-right, bottom-left "
                   "of reference.png (clockwise)")
    cross = [np.cross(q[(i + 1) % 4] - q[i], q[(i + 2) % 4] - q[(i + 1) % 4]) for i in range(4)]
    if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
        out.append("not convex (corners crossed or in the wrong order)")
    return out


# --------------------------------------------------------------------------------------------- interactive states
class _View:
    """Fit a frame into the display and map display <-> frame coordinates."""

    def __init__(self, im: np.ndarray):
        h, w = im.shape[:2]
        self.s = min(1.0, MAX_W / w, MAX_H / h)
        self.base = cv2.resize(im, (int(round(w * self.s)), int(round(h * self.s))), interpolation=cv2.INTER_AREA) if self.s < 1 else im.copy()
        self.full = im

    def to_frame(self, x, y):
        return np.array([x / self.s, y / self.s], float)

    def to_disp(self, p):
        return int(round(p[0] * self.s)), int(round(p[1] * self.s))

    def magnifier(self, canvas, x, y, zoom=4, half=24):
        """Draw a zoomed patch of the full-resolution frame around display point (x, y) in a canvas corner."""
        cx, cy = self.to_frame(x, y).astype(int)
        H, W = self.full.shape[:2]
        x0, y0 = np.clip(cx - half, 0, W - 2 * half), np.clip(cy - half, 0, H - 2 * half)
        patch = cv2.resize(self.full[y0:y0 + 2 * half, x0:x0 + 2 * half], (4 * half * zoom // 2, 4 * half * zoom // 2),
                           interpolation=cv2.INTER_NEAREST)
        ph, pw = patch.shape[:2]
        c = (int((cx - x0) * pw / (2 * half)), int((cy - y0) * ph / (2 * half)))
        cv2.line(patch, (c[0], 0), (c[0], ph), (0, 255, 255), 1); cv2.line(patch, (0, c[1]), (pw, c[1]), (0, 255, 255), 1)
        oy = 0 if y > canvas.shape[0] // 2 else canvas.shape[0] - ph
        ox = canvas.shape[1] - pw
        if ox >= 0 and oy >= 0:
            canvas[oy:oy + ph, ox:ox + pw] = patch
            cv2.rectangle(canvas, (ox, oy), (ox + pw - 1, oy + ph - 1), (255, 255, 255), 1)


def _text(canvas, lines, y=22):
    for i, t in enumerate(lines):
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(canvas, t, (10, y + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, th, cv2.LINE_AA)


class CornerState:
    """One frame of corner annotation. Feed it mouse and key events; ``done`` / ``result`` tell the caller what to do."""

    def __init__(self, im: np.ndarray, frame: int, k: int, n: int, initial: Optional[np.ndarray] = None):
        self.view, self.frame, self.k, self.n = _View(im), frame, k, n
        self.pts: List[np.ndarray] = [] if initial is None else [np.asarray(p, float) for p in initial]
        self.cursor = (0, 0)
        self.result: Optional[str] = None          # "accept", "skip", "quit"
        self.message = ""

    def mouse(self, event, x, y, flags=0):
        self.cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and len(self.pts) < 4:
            self.pts.append(self.view.to_frame(x, y))

    def key(self, k: int):
        if k in (ord("u"), 8, 127) and self.pts:
            self.pts.pop()
        elif k == ord("r"):
            self.pts = []
        elif k in (13, 10, 32):
            if len(self.pts) < 4:
                self.message = "place all four corners first"
                return
            probs = quad_problems(np.array(self.pts), self.view.full.shape[:2])
            if probs:
                self.message = "; ".join(probs) + " (press r to redo, or Enter again to keep)"
                if getattr(self, "_warned", None) == tuple(map(tuple, np.round(self.pts, 1))):
                    self.result = "accept"
                self._warned = tuple(map(tuple, np.round(self.pts, 1)))
                return
            self.result = "accept"
        elif k == ord("n"):
            self.result = "skip"
        elif k == ord("q"):
            self.result = "quit"

    @property
    def corners(self) -> np.ndarray:
        return np.array(self.pts, float)

    def render(self) -> np.ndarray:
        c = self.view.base.copy()
        d = [self.view.to_disp(p) for p in self.pts]
        for i in range(len(d) - 1):
            cv2.line(c, d[i], d[i + 1], (0, 255, 0), 2)
        if len(d) == 4:
            cv2.line(c, d[3], d[0], (0, 255, 0), 2)
        for i, p in enumerate(d):
            cv2.circle(c, p, 6, COLOURS[i], -1); cv2.circle(c, p, 7, (255, 255, 255), 1)
            cv2.putText(c, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOURS[i], 2)
        nxt = CORNER_NAMES[len(self.pts)] if len(self.pts) < 4 else "Enter = accept"
        _text(c, [f"frame {self.frame}  ({self.k}/{self.n})   next: {nxt}",
                  "click corner | u undo | r restart | Enter accept | n skip | q save+quit", self.message])
        self.view.magnifier(c, *self.cursor)
        return c


class MaskState:
    """One frame of mask annotation: polygons and a brush; white = occluder."""

    def __init__(self, im: np.ndarray, frame: int, k: int, n: int, board: Optional[np.ndarray] = None,
                 initial: Optional[np.ndarray] = None):
        self.view, self.frame, self.k, self.n = _View(im), frame, k, n
        h, w = im.shape[:2]
        self.mask = np.zeros((h, w), bool) if initial is None else initial.astype(bool).copy()
        self.board = None if board is None else sp.quad_mask((h, w), board)
        self.poly: List[np.ndarray] = []
        self.mode, self.radius = "polygon", 12
        self.history: List[np.ndarray] = []
        self.cursor = (0, 0)
        self.result: Optional[str] = None
        self.message = ""

    def _push(self):
        self.history.append(self.mask.copy())
        self.history = self.history[-30:]

    def _paint(self, x, y, value: bool):
        p = self.view.to_frame(x, y).astype(int)
        m = np.zeros(self.mask.shape, np.uint8)
        cv2.circle(m, (int(p[0]), int(p[1])), int(round(self.radius / self.view.s)), 1, -1)
        self.mask[m > 0] = value

    def mouse(self, event, x, y, flags=0):
        self.cursor = (x, y)
        if self.mode == "polygon":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.poly.append(self.view.to_frame(x, y))
        else:
            value = self.mode == "brush"
            if event == cv2.EVENT_LBUTTONDOWN:
                self._push(); self._paint(x, y, value)
            elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
                self._paint(x, y, value)

    def close_polygon(self):
        if len(self.poly) >= 3:
            self._push()
            m = np.zeros(self.mask.shape, np.uint8)
            cv2.fillPoly(m, [np.round(np.array(self.poly)).astype(np.int32)], 1)
            self.mask |= m > 0
        self.poly = []

    def key(self, k: int):
        if k in (13, 10):
            self.close_polygon()
        elif k == ord("b"):
            self.close_polygon(); self.mode = "brush"
        elif k == ord("e"):
            self.close_polygon(); self.mode = "eraser"
        elif k == ord("p"):
            self.mode = "polygon"
        elif k in (ord("+"), ord("=")):
            self.radius = min(80, self.radius + 3)
        elif k == ord("-"):
            self.radius = max(2, self.radius - 3)
        elif k in (ord("u"), 8, 127):
            if self.poly:
                self.poly.pop()
            elif self.history:
                self.mask = self.history.pop()
        elif k == ord("r"):
            self._push(); self.mask[:] = False; self.poly = []
        elif k == ord("s"):
            self.close_polygon(); self.result = "accept"
        elif k == ord("n"):
            self.result = "skip"
        elif k == ord("q"):
            self.close_polygon(); self.result = "quit"

    @property
    def final_mask(self) -> np.ndarray:
        return self.mask & self.board if self.board is not None else self.mask.copy()

    def render(self) -> np.ndarray:
        c = self.view.base.copy()
        m = cv2.resize(self.final_mask.astype(np.uint8), (c.shape[1], c.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        c[m] = (0.45 * c[m] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
        if self.board is not None:
            b = cv2.resize(self.board.astype(np.uint8), (c.shape[1], c.shape[0]), interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(c, cnts, -1, (0, 255, 0), 1)
        d = [self.view.to_disp(p) for p in self.poly]
        for i in range(len(d)):
            cv2.circle(c, d[i], 4, (0, 255, 255), -1)
            if i:
                cv2.line(c, d[i - 1], d[i], (0, 255, 255), 2)
        if self.mode != "polygon":
            cv2.circle(c, self.cursor, self.radius, (255, 255, 255) if self.mode == "brush" else (0, 0, 255), 1)
        frac = self.final_mask.mean() * 100
        _text(c, [f"frame {self.frame}  ({self.k}/{self.n})   mode: {self.mode}   mask {frac:.1f}% of frame",
                  "click vertices + Enter = fill polygon | b brush | e eraser | p polygon | +/- size | u undo | r clear",
                  "s save + next | n skip | q save+quit" + ("   (green = board, mask is clipped to it)" if self.board is not None else ""),
                  self.message])
        self.view.magnifier(c, *self.cursor)
        return c


def _require_display():
    """Qt aborts the whole process without a display, so check first and explain."""
    import os, sys
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise SystemExit("no display found: run `corners` / `masks` on your own computer (Windows, macOS, or a Linux desktop), "
                         "not on a remote server or in Colab. `python -m overlaylab.annotate check <clip_id>` works without a display.")


def _close_windows():
    cv2.destroyAllWindows()


def _show_reference(img: np.ndarray):
    cv2.imshow("reference.png: corner order", img)


def _run_window(state, title: str):
    try:
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    except cv2.error as e:
        raise SystemExit("could not open a window (no display?). Run the annotation on your own computer; "
                         "`python -m overlaylab.annotate check` works without a display.\n" + str(e))
    cv2.setMouseCallback(title, lambda ev, x, y, fl, _: state.mouse(ev, x, y, fl))
    while state.result is None:
        cv2.imshow(title, state.render())
        k = cv2.waitKey(20) & 0xFF
        if k != 255:
            state.key(k)
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            state.result = "quit"
    return state.result


# --------------------------------------------------------------------------------------------- commands
def default_frames(n_frames: int, count: int) -> List[int]:
    return sorted({int(round(x)) for x in np.linspace(0, n_frames - 1, count)})


def annotate_corners(clip_id: str, frames: Optional[Sequence[int]], count: int, annotator: str) -> int:
    _require_display()
    video = clip_video(clip_id)
    info = sp.video_info(video)
    frames = list(frames) if frames else default_frames(info["frames"], count)
    ims = read_frames(video, frames)
    manifest = data.MANIFEST_DIR / "audit_corners.csv"
    old = {int(r["frame"]): sp.corners_from_row(r) for r in read_rows(manifest)
           if r["clip_id"] == clip_id and r.get("annotator", "") == annotator}
    ref = data.personal_clip_dir(clip_id) / "reference.png"
    if ref.exists():
        r = cv2.imread(str(ref)); h, w = r.shape[:2]
        for i, p in enumerate([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]):
            cv2.circle(r, p, max(6, w // 60), COLOURS[i], -1)
            cv2.putText(r, str(i + 1), (min(max(p[0], 10), w - 30), min(max(p[1] + 30, 30), h - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOURS[i], 2)
        _show_reference(cv2.resize(r, (480, int(480 * h / w))))
    saved = 0
    for k, f in enumerate(frames, 1):
        st = CornerState(ims[f], f, k, len(frames), old.get(f))
        res = _run_window(st, f"corners: {clip_id}")
        if res == "accept":
            fp = save_frame(clip_id, f, ims[f])
            row = {"clip_id": clip_id, "frame": f, "frame_path": rel(fp), "annotator": annotator,
                   "annotated_before_inference": "yes", **sp.row_from_corners(st.corners)}
            upsert_rows(manifest, CORNER_FIELDS, [row]); saved += 1
        if res == "quit":
            break
    _close_windows()
    print(f"{saved} corner frame(s) written to {manifest}")
    return saved


def annotate_masks(clip_id: str, frames: Sequence[int], occluder: str, annotator: str) -> int:
    if not frames:
        raise SystemExit("give the mask frames with --frames (choose frames where the occluder is on the board, plus empty ones if you like)")
    _require_display()
    video = clip_video(clip_id)
    ims = read_frames(video, frames)
    corners = data.load_audit_corners(clip_id) if (data.MANIFEST_DIR / "audit_corners.csv").exists() else {}
    manifest = data.MANIFEST_DIR / "audit_masks.csv"
    mdir = data.personal_clip_dir(clip_id) / "masks"
    saved = 0
    for k, f in enumerate(frames, 1):
        board = corners.get(f)
        note = "" if board is not None else ("no audit corners on this frame: the mask is NOT clipped to the board, "
                                             "so paint only where the occluder covers the board (or annotate this frame's corners first)")
        mp = mdir / f"mask_{f:06d}.png"
        initial = sp.read_mask(mp) if mp.exists() else None
        st = MaskState(ims[f], f, k, len(frames), board, initial)
        st.message = note
        res = _run_window(st, f"masks: {clip_id}")
        if res in ("accept", "quit") and (res == "accept" or st.final_mask.any() or initial is not None):
            fp = save_frame(clip_id, f, ims[f])
            sp.write_mask(mp, st.final_mask)
            row = {"clip_id": clip_id, "frame": f, "frame_path": rel(fp), "mask_path": rel(mp), "annotator": annotator,
                   "occluder": occluder if st.final_mask.any() else "none", "annotated_before_inference": "yes"}
            upsert_rows(manifest, MASK_FIELDS, [row]); saved += 1
        if res == "quit":
            break
    _close_windows()
    print(f"{saved} mask(s) written to {mdir} and {manifest}")
    return saved


def check(clip_id: str, out_dir: Optional[Path] = None) -> List[str]:
    """Validate the audit annotations of a clip and write preview images. Returns the list of problems."""
    video = clip_video(clip_id)
    info = sp.video_info(video)
    shape = (info["height"], info["width"])
    out_dir = Path(out_dir or data.OUTPUT_DIR / "annotation_check" / clip_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    problems: List[str] = []
    c_rows = [r for r in read_rows(data.MANIFEST_DIR / "audit_corners.csv") if r["clip_id"] == clip_id]
    m_rows = [r for r in read_rows(data.MANIFEST_DIR / "audit_masks.csv") if r["clip_id"] == clip_id]
    by_ann: Dict[str, Dict[int, np.ndarray]] = {}
    for r in c_rows:
        by_ann.setdefault(r.get("annotator", ""), {})[int(r["frame"])] = sp.corners_from_row(r)
    frames_c = sorted({int(r["frame"]) for r in c_rows})
    print(f"{clip_id}: {info['frames']} frames, {info['width']}x{info['height']} @ {info['fps']:g} fps")
    print(f"corner rows: {len(c_rows)} on {len(frames_c)} frames {frames_c}; annotators {sorted(by_ann)}")
    if len(frames_c) < 10:
        problems.append(f"only {len(frames_c)} corner frames (10 required)")
    for f in frames_c:
        if not 0 <= f < info["frames"]:
            problems.append(f"corner frame {f} is outside the clip")
    for ann, cs in by_ann.items():
        for f, q in cs.items():
            for p in quad_problems(q, shape):
                problems.append(f"corners of frame {f} ({ann or 'no annotator'}): {p}")
    for r in c_rows + m_rows:
        if r.get("annotated_before_inference", "").strip().lower() != "yes":
            problems.append(f"frame {r['frame']}: annotated_before_inference is not 'yes'")
    n_occ = 0
    for r in m_rows:
        f = int(r["frame"])
        try:
            m = sp.read_mask(sp.resolve(data.MANIFEST_DIR / "audit_masks.csv", r["mask_path"]))
        except FileNotFoundError:
            problems.append(f"mask of frame {f} not found: {r['mask_path']}"); continue
        if m.shape != shape:
            problems.append(f"mask of frame {f} is {m.shape[1]}x{m.shape[0]}, the clip is {shape[1]}x{shape[0]}")
        n_occ += bool(m.any())
    print(f"mask rows: {len(m_rows)} ({n_occ} with an occluder)")
    no_c = sorted(int(r["frame"]) for r in m_rows if int(r["frame"]) not in frames_c)
    if no_c:
        print(f"note: mask frames {no_c} have no corner annotation, so their masks were not clipped to the board; "
              "annotating their corners as well makes the masks cleaner and the frames usable as clean / occluded frames")
    if len(m_rows) < 5:
        problems.append(f"only {len(m_rows)} masks (5 required)")
    if n_occ < 3:
        problems.append(f"only {n_occ} masks with the occluder on the board (3 required)")
    # previews
    want = sorted(set(frames_c) | {int(r["frame"]) for r in m_rows if 0 <= int(r["frame"]) < info["frames"]})
    ims = read_frames(video, [f for f in want if f < info["frames"]]) if want else {}
    first = by_ann[sorted(by_ann)[0]] if by_ann else {}
    masks = {int(r["frame"]): r for r in m_rows}
    for f, im in ims.items():
        c = im.copy()
        if f in first:
            q = np.round(first[f]).astype(np.int32)
            cv2.polylines(c, [q], True, (0, 255, 0), 2)
            for i, p in enumerate(q):
                cv2.circle(c, tuple(int(v) for v in p), 6, COLOURS[i], -1)
                cv2.putText(c, str(i + 1), (int(p[0]) + 8, int(p[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOURS[i], 2)
        if f in masks:
            try:
                m = sp.read_mask(sp.resolve(data.MANIFEST_DIR / "audit_masks.csv", masks[f]["mask_path"]))
                if m.shape == c.shape[:2]:
                    c[m] = (0.45 * c[m] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
            except FileNotFoundError:
                pass
        cv2.imwrite(str(out_dir / f"check_{f:06d}.jpg"), c)
    print(f"previews: {len(ims)} images in {out_dir}")
    print("OK: annotations look complete" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="Audit annotation for your own clip: corners, masks, check.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("corners", help="click the board corners on audit frames")
    a.add_argument("clip_id"); a.add_argument("--frames", type=int, nargs="+")
    a.add_argument("--count", type=int, default=10, help="evenly spaced frames when --frames is not given")
    a.add_argument("--annotator", default="", help="your name or id (optional, recorded in the manifest)")
    b = sub.add_parser("masks", help="draw occlusion masks")
    b.add_argument("clip_id"); b.add_argument("--frames", type=int, nargs="+", required=True)
    b.add_argument("--occluder", default="", help='what occludes the board, e.g. "blue folder"')
    b.add_argument("--annotator", default="")
    c = sub.add_parser("check", help="validate the annotations and write preview images (no window needed)")
    c.add_argument("clip_id"); c.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "corners":
        annotate_corners(args.clip_id, args.frames, args.count, args.annotator)
    elif args.cmd == "masks":
        annotate_masks(args.clip_id, args.frames, args.occluder, args.annotator)
    else:
        raise SystemExit(1 if check(args.clip_id, args.out) else 0)


if __name__ == "__main__":
    main()
