"""Supplied utilities that student code may call directly.

Nothing in this module is assessed. It covers image and video I/O, grayscale
conversion, Gaussian smoothing and pyramids, the normalised Sobel kernels,
bilinear sampling, and manifest helpers. Images are float32 in [0, 1] unless a
function says otherwise; colour images are H x W x 3 in RGB order.

Conventions used throughout the lab
-----------------------------------
* ``x`` indexes columns and ``y`` indexes rows.
* Pixel centres sit at integer coordinates: pixel (row r, column c) is the
  point (x=c, y=r).
* Points are N x 2 float arrays of (x, y).
* Quadrilateral corners are ordered top-left, top-right, bottom-right,
  bottom-left in the rectified board.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0


def gaussian_kernel(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    """Normalised 1-D Gaussian kernel with radius ceil(3 sigma) by default."""
    if radius is None:
        radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian smoothing with reflected borders (any dtype -> float32)."""
    if sigma <= 0:
        return image.astype(np.float32)
    k = gaussian_kernel(sigma)
    out = cv2.sepFilter2D(image.astype(np.float32), -1, k, k, borderType=cv2.BORDER_REFLECT)
    return out


def gaussian_pyramid(gray: np.ndarray, levels: int, sigma: float = 1.0) -> List[np.ndarray]:
    """List of images, level 0 full resolution, each next level smoothed and halved."""
    pyr = [gray.astype(np.float32)]
    for _ in range(1, levels):
        blurred = gaussian_smooth(pyr[-1], sigma)
        pyr.append(blurred[::2, ::2].copy())
    return pyr


def sobel_gradients(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return gx, gy, magnitude, and direction (radians) from the normalised Sobel kernels.

    Same contract as Lab 1: +y points down in image coordinates, so the
    direction is ``arctan2(gy, gx)``. Correlation with reflected borders.
    """
    g = gray.astype(np.float32)
    gx = cv2.filter2D(g, -1, SOBEL_X, borderType=cv2.BORDER_REFLECT)
    gy = cv2.filter2D(g, -1, SOBEL_Y, borderType=cv2.BORDER_REFLECT)
    magnitude = np.hypot(gx, gy).astype(np.float32)
    return gx, gy, magnitude, np.arctan2(gy, gx).astype(np.float32)


def box_filter(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.blur(image.astype(np.float32), (size, size), borderType=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------------
# Image and video I/O
# --------------------------------------------------------------------------

def read_image(path, as_float: bool = True) -> np.ndarray:
    """Read an image as RGB (H x W x 3). Float32 in [0, 1] by default."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0 if as_float else rgb


def write_image(path, image: np.ndarray) -> None:
    """Write an RGB image (float [0, 1] or uint8) to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = to_uint8(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img)


def read_mask(path) -> np.ndarray:
    """Read a binary mask PNG as a Boolean array."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m > 127


def write_mask(path, mask: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.asarray(mask).astype(np.uint8) * 255))


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(np.asarray(image, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def to_float(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return np.asarray(image, dtype=np.float32)


def to_gray(image: np.ndarray) -> np.ndarray:
    """RGB (float or uint8) -> float32 grayscale in [0, 1] (ITU-R 601 weights)."""
    img = to_float(image)
    if img.ndim == 2:
        return img
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def resize_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w == width:
        return image
    nh = int(round(h * width / w))
    return cv2.resize(image, (width, nh), interpolation=cv2.INTER_AREA)


def video_info(path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def iter_video(path, step: int = 1, width: Optional[int] = None,
               start: int = 0, stop: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_index, RGB float frame) from a video, every ``step`` frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if stop is not None and idx >= stop:
            break
        if idx >= start and (idx - start) % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if width is not None:
                rgb = resize_width(rgb, width)
            yield idx, rgb.astype(np.float32) / 255.0
        idx += 1
    cap.release()


def ffmpeg_available() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


def transcode_for_preview(path, out=None, crf: int = 32, speed: int = 4) -> Optional[Path]:
    """Write a VP9 ``.webm`` copy of ``path`` next to it (or at ``out``) and return its path.

    The mp4v / H.264 files written elsewhere in this package do not play inside VS Code's
    built-in video preview (its Electron build ships no proprietary codecs); VP9 in WebM does.
    Returns None when ffmpeg is not installed. Pixel content is unchanged apart from
    compression; keep the .mp4 as the file you submit and evaluate.
    """
    import subprocess
    if not ffmpeg_available():
        return None
    src = Path(path)
    dst = Path(out) if out else src.with_suffix(".webm")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-an",
           "-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-row-mt", "1", "-cpu-used", str(speed),
           "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


class VideoWriter:
    """Minimal RGB video writer (mp4v). Frames may be float [0, 1] or uint8.

    With ``webm=True`` a VP9 ``.webm`` copy is written on close (see ``transcode_for_preview``)
    so the result can be played inside VS Code; the ``.mp4`` is still the primary output.
    """

    def __init__(self, path, fps: float, size: Tuple[int, int], webm: bool = False):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(path)
        self.webm = webm
        self.size = size  # (width, height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"could not open video writer for {path}")

    def write(self, frame: np.ndarray) -> None:
        img = to_uint8(frame)
        if img.shape[1] != self.size[0] or img.shape[0] != self.size[1]:
            img = cv2.resize(img, self.size)
        self.writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.writer.release()
        if self.webm:
            transcode_for_preview(self.path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------
# Geometry helpers (not assessed)
# --------------------------------------------------------------------------

def rect_corners(width: int, height: int) -> np.ndarray:
    """Corners of a width x height image: TL, TR, BR, BL (pixel-centre convention)."""
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                    dtype=np.float64)


def quad_mask(shape: Tuple[int, int], corners: np.ndarray) -> np.ndarray:
    """Boolean mask of the pixels inside a quadrilateral."""
    m = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.round(np.asarray(corners, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def longer_diagonal(corners: np.ndarray) -> float:
    c = np.asarray(corners, dtype=np.float64)
    return float(max(np.linalg.norm(c[0] - c[2]), np.linalg.norm(c[1] - c[3])))


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------

def read_csv(path) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def corners_from_row(row: dict) -> np.ndarray:
    return np.array([[float(row[f"x{i}"]), float(row[f"y{i}"])] for i in range(1, 5)],
                    dtype=np.float64)


def row_from_corners(corners: np.ndarray) -> dict:
    c = np.asarray(corners, dtype=np.float64)
    return {f"{axis}{i + 1}": f"{c[i, j]:.3f}" for i in range(4) for j, axis in enumerate("xy")}


def resolve(manifest_path, relative: str) -> Path:
    """Resolve a path stored relative to a manifest's directory."""
    return (Path(manifest_path).parent / relative).resolve()
