"""Synthetic board sequences with known ground truth.

The renderer places a textured, planar advertisement board in front of a
textured background under a per-frame homography that mimics a hand-held
broadcast camera: panning, zooming, perspective tilt, and jitter. Optional
events inject a moving occluder, a whip-pan with motion blur, and an
illumination change. Because the homography is known exactly, every rendered
frame comes with exact board corners and an exact occlusion mask, which is
what the unit tests and the debugging notebooks rely on.

Nothing here is assessed; use it to check each stage against known answers
before touching real footage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .supplied import rect_corners, quad_mask, to_uint8

# ---------------------------------------------------------------------------
# Textures
# ---------------------------------------------------------------------------

_PALETTES = {
    "board": [(0.08, 0.20, 0.55), (0.98, 0.84, 0.10), (0.96, 0.96, 0.94), (0.85, 0.12, 0.12)],
    "ad_A": [(0.92, 0.25, 0.10), (0.99, 0.95, 0.90), (0.10, 0.10, 0.12), (0.10, 0.55, 0.85)],
    "ad_B": [(0.06, 0.45, 0.30), (0.95, 0.95, 0.95), (0.98, 0.70, 0.05), (0.15, 0.15, 0.15)],
}


def _put_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float,
              colour: Tuple[float, float, float], thickness: int, font=cv2.FONT_HERSHEY_DUPLEX):
    cv2.putText(img, text, org, font, scale, colour, thickness, cv2.LINE_AA)


def make_board_texture(width: int = 1600, height: int = 400, seed: int = 0,
                       text: str = "OVERLAY LAB", palette: str = "board") -> np.ndarray:
    """A poster-like board: lettering (repetitive structure), a tiled block, logos.

    Returns an RGB float32 image in [0, 1]. The tiled region and the repeated
    letters are deliberate: they produce near-identical nearest and
    second-nearest neighbours for the ratio-test discussion in Stage 1.
    """
    rng = np.random.default_rng(seed)
    bg, accent, light, dark = _PALETTES[palette]
    img = np.empty((height, width, 3), dtype=np.float32)
    img[:] = bg
    # subtle paper texture so that flat regions are not perfectly flat
    img += rng.normal(0, 0.012, size=(height, width, 1)).astype(np.float32)
    # accent band at the bottom
    band = int(height * 0.82)
    img[band:, :] = accent
    # tiled pattern block on the left (repetitive structure)
    tile = int(height * 0.12)
    x0, x1 = int(width * 0.03), int(width * 0.22)
    for yy in range(int(height * 0.1), band - tile, tile):
        for xx in range(x0, x1, tile):
            c = light if ((yy // tile + xx // tile) % 2 == 0) else dark
            cv2.rectangle(img, (xx + 2, yy + 2), (xx + tile - 3, yy + tile - 3), c, -1)
    # big lettering
    scale = height / 110.0
    _put_text(img, text, (int(width * 0.27), int(height * 0.52)), scale, light, max(2, int(scale * 4)))
    _put_text(img, "SEMESTER 1  2026-2027", (int(width * 0.27), int(height * 0.74)),
              scale * 0.38, light, max(1, int(scale * 1.6)))
    # logo: concentric circles and a diamond on the right
    cx, cy = int(width * 0.90), int(height * 0.42)
    r = int(height * 0.28)
    cv2.circle(img, (cx, cy), r, light, -1)
    cv2.circle(img, (cx, cy), int(r * 0.66), dark, -1)
    cv2.circle(img, (cx, cy), int(r * 0.33), accent, -1)
    d = int(height * 0.1)
    pts = np.array([[cx - 3 * r // 2, cy], [cx - 3 * r // 2 + d, cy - d],
                    [cx - 3 * r // 2 + 2 * d, cy], [cx - 3 * r // 2 + d, cy + d]], np.int32)
    cv2.fillPoly(img, [pts], light)
    # scattered small marks so that the band is not textureless
    for _ in range(40):
        x = int(rng.uniform(0, width)); y = int(rng.uniform(band + 4, height - 4))
        cv2.circle(img, (x, y), 2, dark, -1)
    return np.clip(img, 0, 1)


def make_artwork(width: int = 1600, height: int = 400, variant: str = "ad_A",
                 seed: int = 1) -> np.ndarray:
    """Replacement artwork in the board aspect ratio (RGB float32)."""
    rng = np.random.default_rng(seed)
    bg, light, dark, accent = _PALETTES[variant]
    img = np.empty((height, width, 3), dtype=np.float32)
    img[:] = bg
    if variant == "ad_A":
        for i in range(0, width, 80):
            cv2.line(img, (i, 0), (i + 40, height), accent, 2)
        _put_text(img, "VISION 4243", (int(width * 0.08), int(height * 0.6)), height / 95.0,
                  light, int(height / 22))
        cv2.circle(img, (int(width * 0.88), height // 2), int(height * 0.36), light, -1)
        cv2.circle(img, (int(width * 0.88), height // 2), int(height * 0.22), dark, -1)
    else:
        cv2.rectangle(img, (0, 0), (width, int(height * 0.18)), accent, -1)
        _put_text(img, "HOMOGRAPHY", (int(width * 0.05), int(height * 0.62)), height / 110.0,
                  light, int(height / 24))
        _put_text(img, "& FLOW", (int(width * 0.62), int(height * 0.62)), height / 110.0,
                  accent, int(height / 24))
        for k in range(6):
            x = int(width * 0.05 + k * width * 0.15)
            cv2.rectangle(img, (x, int(height * 0.78)), (x + 40, int(height * 0.95)), dark, -1)
    img += rng.normal(0, 0.006, size=(height, width, 1)).astype(np.float32)
    return np.clip(img, 0, 1)


def make_background(width: int = 2600, height: int = 1500, seed: int = 2) -> np.ndarray:
    """A wall-like background with furniture-like rectangles and strong texture."""
    rng = np.random.default_rng(seed)
    img = np.empty((height, width, 3), dtype=np.float32)
    img[:] = (0.62, 0.60, 0.56)
    # low-frequency shading
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    img *= (0.85 + 0.15 * np.cos(xx / width * 3.1) * np.cos(yy / height * 2.0))[..., None]
    # brick-like grid
    for y in range(0, height, 48):
        cv2.line(img, (0, y), (width, y), (0.50, 0.47, 0.44), 2)
    for y in range(0, height, 48):
        off = 48 if (y // 48) % 2 else 0
        for x in range(off, width, 96):
            cv2.line(img, (x, y), (x, y + 48), (0.50, 0.47, 0.44), 2)
    # a few "objects": doors, cabinets, posters
    for _ in range(14):
        x = int(rng.uniform(0, width - 300)); y = int(rng.uniform(0, height - 300))
        w = int(rng.uniform(80, 320)); h = int(rng.uniform(80, 320))
        c = tuple(float(v) for v in rng.uniform(0.15, 0.9, size=3))
        cv2.rectangle(img, (x, y), (x + w, y + h), c, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0.1, 0.1, 0.1), 3)
    # floor strip
    cv2.rectangle(img, (0, int(height * 0.82)), (width, height), (0.33, 0.30, 0.28), -1)
    for x in range(0, width, 64):
        cv2.line(img, (x, int(height * 0.82)), (x, height), (0.26, 0.24, 0.22), 1)
    img += rng.normal(0, 0.02, size=(height, width, 1)).astype(np.float32)
    return np.clip(img, 0, 1)


def checkerboard(rows: int = 8, cols: int = 8, cell: int = 16) -> np.ndarray:
    """Grayscale checkerboard in [0, 1] with the given cell size."""
    y, x = np.mgrid[0:rows * cell, 0:cols * cell]
    return (((y // cell) + (x // cell)) % 2).astype(np.float32)


def rotated_corner(size: int = 64, angle_deg: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """A single bright L-shaped corner on a dark background, rotated about the centre."""
    img = np.zeros((size, size), dtype=np.float32)
    c = size // 2
    img[c - 1:, c - 1:] = 1.0
    M = cv2.getRotationMatrix2D((c - 0.5, c - 0.5), angle_deg, 1.0)
    rot = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.GaussianBlur(rot, (0, 0), sigma)


def translate_image(gray: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Sub-pixel translation by (dx, dy) with bilinear interpolation (reflected borders)."""
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    h, w = gray.shape[:2]
    return cv2.warpAffine(gray.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def textured_patch(size: int = 128, seed: int = 3, sigma: float = 1.5) -> np.ndarray:
    """Smooth random texture (good for flow and descriptor tests)."""
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0, 1, size=(size, size)).astype(np.float32)
    img = cv2.GaussianBlur(noise, (0, 0), sigma)
    img -= img.min(); img /= max(img.max(), 1e-6)
    return img


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def homography_from_corners(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Exact homography mapping four src points to four dst points (OpenCV, not assessed)."""
    H = cv2.getPerspectiveTransform(np.asarray(src, np.float32), np.asarray(dst, np.float32))
    return H.astype(np.float64) / H[2, 2]


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64)
    hom = np.concatenate([p, np.ones((len(p), 1))], axis=1) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def project_board(board_w: float, board_h: float, centre: Tuple[float, float], scale: float,
                  yaw: float, pitch: float, roll: float, focal: float = 1.6) -> np.ndarray:
    """Project a board of physical size (board_w x board_h) with a pinhole model.

    ``scale`` is the on-screen width (px) of the board at zero tilt;
    yaw/pitch/roll are radians. Returns 4 x 2 frame corners (TL, TR, BR, BL).
    """
    X = np.array([[-0.5, -0.5 * board_h / board_w, 0], [0.5, -0.5 * board_h / board_w, 0],
                  [0.5, 0.5 * board_h / board_w, 0], [-0.5, 0.5 * board_h / board_w, 0]])
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    cp_, sp_ = np.cos(pitch), np.sin(pitch)
    cr_, sr_ = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
    Rx = np.array([[1, 0, 0], [0, cp_, -sp_], [0, sp_, cp_]])
    Rz = np.array([[cr_, -sr_, 0], [sr_, cr_, 0], [0, 0, 1]])
    P = X @ (Rz @ Rx @ Ry).T
    z = P[:, 2] + focal
    x = focal * P[:, 0] / z
    y = focal * P[:, 1] / z
    out = np.stack([x, y], axis=1) * scale
    out[:, 0] += centre[0]
    out[:, 1] += centre[1]
    return out


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

@dataclass
class SequenceSpec:
    n_frames: int = 90
    frame_size: Tuple[int, int] = (1280, 720)   # (width, height)
    board_size: Tuple[int, int] = (1600, 400)    # reference texture size (width, height)
    seed: int = 0
    # camera motion
    pan_amplitude: float = 0.14      # fraction of frame width
    zoom_range: Tuple[float, float] = (0.78, 1.12)
    base_scale: float = 0.62         # board width as a fraction of frame width
    tilt_deg: float = 14.0           # peak yaw
    jitter_px: float = 2.5           # hand-held jitter (std, smoothed)
    background_parallax: float = 0.6  # 1 = coplanar with board; <1 = farther away
    # events (frame ranges; None = off)
    occlusion: Optional[Tuple[int, int]] = (40, 70)
    occluder: str = "ball_and_pole"  # "ball", "pole", "ball_and_pole", "hand"
    whip_pan: Optional[Tuple[int, int]] = None
    illumination_step: Optional[int] = None
    illumination_gain: float = 0.6
    illumination_gradient: float = 0.12
    blur_sigma: float = 0.6
    noise_std: float = 0.01
    exit_frame: Optional[Tuple[int, int]] = None  # board leaves the frame
    # outputs
    text: str = "OVERLAY LAB"


def camera_path(spec: SequenceSpec) -> np.ndarray:
    """Per-frame board corners (n x 4 x 2) in frame coordinates."""
    rng = np.random.default_rng(spec.seed + 11)
    n = spec.n_frames
    W, H = spec.frame_size
    t = np.linspace(0, 1, n)
    bw, bh = spec.board_size
    # smooth pan and zoom
    cx = W * (0.5 + spec.pan_amplitude * np.sin(2 * np.pi * t * 0.9 + 0.3))
    cy = H * (0.52 + 0.04 * np.sin(2 * np.pi * t * 0.6 + 1.1))
    z0, z1 = spec.zoom_range
    zoom = z0 + (z1 - z0) * 0.5 * (1 - np.cos(2 * np.pi * t * 0.7))
    yaw = np.deg2rad(spec.tilt_deg) * np.sin(2 * np.pi * t * 0.5 + 0.8)
    pitch = np.deg2rad(spec.tilt_deg * 0.3) * np.sin(2 * np.pi * t * 0.4)
    roll = np.deg2rad(2.0) * np.sin(2 * np.pi * t * 0.8)
    # hand-held jitter: smoothed random walk
    def jitter():
        j = np.cumsum(rng.normal(0, 1, n))
        j = cv2.GaussianBlur(j.reshape(-1, 1).astype(np.float32), (0, 0), 2.5).ravel()
        j -= j.mean()
        return j / (j.std() + 1e-6) * spec.jitter_px
    cx = cx + jitter(); cy = cy + jitter()
    if spec.whip_pan is not None:
        a, b = spec.whip_pan
        ramp = np.clip((np.arange(n) - a) / max(1, b - a), 0, 1)
        swing = 0.5 - 0.5 * np.cos(np.pi * ramp)  # 0 -> 1 smoothly
        cx = cx + 0.30 * W * np.sin(np.pi * swing)   # out and back
    if spec.exit_frame is not None:
        a, b = spec.exit_frame
        ramp = np.clip((np.arange(n) - a) / max(1, b - a), 0, 1)
        cx = cx + 0.9 * W * np.sin(np.pi * ramp)
    corners = np.zeros((n, 4, 2))
    for i in range(n):
        corners[i] = project_board(bw, bh, (cx[i], cy[i]), spec.base_scale * W * zoom[i],
                                   yaw[i], pitch[i], roll[i])
    return corners


def background_path(spec: SequenceSpec, corners: np.ndarray) -> List[np.ndarray]:
    """Per-frame 3x3 similarity for the background plane (reduced parallax)."""
    W, H = spec.frame_size
    centres = corners.mean(axis=1)
    widths = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)
    c0 = centres[0]; w0 = widths[0]
    out = []
    for i in range(len(corners)):
        s = 1.0 + spec.background_parallax * (widths[i] / w0 - 1.0)
        d = spec.background_parallax * (centres[i] - c0)
        # keep the background centred roughly on the frame; bg texture is larger than frame
        tx = W / 2 - s * 1300 + d[0]
        ty = H / 2 - s * 750 + d[1]
        out.append(np.array([[s, 0, tx], [0, s, ty], [0, 0, 1]], dtype=np.float64))
    return out


def _occluder_shapes(spec: SequenceSpec, i: int, corners: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    if spec.occlusion is None:
        return []
    a, b = spec.occlusion
    if i < a or i > b:
        return []
    u = (i - a) / max(1, b - a)
    c = corners[i]
    left = c[[0, 3]].mean(axis=0); right = c[[1, 2]].mean(axis=0)
    top = c[[0, 1]].mean(axis=0); bottom = c[[2, 3]].mean(axis=0)
    height = np.linalg.norm(top - bottom)
    out = []
    if spec.occluder in ("ball", "ball_and_pole", "hand"):
        p = left + (right - left) * (u * 1.3 - 0.15)
        p[1] += -0.9 * height * np.sin(np.pi * u) + 0.2 * height
        r = 0.38 * height
        out.append(("ball", np.array([p[0], p[1], r])))
    if spec.occluder in ("pole", "ball_and_pole"):
        x = left[0] + (right[0] - left[0]) * (1.1 - u * 1.2)
        out.append(("pole", np.array([x, c[:, 1].min() - 0.6 * height, x + 0.15 * height,
                                      c[:, 1].max() + 0.6 * height, 0.045 * height])))
    if spec.occluder == "hand":
        p = left + (right - left) * (u * 1.2 - 0.1)
        out.append(("hand", np.array([p[0], p[1] + 0.3 * height, 0.45 * height, 0.22 * height])))
    return out


def _draw_occluders(frame: np.ndarray, shapes, rng: np.random.Generator) -> np.ndarray:
    """Draw occluders onto ``frame`` (in place) and return their Boolean mask."""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for kind, p in shapes:
        if kind == "ball":
            cx, cy, r = p
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            inside = d <= r
            shade = np.clip(1.0 - 0.9 * ((xx - cx + 0.35 * r) ** 2 + (yy - cy + 0.35 * r) ** 2) / (r * r), 0.15, 1)
            stripes = 0.5 + 0.5 * np.sign(np.sin((xx - cx) / r * 6.0 + (yy - cy) / r * 2.0))
            colour = np.stack([0.95 * shade, (0.35 + 0.5 * stripes) * shade, 0.15 * shade], axis=-1)
            frame[inside] = colour[inside]
            mask[inside] = 1
        elif kind == "pole":
            x0, y0, x1, y1, half = p
            # thin diagonal pole
            pts = np.array([[x0 - half, y0], [x0 + half, y0], [x1 + half, y1], [x1 - half, y1]], np.int32)
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [pts], 1)
            inside = m.astype(bool)
            tex = 0.35 + 0.25 * np.sign(np.sin(yy / 9.0))
            colour = np.stack([0.25 * tex, 0.22 * tex, 0.20 * tex], axis=-1)
            frame[inside] = colour[inside]
            mask[inside] = 1
        elif kind == "hand":
            cx, cy, a, b = p
            inside = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1
            colour = np.stack([0.80 + 0.02 * np.sin(xx / 3), 0.60 + 0.03 * np.sin(yy / 4), 0.50 * np.ones_like(xx)], -1)
            frame[inside] = colour[inside]
            mask[inside] = 1
    return mask.astype(bool)


def render_frame(board: np.ndarray, background: np.ndarray, H_board: np.ndarray, H_bg: np.ndarray,
                 frame_size: Tuple[int, int], gain: float = 1.0, gradient: float = 0.0,
                 blur_sigma: float = 0.0, motion_blur: Tuple[float, float] = (0.0, 0.0),
                 noise_std: float = 0.0, occluders=(), rng: Optional[np.random.Generator] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one frame. Returns (frame RGB float, occluder mask, board coverage mask)."""
    W, H = frame_size
    rng = rng or np.random.default_rng(0)
    bg = cv2.warpPerspective(background, H_bg, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fg = cv2.warpPerspective(board, H_board, (W, H), flags=cv2.INTER_LINEAR)
    cov = cv2.warpPerspective(np.ones(board.shape[:2], np.float32), H_board, (W, H), flags=cv2.INTER_NEAREST) > 0.5
    frame = bg.copy()
    frame[cov] = fg[cov]
    occ = _draw_occluders(frame, occluders, rng)
    # illumination: global gain and a horizontal gradient
    if gradient != 0.0:
        ramp = np.linspace(1 - gradient, 1 + gradient, W, dtype=np.float32)[None, :, None]
        frame = frame * ramp
    frame = frame * gain
    if blur_sigma > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), blur_sigma)
    mx, my = motion_blur
    length = int(np.hypot(mx, my))
    if length >= 2:
        k = np.zeros((length, length), np.float32)
        cv2.line(k, (0, 0), (max(0, int(mx)), max(0, int(my))) if mx >= 0 and my >= 0 else (length - 1, 0), 1, 1)
        k /= max(k.sum(), 1)
        frame = cv2.filter2D(frame, -1, k, borderType=cv2.BORDER_REFLECT)
    if noise_std > 0:
        frame = frame + rng.normal(0, noise_std, size=frame.shape).astype(np.float32)
    return np.clip(frame, 0, 1).astype(np.float32), occ, cov


@dataclass
class RenderedSequence:
    frames: List[np.ndarray]
    homographies: List[np.ndarray]        # board reference -> frame
    corners: np.ndarray                   # n x 4 x 2
    occlusion_masks: List[np.ndarray]     # occluder AND board quad, per frame
    board: np.ndarray
    spec: SequenceSpec
    events: Dict[str, Tuple[int, int]] = field(default_factory=dict)


def iter_sequence(spec: SequenceSpec, board: Optional[np.ndarray] = None,
                  background: Optional[np.ndarray] = None
                  ) -> Iterator[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (index, frame, H, corners, occlusion_mask) one frame at a time."""
    bw, bh = spec.board_size
    board = make_board_texture(bw, bh, spec.seed, spec.text) if board is None else board
    background = make_background(seed=spec.seed + 2) if background is None else background
    corners = camera_path(spec)
    bgs = background_path(spec, corners)
    ref = rect_corners(bw, bh)
    rng = np.random.default_rng(spec.seed + 99)
    W, H = spec.frame_size
    for i in range(spec.n_frames):
        Hb = homography_from_corners(ref, corners[i])
        gain = 1.0
        if spec.illumination_step is not None and i >= spec.illumination_step:
            gain = spec.illumination_gain
        mblur = (0.0, 0.0)
        if spec.whip_pan is not None and i > 0:
            v = corners[i].mean(axis=0) - corners[i - 1].mean(axis=0)
            if np.linalg.norm(v) > 8:
                mblur = (abs(float(v[0])) * 0.9, abs(float(v[1])) * 0.9)
        shapes = _occluder_shapes(spec, i, corners)
        frame, occ, cov = render_frame(board, background, Hb, bgs[i], (W, H), gain=gain,
                                       gradient=spec.illumination_gradient, blur_sigma=spec.blur_sigma,
                                       motion_blur=mblur, noise_std=spec.noise_std,
                                       occluders=shapes, rng=rng)
        gt_mask = occ & quad_mask((H, W), corners[i])
        yield i, frame, Hb, corners[i], gt_mask


def render_sequence(spec: SequenceSpec, board: Optional[np.ndarray] = None,
                    background: Optional[np.ndarray] = None) -> RenderedSequence:
    """Render every frame into memory (use iter_sequence for long clips)."""
    bw, bh = spec.board_size
    board = make_board_texture(bw, bh, spec.seed, spec.text) if board is None else board
    frames, Hs, masks, corners = [], [], [], []
    for i, f, Hb, c, m in iter_sequence(spec, board, background):
        frames.append(f); Hs.append(Hb); corners.append(c); masks.append(m)
    events = {}
    if spec.occlusion: events["occlusion"] = tuple(spec.occlusion)
    if spec.whip_pan: events["whip_pan"] = tuple(spec.whip_pan)
    if spec.illumination_step is not None: events["illumination"] = (spec.illumination_step, spec.illumination_step)
    if spec.exit_frame: events["exit"] = tuple(spec.exit_frame)
    return RenderedSequence(frames, Hs, np.array(corners), masks, board, spec, events)


def common_clip_spec(n_frames: int = 450, frame_size: Tuple[int, int] = (1920, 1080)) -> SequenceSpec:
    """The specification used to render the supplied common clip."""
    return SequenceSpec(n_frames=n_frames, frame_size=frame_size, seed=2026,
                        pan_amplitude=0.13, zoom_range=(0.80, 1.15), base_scale=0.60,
                        tilt_deg=15.0, jitter_px=3.0, background_parallax=0.62,
                        occlusion=(188, 228), occluder="ball_and_pole",
                        illumination_gradient=0.10, blur_sigma=0.7, noise_std=0.012)


def stress_specs() -> Dict[str, SequenceSpec]:
    """Short synthetic stress sequences for Notebook 2 and the tests."""
    base = dict(n_frames=90, frame_size=(960, 540), seed=7)
    return {
        "whip_pan": SequenceSpec(**base, occlusion=None, whip_pan=(30, 50)),
        "illumination": SequenceSpec(**base, occlusion=None, illumination_step=45, illumination_gain=0.55),
        "near_total_occlusion": SequenceSpec(**base, occlusion=(25, 60), occluder="hand", base_scale=0.5),
        "long_track": SequenceSpec(n_frames=240, frame_size=(960, 540), seed=8, occlusion=None),
        "exit": SequenceSpec(**base, occlusion=None, exit_frame=(30, 60)),
    }


# ---------------------------------------------------------------------------
# Correspondence fixtures (geometry tests)
# ---------------------------------------------------------------------------

def synthetic_correspondences(H: np.ndarray, n_inliers: int = 60, n_outliers: int = 0,
                              noise: float = 0.0, seed: int = 0,
                              src_range: Tuple[float, float, float, float] = (0, 1600, 0, 400),
                              dst_shape: Tuple[int, int] = (720, 1280)
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (src, dst, is_inlier) point sets related by H with injected outliers."""
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = src_range
    src = np.stack([rng.uniform(x0, x1, n_inliers + n_outliers),
                    rng.uniform(y0, y1, n_inliers + n_outliers)], axis=1)
    dst = apply_homography(H, src)
    dst += rng.normal(0, noise, size=dst.shape) if noise > 0 else 0
    is_inlier = np.ones(len(src), dtype=bool)
    if n_outliers:
        idx = rng.choice(len(src), n_outliers, replace=False)
        dst[idx, 0] = rng.uniform(0, dst_shape[1], n_outliers)
        dst[idx, 1] = rng.uniform(0, dst_shape[0], n_outliers)
        is_inlier[idx] = False
    return src, dst, is_inlier


def random_homography(seed: int = 0, frame_size: Tuple[int, int] = (1280, 720),
                      board_size: Tuple[int, int] = (1600, 400)) -> Tuple[np.ndarray, np.ndarray]:
    """A plausible board-to-frame homography and its frame corners."""
    rng = np.random.default_rng(seed)
    W, H = frame_size
    c = project_board(board_size[0], board_size[1], (W * rng.uniform(0.4, 0.6), H * rng.uniform(0.45, 0.55)),
                      W * rng.uniform(0.5, 0.7), np.deg2rad(rng.uniform(-18, 18)),
                      np.deg2rad(rng.uniform(-6, 6)), np.deg2rad(rng.uniform(-3, 3)))
    return homography_from_corners(rect_corners(*board_size), c), c


__all__ = [
    "SequenceSpec", "RenderedSequence", "make_board_texture", "make_artwork", "make_background",
    "checkerboard", "rotated_corner", "translate_image", "textured_patch",
    "homography_from_corners", "apply_homography", "project_board", "camera_path",
    "render_frame", "iter_sequence", "render_sequence", "common_clip_spec", "stress_specs",
    "synthetic_correspondences", "random_homography",
]
