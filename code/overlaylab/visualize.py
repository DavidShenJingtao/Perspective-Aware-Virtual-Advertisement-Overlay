"""Supplied plotting and drawing helpers (not assessed).

Drawing functions take and return RGB float images; plotting functions use
matplotlib and return the figure so that notebooks can save it.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .supplied import to_uint8, to_float

GREEN = (0.1, 0.8, 0.2)
RED = (0.9, 0.15, 0.15)
BLUE = (0.2, 0.4, 0.95)
YELLOW = (0.98, 0.85, 0.1)


def draw_quad(image: np.ndarray, corners: np.ndarray, colour=GREEN, thickness: int = 2) -> np.ndarray:
    out = to_float(image).copy()
    c = np.asarray(corners, dtype=np.float64)
    if not np.all(np.isfinite(c)):
        return out
    pts = np.round(c).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, colour, thickness, cv2.LINE_AA)
    return out


def draw_points(image: np.ndarray, points: np.ndarray, colour=YELLOW, radius: int = 3,
                valid: Optional[np.ndarray] = None, invalid_colour=RED) -> np.ndarray:
    out = to_float(image).copy()
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    for i, (x, y) in enumerate(pts):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        c = colour if valid is None or valid[i] else invalid_colour
        cv2.circle(out, (int(round(x)), int(round(y))), radius, c, -1, cv2.LINE_AA)
    return out


def draw_flow(image: np.ndarray, points: np.ndarray, vectors: np.ndarray, valid: Optional[np.ndarray] = None,
              scale: float = 1.0) -> np.ndarray:
    out = to_float(image).copy()
    pts = np.asarray(points, dtype=np.float64); vec = np.asarray(vectors, dtype=np.float64)
    for i in range(len(pts)):
        p = pts[i]; q = p + scale * vec[i]
        if not np.all(np.isfinite(q)):
            continue
        c = GREEN if valid is None or valid[i] else RED
        cv2.arrowedLine(out, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), c, 1, cv2.LINE_AA, tipLength=0.3)
    return out


def draw_matches(ref: np.ndarray, frame: np.ndarray, kp_ref: np.ndarray, kp_frame: np.ndarray,
                 pairs: np.ndarray, inliers: Optional[np.ndarray] = None, max_lines: int = 300) -> np.ndarray:
    """Reference stacked above the frame with match lines (green inliers, red outliers)."""
    r = to_float(ref); f = to_float(frame)
    w = max(r.shape[1], f.shape[1])
    canvas = np.zeros((r.shape[0] + f.shape[0], w, 3), np.float32)
    canvas[:r.shape[0], :r.shape[1]] = r
    canvas[r.shape[0]:, :f.shape[1]] = f
    off = r.shape[0]
    pairs = np.asarray(pairs).reshape(-1, 2)
    if inliers is None:
        inliers = np.ones(len(pairs), bool)
    order = np.argsort(~inliers)  # draw outliers first, inliers last
    for k in order[:max_lines]:
        i, j = pairs[k]
        p = kp_ref[i]; q = kp_frame[j]
        c = GREEN if inliers[k] else RED
        cv2.line(canvas, (int(p[0]), int(p[1])), (int(q[0]), int(q[1]) + off), c, 1, cv2.LINE_AA)
        cv2.circle(canvas, (int(p[0]), int(p[1])), 2, c, -1)
        cv2.circle(canvas, (int(q[0]), int(q[1]) + off), 2, c, -1)
    return canvas


def overlay_mask(image: np.ndarray, mask: np.ndarray, colour=RED, alpha: float = 0.45) -> np.ndarray:
    out = to_float(image).copy()
    m = np.asarray(mask, dtype=bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.asarray(colour, np.float32)
    return out


def heatmap(values: np.ndarray, vmax: Optional[float] = None) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    vmax = vmax or max(float(np.nanmax(v)), 1e-6)
    u8 = to_uint8(np.clip(v / vmax, 0, 1))
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def montage(images: Sequence[np.ndarray], cols: int = 3, pad: int = 4, width: Optional[int] = None) -> np.ndarray:
    ims = [to_float(i) if i.ndim == 3 else np.repeat(to_float(i)[..., None], 3, -1) for i in images]
    if width:
        ims = [cv2.resize(i, (width, int(i.shape[0] * width / i.shape[1]))) for i in ims]
    h = max(i.shape[0] for i in ims); w = max(i.shape[1] for i in ims)
    rows = int(np.ceil(len(ims) / cols))
    canvas = np.ones((rows * (h + pad) - pad, cols * (w + pad) - pad, 3), np.float32)
    for k, im in enumerate(ims):
        r, c = divmod(k, cols)
        canvas[r * (h + pad):r * (h + pad) + im.shape[0], c * (w + pad):c * (w + pad) + im.shape[1]] = im
    return canvas


def label(image: np.ndarray, text: str, org=(8, 24), scale: float = 0.7) -> np.ndarray:
    out = to_float(image).copy()
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (1, 1, 1), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# matplotlib
# ---------------------------------------------------------------------------

def plot_monitor_timeline(stats: Sequence[Dict], flags: Optional[Sequence[Tuple[str, str]]] = None,
                          events: Optional[Dict[str, Tuple[int, int]]] = None,
                          errors: Optional[Dict[int, float]] = None,
                          thresholds: Optional[Dict[str, float]] = None, title: str = ""):
    """Inlier ratio, mean flow, brightness change (online) and corner error (audit)."""
    import matplotlib.pyplot as plt
    keys = ["inlier_ratio", "mean_flow", "brightness_change"]
    n = len(keys) + (1 if errors else 0)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.9 * n), sharex=True)
    axes = np.atleast_1d(axes)
    t = np.arange(len(stats))
    for ax, k in zip(axes, keys):
        y = np.array([s.get(k, np.nan) for s in stats], dtype=float)
        ax.plot(t, y, lw=1.2)
        ax.set_ylabel(k.replace("_", " "))
        if thresholds and k in thresholds:
            ax.axhline(thresholds[k], color="k", ls="--", lw=0.8)
    if errors:
        ax = axes[-1]
        fr = sorted(errors)
        ax.plot(fr, [errors[f] * 100 for f in fr], "o-", color="darkgreen", ms=3, lw=1)
        ax.set_ylabel("corner error (%)")
    for ax in axes:
        if events:
            for name, (a, b) in events.items():
                ax.axvspan(a, b, alpha=0.12, label=name)
        if flags:
            for i, (flag, _) in enumerate(flags):
                if flag == "HIDE":
                    ax.axvline(i, color="red", alpha=0.08, lw=1)
                elif flag == "REINIT":
                    ax.axvline(i, color="orange", alpha=0.15, lw=1)
    axes[-1].set_xlabel("frame index")
    if title:
        fig.suptitle(title)
    if events:
        axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_residual_histogram(residuals: np.ndarray, inliers: np.ndarray, threshold: float, title: str = ""):
    import matplotlib.pyplot as plt
    r = np.sqrt(np.asarray(residuals, dtype=float) / 2.0)  # px per direction
    fig, ax = plt.subplots(figsize=(5, 3))
    bins = np.linspace(0, max(np.percentile(r[np.isfinite(r)], 98), 1), 40)
    ax.hist(r[inliers], bins=bins, color="green", alpha=0.7, label="inliers")
    ax.hist(r[~inliers], bins=bins, color="red", alpha=0.5, label="outliers")
    ax.axvline(np.sqrt(threshold / 2.0), color="k", ls="--", lw=0.8)
    ax.set_xlabel("symmetric transfer error (px)"); ax.set_ylabel("count"); ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_singular_values(svals: Sequence[np.ndarray], labels: Sequence[str], title: str = ""):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    for s, l in zip(svals, labels):
        ax.semilogy(np.arange(1, len(s) + 1), s / s[0], "o-", ms=3, label=l)
    ax.set_xlabel("index"); ax.set_ylabel("sigma_i / sigma_1"); ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_eigenvalues(eig: np.ndarray, valid: np.ndarray, min_eig: float, title: str = ""):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    e = np.asarray(eig)
    ax.loglog(e[valid, 0] + 1e-9, e[valid, 1] + 1e-9, ".", color="green", label="valid")
    ax.loglog(e[~valid, 0] + 1e-9, e[~valid, 1] + 1e-9, ".", color="red", label="invalid")
    ax.axhline(min_eig, color="k", ls="--", lw=0.8)
    ax.set_xlabel("lambda_1"); ax.set_ylabel("lambda_2"); ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


__all__ = ["draw_quad", "draw_points", "draw_flow", "draw_matches", "overlay_mask", "heatmap",
           "montage", "label", "plot_monitor_timeline", "plot_residual_histogram",
           "plot_singular_values", "plot_eigenvalues"]
