"""Configuration objects for every stage of the overlay pipeline.

All tunable behaviour is expressed here so that ablations change *only* the
intended factor. Student code must read settings from these objects rather
than from hard-coded constants. Fields are documented inline; the defaults
are reasonable starting points for the 640-pixel notebook subset and are
not tuned for the full-resolution clip.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Optional, Tuple
import json

import numpy as np


@dataclass
class DetectorConfig:
    """Harris corner detector (Stage 1, keypoint branch)."""
    sigma_w: float = 2.0          # window scale of the Gaussian weight w(x, y)
    kappa: float = 0.05           # Harris sensitivity constant
    response_fraction: float = 0.005  # discard R below this fraction of max(R)
    nms_mode: str = "radius"      # "window" or "radius"
    nms_radius: int = 5           # r: (2r+1)x(2r+1) window, or greedy radius
    max_corners: int = 800
    border: int = 12              # no corners within this many pixels of the edge


@dataclass
class DescriptorConfig:
    """Gradient-histogram descriptor (Stage 1, keypoint branch)."""
    patch_scale: float = 1.0      # 16x16 window scaled by this factor
    orientation_bins: int = 36
    cells: int = 4                # 4x4 spatial grid
    cell_bins: int = 8            # 8 orientation bins per cell -> 128-D
    gaussian_weighting: bool = True
    clamp: float = 0.2


@dataclass
class MatchConfig:
    """Ratio-test matching (Stage 1, keypoint branch)."""
    ratio: float = 0.8
    mutual: bool = False


@dataclass
class RansacConfig:
    """Robust homography estimation (Stage 1, geometry branch)."""
    threshold: float = 9.0        # symmetric transfer error threshold (px^2)
    confidence: float = 0.995     # p in N = log(1-p)/log(1-(1-eps)^4)
    max_iterations: int = 2000
    min_inliers: int = 12
    seed: int = 0                 # seed for the RANSAC random generator

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


@dataclass
class WarpConfig:
    """Artwork warping and single-frame overlay (Stage 1)."""
    feather: int = 0              # feather width in px at the quadrilateral edge
    match_gain: bool = False      # optional brightness matching of the artwork
    reference_shape: Optional[Tuple[int, int]] = None  # (H, W) of the reference board; set per clip


@dataclass
class LocalisationConfig:
    """Stage 1 Localise: reference-to-frame localisation."""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    descriptor: DescriptorConfig = field(default_factory=DescriptorConfig)
    matcher: MatchConfig = field(default_factory=MatchConfig)
    ransac: RansacConfig = field(default_factory=RansacConfig)
    warp: WarpConfig = field(default_factory=WarpConfig)
    success_threshold: float = 0.02   # normalised corner error; frozen on validation


@dataclass
class FlowConfig:
    """Single-level iterative Lucas-Kanade (Stage 2 Track)."""
    window: int = 7               # w: (2w+1)x(2w+1) window
    iterations: int = 20
    min_eigenvalue: float = 2e-4  # lambda_2 threshold on the normalised structure tensor
    max_residual: float = 0.08    # mean absolute residual (intensities in [0, 1])
    convergence: float = 0.01     # stop when the update is smaller than this (px)
    pyramid_levels: int = 1       # >1 enables the optional coarse-to-fine extension


@dataclass
class MonitorConfig:
    """Failure monitor thresholds (Stage 2 Track). Set on validation frames and frozen."""
    min_inlier_count: int = 12
    min_inlier_ratio: float = 0.4
    max_median_residual: float = 3.0      # px
    max_corner_displacement: float = 60.0  # px per frame
    max_mean_flow: float = 25.0           # px per frame
    max_brightness_change: float = 0.15   # mean |dI| over the board, [0, 1] scale
    hide_after_consecutive: int = 2       # frames of trouble before HIDE


@dataclass
class TrackConfig:
    """Stage 2 Track temporal variants."""
    variant: str = "flow_assisted"        # "independent" or "flow_assisted"
    flow: FlowConfig = field(default_factory=FlowConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    reinit_every: int = 60                # fixed-interval reinitialisation (0 = never)
    reinit_min_points: int = 20           # reinitialise when fewer valid points remain
    propagate: str = "inliers"            # "inliers" or "corners"


@dataclass
class OcclusionConfig:
    """Motion-compensated differencing (Stage 3 Composite)."""
    mode: str = "difference"              # "difference" or "colour" (ablation)
    strong: float = 0.25                  # strong hysteresis threshold, [0, 1] scale
    weak: float = 0.12                    # weak hysteresis threshold
    min_area: int = 150                   # px at FULL resolution; run_clip rescales it by (width/full)^2
    open_radius: int = 2                  # opening radius r: deletes structures thinner than 2r+1 px (0 disables)
    connectivity: int = 8
    gain_quantile: float = 0.5            # pixels below this |diff| quantile fit gain/offset
    feather: int = 3


@dataclass
class CompositeConfig:
    feather: int = 3
    match_gain: bool = False


@dataclass
class PipelineConfig:
    """Everything needed by overlaylab.pipelines.run_clip."""
    localisation: LocalisationConfig = field(default_factory=LocalisationConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    composite: CompositeConfig = field(default_factory=CompositeConfig)
    resize_width: Optional[int] = None    # None = full resolution

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def thresholds(self) -> dict:
        """Export every algorithm setting, excluding per-run resolution and reference shape.

        Pixel distances are in processing pixels; min_area is in original-video
        pixels. Preview and full-resolution results must be labelled separately.
        """
        frozen = asdict(self)
        frozen["resize_width"] = None
        frozen["localisation"]["warp"]["reference_shape"] = None
        return frozen

    @classmethod
    def from_json(cls, text: str) -> "PipelineConfig":
        """Build a config from a full or *partial* JSON dict (deep-merged onto the defaults)."""
        d = json.loads(text)
        if "monitor" in d and "track" not in d:          # accept a top-level monitor block
            d = {**d, "track": {"monitor": d["monitor"]}}
        cfg = cls()
        _merge_into(cfg, d)
        return cfg


def _merge_into(obj, d: dict) -> None:
    import dataclasses
    for k, v in d.items():
        if not dataclasses.is_dataclass(obj) or k not in {f.name for f in dataclasses.fields(obj)}:
            continue
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _merge_into(cur, v)
        elif k == "reference_shape" and v is not None:
            setattr(obj, k, tuple(v))
        else:
            setattr(obj, k, v)


def notebook_config() -> PipelineConfig:
    """Settings for the deterministic 640-pixel notebook subset."""
    cfg = PipelineConfig(resize_width=640)
    cfg.localisation.detector.max_corners = 600
    return cfg


__all__ = [
    "DetectorConfig", "DescriptorConfig", "MatchConfig", "RansacConfig", "WarpConfig",
    "LocalisationConfig", "FlowConfig", "MonitorConfig", "TrackConfig",
    "OcclusionConfig", "CompositeConfig", "PipelineConfig", "notebook_config",
    "replace",
]
