"""Task B tests: LK on a synthetic translation, validity flags, monitor logic, tracking interfaces."""
import unittest

import numpy as np

from overlaylab.config import FlowConfig, MonitorConfig, TrackConfig, LocalisationConfig
from overlaylab.temporal import lk_flow, failure_monitor, track_board, TrackInit, OK, REINIT, HIDE
from overlaylab.synthetic import textured_patch, translate_image, SequenceSpec, render_sequence
from overlaylab import metrics
from overlaylab import supplied as sp


class TestLucasKanade(unittest.TestCase):
    def test_recovers_subpixel_translation(self):
        img = textured_patch(160, seed=4, sigma=1.6)
        dx, dy = 1.7, -1.2
        nxt = translate_image(img, dx, dy)  # content moves by (+dx, +dy)
        ys, xs = np.mgrid[30:130:12, 30:130:12]
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
        cfg = FlowConfig(window=7, iterations=30, min_eigenvalue=1e-4, max_residual=0.1)
        d, eig, resid, valid = lk_flow(img, nxt, pts, cfg)
        self.assertEqual(d.shape, (len(pts), 2)); self.assertEqual(eig.shape, (len(pts), 2))
        self.assertTrue(np.all(eig[:, 0] >= eig[:, 1] - 1e-9))
        self.assertGreater(valid.mean(), 0.8)
        err = np.linalg.norm(d[valid] - np.array([dx, dy]), axis=1)
        self.assertLess(np.median(err), 0.15)

    def test_flat_window_is_invalid(self):
        img = np.full((64, 64), 0.5, np.float32)
        d, eig, resid, valid = lk_flow(img, img, np.array([[32.0, 32.0]]), FlowConfig())
        self.assertFalse(valid[0])
        self.assertLess(eig[0, 1], FlowConfig().min_eigenvalue)

    def test_aperture_problem_flags_small_second_eigenvalue(self):
        x = np.arange(96, dtype=np.float32)
        stripes = np.repeat((0.5 + 0.5 * np.sin(x / 3.0))[None, :], 96, axis=0)  # only Ix
        d, eig, resid, valid = lk_flow(stripes, stripes, np.array([[48.0, 48.0]]), FlowConfig(min_eigenvalue=1e-3))
        self.assertGreater(eig[0, 0], 1e-3)
        self.assertLess(eig[0, 1], 1e-6)
        self.assertFalse(valid[0])

    def test_point_leaving_image_is_invalid(self):
        img = textured_patch(64, seed=2)
        d, eig, resid, valid = lk_flow(img, img, np.array([[62.0, 32.0], [-3.0, 10.0]]), FlowConfig(window=5))
        self.assertFalse(valid[0]); self.assertFalse(valid[1])

    def test_pyramid_extension_handles_larger_motion(self):
        """Optional coarse-to-fine extension: skipped (not failed) when it is not implemented."""
        img = textured_patch(200, seed=8, sigma=1.2)
        nxt = translate_image(img, 11.0, -7.0)         # far beyond a single-level 5 px window
        pts = np.array([[80.0, 90.0], [120.0, 110.0], [100.0, 70.0]])
        cfg = FlowConfig(window=5, iterations=40, pyramid_levels=4, max_residual=0.15, min_eigenvalue=1e-4)
        d, _, _, valid = lk_flow(img, nxt, pts, cfg)
        single, _, _, valid_single = lk_flow(img, nxt, pts, FlowConfig(window=5, iterations=40, pyramid_levels=1,
                                                                        max_residual=0.15, min_eigenvalue=1e-4))
        if valid.sum() == 0 or np.allclose(d, single):
            self.skipTest("coarse-to-fine estimation not implemented (optional extension)")
        err = np.linalg.norm(d[valid] - np.array([11.0, -7.0]), axis=1)
        self.assertLess(np.median(err), 0.5)


class TestFailureMonitor(unittest.TestCase):
    def _s(self, **kw):
        base = dict(inlier_count=80, inlier_ratio=0.8, median_residual=0.8, corner_displacement=3.0,
                    mean_flow=2.0, brightness_change=0.02, n_valid_points=90, status="ok")
        base.update(kw)
        return base

    def test_flags_and_triggers(self):
        cfg = MonitorConfig(min_inlier_ratio=0.4, max_brightness_change=0.15, hide_after_consecutive=2,
                            max_corner_displacement=60)
        stats = [self._s(), self._s(inlier_ratio=0.3), self._s(inlier_ratio=0.2), self._s(),
                 self._s(brightness_change=0.4), self._s(status="lost"), self._s(corner_displacement=120)]
        out = failure_monitor(stats, cfg)
        self.assertEqual(len(out), len(stats))
        self.assertEqual(out[0][0], OK)
        self.assertEqual(out[1][0], REINIT); self.assertEqual(out[1][1], "inlier_ratio")
        self.assertEqual(out[2][0], HIDE)
        self.assertEqual(out[3][0], OK)
        self.assertEqual(out[4], (REINIT, "brightness_change"))
        self.assertEqual(out[5][0], HIDE)
        self.assertEqual(out[6], (REINIT, "corner_displacement"))

    def test_nan_statistics_are_ignored(self):
        out = failure_monitor([self._s(corner_displacement=float("nan"), mean_flow=float("nan"),
                                       brightness_change=float("nan"))], MonitorConfig())
        self.assertEqual(out[0][0], OK)


class TestTrackBoard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # gentle motion (a few px per frame) so that single-level LK can follow it
        spec = SequenceSpec(n_frames=6, frame_size=(640, 360), seed=12, occlusion=None,
                            pan_amplitude=0.012, zoom_range=(0.96, 1.02), tilt_deg=4.0,
                            jitter_px=0.6, base_scale=0.7, noise_std=0.005)
        cls.seq = render_sequence(spec)
        # the reference must be supplied at a resolution comparable to the on-screen board
        cls.ref = sp.resize_width(cls.seq.board, 480)
        cls.loc = LocalisationConfig()
        cls.loc.detector.max_corners = 500
        cls.loc.detector.nms_mode = "radius"
        cls.loc.detector.nms_radius = 5

    def _check(self, result):
        self.assertEqual(len(result.homographies), 6)
        self.assertEqual(len(result.corners), 6)
        self.assertEqual(len(result.stats), 6)
        self.assertEqual(set(result.status) - {"ok", "lost"}, set())
        errs = [metrics.corner_error(result.corners[i], self.seq.corners[i]) for i in range(6)
                if result.status[i] == "ok"]
        self.assertGreaterEqual(len(errs), 4)
        self.assertLess(np.median(errs), 0.03)
        for s in result.stats:
            for k in ("inlier_count", "inlier_ratio", "median_residual", "corner_displacement",
                      "mean_flow", "brightness_change", "status"):
                self.assertIn(k, s)

    def test_independent_variant(self):
        cfg = TrackConfig(variant="independent")
        self._check(track_board(self.seq.frames, TrackInit(self.ref, self.loc), cfg))

    def test_flow_assisted_variant_exposes_flow(self):
        cfg = TrackConfig(variant="flow_assisted", reinit_every=0)
        res = track_board(self.seq.frames, TrackInit(self.ref, self.loc), cfg)
        self._check(res)
        self.assertEqual(res.reinit_frames[0], 0)
        self.assertEqual(len(res.flow), 6)
        rec = res.flow[1]
        for k in ("points", "vectors", "eigenvalues", "valid"):
            self.assertIn(k, rec)
        self.assertEqual(rec["vectors"].shape[1], 2)


if __name__ == "__main__":
    unittest.main()
