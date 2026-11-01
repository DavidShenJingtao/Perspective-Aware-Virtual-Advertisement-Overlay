"""Task C tests: hysteresis connectivity, occlusion mask on a synthetic occluder, alpha bounds."""
import unittest

import numpy as np

from overlaylab.config import OcclusionConfig, CompositeConfig
from overlaylab.compositing import hysteresis, occlusion_mask, select_difference_thresholds, composite
from overlaylab.synthetic import (make_board_texture, make_background, render_frame,
                                  homography_from_corners, project_board)
from overlaylab.supplied import rect_corners, quad_mask
from overlaylab import metrics


class TestHysteresis(unittest.TestCase):
    def test_diagonal_chain_needs_eight_connectivity(self):
        strong = np.zeros((7, 7), bool); strong[0, 0] = True
        weak = np.eye(7, dtype=bool)
        self.assertEqual(hysteresis(strong, weak, 8).sum(), 7)
        self.assertEqual(hysteresis(strong, weak, 4).sum(), 1)

    def test_unreachable_weak_pixels_are_dropped(self):
        strong = np.zeros((5, 5), bool); strong[2, 2] = True
        weak = strong.copy(); weak[0, 4] = True; weak[2, 3] = True
        out = hysteresis(strong, weak, 4)
        self.assertTrue(out[2, 2] and out[2, 3]); self.assertFalse(out[0, 4])

    def test_validation(self):
        with self.assertRaises(ValueError):
            hysteresis(np.zeros((3, 3), bool), np.zeros((4, 4), bool), 8)
        with self.assertRaises(ValueError):
            hysteresis(np.zeros((3, 3), bool), np.zeros((3, 3), bool), 6)


class _Scene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.board = make_board_texture(800, 200, seed=1)
        cls.bg = make_background(1400, 800, seed=2)
        W, H = 640, 360
        cls.corners = project_board(800, 200, (320, 180), 420, np.deg2rad(10), 0.05, 0.0)
        cls.H = homography_from_corners(rect_corners(800, 200), cls.corners)
        Hbg = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1.0]])
        rng = np.random.default_rng(0)
        ball = ("ball", np.array([330.0, 170.0, 35.0]))
        cls.clean, _, cls.cov = render_frame(cls.board, cls.bg, cls.H, Hbg, (W, H), gain=0.85, gradient=0.08,
                                             noise_std=0.01, rng=rng)
        cls.occluded, occ, _ = render_frame(cls.board, cls.bg, cls.H, Hbg, (W, H), gain=0.85, gradient=0.08,
                                            noise_std=0.01, occluders=[ball], rng=rng)
        cls.gt = occ & quad_mask((H, W), cls.corners)


class TestOcclusionMask(_Scene):
    def test_detects_ball_and_ignores_illumination(self):
        cfg = OcclusionConfig(strong=0.25, weak=0.12, min_area=60, open_radius=1)
        m, diff, comp = occlusion_mask(self.occluded, self.H, self.board, cfg)
        self.assertEqual(m.shape, self.gt.shape); self.assertEqual(diff.shape, self.gt.shape)
        self.assertGreater(metrics.mask_iou(m, self.gt), 0.5)
        self.assertEqual(comp.shape[1], 6)
        self.assertGreaterEqual(len(comp), 1)
        self.assertTrue(np.all(comp[:, 5] >= cfg.min_area))
        m0, _, _ = occlusion_mask(self.clean, self.H, self.board, cfg)
        self.assertLess(m0.sum() / self.cov.sum(), 0.02)   # gain/offset absorbed the illumination change

    def test_mask_is_inside_coverage(self):
        cfg = OcclusionConfig(strong=0.25, weak=0.12, min_area=60)
        m, _, _ = occlusion_mask(self.occluded, self.H, self.board, cfg)
        self.assertEqual(np.logical_and(m, ~self.cov).sum(), 0)

    def test_min_area_removes_small_components(self):
        big = OcclusionConfig(strong=0.25, weak=0.12, min_area=10 ** 6)
        m, _, comp = occlusion_mask(self.occluded, self.H, self.board, big)
        self.assertEqual(m.sum(), 0); self.assertEqual(len(comp), 0)

    def test_threshold_selection_returns_frozen_config(self):
        cfg = OcclusionConfig(min_area=60)
        frozen, log = select_difference_thresholds([(self.occluded, self.H, self.board)], [self.gt], cfg)
        self.assertLess(frozen.weak, frozen.strong)
        self.assertGreater(len(log), 5)
        self.assertIn("mean_iou", log[0])
        m, _, _ = occlusion_mask(self.occluded, self.H, self.board, frozen)
        self.assertGreater(metrics.mask_iou(m, self.gt), 0.5)


class TestComposite(unittest.TestCase):
    def test_alpha_formula_and_bounds(self):
        rng = np.random.default_rng(0)
        frame = rng.uniform(0, 1, (40, 60, 3)).astype(np.float32)
        warped = rng.uniform(0, 1, (40, 60, 3)).astype(np.float32)
        C = np.zeros((40, 60), bool); C[5:35, 10:50] = True
        O = np.zeros((40, 60), bool); O[15:25, 20:30] = True
        out, alpha = composite(frame, warped, C, O, CompositeConfig(feather=0))
        self.assertEqual(out.shape, frame.shape); self.assertEqual(alpha.shape, (40, 60))
        self.assertTrue(np.all(alpha >= 0) and np.all(alpha <= 1))
        np.testing.assert_allclose(alpha, (C & ~O).astype(np.float32))
        np.testing.assert_allclose(out[~C], frame[~C])
        np.testing.assert_allclose(out[O], frame[O])
        np.testing.assert_allclose(out[C & ~O], warped[C & ~O], atol=1e-6)

    def test_feathering_softens_boundaries(self):
        frame = np.zeros((40, 60, 3), np.float32); warped = np.ones((40, 60, 3), np.float32)
        C = np.zeros((40, 60), bool); C[5:35, 10:50] = True
        O = np.zeros((40, 60), bool)
        out, alpha = composite(frame, warped, C, O, CompositeConfig(feather=4))
        self.assertTrue(np.all(alpha >= 0) and np.all(alpha <= 1))
        self.assertLess(alpha[5, 10], 0.6)       # near the quad edge
        self.assertGreater(alpha[20, 30], 0.99)  # interior
        self.assertEqual(alpha[~C].max(), 0.0)


if __name__ == "__main__":
    unittest.main()
