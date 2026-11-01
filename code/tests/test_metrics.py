"""Supplied-metric tests (these pass out of the box; they document the metric semantics)."""
import unittest

import numpy as np

from overlaylab import metrics
from overlaylab import evaluate as E
from overlaylab.synthetic import make_artwork, homography_from_corners
from overlaylab.supplied import rect_corners
import cv2


class TestCornerMetrics(unittest.TestCase):
    def setUp(self):
        self.gt = np.array([[100, 100], [700, 120], [690, 300], [110, 280]], float)

    def test_corner_error_identity_and_translation(self):
        self.assertAlmostEqual(metrics.corner_error(self.gt, self.gt), 0.0)
        d = metrics.corner_error(self.gt + np.array([3.0, 4.0]), self.gt)
        diag = max(np.linalg.norm(self.gt[0] - self.gt[2]), np.linalg.norm(self.gt[1] - self.gt[3]))
        self.assertAlmostEqual(d, 5.0 / diag)
        self.assertEqual(metrics.corner_error(np.full((4, 2), np.nan), self.gt), float("inf"))

    def test_success_and_summary(self):
        e = [0.01, 0.02, 0.05, float("inf")]
        self.assertAlmostEqual(metrics.success_rate(e, 0.02), 0.5)
        s = metrics.summarise_errors(e, 0.02)
        self.assertEqual(s["n"], 4); self.assertEqual(s["failed"], 1)
        self.assertAlmostEqual(s["median"], 0.02)

    def test_interpolation_and_jitter(self):
        gt_frames = [0, 10]
        gt = [self.gt, self.gt + 10.0]
        mid = metrics.interpolate_corners(gt_frames, gt, [5])[0]
        np.testing.assert_allclose(mid, self.gt + 5.0)
        pred = {f: metrics.interpolate_corners(gt_frames, gt, [f])[0] for f in range(11)}
        self.assertAlmostEqual(metrics.jitter(pred, gt_frames, gt), 0.0)
        rng = np.random.default_rng(0)
        noisy = {f: c + rng.normal(0, 1.0, c.shape) for f, c in pred.items()}
        self.assertGreater(metrics.jitter(noisy, gt_frames, gt), 0.3)

    def test_drift_and_recovery(self):
        errors = {0: 0.0, 15: 0.01, 30: 0.02, 45: 0.005}
        d = dict(metrics.drift(errors, reinit_frames=[0, 40]))
        self.assertEqual(d, {0: 0.0, 15: 0.01, 30: 0.02, 5: 0.005})
        self.assertEqual(metrics.recovery_time({50: 0.1, 51: 0.05, 52: 0.01}, 50, 0.02), 2)
        self.assertIsNone(metrics.recovery_time({50: 0.1}, 50, 0.02, max_wait=3))


class TestMaskMetrics(unittest.TestCase):
    def test_iou_f1_leakage(self):
        a = np.zeros((10, 10), bool); a[:5] = True
        b = np.zeros((10, 10), bool); b[2:7] = True
        self.assertAlmostEqual(metrics.mask_iou(a, b), 30 / 70)
        self.assertAlmostEqual(metrics.mask_f1(a, b), 2 * 30 / 100)
        alpha = np.zeros((10, 10), np.float32); alpha[:3] = 1.0   # artwork painted on rows 0-2
        self.assertAlmostEqual(metrics.leakage(alpha, b), 10 / 50)
        self.assertTrue(np.isnan(metrics.leakage(alpha, np.zeros((10, 10), bool))))

    def test_boundary_f1_perfect(self):
        m = np.zeros((20, 20), bool); m[5:15, 5:15] = True
        self.assertAlmostEqual(metrics.boundary_f1(m, m), 1.0)


class TestContentMetrics(unittest.TestCase):
    def test_outside_change_ignores_inside_edits(self):
        frame = np.random.default_rng(1).uniform(0, 1, (120, 200, 3)).astype(np.float32)
        gt = np.array([[40, 30], [160, 30], [160, 90], [40, 90]], float)
        edited = frame.copy(); edited[40:80, 60:140] = 0.0
        self.assertAlmostEqual(metrics.outside_board_change(frame, edited, gt), 0.0)
        self.assertGreater(metrics.outside_board_change(frame, 1 - frame, gt), 0.1)

    def test_rectified_fidelity_high_for_exact_artwork(self):
        art = make_artwork(400, 100, "ad_A")
        gt = np.array([[50, 40], [330, 60], [320, 130], [60, 120]], float)
        H = homography_from_corners(rect_corners(400, 100), gt)
        frame = cv2.warpPerspective(art, H, (400, 200))
        f = metrics.rectified_content_fidelity(frame, gt, art)
        self.assertGreater(f["ncc"], 0.9)
        self.assertLess(f["mae"], 0.08)
        other = metrics.rectified_content_fidelity(frame, gt, make_artwork(400, 100, "ad_B"))
        self.assertLess(other["ncc"], f["ncc"])


class TestVideoEvaluation(unittest.TestCase):
    def test_psnr_ssim_and_artwork_relocalisation(self):
        art = make_artwork(800, 200, "ad_A")
        rng = np.random.default_rng(3)
        bg = rng.uniform(0.2, 0.8, (360, 640, 3)).astype(np.float32)
        gt = np.array([[80, 90], [560, 120], [540, 260], [100, 240]], float)
        H = homography_from_corners(rect_corners(800, 200), gt)
        frame = bg.copy()
        warped = cv2.warpPerspective(art, H, (640, 360)); cov = cv2.warpPerspective(np.ones((200, 800), np.float32), H, (640, 360)) > 0.5
        frame[cov] = warped[cov]
        self.assertAlmostEqual(E.ssim(frame, frame), 1.0, places=4)
        self.assertGreater(E.psnr(frame, frame + 0.01), 35)
        loc = E.ArtworkLocaliser(art)
        corners, n_in = loc(frame)
        self.assertIsNotNone(corners)
        self.assertLess(metrics.corner_error(corners, gt), 0.01)
        self.assertIsNone(loc(bg)[0])
        self.assertLess(E.background_drift(bg, frame, gt), 1.0)


if __name__ == "__main__":
    unittest.main()
