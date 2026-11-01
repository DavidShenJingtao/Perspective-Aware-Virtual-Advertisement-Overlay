"""Part 2 tests: normalisation, DLT recovery and degeneracy, RANSAC with outliers, warp direction."""
import unittest

import numpy as np

from overlaylab.config import RansacConfig
from overlaylab.geometry import (bilinear_sample, normalise_points, dlt_homography, transfer_points, ransac_homography,
                                 warp_artwork, symmetric_transfer_error)
from overlaylab.synthetic import (random_homography, synthetic_correspondences, apply_homography,
                                  checkerboard)


def _same_homography(H1, H2, tol=1e-4):
    H1 = H1 / H1[2, 2]; H2 = H2 / H2[2, 2]
    return np.abs(H1 - H2).max() / max(np.abs(H2).max(), 1) < tol


class TestBilinear(unittest.TestCase):
    def test_plane_is_exact_and_colour_is_supported(self):
        y, x = np.mgrid[:8, :9]; plane = (2 * y + 3 * x).astype(np.float32)
        yy = np.array([1.2, 4.5]); xx = np.array([2.4, 5.25])
        np.testing.assert_allclose(bilinear_sample(plane, yy, xx), 2 * yy + 3 * xx, atol=1e-5)
        colour = np.stack([plane, 2 * plane, -plane], axis=-1)
        out = bilinear_sample(colour, yy, xx)
        self.assertEqual(out.shape, (2, 3))
        np.testing.assert_allclose(out[:, 1], 2 * (2 * yy + 3 * xx), atol=1e-5)
        np.testing.assert_allclose(bilinear_sample(plane, np.array([7.0]), np.array([8.0])), [plane[7, 8]])


class TestNormalisation(unittest.TestCase):
    def test_centroid_and_scale(self):
        rng = np.random.default_rng(1)
        pts = rng.uniform(0, 1000, size=(30, 2))
        T, n = normalise_points(pts)
        np.testing.assert_allclose(n.mean(axis=0), 0, atol=1e-9)
        self.assertAlmostEqual(np.linalg.norm(n, axis=1).mean(), np.sqrt(2), places=9)
        hom = np.concatenate([pts, np.ones((30, 1))], axis=1) @ T.T
        np.testing.assert_allclose(hom[:, :2] / hom[:, 2:3], n, atol=1e-9)
        self.assertEqual(T.shape, (3, 3))


class TestDLT(unittest.TestCase):
    def test_exact_recovery_from_four_and_many_points(self):
        H, _ = random_homography(seed=3)
        for n in (4, 25):
            src, dst, _ = synthetic_correspondences(H, n_inliers=n, seed=n)
            Hest, svals, ok = dlt_homography(src, dst)
            self.assertTrue(ok)
            self.assertAlmostEqual(Hest[2, 2], 1.0)
            self.assertTrue(_same_homography(Hest, H), msg=f"n={n}")
            self.assertEqual(len(svals), min(2 * n, 9))

    def test_refuses_fewer_than_four(self):
        with self.assertRaises(ValueError):
            dlt_homography(np.zeros((3, 2)), np.zeros((3, 2)))

    def test_collinear_points_report_failure(self):
        src = np.array([[0, 0], [10, 10], [20, 20], [30, 30]], float)
        dst = src * 2 + 5
        H, svals, ok = dlt_homography(src, dst)
        self.assertFalse(ok)
        self.assertIsNone(H)
        self.assertTrue(np.all(np.isfinite(svals)))

    def test_transfer_points_matches_reference(self):
        H, _ = random_homography(seed=4)
        pts = np.random.default_rng(0).uniform(0, 500, size=(12, 2))
        np.testing.assert_allclose(transfer_points(H, pts), apply_homography(H, pts), atol=1e-9)

    def test_symmetric_transfer_error_zero_for_exact(self):
        H, _ = random_homography(seed=5)
        src, dst, _ = synthetic_correspondences(H, n_inliers=10, seed=2)
        np.testing.assert_allclose(symmetric_transfer_error(H, src, dst), 0, atol=1e-6)


class TestRANSAC(unittest.TestCase):
    def test_recovers_homography_with_forty_percent_outliers(self):
        H, _ = random_homography(seed=6)
        src, dst, is_in = synthetic_correspondences(H, n_inliers=60, n_outliers=40, noise=0.3, seed=7)
        cfg = RansacConfig(threshold=9.0, confidence=0.999, max_iterations=3000, min_inliers=12, seed=1)
        Hest, mask, resid, iters, ok = ransac_homography(src, dst, cfg)
        self.assertTrue(ok)
        self.assertLessEqual(iters, cfg.max_iterations)
        self.assertGreater(np.mean(mask == is_in), 0.95)
        self.assertEqual(resid.shape, (100,))
        corners = np.array([[0, 0], [1599, 0], [1599, 399], [0, 399]], float)
        err = np.linalg.norm(transfer_points(Hest, corners) - transfer_points(H, corners), axis=1).mean()
        self.assertLess(err, 1.5)

    def test_adaptive_iterations_stop_early_without_outliers(self):
        H, _ = random_homography(seed=8)
        src, dst, _ = synthetic_correspondences(H, n_inliers=40, noise=0.1, seed=9)
        cfg = RansacConfig(max_iterations=5000, seed=3)
        _, mask, _, iters, ok = ransac_homography(src, dst, cfg)
        self.assertTrue(ok)
        self.assertLess(iters, 50)
        self.assertGreater(mask.mean(), 0.9)

    def test_status_false_when_too_few_inliers(self):
        rng = np.random.default_rng(0)
        src = rng.uniform(0, 1000, (30, 2)); dst = rng.uniform(0, 1000, (30, 2))  # pure noise
        cfg = RansacConfig(threshold=4.0, max_iterations=200, min_inliers=12, seed=0)
        H, mask, resid, iters, ok = ransac_homography(src, dst, cfg)
        self.assertFalse(ok)
        self.assertLess(mask.sum(), 12)

    def test_deterministic_with_seed(self):
        H, _ = random_homography(seed=10)
        src, dst, _ = synthetic_correspondences(H, n_inliers=50, n_outliers=30, noise=0.5, seed=11)
        cfg = RansacConfig(seed=42)
        a = ransac_homography(src, dst, cfg); b = ransac_homography(src, dst, cfg)
        np.testing.assert_allclose(a[0], b[0]); self.assertEqual(a[3], b[3])


class TestWarp(unittest.TestCase):
    def test_integer_translation_lands_on_expected_pixels(self):
        art = np.repeat(checkerboard(8, 8, 16)[..., None], 3, axis=-1)
        H = np.array([[1, 0, 50], [0, 1, 30], [0, 0, 1]], float)
        warped, cov = warp_artwork(art, H, (240, 320, 3))
        self.assertEqual(warped.shape, (240, 320, 3)); self.assertEqual(cov.shape, (240, 320))
        self.assertEqual(int(cov.sum()), 128 * 128)
        self.assertTrue(cov[30, 50] and cov[157, 177])
        self.assertFalse(cov[29, 50] or cov[30, 49] or cov[158, 177])
        np.testing.assert_allclose(warped[30:158, 50:178], art, atol=1e-6)
        self.assertEqual(warped[0, 0, 0], 0.0)

    def test_scale_maps_corners_in_destination_to_source_direction(self):
        art = np.repeat(checkerboard(4, 4, 8)[..., None], 3, axis=-1)   # 32 x 32
        H = np.array([[2, 0, 10], [0, 2, 20], [0, 0, 1]], float)       # artwork -> frame
        warped, cov = warp_artwork(art, H, (120, 120, 3))
        ys, xs = np.nonzero(cov)
        self.assertEqual((xs.min(), ys.min()), (10, 20))
        self.assertEqual((xs.max(), ys.max()), (10 + 62, 20 + 62))
        # frame pixel (10 + 2*8, 20 + 2*8) maps back to artwork pixel (8, 8): second cell, value 0
        self.assertAlmostEqual(float(warped[20 + 16, 10 + 16, 0]), float(art[8, 8, 0]), places=5)

    def test_projective_warp_covers_quadrilateral(self):
        from overlaylab.synthetic import homography_from_corners
        from overlaylab.supplied import rect_corners, quad_mask
        art = np.repeat(checkerboard(4, 16, 8)[..., None], 3, axis=-1)  # 32 x 128
        dst = np.array([[40, 30], [200, 50], [190, 110], [50, 100]], float)
        H = homography_from_corners(rect_corners(128, 32), dst)
        warped, cov = warp_artwork(art, H, (160, 260, 3))
        inside = quad_mask((160, 260), dst)
        self.assertGreater(np.logical_and(cov, inside).sum() / inside.sum(), 0.95)
        self.assertLess(np.logical_and(cov, ~inside).sum() / cov.sum(), 0.05)


if __name__ == "__main__":
    unittest.main()
