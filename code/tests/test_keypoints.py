"""Part 1 tests: Harris response, both NMS modes, descriptor invariants, ratio test."""
import unittest

import numpy as np

from overlaylab.config import DetectorConfig, DescriptorConfig, MatchConfig
from overlaylab.keypoints import harris_response, detect_corners, describe_keypoints, match_descriptors
from overlaylab.synthetic import rotated_corner, checkerboard, textured_patch


class TestHarris(unittest.TestCase):
    def test_response_peaks_at_corner_and_not_on_flat_regions(self):
        img = rotated_corner(64, 0.0)
        R = harris_response(img, DetectorConfig(sigma_w=1.5, kappa=0.05))
        self.assertEqual(R.shape, img.shape)
        y, x = np.unravel_index(np.argmax(R), R.shape)
        self.assertLess(abs(x - 31.5), 3.5); self.assertLess(abs(y - 31.5), 3.5)
        self.assertLessEqual(R[5:15, 5:15].max(), 1e-6)          # flat background
        self.assertLessEqual(R[48:60, 48:60].max(), 1e-6)        # flat foreground

    def test_response_is_rotation_invariant(self):
        cfg = DetectorConfig(sigma_w=1.5, kappa=0.05)
        peaks = [harris_response(rotated_corner(64, a), cfg).max() for a in (0, 30, 60, 90)]
        self.assertGreater(min(peaks), 0)
        self.assertLess(max(peaks) / min(peaks), 1.3)

    def test_edges_have_negative_or_small_response(self):
        img = np.zeros((64, 64), np.float32); img[:, 32:] = 1.0
        R = harris_response(img, DetectorConfig(sigma_w=1.5, kappa=0.05))
        self.assertLessEqual(R[20:44, 30:34].max(), 1e-6)


class TestNMS(unittest.TestCase):
    def test_window_mode_finds_checkerboard_crossings(self):
        img = checkerboard(8, 8, 16)
        cfg = DetectorConfig(sigma_w=1.5, nms_mode="window", nms_radius=4, border=8, max_corners=200)
        kps, resp = detect_corners(img, cfg)
        self.assertGreaterEqual(len(kps), 30)
        self.assertEqual(kps.shape[1], 2)
        self.assertTrue(np.all(np.diff(resp) <= 1e-9))                      # sorted by response
        # every corner sits near a cell crossing (crossings are at 16k - 0.5)
        for x, y in kps:
            self.assertLess(abs(((x + 8.5) % 16) - 8), 2.0, msg=f"x={x}")
            self.assertLess(abs(((y + 8.5) % 16) - 8), 2.0, msg=f"y={y}")

    def test_radius_mode_enforces_spacing(self):
        img = textured_patch(160, seed=5, sigma=1.2)
        cfg = DetectorConfig(sigma_w=1.5, nms_mode="radius", nms_radius=7, border=6, max_corners=120,
                             response_fraction=0.001)
        kps, resp = detect_corners(img, cfg)
        self.assertGreaterEqual(len(kps), 10)
        d = np.linalg.norm(kps[:, None, :] - kps[None, :, :], axis=-1) + np.eye(len(kps)) * 1e9
        self.assertGreater(d.min(), 7.0)
        self.assertTrue(np.all(np.diff(resp) <= 1e-9))

    def test_border_and_max_corners(self):
        img = textured_patch(128, seed=6, sigma=1.2)
        cfg = DetectorConfig(sigma_w=1.5, nms_mode="window", nms_radius=2, border=10, max_corners=25,
                             response_fraction=0.001)
        kps, resp = detect_corners(img, cfg)
        self.assertLessEqual(len(kps), 25)
        self.assertTrue(np.all(kps[:, 0] >= 10) and np.all(kps[:, 0] <= 117))
        self.assertTrue(np.all(kps[:, 1] >= 10) and np.all(kps[:, 1] <= 117))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            detect_corners(textured_patch(64), DetectorConfig(nms_mode="bogus"))


class TestDescriptor(unittest.TestCase):
    def setUp(self):
        self.img = textured_patch(160, seed=9, sigma=1.4)
        self.kps, _ = detect_corners(self.img, DetectorConfig(sigma_w=1.5, nms_mode="radius",
                                                             nms_radius=6, border=14, max_corners=60,
                                                             response_fraction=0.001))

    def test_shape_norm_and_clamp(self):
        desc, kept = describe_keypoints(self.img, self.kps, DescriptorConfig())
        self.assertEqual(desc.shape[1], 128)
        self.assertEqual(desc.shape[0], len(kept))
        np.testing.assert_allclose(np.linalg.norm(desc, axis=1), 1.0, atol=1e-4)
        self.assertLessEqual(desc.max(), 0.5)
        self.assertTrue(np.all(desc >= 0))

    def test_window_leaving_image_is_dropped(self):
        kps = np.array([[2.0, 2.0], [80.0, 80.0], [157.0, 60.0]])
        desc, kept = describe_keypoints(self.img, kps, DescriptorConfig())
        self.assertEqual(list(kept), [1])

    def test_rotation_consistency(self):
        # rotate by exactly 90 degrees: (x, y) -> (y, W-1-x) under np.rot90
        img90 = np.ascontiguousarray(np.rot90(self.img))
        W = self.img.shape[1]
        kps90 = np.stack([self.kps[:, 1], W - 1 - self.kps[:, 0]], axis=1)
        cfg = DescriptorConfig()
        d0, k0 = describe_keypoints(self.img, self.kps, cfg)
        d1, k1 = describe_keypoints(img90, kps90, cfg)
        common = sorted(set(k0) & set(k1))
        self.assertGreater(len(common), 10)
        i0 = [list(k0).index(c) for c in common]; i1 = [list(k1).index(c) for c in common]
        a, b = d0[i0], d1[i1]
        dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        correct = np.mean(np.argmin(dist, axis=1) == np.arange(len(a)))
        self.assertGreater(correct, 0.7)


class TestMatching(unittest.TestCase):
    def test_ratio_test_rejects_ambiguous_matches(self):
        rng = np.random.default_rng(0)
        a0 = rng.normal(size=128); a0 /= np.linalg.norm(a0)
        a1 = rng.normal(size=128); a1 /= np.linalg.norm(a1)
        A = np.stack([a0, a1])
        B = np.stack([a0 + 0.01 * rng.normal(size=128),        # clear match for a0
                      a1 + 0.02 * rng.normal(size=128),        # near-duplicate 1 for a1
                      a1 + 0.02 * rng.normal(size=128),        # near-duplicate 2 for a1 -> ambiguous
                      rng.normal(size=128)])
        pairs, dist, ratio = match_descriptors(A, B, MatchConfig(ratio=0.8, mutual=False))
        self.assertEqual(pairs.tolist(), [[0, 0]])
        self.assertEqual(len(dist), 1); self.assertEqual(len(ratio), 1)
        self.assertLess(ratio[0], 0.8)
        pairs_loose, _, _ = match_descriptors(A, B, MatchConfig(ratio=1.01))
        self.assertEqual(len(pairs_loose), 2)

    def test_mutual_check(self):
        A = np.array([[0.0, 0.0], [0.6, 0.0]])
        B = np.array([[0.5, 0.0], [5.0, 5.0], [9.0, 0.0]])
        pairs, _, _ = match_descriptors(A, B, MatchConfig(ratio=0.9, mutual=False))
        self.assertIn([0, 0], pairs.tolist()); self.assertIn([1, 0], pairs.tolist())
        pairs_m, _, _ = match_descriptors(A, B, MatchConfig(ratio=0.9, mutual=True))
        self.assertEqual(pairs_m.tolist(), [[1, 0]])    # B0's nearest in A is A1

    def test_empty_inputs(self):
        pairs, dist, ratio = match_descriptors(np.zeros((0, 128)), np.zeros((5, 128)), MatchConfig())
        self.assertEqual(pairs.shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
