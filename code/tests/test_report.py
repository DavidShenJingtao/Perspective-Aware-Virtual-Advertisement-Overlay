"""The stress-table builder: tracking summaries from a stats dict, mask scoring, and the Markdown rendering."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from overlaylab import report as R
from overlaylab import supplied as sp


def _stats(n=64, reinit=(0, 40), hide=(20, 21), lost=(22,), width=640):
    corners = [np.array([[100.0 + t, 100.0], [300.0 + t, 100.0], [300.0 + t, 260.0], [100.0 + t, 260.0]]) for t in range(n)]
    status = ["ok"] * n
    for t in lost:
        status[t] = "lost"; corners[t] = np.full((4, 2), np.nan)
    flags = [["OK", "none"]] * n
    flags = [["HIDE", "inlier_ratio"] if t in hide else ["REINIT", "inlier_count"] if t in reinit else ["OK", "none"] for t in range(n)]
    return {"resize_width": width, "corners": [c.tolist() for c in corners], "status": status, "flags": flags,
            "reinit_frames": list(reinit)}


class TestSummariseTracking(unittest.TestCase):
    def test_rescales_and_counts(self):
        st = _stats()
        # ground truth at full resolution (1280 wide => factor 2), exact on even frames, 1 px off on frame 48
        gt = {f: np.array([[100.0 + f, 100.0], [300.0 + f, 100.0], [300.0 + f, 260.0], [100.0 + f, 260.0]]) * 2 for f in (16, 32, 48, 60)}
        out = R.summarise_tracking(st, gt, threshold=0.02, full_width=1280, drift_after=8)
        self.assertEqual(out["n"], 4); self.assertEqual(out["failed"], 0)
        self.assertAlmostEqual(out["median_error"], 0.0, places=6); self.assertAlmostEqual(out["p90_error"], 0.0, places=6)
        self.assertEqual(out["success"], 1.0)
        self.assertEqual(out["reinit"], 2); self.assertEqual(out["hide"], 2)
        self.assertEqual(out["frames_tracked"], 63); self.assertEqual(out["frames_total"], 64)
        self.assertAlmostEqual(out["jitter_px"], 0.0, places=6)
        # drift: frames >= 8 after the last reinit (0 or 40): 16, 32 (since 0), 48 and 60 (since 40) -> all four
        self.assertEqual(out["drift_frames"], 4)
        self.assertEqual(out["longest_run"], 40)          # reinits at 0 and 40 in 64 frames

    def test_hidden_and_lost_frames_fail(self):
        st = _stats(hide=(16,), lost=(32,))
        gt = {f: np.array([[100.0 + f, 100.0], [300.0 + f, 100.0], [300.0 + f, 260.0], [100.0 + f, 260.0]]) for f in (16, 32, 48)}
        out = R.summarise_tracking(st, gt, threshold=0.02, full_width=640)
        self.assertEqual(out["failed"], 2); self.assertAlmostEqual(out["success"], 1 / 3)


class TestMaskScores(unittest.TestCase):
    def test_iou_f1_hidden_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            gt = {10: np.zeros((40, 60), bool), 20: np.zeros((40, 60), bool), 30: np.zeros((40, 60), bool), 40: np.zeros((40, 60), bool)}
            gt[10][5:25, 5:35] = True; gt[20][0:10, 0:10] = True; gt[30][0:10, 0:10] = True      # 40 stays empty
            pred = np.zeros((40, 60), bool); pred[5:25, 5:35] = True
            sp.write_mask(Path(d) / "mask_000010.png", pred)
            stats = {"mask_dir": d, "mask_frames": [10], "mask_frames_hidden": [20]}
            out = R.mask_scores(stats, gt, [10, 20, 30, 40])
            self.assertAlmostEqual(out["iou"], 1.0); self.assertAlmostEqual(out["f1"], 1.0)
            self.assertEqual(out["mask_frames_scored"], 1)
            self.assertEqual(out["mask_frames_hidden"], [20]); self.assertEqual(out["mask_frames_missing"], [30])

    def test_resizes_ground_truth_to_mask_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            gt = {1: np.zeros((80, 120), bool)}; gt[1][10:50, 10:70] = True
            pred = np.zeros((40, 60), bool); pred[5:25, 5:35] = True
            sp.write_mask(Path(d) / "mask_000001.png", pred)
            out = R.mask_scores({"mask_dir": d, "mask_frames": [1], "mask_frames_hidden": []}, gt, [1])
            self.assertGreater(out["iou"], 0.9)


class TestRendering(unittest.TestCase):
    def test_markdown_table_and_missing_rows(self):
        rows = [{"label": "common · test", "n": 13, "median_error": 0.0042, "p90_error": 0.01, "success": 1.0,
                 "median_inliers": 80, "median_inlier_ratio": 0.7, "affine_median_error": 0.03, "affine_success": 0.2},
                {"label": "scale (coke_scale)", "note": "missing run"}]
        md = R.markdown_table(R.STAGE1_COLUMNS, rows)
        lines = md.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("| common · test | 13 | 0.42% | 1.00% | 100% | 80 | 0.700 | 3.00% | 20% |", lines[2])
        self.assertTrue(lines[3].startswith("| scale (coke_scale) | missing run |"))

    def test_affine_from_points_recovers_an_affine_map(self):
        rng = np.random.default_rng(0)
        src = rng.uniform(0, 100, (20, 2))
        A = np.array([[1.2, 0.1, 5.0], [-0.2, 0.9, -3.0], [0, 0, 1.0]])
        dst = (A @ np.c_[src, np.ones(20)].T).T[:, :2]
        np.testing.assert_allclose(R.affine_from_points(src, dst), A, atol=1e-9)

    def test_row_label_uses_factor_names(self):
        self.assertEqual(R.row_label("common_clip", "test", "flow_assisted"), "common · test · flow-assisted")
        label = R.row_label("not_a_known_clip")
        self.assertEqual(label, "not_a_known_clip (not_a_known_clip)")


if __name__ == "__main__":
    unittest.main()
