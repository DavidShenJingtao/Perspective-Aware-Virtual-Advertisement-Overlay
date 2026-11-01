"""Supplied evaluator in frames mode (``overlaylab.evaluate.evaluate_frames``) on untouched common-clip frames.

Needs the data bundle under ``data/``; skipped otherwise. No student code is called.
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from overlaylab import data
from overlaylab import evaluate as E
from overlaylab import supplied as sp


@unittest.skipUnless(data.COMMON_CLIP.exists() and data.CORNERS_MANIFEST.exists(), "common clip bundle not found under data/")
class TestEvaluateFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="lab2_frames_"))
        masks = data.load_gt_masks()
        f0 = sorted(masks)[0]                       # a mask frame; it is also a ground-truth corner frame
        cls.f0 = f0
        for i, frame in sp.iter_video(data.COMMON_CLIP, start=f0, stop=f0 + 2):
            sp.write_image(cls.tmp / f"frame_{i:06d}.png", frame)   # unchanged frames written back
        cls.paths = sorted(cls.tmp.glob("frame_*.png"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_frame_index_from_name(self):
        self.assertEqual(E.frame_index_from_name("outputs/genai/frame_000096.png"), 96)
        with self.assertRaises(ValueError):
            E.frame_index_from_name("genai/output.png")

    def test_unchanged_frames(self):
        self.assertEqual(len(self.paths), 2)
        rep = E.evaluate_frames(self.paths, verbose=False)
        self.assertEqual(rep["mode"], "frames")
        self.assertEqual(rep["frames_with_corners"], [self.f0])
        self.assertEqual(rep["frames_with_masks"], [self.f0])
        self.assertFalse(rep["resized"])
        # nothing was painted: the artwork is not found (inf error, no detection) and the evaluator does not crash
        p = rep["placement"]
        self.assertEqual(p["n"], 1)
        self.assertTrue(p["detection_rate"] == 0.0 or (p["median"] is not None and np.isfinite(p["median"])))
        # the frames are the original pixels up to PNG round-off
        self.assertGreaterEqual(rep["preservation"]["psnr_outside"], 35.0)
        # occlusion: nothing was painted, so the artwork is not detected -> overlay absent on the mask frame;
        # coverage / false occlusion are not computed (nan), leakage still is (zero: a hidden overlay leaks nothing)
        o = rep["occlusion"]
        self.assertEqual(o["mask_frames"], 1); self.assertEqual(o["empty_mask_frames"], 0)
        self.assertTrue(o["per_frame"][0]["overlay_absent"])
        self.assertEqual(o["overlay_absent_frames"], [self.f0]); self.assertEqual(o["n_coverage_frames"], 0)
        self.assertTrue(np.isnan(o["coverage_visible"])); self.assertTrue(np.isnan(o["false_occlusion"]))
        self.assertTrue(np.isnan(o["per_frame"][0]["false_occlusion"]))
        self.assertLess(o["leakage"], 0.01)
        self.assertIn("overlay absent on 1", E.format_report(rep))
        # temporal terms on the one supplied pair (f0, f0 + 1)
        self.assertEqual(rep["temporal"]["pairs"], 1)
        self.assertTrue(np.isfinite(rep["temporal"]["flicker"]))

    def test_resized_input_is_handled(self):
        import cv2
        small = self.tmp / "small"; small.mkdir(exist_ok=True)
        for p in self.paths:
            img = sp.read_image(p)
            sp.write_image(small / p.name, cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA))
        rep = E.evaluate_frames(sorted(small.glob("frame_*.png")), verbose=False)
        self.assertTrue(rep["resized"])
        self.assertEqual(len(rep["resized_frames"]), 2)
        self.assertGreaterEqual(rep["preservation"]["psnr_outside"], 25.0)
        self.assertTrue(E.format_report(rep).startswith("Evaluation of 2 frames"))


if __name__ == "__main__":
    unittest.main()
