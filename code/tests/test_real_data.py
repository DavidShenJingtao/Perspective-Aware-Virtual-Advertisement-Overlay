"""Manifest counts, split boundaries, the stress clips, and ground-truth correspondence on the common clip.

These tests need the data bundle extracted to ``data/`` beside ``code/``; they
are skipped when it is missing so that the pure-synthetic tests still run.
"""
import unittest

import numpy as np

from overlaylab import data
from overlaylab import supplied as sp


def _bundle_present():
    return data.COMMON_CLIP.exists() and data.CORNERS_MANIFEST.exists()


@unittest.skipUnless(_bundle_present(), "common clip bundle not found under data/")
class TestManifests(unittest.TestCase):
    def test_corner_manifest_counts_and_splits(self):
        rows = sp.read_csv(data.CORNERS_MANIFEST)
        s = sp.read_json(data.DATASET_SUMMARY)
        self.assertEqual(len(rows), s["common_clip"]["annotated_corner_frames"])
        self.assertGreaterEqual(len(rows), 26)
        frames = [int(r["frame"]) for r in rows]
        self.assertEqual(frames, sorted(frames))
        self.assertTrue(all(f % data.GT_STRIDE == 0 for f in frames))
        splits = [r["split"] for r in rows]
        self.assertEqual(splits.count("validation"), splits.count("test"))
        for k, r in enumerate(rows):
            self.assertEqual(r["split"], data.split_of_annotation_index(k))

    def test_mask_manifest_counts_and_files(self):
        rows = sp.read_csv(data.MASKS_MANIFEST)
        self.assertEqual(len(rows), 6)
        self.assertEqual([r["split"] for r in rows].count("validation"), 3)
        for r in rows:
            self.assertTrue(sp.resolve(data.MASKS_MANIFEST, r["mask_path"]).exists(), r["mask_path"])

    def test_artwork_and_reference_present(self):
        s = sp.read_json(data.DATASET_SUMMARY)
        aspect = s["common_clip"].get("board_aspect", 4.0)
        for name in ("ad",):
            art = data.load_artwork(name)
            self.assertAlmostEqual(art.shape[1] / art.shape[0], aspect, places=2)
        ref = data.load_reference()
        self.assertEqual(ref.ndim, 3)
        self.assertAlmostEqual(ref.shape[1] / ref.shape[0], aspect, places=2)

    def test_diagnostic_clips_if_present(self):
        for clip_id in data.diagnostic_clip_ids():
            video, ref = data.clip_paths(clip_id)
            self.assertTrue(video.exists(), video); self.assertTrue(ref.exists(), ref)
            gt = data.load_diagnostic_corners(clip_id)
            self.assertGreaterEqual(len(gt), 10, clip_id)
            self.assertTrue(all(f % data.GT_STRIDE == 0 for f in gt), clip_id)
            self.assertGreaterEqual(sp.video_info(video)["frames"], max(gt) + 1, clip_id)

    def test_stress_clips_if_present(self):
        if not data.diagnostic_clip_ids():
            self.skipTest("no stress clips")
        # every stress clip isolates one capture factor; each Stage has exactly two stress clips, named by factor
        summary = sp.read_json(data.DATASET_SUMMARY)
        factors = summary["factors"]
        self.assertEqual(factors["common_clip"], "common"); self.assertEqual(data.clip_factor("common_clip"), "common")
        for clip_id in data.diagnostic_clip_ids():
            f = data.clip_factor(clip_id)
            self.assertEqual(f, factors[clip_id], clip_id)
            self.assertEqual(data.diagnostic_clip_ids(factor=f), [clip_id], f"factor {f!r} must name exactly one clip")
            self.assertEqual(summary["diagnostic_clips"][clip_id]["factor"], f)
        stress = data.stress_clips()
        self.assertEqual(sorted(stress), [1, 2, 3])
        for stage, clips in stress.items():
            self.assertEqual(len(clips), 2, f"Stage {stage} must have two stress clips, got {clips}")
            for factor, clip_id in clips.items():
                self.assertEqual(data.clip_factor(clip_id), factor, (stage, factor, clip_id))
                self.assertIn(clip_id, data.diagnostic_clip_ids())
                self.assertIn(stage, summary["diagnostic_clips"][clip_id]["stages"])
        self.assertEqual(data.stress_clips(1), stress[1])
        self.assertEqual(set(data.stress_clip_ids()), {c for m in stress.values() for c in m.values()})
        # Stage 3 masks: the combined clip has real masks, the viewpoint clip empty ones
        for factor, clip_id in stress[3].items():
            self.assertEqual(sorted(data.load_diagnostic_masks(clip_id)), summary["diagnostic_clips"][clip_id]["mask_frames"])
            self.assertEqual(data.mask_frames_for_clip(clip_id), summary["diagnostic_clips"][clip_id]["mask_frames"])
        self.assertEqual(data.mask_frames_for_clip("common_clip"), summary["common_clip"]["mask_frames"])
        self.assertEqual(data.mask_frames_for_clip(data.stress_clips(2)["blur"]), [])

    def test_diagnostic_masks_exist_and_lie_in_their_quad(self):
        if not data.DIAGNOSTIC_MASKS_MANIFEST.exists():
            self.skipTest("no diagnostic masks manifest")
        rows = sp.read_csv(data.DIAGNOSTIC_MASKS_MANIFEST)
        self.assertGreater(len(rows), 0)
        for r in rows:
            path = sp.resolve(data.DIAGNOSTIC_MASKS_MANIFEST, r["mask_path"])
            self.assertTrue(path.exists(), r["mask_path"])
            self.assertEqual(r["factor"], data.clip_factor(r["clip_id"]), r["clip_id"])
            self.assertIn(r["factor"], data.stress_clips(3), r["clip_id"])
            self.assertEqual(r["split"], "test")
            info = sp.video_info(data.clip_paths(r["clip_id"])[0])
            m = sp.read_mask(path)
            self.assertEqual(m.shape, (info["height"], info["width"]), r["mask_path"])
            self.assertEqual(int(m.sum()), int(r["foreground_pixels"]), r["mask_path"])
            if r["occluder"] == "none":
                self.assertEqual(int(m.sum()), 0, f"{r['clip_id']} frame {r['frame']}: occluder 'none' must be an empty mask")
                continue
            self.assertGreater(m.sum(), 200)
            inside = sp.quad_mask(m.shape, sp.corners_from_row(r))      # the row's own POT corners
            self.assertGreater(np.logical_and(m, inside).sum() / m.sum(), 0.9, f"{r['clip_id']} frame {r['frame']}")
        # the loaders agree with the manifest
        for clip_id in {r["clip_id"] for r in rows}:
            masks = data.load_diagnostic_masks(clip_id); corners = data.load_diagnostic_mask_corners(clip_id)
            self.assertEqual(sorted(masks), sorted(corners))
            self.assertTrue(all(c.shape == (4, 2) for c in corners.values()))

    def test_summary_matches_manifests(self):
        s = sp.read_json(data.DATASET_SUMMARY)
        self.assertEqual(s["common_clip"]["annotated_corner_frames"], len(sp.read_csv(data.CORNERS_MANIFEST)))
        self.assertEqual(s["common_clip"]["occlusion_masks"], 6)
        info = sp.video_info(data.COMMON_CLIP)
        self.assertEqual(s["common_clip"]["width"], info["width"])
        self.assertEqual(s["common_clip"]["height"], info["height"])


@unittest.skipUnless(_bundle_present(), "common clip bundle not found under data/")
class TestGroundTruthCorrespondence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info = sp.video_info(data.COMMON_CLIP)
        cls.gt = data.load_gt_corners()
        cls.masks = data.load_gt_masks()

    def test_quadrilaterals_are_convex_and_inside_frame(self):
        W, H = self.info["width"], self.info["height"]
        for f, c in self.gt.items():
            self.assertTrue(np.all(c[:, 0] >= -1) and np.all(c[:, 0] <= W), f)
            self.assertTrue(np.all(c[:, 1] >= -1) and np.all(c[:, 1] <= H), f)
            # consistent orientation of consecutive edges => convex
            d = np.roll(c, -1, axis=0) - c
            cross = d[:, 0] * np.roll(d, -1, axis=0)[:, 1] - d[:, 1] * np.roll(d, -1, axis=0)[:, 0]
            self.assertTrue(np.all(cross > 0) or np.all(cross < 0), f)
            # TL is left of TR and above BL
            self.assertLess(c[0, 0], c[1, 0]); self.assertLess(c[0, 1], c[3, 1])

    def test_board_is_prominent(self):
        W, H = self.info["width"], self.info["height"]
        for f, c in self.gt.items():
            self.assertGreater(sp.quad_mask((H, W), c).mean(), 0.05, f)

    def test_annotated_frames_exist_in_video(self):
        self.assertGreaterEqual(self.info["frames"], max(self.gt) + 1)
        self.assertGreaterEqual(self.info["frames"], max(self.masks) + 1)

    def test_masks_match_resolution_and_lie_on_the_board(self):
        W, H = self.info["width"], self.info["height"]
        gt_all = data.load_gt_corners()
        from overlaylab import metrics
        for f, m in self.masks.items():
            self.assertEqual(m.shape, (H, W))
            self.assertGreater(m.sum(), 200)
            # masks are only annotated inside the board: interpolate corners to the mask frame
            frames = sorted(gt_all)
            c = metrics.interpolate_corners(frames, [gt_all[k] for k in frames], [f])[0]
            inside = sp.quad_mask((H, W), c)
            self.assertGreater(np.logical_and(m, inside).sum() / m.sum(), 0.9, f)

    def test_genai_template_structure(self):
        cond = data.load_genai_conditions()
        self.assertEqual(cond["mode"], "personal_clip_frames")
        self.assertNotIn("conditions", cond); self.assertNotIn("condition_order", cond)
        for key in ("clean", "occluded", "perspective", "pair"):
            self.assertIn(key, cond["frames"])
        self.assertEqual(len(cond["frames"]["pair"]), 2)
        self.assertIn("template", cond["prompt"]); self.assertIn("inputs", cond["prompt"]); self.assertIn("seed", cond["model"])
        summary = sp.read_json(data.DATASET_SUMMARY)
        self.assertEqual(summary["genai"]["mode"], "personal_clip_frames")
        self.assertEqual(summary["genai"]["runs"], 1)
        self.assertEqual(summary["personal_clips"]["per_submission"], 1)


if __name__ == "__main__":
    unittest.main()
