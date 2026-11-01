"""The annotation tool's logic, driven without a window (mouse and key events are fed to the state objects)."""
import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from overlaylab import annotate as A


class TestAnnotate(unittest.TestCase):
    def setUp(self):
        self.im = np.full((720, 1280, 3), 128, np.uint8)
        self.q = np.array([[300, 200], [900, 180], [930, 560], [320, 600]], float)

    def test_quad_order_checks(self):
        self.assertEqual(A.quad_problems(self.q, (720, 1280)), [])
        self.assertTrue(any("anticlockwise" in p for p in A.quad_problems(self.q[[0, 3, 2, 1]], (720, 1280))))
        self.assertTrue(any("convex" in p for p in A.quad_problems(self.q[[0, 2, 1, 3]], (720, 1280))))

    def test_corner_state_clicks_undo_accept(self):
        st = A.CornerState(self.im, 0, 1, 1)
        st.mouse(cv2.EVENT_LBUTTONDOWN, 5, 5); st.key(ord("u"))            # a wrong click, undone
        for p in self.q:
            x, y = p * st.view.s
            st.mouse(cv2.EVENT_LBUTTONDOWN, int(round(x)), int(round(y)))
        st.key(13)
        self.assertEqual(st.result, "accept")
        self.assertLess(np.abs(st.corners - self.q).max(), 1.0 / st.view.s + 1e-6)
        self.assertEqual(st.render().ndim, 3)

    def test_corner_state_needs_four(self):
        st = A.CornerState(self.im, 0, 1, 1)
        st.mouse(cv2.EVENT_LBUTTONDOWN, 10, 10); st.key(13)
        self.assertIsNone(st.result)

    def test_mask_state_polygon_brush_and_clip(self):
        st = A.MaskState(self.im, 0, 1, 1, board=self.q)
        for x, y in [(100, 100), (600, 100), (600, 400), (100, 400)]:               # half outside the board
            st.mouse(cv2.EVENT_LBUTTONDOWN, int(x * st.view.s), int(y * st.view.s))
        st.key(13)
        m = st.final_mask
        self.assertTrue(m[300, 500]); self.assertFalse(m[150, 150])               # clipped to the board
        st.key(ord("b")); st.mouse(cv2.EVENT_LBUTTONDOWN, int(800 * st.view.s), int(400 * st.view.s))
        self.assertTrue(st.final_mask[400, 800])
        st.key(ord("u")); self.assertFalse(st.final_mask[400, 800])
        st.key(ord("s")); self.assertEqual(st.result, "accept")
        self.assertEqual(st.render().shape[2], 3)

    def test_upsert_replaces_same_frame_and_annotator(self):
        with tempfile.TemporaryDirectory() as d:
            man = Path(d) / "audit_corners.csv"
            row = {"clip_id": "c", "frame": 5, "frame_path": "f", "annotator": "a", "annotated_before_inference": "yes",
                   **{f"{ax}{i}": "1.000" for i in range(1, 5) for ax in "xy"}}
            A.upsert_rows(man, A.CORNER_FIELDS, [row])
            A.upsert_rows(man, A.CORNER_FIELDS, [{**row, "x1": "2.000"}])
            A.upsert_rows(man, A.CORNER_FIELDS, [{**row, "annotator": "b"}])
            with open(man) as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            self.assertEqual([r["x1"] for r in rows if r["annotator"] == "a"], ["2.000"])

    def test_default_frames_spread(self):
        f = A.default_frames(450, 10)
        self.assertEqual(len(f), 10); self.assertEqual(f[0], 0); self.assertEqual(f[-1], 449)


if __name__ == "__main__":
    unittest.main()
