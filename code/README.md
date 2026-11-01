# CS4243 Lab 2 — Perspective-Aware Virtual Advertisement Overlay

The lab has four stages: **Stage 1 Localise** (`keypoints.py`, `geometry.py`, `localisation.py`), **Stage 2 Track**
(`temporal.py`), **Stage 3 Composite** (`compositing.py`) and **Stage 4 Your clip** (one personal clip, its Overlay
Scorecard and the generative comparison). Stages 1-3 are developed on the **common clip** and each is reported on the
same shape of table: the common clip (validation and test frames) plus the Stage's two **stress clips**, named by the
capture factor they isolate. `python -m overlaylab.report` builds those three tables for you.

**What to read where.** The handout (`CS4243_Lab_2.pdf`) says what each Stage is, why, and how it is marked; this README holds the data layout, clip identifiers, commands, the interfaces table and the submission tree; the docstring of every `YOUR CODE HERE` function is its specification (steps, return values, status flags); `REPORT_TEMPLATE.md` lists what every report section and Question must contain.

## Layout
```
code/            this folder: overlaylab/, tests/, notebooks/, manifests/, outputs/, student_files/
data/            beside code/: common_clip/, artwork/, diagnostic/<clip_id>/, personal/<clip_id>/
```
`data/` sits beside `code/` in the release and holds the following paths (the `frames/` folders, your artwork and
`personal/` are created by you):
```
data/common_clip/board_clip.mp4                 # the common clip
data/common_clip/board_reference.png
data/common_clip/frames/                        # python -m overlaylab.extract_frames ../data/common_clip
data/common_clip/masks/                         # six ground-truth occlusion masks
data/artwork/ad.png                             # the supplied replacement artwork (1200 x 1000, 6:5)
data/artwork/<your_artwork>.png                 # yours, for Stage 4 (submitted with your data)
data/diagnostic/<clip_id>/clip.mp4              # the stress clips (same poster, one capture factor each)
data/diagnostic/<clip_id>/reference.png
data/diagnostic/<clip_id>/masks/                # masks of the combined and viewpoint clips (manifests/diagnostic_masks.csv)
data/personal/<clip_id>/clip.mp4                # your clip (Stage 4)
data/personal/<clip_id>/reference.png           # rectified board at its on-screen width
data/personal/<clip_id>/frames/
data/personal/<clip_id>/masks/
```
Replace angle-bracket placeholders in commands with your own values; do not type the brackets.
The supplied frame extractor writes PNG frames with zero-padded indices; do not rename frames, masks or artwork.

## The clips
All clips show the same framed poster (POT-280 benchmark, object "Coke"), hand-held at 1280x720, and share one rectified
reference image. The common clip is the only one with **validation** frames: every threshold is chosen there (even
annotation index of `manifests/common_clip_corners.csv`), frozen into `outputs/thresholds.json`, and only *applied* to
the stress clips (`manifests/diagnostic_corners.csv`, `diagnostic_masks.csv`, all `test`) and to your own clip.

| factor | clip id | what it isolates | used by | masks |
|---|---|---|---|---|
| common | `common_clip` | hand-held, mild geometry, blue card crosses 11 times; development + validation | every Stage | 6 (3 validation + 3 test) |
| viewpoint | `coke_perspective` | camera orbits to a ~60 degree side view: only a full homography fits; nothing in front of the board | Stage 1, Stage 3 | 3 *empty* masks (false-occlusion test) |
| scale | `coke_scale` | camera approaches and retreats (board 0.29-1.0 of its reference size): the descriptor is not scale invariant | Stage 1 | none |
| blur | `coke_blur` | motion blur: brightness constancy fails, aperture problem on the board edges | Stage 2 | none |
| exit | `coke_out_of_view` | the board leaves the frame and re-enters: HIDE / REINIT | Stage 2 | none |
| combined | `coke_unconstrained` | rotation -79..80 degrees, orbit, fast motion, exits, blue card crosses twice | Stage 3 (and Question 3) | 6 (all test) |

`data.stress_clips()` returns `{1: {"viewpoint": "coke_perspective", "scale": "coke_scale"}, 2: {"blur": "coke_blur",
"exit": "coke_out_of_view"}, 3: {"viewpoint": "coke_perspective", "combined": "coke_unconstrained"}}` and
`data.clip_factor(clip_id)` the factor name. `coke_rotation` (in-plane rotation only) is shipped as supplementary material
and appears in no table.

## Setup
Use Python 3.11 (the version used for validation). Extract the entire release and keep `code/` and `data/` beside each other.
Run all commands below from `code/`. Use a local graphical environment for annotation; a notebook environment
(e.g. `pip install jupyterlab`) is needed for the notebooks. Install the FFmpeg command-line program for
video conversion and segmented-generation joins; `ffmpeg -version` checks it. It is separate from pip packages.

```
pip install -r requirements.txt
python -m overlaylab.extract_frames ../data/common_clip        # writes data/common_clip/frames/
python -m unittest discover -s tests -v                       # 82 tests; the 37 metric, evaluator, report and data tests pass before you start
```

## What you implement
`YOUR CODE HERE` regions in `overlaylab/keypoints.py`, `geometry.py`, `localisation.py` (Stage 1), `temporal.py`
(Stage 2), `compositing.py` (Stage 3) and the required cells of Notebooks 2-3. Do not change public signatures. Allowed
library calls are listed at the top of each file and in the handout; `cv2.SIFT`, `cv2.findHomography`,
`cv2.warpPerspective`, `cv2.calcOpticalFlowPyrLK` and `cv2.goodFeaturesToTrack` may appear only in the labelled
comparison cells of Notebooks 1-2; a learned flow such as RAFT, if you compare one, belongs in the same Notebook 2 cell (and `cv2.getPerspectiveTransform` only as the labelled ground-truth fallback of
Notebook 3, section 6).

## Notebooks
Run `01` -> `04` from a fresh kernel. Notebooks 1-3 use a deterministic subset of the common clip (every eighth frame at
640 px, cached under `data/common_clip/subset_640/`). Notebook 1 explores Stage 1 (parameter sweeps, DLT conditioning,
RANSAC iterations: explorations, not report questions) and the matching rate on the viewpoint and scale clips. Notebook 2
freezes the Stage 1 and Stage 2 thresholds and runs the affine-versus-homography comparison on the viewpoint clip.
Notebook 3 freezes the Stage 3 thresholds and previews the combined and viewpoint clips with them. Notebook 4 is Stage 4:
your clip, its Scorecard, and the generative comparison on your own audit frames with an open-weight video model that returns the edited video itself, run on your clip (cut it into a shorter interval or into segments joined afterwards if the model does not accept it), one attempt with fixed settings and seed (possibly several duration-limited segments). Use a model your hardware supports (the handout and `manifests/genai_conditions.json` list examples with citations: MiniMax-H3, JoyAI-Video-Edit, LTX-2.3, Wan2.1-VACE-14B; Wan2.1-VACE-1.3B, arXiv:2503.20314 / arXiv:2503.07598, if none fits on your GPU); a staff example with MiniMax-H3 is in `data/genai_oracle/`.

## Annotate your clip (Stage 4)

Prepare the final video **before annotation**. Record in landscape and use a constant frame rate so source timestamps
match frame indices. For phone recordings, this example creates a 30 fps clip and removes source metadata/audio:

```
ffmpeg -i <recording.mp4> -map_metadata -1 -map_chapters -1 -an -vf fps=30 -c:v libx264 -crf 18 -pix_fmt yuv420p ../data/personal/<clip_id>/clip.mp4
```

Create the clip directory first. Replace placeholders with actual paths. Do not trim, resize, retime or re-encode this
source clip after recording the audit annotations; that would invalidate their frame indices or pixel coordinates.

Plan the audit frames and annotate **before** you run anything on your clip. The five mask frames must be among the ten corner frames; also include the clean and strongest-perspective comparison frames. Put the video and the rectified board in place first:
`data/personal/<clip_id>/clip.mp4` and `data/personal/<clip_id>/reference.png`. Film in landscape: the video models
used in Stage 4 generate landscape video. Then, on your own computer (the tool opens a window, so it does not work
on a remote server or in Colab):
```
python -m overlaylab.annotate corners <clip_id> --annotator <name>                 # 10 evenly spaced frames
python -m overlaylab.annotate corners <clip_id> --frames <ten valid frame indices> --annotator <name>
python -m overlaylab.annotate masks <clip_id> --frames <five of those frame indices> --occluder "<object>" --annotator <name>
python -m overlaylab.annotate check <clip_id>                                      # validation + previews, no window
```
* **Corners** (10 frames, spread over the clip, including the frames you will use as clean and strongest-perspective
  frames): click the four board corners in the order of `reference.png`: 1 top-left, 2 top-right, 3 bottom-right,
  4 bottom-left. That is clockwise on screen, and the window shows the order on the reference. A magnifier follows the
  cursor. Keys: `u` undo, `r` restart the frame, `Enter` accept, `n` skip, `q` save and quit. A corner that the occluder
  hides is still clicked where the board's corner is.
* **Masks** (5 frames, at least 3 with the occluder on the board): outline the occluder with clicks and press `Enter`
  to fill the polygon (several polygons are allowed). `b` switches to a brush and `e` to an eraser (drag to paint;
  `+` / `-` change the size), and `p` returns to polygons. `u` undo, `r` clear, `s` save and go to the next frame,
  `q` save and quit. Annotate the corners of a mask frame first: the mask is then clipped to the board, like the
  supplied masks. A frame saved with an empty mask means "nothing in front of the board". Draw masks by hand: a
  promptable segmenter must not touch the audit frames.
* **What is written**: the frames go to `data/personal/<clip_id>/frames/` and the masks to `masks/`. Rows go to
  `manifests/audit_corners.csv` and `manifests/audit_masks.csv`, with the same columns as the templates and
  `annotated_before_inference = yes`. Re-annotating a frame replaces that annotator's row. Use one designated annotator name in the scored audit manifests; put independent secondary annotations in student_files/. `check` flags missing frames, the wrong
  corner order, crossed corners and wrong mask sizes, and writes previews to `outputs/annotation_check/<clip_id>/`.
  Look at them before you run the pipeline.
* Fill `manifests/capture_manifest.csv` from its template by hand.

## Reproduce the submitted videos and the stress tables
```
python -m overlaylab.run_pipeline --all --thresholds outputs/thresholds.json      # the eight videos below (~30 min)
python -m overlaylab.report                                                       # outputs/stress_tables.md + .json
python -m overlaylab.run_pipeline --clip <your clip id> --variant independent   --artwork <your_artwork> --thresholds outputs/thresholds.json
python -m overlaylab.run_pipeline --clip <your clip id> --variant flow_assisted --artwork <your_artwork> --thresholds outputs/thresholds.json
```
`thresholds.json` saves **every algorithm setting and seed**. It clears the per-run image width and reference
shape, so the documented submission commands run at full resolution. Values expressed in pixels refer to the
processed image; only `occlusion.min_area` is stored in original-video pixels and rescaled by area. Other pixel
settings are deliberately fixed numerical algorithm parameters, not promises of resolution invariance. Validate
final full-resolution settings on common-clip validation frames before final test/stress reporting. A preview
run may behave differently; label its resolution. Do not retune on stress or personal clips.

`--all` runs, with the frozen thresholds and the supplied artwork, the common clip with both temporal variants and the
no-occlusion baseline (`common_clip_independent`, `common_clip_flow_assisted`, `common_clip_flow_assisted_no_occlusion`)
and each stress clip with the flow-assisted variant (`coke_perspective`, `coke_scale`, `coke_blur`, `coke_out_of_view`,
`coke_unconstrained`); complete videos with matching settings are skipped (`--force` to redo). Changed settings, source data, implementation code or partial runs trigger a rerun. Each run writes
`outputs/videos/<clip_id>_<variant>.mp4`, `<clip_id>_<variant>_stats.json` (per-frame corners, statistics, monitor flags),
the occlusion masks of the clip's ground-truth mask frames under `<clip_id>_<variant>_masks/`, and a `.webm` copy of the
video when `ffmpeg` is installed (`--no-webm` to skip). VS Code's built-in preview cannot play `.mp4` but plays the
`.webm`; convert any other video with `python -m overlaylab.to_webm <video.mp4>`. Submit and evaluate the `.mp4`. Single
runs take `--clip`, `--variant`, `--no-occlusion` and `--width 960` while developing (the reference is rescaled with the
frames); submit full-resolution runs.

`report` writes the three **stress tables** (one per Stage, same rows everywhere: common clip validation, common clip
test, the Stage's two stress clips). Stage 1 success and affine success both use all eligible annotated frames. Median/p90 errors describe finite estimates; read them with success and failed counts.
Stage 1 is computed by running your `localise_board` on every annotated frame with
the frozen configuration (median / p90 corner error, success, inliers, and the error of an affine model fitted to the same
RANSAC inliers); Stage 2 from the `_stats.json` of the runs (corner metrics, jitter, drift, longest run, frames tracked, REINIT / HIDE
counts; common clip with both variants, blur and exit clips flow-assisted); Stage 3 from the saved masks (IoU, F1) and the
evaluator reports (leakage, baseline leakage on the common clip, false occlusion on the viewpoint clip). Missing evaluator
reports are computed on the fly. Paste `outputs/stress_tables.md` into the report; the grader rebuilds it from your runs.

## Evaluate a finished video or a set of edited frames
```
python -m overlaylab.evaluate --video outputs/videos/common_clip_flow_assisted.mp4
python -m overlaylab.evaluate --video outputs/videos/coke_blur_flow_assisted.mp4 --clip coke_blur
python -m overlaylab.evaluate --video outputs/videos/<clip_id>_flow_assisted.mp4 --clip <clip_id> --artwork <your_artwork>
python -m overlaylab.evaluate --frames outputs/genai/frame_*.png --clip <clip_id> --artwork <your_artwork>
```
The report (placement, fidelity, temporal consistency, preservation outside the board, occlusion) is computed from the
output, the original clip and the ground truth only, so it applies equally to your pipeline and to the generative outputs.
`--video` scores a finished video (all annotated frames inside `--interval`; `--offset` is the clip index of its first
frame). `--frames` scores edited still frames named `frame_XXXXXX.png` (the index is the clip frame number): placement /
fidelity / preservation on the supplied frames with ground-truth corners, occlusion on the supplied mask frames, and the
temporal terms on every supplied consecutive pair; frames of another resolution are resized to the clip's. The occlusion
block reports leakage and foreground PSNR on every frame with a non-empty mask (a hidden overlay leaks nothing), the
**visible-board coverage** (fraction of the unoccluded board that was painted) and, on frames whose mask is empty
(the viewpoint clip), the **false-occlusion fraction** (`1 - coverage`). Coverage and false occlusion are only computed on mask
frames where the artwork is actually detected in the output; frames where the overlay is absent (hidden by the monitor,
tracker lost) are listed as `overlay_absent_frames` and excluded, and the printed line says how many frames each mean
is based on. JSON reports go to `outputs/evaluation/`. The same tool is used for grading.

## Required Public Interfaces
| Function | Contract |
|---|---|
| `harris_response(gray, config)` | Harris response map R |
| `detect_corners(gray, config)` | NMS-thinned N x 2 keypoints ordered by response, and responses |
| `describe_keypoints(gray, keypoints, config)` | N' x 128 orientation-normalised, clamped, unit-norm descriptors and kept indices |
| `match_descriptors(desc_a, desc_b, config)` | Ratio-tested (optionally mutual) index pairs, distances, and ratios |
| `bilinear_sample(image, y, x)` | Lab 1 bilinear interpolation, extended to colour images |
| `normalise_points(pts)` | Similarity T and normalised points with zero centroid and mean distance sqrt(2) |
| `dlt_homography(src, dst)` | 3 x 3 homography with h9 = 1, singular values, and status |
| `symmetric_transfer_error(H, src, dst)` | Squared forward error plus squared backward error, in pixel squared units |
| `ransac_homography(src, dst, config)` | Homography, Boolean inlier mask, residuals, iterations used, and status |
| `transfer_points(H, pts)` | Projected points pi(Hx) for N x 2 input |
| `warp_artwork(artwork, H, frame_shape)` | Destination-to-source bilinear warp and Boolean coverage mask |
| `localise_board(reference, frame, config)` | Homography, four projected corners, matches, inlier mask, and diagnostics |
| `artwork_to_frame(artwork, H, config)` | Composed artwork-to-frame homography |
| `overlay_frame(frame, artwork, H, config)` | Single overlaid frame without occlusion handling |
| `evaluate_audit_frames(reference, frames, gt_corners, config)` | Per-frame errors and summary on annotated frames |
| `lk_flow(prev_gray, next_gray, points, config)` | Displacements, eigenvalues (lambda_1, lambda_2), residuals, and validity flags |
| `track_board(frames, init, config)` | Per-frame homography, corners, status, and monitor statistics |
| `failure_monitor(stats, config)` | Per-frame OK / REINIT / HIDE flag and the triggering statistic |
| `hysteresis(strong, weak, connectivity)` | Lab 1 hysteresis: weak pixels reachable from strong seeds |
| `compensated_difference(frame, warped_reference, coverage, config)` | Gain/offset-compensated difference map and the fitted (a, b) |
| `occlusion_mask(frame, H, reference, config)` | Occlusion mask inside coverage, difference map, and component statistics |
| `select_difference_thresholds(frames, masks, config)` | Frozen hysteresis levels and minimum area chosen on validation masks only |
| `composite(frame, warped, coverage, occlusion, config)` | Composited frame and alpha map |

## Report Traceability Table
Use `REPORT_TEMPLATE.md`. The prose must mention every function containing a `YOUR CODE HERE` region at least once, and
the final evidence table must retain and complete every row. A test name alone is insufficient: cite the corresponding
report discussion and notebook figure, table, video, or numerical result.

| Student file | Required sections | Minimum evidence location |
|---|---|---|
| keypoints.py | harris_response, detect_corners, describe_keypoints, match_descriptors | Report Stage 1 section and Notebook 1 |
| geometry.py | bilinear_sample, normalise_points, dlt_homography, symmetric_transfer_error, ransac_homography, transfer_points, warp_artwork | Report Stage 1 section and Notebook 1 |
| localisation.py | localise_board, artwork_to_frame, overlay_frame, evaluate_audit_frames | Report Stage 1 section and Notebook 2 |
| temporal.py | lk_flow, track_board, failure_monitor | Report Stage 2 section, Notebook 2, and the common-clip videos of both variants |
| compositing.py | hysteresis, compensated_difference, occlusion_mask, select_difference_thresholds, composite | Report Stage 3 section and Notebook 3 |

## Submission
Submit one zip with two folders, laid out as in the release: `code/` (this whole folder) and `data/` holding only
the files you created, `data/personal/<clip_id>/` (your clip, reference, frames and masks) and
`data/artwork/<your_artwork>.png`. Leave out the supplied data (`common_clip/`, `diagnostic/`, `genai_oracle/`,
`artwork/ad.png`). Add any other files (the AI provenance appendix, helper scripts) only in `code/student_files/`;
do not add files anywhere else.
```
data
    personal/<clip_id>/
        clip.mp4, reference.png, frames/, masks/
    artwork/<your_artwork>.png
code
    manifests
        dataset_summary.json
        common_clip_corners.csv
        common_clip_masks.csv
        diagnostic_corners.csv
        diagnostic_masks.csv
        capture_manifest_template.csv
        audit_corners_template.csv
        audit_masks_template.csv
        genai_conditions.json          # frames and prompts filled in
        genai_log_template.csv
        capture_manifest.csv           # filled from the templates
        audit_corners.csv
        audit_masks.csv
        genai_log.csv
    notebooks
        01_keypoints_and_geometry.ipynb
        02_localisation_and_temporal.ipynb
        03_occlusion_and_compositing.ipynb
        04_personal_clips_and_genai_analysis.ipynb
    outputs
        videos
            common_clip_independent.mp4, common_clip_flow_assisted.mp4, common_clip_flow_assisted_no_occlusion.mp4
            coke_perspective_flow_assisted.mp4, coke_scale_flow_assisted.mp4, coke_blur_flow_assisted.mp4,
            coke_out_of_view_flow_assisted.mp4, coke_unconstrained_flow_assisted.mp4
            <clip_id>_independent.mp4, <clip_id>_flow_assisted.mp4      # your clip
            <clip_id>_<variant>_stats.json, <clip_id>_<variant>_masks/  # one per video (run_pipeline writes them)
        genai
            inputs/                    # exported clip (or cut) and board masks
            output.mp4, frame_XXXXXX.png   # the model's video and the audit frames extracted from it
        evaluation
            <video>_report.json, genai_frames_report.json, stage4_genai_reports.json
        scorecards
            <clip_id>_scorecard.md
        stress_tables.md, stress_tables.json    # python -m overlaylab.report
        thresholds.json
        stage1_settings.json            # Notebook 1 -> Notebook 2 settings handoff
    student_files
        <AI provenance appendix, any other files you add>
    tests
        test_keypoints.py, test_geometry.py, test_flow.py, test_compositing.py, test_metrics.py, test_evaluate.py, test_report.py, test_real_data.py, test_annotate.py
    overlaylab
        config.py, data.py, keypoints.py, geometry.py, localisation.py, temporal.py, compositing.py,
        metrics.py, evaluate.py, report.py, pipelines.py, run_pipeline.py, extract_frames.py, annotate.py, to_webm.py, supplied.py, synthetic.py, visualize.py, __init__.py
    README.md
    REPORT.md
    REPORT_TEMPLATE.md
    requirements.txt
```
Record every threshold in `outputs/thresholds.json`, the generative run in `manifests/genai_log.csv` (one open-weight video model, direct pixel generation, one attempt with fixed settings and seed; see `manifests/genai_conditions.json`;
`[AI-GENERATION]` on every generated pixel), and complete `REPORT.md` from
`REPORT_TEMPLATE.md`, including the evidence table and the AI-use disclosure table.

The supplied annotation manifests retain only valid benchmark corner rows. A 16-frame annotation grid does not
mean that every multiple of 16 appears; the common clip has 26 retained rows, split 13/13 by row order.
The viewpoint clip's empty masks are tests of false occlusion, not missing labels.

## Boundary rules and interpretation
Finish validation tuning before looking at test/stress results. Once a Stage is frozen, later Stages must not
retune it using test evidence. Run Notebook 3 before launching the final `--all` videos and stress tables.
Synthetic ablations and validation-frame sweeps are for development. Stress-frame plots are explanations of
frozen results; they must not change the submitted settings.

Notebook 3 section 6 has one explicitly labelled **evaluation-only** exception: a ground-truth homography may
isolate occlusion behaviour when localisation fails. Label every such result `GT-fallback`. It is not a pipeline
prediction and must never enter submitted videos, tracking results or the generated stress tables.
Notebook 2 section 6 is a labelled `cv2.warpPerspective` comparison for parallax; it is not the assessed warp.

Use the two worst cases for failure analysis. If fewer than two fail your frozen threshold, say so; do not
intentionally damage a method to manufacture a failure. The same rule applies to the personal Scorecard.

For Stage 4, choose three distinct audit frames (clean, occluded, strongest perspective) and a consecutive pair
before generation. Include the occluded frame in both audit manifests. The temporal pair need not add extra
corner annotations: the evaluator interpolates corners, using the same annotation dictionary for both methods.
Use one attempt; duration-limited segments belong to that attempt. Keep settings and seed fixed, log every
segment and preserve each raw output. A failed technical invocation may be retried unchanged and logged.
Inference scripts are permitted; generated code or a classical overlay cannot perform the generative edit.
Sample at nominal source timestamps and report retiming as an error, not as a reason to select better frames.

The supplied `data/genai_oracle/` is the example explicitly required by the handout. It contains a generative
output and measurements, not an implementation of any assessed function. None of the 23 assessed functions
is filled in, and the notebooks have no saved outputs.

The Stage 2 `tracked` count means tracker status `ok`; some of those frames may still be hidden by the monitor. Read it alongside `HIDE`. The monitor timeline and per-frame flags show which frames were actually painted.

`propagate="corners"` is optional. It supplies only four points, so it requires compatible settings:
`track.reinit_min_points <= 4`, `localisation.ransac.min_inliers = 4` and a monitor minimum inlier count of at most four.
The required default uses inlier correspondences. Choose optional-mode settings on validation/synthetic data only.

Finished-video leakage and visible-board coverage are **pixel-change proxies**: the evaluator calls a pixel painted
when mean RGB difference from the original exceeds 0.08. Thus `false_occlusion = 1 - coverage` on empty-mask
frames can include feathering, resampling and artwork colours similar to the original, as well as true false masks.
Direct mask fractions / alpha leakage in Notebook 3 are separate measurements and need not equal the output-video
proxies. The same video metric is used for classical and generative outputs. Coverage is excluded when SIFT does not
detect the artwork; that flag can also mean failed detection, not only deliberate HIDE. Quote counts and inspect frames.

Stages 1 and 3 have four rows. Stage 2 has six because its common validation/test rows each appear for both temporal variants; blur and exit each have one flow-assisted row.

Jitter uses all available annotations for the clip, not just the row's validation/test subset. It is therefore identical between those two rows for a given variant. `(re)localisations` counts actual localiser calls; `REINIT flags` counts monitor requests, which can differ.

Exclude model checkpoints, Python environments, caches, notebook checkpoint folders and optional `.webm` previews from submission. Keep model/version/settings and provenance in the log and appendix instead. The submission contains the required MP4s, not the preview copies.

## Complete the evaluator reports
The stress-table builder evaluates the four videos needed for Stage 3. Evaluate the remaining four supplied videos too:

```
python -m overlaylab.evaluate --video outputs/videos/common_clip_independent.mp4
python -m overlaylab.evaluate --video outputs/videos/coke_scale_flow_assisted.mp4 --clip coke_scale
python -m overlaylab.evaluate --video outputs/videos/coke_blur_flow_assisted.mp4 --clip coke_blur
python -m overlaylab.evaluate --video outputs/videos/coke_out_of_view_flow_assisted.mp4 --clip coke_out_of_view
```

Notebook 4 saves both full personal-video reports and the reports comparing identical selected frames.
Development previews under `outputs/annotation_check/` are optional; omit them from submission or put selected evidence
in `student_files/`. Include the required generated reports and outputs in their listed locations.
