# CS4243 Lab 2 report

This compact report records how every `YOUR CODE HERE` section is exercised. Elaborate on your implementation and the results you observed; add figures, plots and grounded reasoning that show your understanding. Every reported number must name its clip (common / viewpoint / scale / blur / exit / combined / your clip), frame subset (validation / test / audit) and method variant, and must come from `overlaylab.metrics` or `overlaylab.evaluate`. Thresholds are frozen on the common clip's validation frames; each Stage's **stress table** (`python -m overlaylab.report`, `outputs/stress_tables.md`) reports the common clip (validation and test) and the Stage's two stress clips with that frozen configuration. Paste the generated table into the Stage's section unchanged and explain the row the Stage's question points at; the other rows need no interpretation.

## Stage 1 — Localise
<!-- harris_response, detect_corners, describe_keypoints, match_descriptors; bilinear_sample, normalise_points, dlt_homography, symmetric_transfer_error, ransac_homography, transfer_points, warp_artwork; localise_board, artwork_to_frame, overlay_frame, evaluate_audit_frames; frozen success threshold -->
### Stress table (Stage 1)
<!-- generated rows: common · validation, common · test, viewpoint, scale — median / p90 corner error, success at the frozen threshold, inliers, inlier ratio, affine median error / success -->
### Question 1 — Matching
<!-- a frame pair on the viewpoint or scale clip where the matching rate collapses; rotation absorbed by the dominant orientation vs viewpoint / scale not absorbed; which descriptor component fails; the labelled SIFT cell -->
### Question 2 — Geometry
<!-- the affine column on the viewpoint clip: why the affine fit breaks as the camera orbits, related to the keystone; whole-frame warp and parallax; why parallax does not harm the overlay and one capture condition under which it would -->

## Stage 2 — Track
<!-- lk_flow, track_board (both variants on the common clip), failure_monitor; jitter and drift; frozen monitor thresholds -->
### Stress table (Stage 2)
<!-- generated rows: common · validation / test for both variants, blur, exit — corner metrics, jitter, drift, frames tracked, REINIT / HIDE counts -->
### Question 3 — Tracking
<!-- flicker (independent) vs drift (flow-assisted); eigenvalues: aperture vs brightness constancy, where each kind lies on the board (blur clip); monitor timeline of the combined clip: which statistic fired first at the card crossing and at the fast rotation, occlusion vs drift; what the exit clip's HIDE / REINIT sequence shows -->

## Stage 3 — Composite
<!-- hysteresis, compensated_difference, occlusion_mask, select_difference_thresholds, composite; frozen difference thresholds; no-occlusion baseline -->
### Stress table (Stage 3)
<!-- generated rows: common · validation / test, combined, viewpoint — mask IoU / F1 / leakage, baseline leakage (common), false occlusion (viewpoint) -->
| row | eligible masks | scored masks | hidden masks | missing masks | overlay absent | IoU | F1 | leakage | baseline leakage | false occlusion |
|---|---|---|---|---|---|---|---|---|---|---|
| common · validation | 3 | 3 | 0 | 0 | 0 | 0.917 | 0.956 | 1.56% | 96.40% | n/a |
| common · test | 3 | 2 | 1 | 0 | 1 | 0.944 | 0.971 | 0.61% | 61.74% | n/a |
| combined (coke_unconstrained) | 6 | 4 | 2 | 0 | 2 | 0.935 | 0.966 | 0.22% | n/a | n/a |
| viewpoint (coke_perspective) | 3 | 0 | 1 | 0 | 1 | n/a | n/a | n/a | n/a | 10.65% |

IoU / F1 from the occlusion masks saved by `run_pipeline` on the ground-truth mask frames (frames on which the overlay is hidden are not scored); leakage, baseline leakage and false occlusion from `overlaylab.evaluate` on the finished videos. The viewpoint clip has empty masks: only false occlusion applies.

At full resolution, occlusion handling reduced finished-video foreground leakage from **96.40%** to **1.56%** on common validation and from **61.74%** to **0.61%** on common test. The common test mask IoU describes two scored frames, with one hidden. The combined clip achieved **0.935** IoU on four scored frames, with two hidden. There were no missing masks.

### Question 4 — Occlusion
<!-- which occluders differencing recovers and which stay ambiguous; effect of the strong threshold and the minimum area; why hysteresis matters for thin occluders and why opening works against it (thinnest occluder that survives the frozen open_radius; what the radius buys over radius 0 on the validation masks); why the viewpoint clip produces false occlusion under strong perspective with a correct homography, and which part of the differencing is responsible -->
1. **Which occluders are recovered, and which remain ambiguous?**
The blue card is recovered where its colour contrasts with the poster. An occluder coloured like the board is ambiguous and can be missed. A shadow or specular highlight can be mistaken for foreground because it changes the board’s appearance. An occluder that stops moving remains detectable if it contrasts with the board, since each frame is compared with the clean reference.
2. **What happens when the strong threshold or minimum component area is raised?** On common validation frame 192 in section 5 of notebook 3, raising the strong threshold from **0.25** to **0.375** or the minimum area from **38** to **228** pixels kept the mask at **15997** pixels and alpha leakage at **0.1%**. Enough strong seeds remained to recover the same region, and the card component exceeded both area cutoffs. Higher cutoffs can increase leakage if they remove genuine foreground.
3. **Why does hysteresis help thin occluders, and why does opening work against it?** Hysteresis retains weak pixels connected to a strong seed, which helps preserve faint sections of a thin occluder such as a pole. On common validation frame 192, disabling hysteresis shrank the mask from **15997** to **9525** pixels and raised alpha leakage from **0.1%** to **29.0%**. Opening erodes the mask with a disc and then dilates what survives. During erosion, a pixel survives only if the entire disc centred on it fits inside the foreground. If the disc cannot fit anywhere within a thin component, erosion removes it entirely and dilation cannot restore it. With the frozen radius of 2 pixels (Section 3), the thinnest horizontal strip to survive a separate synthetic check was 5 pixels wide.
4. **Why does the viewpoint clip produce false occlusion even with a correct homography?** A correct homography aligns the board geometrically. Under strong foreshortening, however, fine texture can be sampled or blurred differently in the captured frame and the warped reference. The global gain and offset cannot account for these local changes. The compensated per-pixel colour difference can therefore exceed the foreground thresholds even where the board is unobstructed. In Notebook 3, Section 6, viewpoint frame 288 has a ground-truth occlusion mask. Even with the evaluation-only ground-truth homography, the algorithm falsely masks **2.7%** of the visible board. The full-resolution stress table reports **10.65%** false occlusion rate using a separate finished-video coverage proxy over two empty-mask frames with detected overlays.

## Stage 4 — Your clip
### Capture and audit
<!-- capture conditions (resolution, fps, duration, board fraction, pan / distance / angle / occlusion crossing), privacy, audit annotations recorded before any method ran -->
### Overlay Scorecard
<!-- outputs/scorecards/<clip_id>_scorecard.md: both temporal variants, overlaylab.evaluate --clip <clip_id> -->
### Generative comparison
<!-- the open-weight video model you chose (direct pixel generation; inference scripts permitted), its settings, the video you gave it (whole clip, a cut, or joined segments), and why (cite the model); the seed, the log, the overlaylab.evaluate --frames report, comparison table against the classical output on the same frames, [AI-GENERATION] markers -->
### Question 5 — Reflection
<!-- classical vs your generative model on your clip: geometric placement, artwork identity, occlusion order, background preservation (what the background PSNR says about generated vs composited pixels); controllability, cost; what the model generated outside the board and in front of it; whether it kept the clip's timing (compare with the staff MiniMax-H3 example in data/genai_oracle/) -->

## Evidence table
| Student file | Function | Report section | Notebook figure / table / video |
|---|---|---|---|
| keypoints.py | harris_response, detect_corners, describe_keypoints, match_descriptors | | |
| geometry.py | bilinear_sample, normalise_points, dlt_homography, symmetric_transfer_error, ransac_homography, transfer_points, warp_artwork | | |
| localisation.py | localise_board, artwork_to_frame, overlay_frame, evaluate_audit_frames | | |
| temporal.py | lk_flow, track_board, failure_monitor | | |
| compositing.py | hysteresis, compensated_difference, occlusion_mask, select_difference_thresholds, composite | | |

## AI-use disclosure table
| Tool / model | Affected artefacts (file, cell, figure, section) | Markers | What the tool contributed | How it was verified |
|---|---|---|---|---|
| | | [AI-CODE] [AI-DESIGN] [AI-VISION] [AI-GENERATION] [AI-LANGUAGE] [HUMAN-DATA] [HUMAN-CHECK] | | |
