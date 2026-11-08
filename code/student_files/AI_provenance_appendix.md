# AI provenance appendix

## Kiro CLI — Stages 1 and 2

Kiro CLI helped draft and revise the Stage 1 and Stage 2 implementation discussion and the answers to Questions 1 and 2. The numerical claims were checked against the saved notebook outputs and `outputs/stress_tables.md`.

## OpenAI Codex — Stages 3 and 4

Codex suggested code for the Stage 3 compositing functions and helped design the threshold, hysteresis, and opening-radius comparisons. The implementation was checked using the compositing unit tests, and the experimental results were checked against Notebook 3 and the generated stress table.

Codex helped write the Notebook 4 scorecard, monitor plots, representative and failure-frame panels, and matched-frame comparison table. The scorecard and comparison values were checked against the saved tracker statistics and evaluator JSON reports.

Codex helped revise Questions 3–5 and the Stage 3 and Stage 4 discussion. The Q3 revisions used the saved monitor flags and supplementary checks on `coke_blur` frames 64–65 and common-clip frames 368–384. The Stage 3 and Stage 4 descriptions were checked against the notebook outputs, figures, and evaluation reports.

## Wan2.1-VACE-1.3B — Generative comparison

**Exact prompt**

> Replace the advertisement on the board in this video with the attached artwork.

The model received `my_clip` source frames **208–359**, resampled to **81 frames at 16 fps**, and `my_artwork.png` as a reference image. No input mask was supplied. The run used the official **Wan-AI/Wan2.1-VACE-1.3B** checkpoint, seed **42**, **50 UniPC steps**, shift **16**, guidance **5.0**, and bfloat16 on one **NVIDIA H100 NVL**. One earlier invocation stopped before generation and produced no video. The successful retry used the same inputs and settings and took **288.651 s**.

The raw video is `outputs/genai/output.mp4`. The five audit extracts in `outputs/genai/` retain source frame indices **210, 285, 315, 316, and 356**. FFprobe confirmed **81 frames at 832 × 480 and 16 fps**. Notebook 4 scored the extracts and both classical variants on the same source indices. The comparison is saved in `outputs/evaluation/stage4_genai_reports.json`.

## Human capture and annotations

Shen Jingtao captured `my_clip` and manually annotated **10 corner frames** and **5 masks**, including **3 occluded frames**, before running the methods. The annotations are recorded in `manifests/audit_corners.csv` and `manifests/audit_masks.csv`.