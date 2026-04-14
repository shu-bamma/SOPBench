# sopbench/archive/ — superseded scripts

These scripts represent the **iterative path** of getting from the naive baseline (~12% IoU) to the final config (~44% IoU). They are kept here for transparency about what was tried, not for production use. Don't import from these — use the canonical scripts in the parent directory.

## What's here and why it was superseded

| Script | Era | Why archived |
|--------|-----|--------------|
| `gemini_baseline.py` | v1 | Naive baseline (default fps=1, decimal seconds). Out-of-bounds predictions. Fixed in v2. |
| `gemini_baseline_v2.py` | v2 | First good version (fps=4 + MM:SS + duration constraint). Now subsumed by `run_experiment.py`. |
| `gemini_baseline_frames.py` | Approach B | Two-pass frame-extraction + caption matching. Performed dramatically worse (~6% IoU vs 43%). |
| `run_eval_v2.py`, `run_eval_frames.py` | various | Evaluation runners for the above superseded experiments. |
| `run_fps_experiment.py` | mid | FPS sweep that compared uniform vs adaptive frame selection. Superseded by `run_experiment.py` for the FPS×Format study. Adaptive lost decisively. |
| `run_experiment_old.py` | mid | Earlier 3-way comparison script (native vs manual MM:SS vs manual MM:SS.ss). |
| `probe_video.py` | v1 | First synthetic probe video. Had a sampling-bias bug — only queried first half of each second. Fixed in `ablations/probe_balanced.py` (v3). |
| `run_pro_noaudio.py` | exploratory | First Pro experiment with audio-off. Superseded by `ablations/run_pro_top_configs.py` after the audio-off ablation showed audio helps. |
| `check_audio_tokens.py` | utility | One-off script to verify audio-stripped videos had fewer tokens. Result: yes, exactly 32 tokens/sec less. |

For the canonical scripts and what to actually run, see the parent `sopbench/README.md`.
