# sopbench/ — SOPBench framework code

The Python package that runs all Gemini step-boundary detection experiments.

## Top-level scripts (core)

| Script | Purpose |
|--------|---------|
| `run_experiment.py` | **Main entry point.** Runs the FPS × Format sweep (10 configs × 5 videos) for the headline result. Use `--fps` and `--format` to run a single config. |
| `run_eval.py` | Loads existing result JSONs and computes/prints metrics. |
| `metrics.py` | Temporal IoU, R@k, ordering compliance, step detection rate. |
| `visualizer.py` | Local HTML viewer at `http://localhost:8080` showing video + GT segments + Gemini predictions side by side. |
| `frame_extractor.py` | OpenCV-based frame extraction utility (used by ablations that need manual frames). |

## `ablations/` — focused ablation experiments

| Script | What it tests |
|--------|---------------|
| `run_subsec_experiment.py` | MM:SS vs MM:SS.ss output format at different FPS values. |
| `run_variance_verification.py` | Re-runs the full sweep sequentially to measure run-to-run variance. |
| `run_noaudio_experiment.py` | Same sweep but on audio-stripped videos (does audio help?). |
| `run_pro_top_configs.py` | Runs `gemini-2.5-pro` on the best Flash configs to test if reasoning helps. |
| `probe_balanced.py` | Synthetic probe video test — verifies how Gemini actually samples frames at different FPS, with both halves of each second queried (no sampling bias). |

## `archive/` — superseded scripts

Earlier versions of experiments, kept for posterity / context. See `archive/README.md`.

## Usage

```bash
# Set API key (only needed once)
cp ../.env.example ../.env  # then edit

# Smoke test on one config
python -m sopbench.run_experiment --fps 1 --format sub

# Full sweep (50 calls, ~$3)
python -m sopbench.run_experiment

# Specific ablation
python -m sopbench.ablations.run_noaudio_experiment

# View results
python -m sopbench.visualizer
```
