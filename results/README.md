# results/ — all experiment outputs

Each subdirectory under `results/captaincook4d/` corresponds to one experiment configuration. Each contains 5 JSON files (one per video) plus a `_summary.json` aggregate.

## Result file structure

```json
{
  "recording_id": "12_51",
  "video": "12_51_tomatomozzarellasalad.mp4",
  "model": "gemini-2.5-flash",
  "fps_tag": "fps1",
  "fps_actual": 1.0,
  "ts_format": "MM:SS.ss",
  "ground_truth": [{"start_time": 17.1, "end_time": 23.4, "description": "Rinse-Rinse a tomato", ...}, ...],
  "predictions": [{"start_time": 18.0, "end_time": 23.0, "raw_start": "00:18", "raw_end": "00:23", ...}, ...],
  "metrics": {"mean_iou": 0.66, "recall_at_1_iou_0.3": 1.0, "recall_at_1_iou_0.5": 0.78, ...}
}
```

## Experiment-to-directory mapping

The `docs/exp_log.md` file describes each experiment in detail. Here is which directories belong to which.

### Main FPS × Format sweep (Experiments 8 + 9)

10 configs × 5 videos = 50 results per run.

| Run | Directory pattern | Description |
|-----|------------------|-------------|
| Run 1 | `clean-fps{1,2,4,8,max}-{mmss,sub}/` | First parallel run (10 dirs) |
| Run 2 | `run2-fps{1,2,4,8,max}-{mmss,sub}/` | Sequential variance verification (10 dirs) |

### Audio-OFF ablation (Experiment 10)

| Directory pattern | Description |
|-------------------|-------------|
| `noaudio-fps{1,2,4,8,max}-{mmss,sub}/` | Same 10 configs on audio-stripped videos (10 dirs) |

### Pro vs Flash (Experiment 11)

| Directory | Description |
|-----------|-------------|
| `pro-fps1-mmss/` | Gemini 2.5 Pro at fps=1, MM:SS |
| `pro-fps1-sub/` | Gemini 2.5 Pro at fps=1, MM:SS.ss |
| `pro-fps2-mmss/` | Gemini 2.5 Pro at fps=2, MM:SS |

### Earlier experiments (kept for posterity)

| Directory | Experiment in `exp_log.md` |
|-----------|---------------------------|
| `gemini-2.5-flash/` | Experiment 1 (v1 baseline) |
| `gemini-2.5-flash-v2/` | Experiment 2 (v2 baseline) |
| `gemini-2.5-flash-frames/` | Experiment 3 (frame extraction two-pass) |
| `gemini-2.5-flash-manual-{intsec,subsec}/` | Experiment 4 (manual frames vs native) |
| `gemini-2.5-flash-{adaptive,uniform}-fps*/` | Experiment 5 (uniform vs adaptive frame selection) |
| `gemini-2.5-flash-subsec-fps*/` | Experiment 7 (sub-second output format CC4D) |
| `gemini-2.5-flash-native/`, `gemini-2.5-flash-manual-{subsec,intsec}/` | Earlier 3-way comparison |

## Total size

~3 MB across 55 result directories (~245 individual JSON files). All small, plain JSON — committed to git directly (no LFS needed).

## Re-computing metrics from raw predictions

If you want to recompute metrics from the raw predictions (e.g., with a different IoU threshold):

```python
import json
from sopbench.metrics import compute_all_metrics

with open("results/captaincook4d/clean-fps1-sub/12_51.json") as f:
    r = json.load(f)
metrics = compute_all_metrics(r["predictions"], r["ground_truth"])
print(metrics)
```
