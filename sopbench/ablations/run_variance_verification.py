"""Sequential verification run: replay all 10 configs × 5 videos to measure variance.

Saves results to `results/captaincook4d/run2-{config}/` for direct comparison
with Run 1 (the parallel run).

Runs sequentially with 5s delays between calls — slower but no rate-limit issues.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from sopbench.run_experiment import (
    compute_max_fps, fmt_mmss, fmt_sub, get_duration,
    load_samples, run_one,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "captaincook4d"


def main():
    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    samples = load_samples()

    fps_configs = {
        "fps1": lambda dur: 1.0,
        "fps2": lambda dur: 2.0,
        "fps4": lambda dur: 4.0,
        "fps8": lambda dur: 8.0,
        "fpsmax": lambda dur: compute_max_fps(dur),
    }
    format_configs = {
        "mmss": (fmt_mmss, "MM:SS", "01:23"),
        "sub": (fmt_sub, "MM:SS.ss", "01:23.50"),
    }

    total_configs = len(fps_configs) * len(format_configs)
    total_runs = total_configs * len(samples)
    run_idx = 0

    all_aggregates = {}

    for fps_tag, fps_func in fps_configs.items():
        for fmt_tag, (fmt_func, ts_label, ts_example) in format_configs.items():
            out_dir = RESULTS / f"run2-{fps_tag}-{fmt_tag}"
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"CONFIG: {fps_tag} + {ts_label}  (output: {out_dir.name})")
            print(f"{'='*60}")

            config_results = []
            for video_path, rec_id, steps in samples:
                run_idx += 1
                dur = get_duration(video_path)
                fps_val = fps_func(dur)

                print(f"  [{run_idx}/{total_runs}] {rec_id} (fps={fps_val:.1f})...",
                      end=" ", flush=True)

                try:
                    aligned, metrics = run_one(
                        client, video_path, steps, fps_val,
                        fmt_func, ts_label, ts_example,
                        max_retries=3,
                    )
                    m = metrics
                    print(f"IoU={m['mean_iou']:.1%} "
                          f"R@1(.3)={m['recall_at_1_iou_0.3']:.1%} "
                          f"R@1(.5)={m['recall_at_1_iou_0.5']:.1%}")

                    result = {
                        "recording_id": rec_id,
                        "video": video_path.name,
                        "dataset": "captaincook4d",
                        "model": "gemini-2.5-flash",
                        "fps_tag": fps_tag,
                        "fps_actual": fps_val,
                        "ts_format": ts_label,
                        "format_tag": fmt_tag,
                        "run": "run2",
                        "ground_truth": steps,
                        "predictions": aligned,
                        "metrics": metrics,
                    }
                    with open(out_dir / f"{rec_id}.json", "w") as f:
                        json.dump(result, f, indent=2)
                    config_results.append(result)

                except Exception as e:
                    print(f"FAILED: {str(e)[:100]}")

                time.sleep(5)  # Polite between calls

            # Aggregate for this config
            if config_results:
                n = len(config_results)
                agg = {
                    k: sum(r["metrics"][k] for r in config_results) / n
                    for k in ["mean_iou", "recall_at_1_iou_0.3",
                              "recall_at_1_iou_0.5", "recall_at_1_iou_0.7"]
                }
                print(f"  AGGREGATE ({n}v): IoU={agg['mean_iou']:.1%} "
                      f"R@1(.3)={agg['recall_at_1_iou_0.3']:.1%} "
                      f"R@1(.5)={agg['recall_at_1_iou_0.5']:.1%}")
                all_aggregates[f"{fps_tag}-{fmt_tag}"] = agg

                with open(out_dir / "_summary.json", "w") as f:
                    json.dump({
                        "config": f"{fps_tag}-{fmt_tag}",
                        "run": "run2",
                        "num_videos": n,
                        "aggregate": agg,
                    }, f, indent=2)

    # Final comparison vs Run 1
    print(f"\n\n{'='*75}")
    print("RUN 2 vs RUN 1 COMPARISON")
    print(f"{'='*75}")
    print(f"{'Config':<18s} | {'Run1 IoU':>9s} | {'Run2 IoU':>9s} | {'Delta':>6s}")
    print("-" * 50)

    for config, agg2 in all_aggregates.items():
        run1_dir = RESULTS / f"clean-{config}"
        run1_iou = 0.0
        if run1_dir.exists():
            run1_results = []
            for f in run1_dir.glob("*.json"):
                if f.name.startswith("_"):
                    continue
                with open(f) as fh:
                    run1_results.append(json.load(fh))
            if run1_results:
                run1_iou = sum(r["metrics"]["mean_iou"] for r in run1_results) / len(run1_results)
        run2_iou = agg2["mean_iou"]
        delta = run2_iou - run1_iou
        print(f"{config:<18s} | {run1_iou:>8.1%} | {run2_iou:>8.1%} | {delta:>+5.1%}")


if __name__ == "__main__":
    main()
