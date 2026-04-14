"""Gemini 2.5 Pro with MAX thinking on top 3 configs (AUDIO ON).

Based on Flash results, audio helps on 7/10 configs — so we use audio-ON videos.

Top 3 configs to test with Pro:
  fps=1 + MM:SS.ss   (best R1: 44.4%)
  fps=1 + MM:SS      (most stable: 42.7% / 42.9%)
  fps=2 + MM:SS      (peak R2: 53.6%, avg 45.6%)

Uses thinking_budget=-1 (dynamic — model decides how much to think).

Expected cost: ~$5-10 for 15 calls (3 configs x 5 videos).
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from sopbench.run_experiment import (
    fmt_mmss, fmt_sub, load_samples, run_one,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "captaincook4d"

THINKING_BUDGET = -1  # dynamic — Pro decides


def main():
    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    samples = load_samples()  # audio-ON videos

    configs = [
        ("fps1", 1.0, "sub", fmt_sub, "MM:SS.ss", "01:23.50"),
        ("fps1", 1.0, "mmss", fmt_mmss, "MM:SS", "01:23"),
        ("fps2", 2.0, "mmss", fmt_mmss, "MM:SS", "01:23"),
    ]

    model = "gemini-2.5-pro"
    total = len(configs) * len(samples)
    idx = 0

    all_aggregates = {}

    for fps_tag, fps_val, fmt_tag, fmt_func, ts_label, ts_example in configs:
        out_dir = RESULTS / f"pro-{fps_tag}-{fmt_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"CONFIG: {model} | {fps_tag} + {ts_label} | AUDIO ON")
        print(f"Output: {out_dir.name}")
        print(f"{'='*60}")

        config_results = []
        for video_path, rec_id, steps in samples:
            idx += 1
            print(f"  [{idx}/{total}] {rec_id} (fps={fps_val:.1f})...",
                  end=" ", flush=True)

            try:
                aligned, metrics = run_one(
                    client, video_path, steps, fps_val,
                    fmt_func, ts_label, ts_example,
                    model=model,
                    max_retries=3,
                    thinking_budget=THINKING_BUDGET,
                )
                m = metrics
                print(f"IoU={m['mean_iou']:.1%} "
                      f"R@1(.3)={m['recall_at_1_iou_0.3']:.1%} "
                      f"R@1(.5)={m['recall_at_1_iou_0.5']:.1%}")

                result = {
                    "recording_id": rec_id,
                    "video": video_path.name,
                    "dataset": "captaincook4d",
                    "model": model,
                    "fps_tag": fps_tag,
                    "fps_actual": fps_val,
                    "ts_format": ts_label,
                    "format_tag": fmt_tag,
                    "audio": "on",
                    "thinking_budget": THINKING_BUDGET,
                    "ground_truth": steps,
                    "predictions": aligned,
                    "metrics": metrics,
                }
                with open(out_dir / f"{rec_id}.json", "w") as f:
                    json.dump(result, f, indent=2)
                config_results.append(result)

            except Exception as e:
                print(f"FAILED: {str(e)[:100]}")

            time.sleep(5)

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
                    "config": f"pro-{fps_tag}-{fmt_tag}",
                    "model": model,
                    "audio": "on",
                    "num_videos": n,
                    "aggregate": agg,
                }, f, indent=2)

    # Final comparison
    print(f"\n\n{'='*80}")
    print("PRO vs FLASH (Audio ON) Mean IoU")
    print(f"{'='*80}")
    print(f"{'Config':<18s} | {'Flash R1':>9s} | {'Flash R2':>9s} | "
          f"{'Pro':>9s} | {'Pro - avg':>10s}")
    print("-" * 65)

    for config, agg in all_aggregates.items():
        r1_dir = RESULTS / f"clean-{config}"
        r2_dir = RESULTS / f"run2-{config}"
        r1_iou = r2_iou = 0.0
        if r1_dir.exists():
            vals = [json.load(open(f))["metrics"]["mean_iou"]
                    for f in r1_dir.glob("*.json")
                    if not f.name.startswith("_")]
            r1_iou = sum(vals) / len(vals) if vals else 0
        if r2_dir.exists():
            vals = [json.load(open(f))["metrics"]["mean_iou"]
                    for f in r2_dir.glob("*.json")
                    if not f.name.startswith("_")]
            r2_iou = sum(vals) / len(vals) if vals else 0
        pro_iou = agg["mean_iou"]
        flash_avg = (r1_iou + r2_iou) / 2
        delta = pro_iou - flash_avg
        print(f"{config:<18s} | {r1_iou:>8.1%} | {r2_iou:>8.1%} | "
              f"{pro_iou:>8.1%} | {delta:>+9.1%}")


if __name__ == "__main__":
    main()
