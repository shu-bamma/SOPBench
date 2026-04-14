"""Gemini 2.5 Pro with MAX thinking on top configs with audio-off.

Tests whether Pro's deeper reasoning improves step boundary detection
on our best-performing configs from Flash experiments.

Top configs (averaged across 2 Flash runs):
  fps2-mmss:  45.6% (best)
  fps1-mmss:  42.8%
  fps1-sub:   41.7%
  fps2-sub:   40.0%

Also tests fps4-mmss as a control (was 36.7% avg on Flash).

Uses thinking_budget=32768 (max) for maximum reasoning depth.
Thinking tokens are billed as output tokens ($10/1M for Pro).

Expected cost: ~$15-20 for 20 calls (4 configs x 5 videos) with max thinking.
"""

THINKING_BUDGET = 32768  # max for gemini-2.5-pro

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from sopbench.run_experiment import (
    fmt_mmss, fmt_sub, get_duration, run_one,
)
from sopbench.run_noaudio_experiment import load_noaudio_samples

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "captaincook4d"


def main():
    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    samples = load_noaudio_samples()

    # Top 4 configs: fps={1,2} x format={MM:SS, MM:SS.ss}
    configs = [
        ("fps1", 1.0, "mmss", fmt_mmss, "MM:SS", "01:23"),
        ("fps1", 1.0, "sub", fmt_sub, "MM:SS.ss", "01:23.50"),
        ("fps2", 2.0, "mmss", fmt_mmss, "MM:SS", "01:23"),
        ("fps2", 2.0, "sub", fmt_sub, "MM:SS.ss", "01:23.50"),
    ]

    model = "gemini-2.5-pro"
    total = len(configs) * len(samples)
    idx = 0

    all_aggregates = {}

    for fps_tag, fps_val, fmt_tag, fmt_func, ts_label, ts_example in configs:
        out_dir = RESULTS / f"pro-noaudio-{fps_tag}-{fmt_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"CONFIG: {model} | {fps_tag} + {ts_label} | AUDIO OFF")
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
                    "audio": "off",
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
                    "config": f"pro-noaudio-{fps_tag}-{fmt_tag}",
                    "model": model,
                    "audio": "off",
                    "num_videos": n,
                    "aggregate": agg,
                }, f, indent=2)

    # Final Pro vs Flash comparison
    print(f"\n\n{'='*75}")
    print("PRO (audio-off) vs FLASH (audio-on Run 1) COMPARISON")
    print(f"{'='*75}")
    print(f"{'Config':<18s} | {'Flash IoU':>9s} | {'Pro IoU':>9s} | {'Delta':>6s}")
    print("-" * 50)

    for config, agg in all_aggregates.items():
        flash_dir = RESULTS / f"clean-{config}"
        flash_iou = 0.0
        if flash_dir.exists():
            flash_results = []
            for f in flash_dir.glob("*.json"):
                if f.name.startswith("_"):
                    continue
                with open(f) as fh:
                    flash_results.append(json.load(fh))
            if flash_results:
                flash_iou = sum(r["metrics"]["mean_iou"] for r in flash_results) / len(flash_results)
        pro_iou = agg["mean_iou"]
        delta = pro_iou - flash_iou
        print(f"{config:<18s} | {flash_iou:>8.1%} | {pro_iou:>8.1%} | {delta:>+5.1%}")


if __name__ == "__main__":
    main()
