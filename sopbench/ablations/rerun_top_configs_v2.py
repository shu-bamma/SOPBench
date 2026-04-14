"""Re-run top 4 configs (fps=1/2 x MM:SS/MM:SS.ss) with correct cv2 durations.

The original `clean-fps*` runs were done with an older cv2 that misread GoPro
frame rates as 29.178 fps instead of 29.97 fps, inflating prompt durations by
~2.68%. Saves to `clean-fps{1,2}-{mmss,sub}-v2/` so we can compare directly.
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from sopbench.run_experiment import (
    fmt_mmss, fmt_sub, get_duration, load_samples, run_one,
)

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results" / "captaincook4d"


def run_config_to_tag(client, samples, fps_val, fmt_func, ts_label, ts_example, tag):
    out_dir = RESULTS / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\nCONFIG: {tag} (fps={fps_val}, {ts_label})\n{'='*60}")

    results = []
    for video_path, rec_id, steps in samples:
        dur = get_duration(video_path)
        print(f"  {rec_id} ({dur:.1f}s, fps={fps_val})...", end=" ", flush=True)
        try:
            aligned, metrics = run_one(
                client, video_path, steps, fps_val,
                fmt_func, ts_label, ts_example, max_retries=3,
            )
            m = metrics
            print(f"IoU={m['mean_iou']:.1%} R@1(.3)={m['recall_at_1_iou_0.3']:.1%} "
                  f"R@1(.5)={m['recall_at_1_iou_0.5']:.1%}")
            result = {
                "recording_id": rec_id, "video": video_path.name,
                "dataset": "captaincook4d", "model": "gemini-2.5-flash",
                "fps_tag": tag.split("-")[1], "fps_actual": fps_val,
                "ts_format": ts_label, "duration_seconds": dur,
                "ground_truth": steps, "predictions": aligned, "metrics": metrics,
            }
            with open(out_dir / f"{rec_id}.json", "w") as f:
                json.dump(result, f, indent=2)
            results.append(result)
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(3)

    if results:
        n = len(results)
        agg = {
            k: sum(r["metrics"][k] for r in results) / n
            for k in ["mean_iou", "recall_at_1_iou_0.3", "recall_at_1_iou_0.5",
                      "recall_at_1_iou_0.7", "step_detection_rate"]
        }
        print(f"  AGGREGATE: IoU={agg['mean_iou']:.1%} R@1(.3)={agg['recall_at_1_iou_0.3']:.1%} "
              f"R@1(.5)={agg['recall_at_1_iou_0.5']:.1%}")
        with open(out_dir / "_summary.json", "w") as f:
            json.dump({"config": tag, "aggregate": agg, "num_videos": n}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=str, required=True, choices=["1", "2"])
    parser.add_argument("--format", type=str, required=True, choices=["mmss", "sub"])
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    samples = load_samples()

    fps_val = float(args.fps)
    if args.format == "mmss":
        fmt_func, ts_label, ts_example = fmt_mmss, "MM:SS", "01:23"
    else:
        fmt_func, ts_label, ts_example = fmt_sub, "MM:SS.ss", "01:23.50"

    tag = f"clean-fps{args.fps}-{args.format}-v2"
    run_config_to_tag(client, samples, fps_val, fmt_func, ts_label, ts_example, tag)


if __name__ == "__main__":
    main()
