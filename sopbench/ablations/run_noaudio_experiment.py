"""Audio-OFF variant: run all 10 configs on audio-stripped videos.

Uses videos from `videos/captaincook4d_samples_noaudio/` (pre-stripped with ffmpeg).
Saves results to `results/captaincook4d/noaudio-{fps_tag}-{format_tag}/`.
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from sopbench.run_experiment import (
    compute_max_fps, fmt_mmss, fmt_sub, get_duration, run_one,
)

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "videos"
RESULTS = ROOT / "results"


def load_noaudio_samples():
    """Load samples from the audio-stripped videos directory."""
    ann_path = VIDEOS / "captaincook4d_samples_noaudio" / "annotations.json"
    with open(ann_path) as f:
        annotations = json.load(f)
    samples = []
    for rec_id, ann in annotations.items():
        matches = list(
            (VIDEOS / "captaincook4d_samples_noaudio").glob(f"{rec_id}_*.mp4")
        )
        if matches:
            steps = [s for s in ann["steps"] if s.get("start_time", -1) >= 0]
            samples.append((matches[0], rec_id, steps))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=str, default="all",
                        help="FPS: 1, 2, 4, 8, max, or 'all'")
    parser.add_argument("--format", type=str, default="all",
                        choices=["mmss", "sub", "all"])
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    samples = load_noaudio_samples()

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

    if args.fps != "all":
        key = f"fps{args.fps}"
        fps_configs = {key: fps_configs[key]}
    if args.format != "all":
        format_configs = {args.format: format_configs[args.format]}

    total = len(fps_configs) * len(format_configs) * len(samples)
    idx = 0

    for fps_tag, fps_func in fps_configs.items():
        for fmt_tag, (fmt_func, ts_label, ts_example) in format_configs.items():
            out_dir = RESULTS / "captaincook4d" / f"noaudio-{fps_tag}-{fmt_tag}"
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"CONFIG: {fps_tag} + {ts_label} (NO AUDIO)")
            print(f"Output: {out_dir.name}")
            print(f"{'='*60}")

            config_results = []
            for video_path, rec_id, steps in samples:
                idx += 1
                dur = get_duration(video_path)
                fps_val = fps_func(dur)

                print(f"  [{idx}/{total}] {rec_id} (fps={fps_val:.1f})...",
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
                        "audio": "off",
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
                with open(out_dir / "_summary.json", "w") as f:
                    json.dump({
                        "config": f"noaudio-{fps_tag}-{fmt_tag}",
                        "audio": "off",
                        "num_videos": n,
                        "aggregate": agg,
                    }, f, indent=2)


if __name__ == "__main__":
    main()
