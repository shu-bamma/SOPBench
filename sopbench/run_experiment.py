"""Clean FPS × Output Format experiment — main SOPBench entry point.

10 configs: fps={1,2,4,8,max} × format={MM:SS, MM:SS.ss}
All native video upload. Same prompt structure. Fair evaluation.

Usage:
    python -m sopbench.run_experiment                       # all 10 configs
    python -m sopbench.run_experiment --fps 2 --format mmss # single config
    python -m sopbench.run_experiment --fps 2 --format sub  # single config
"""

import argparse
import cv2
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from sopbench.metrics import compute_all_metrics

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "videos"
RESULTS = ROOT / "results"

CONTEXT_LIMIT = 1_000_000
PROMPT_RESERVE = 5_000


# ---- Formatters ----

def fmt_mmss(ts):
    m = int(ts // 60)
    s = int(ts % 60)
    return f"{m:02d}:{s:02d}"


def fmt_sub(ts):
    m = int(ts // 60)
    s = ts % 60
    return f"{m:02d}:{s:05.2f}"


def parse_ts(ts_str):
    ts_str = str(ts_str).strip().strip('"')
    if ts_str.lower() in ("not_found", "n/a", "-1", ""):
        return -1.0
    parts = ts_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(ts_str)


# ---- Helpers ----

def get_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total / fps if fps > 0 else 0


def compute_max_fps(duration_secs):
    # Per Gemini docs: 258 tokens/frame (default res) + 32 audio tokens/sec
    # + ~7 timestamp tokens/sec (negligible, lumped into frame cost conservatively)
    available = CONTEXT_LIMIT - PROMPT_RESERVE
    max_fps = (available / duration_secs - 32) / 258
    return min(max_fps, 24.0)


def load_samples():
    ann_path = VIDEOS / "captaincook4d_samples" / "annotations.json"
    with open(ann_path) as f:
        annotations = json.load(f)
    samples = []
    for rec_id, ann in annotations.items():
        matches = list((VIDEOS / "captaincook4d_samples").glob(f"{rec_id}_*.mp4"))
        if matches:
            steps = [s for s in ann["steps"] if s.get("start_time", -1) >= 0]
            samples.append((matches[0], rec_id, steps))
    return samples


# ---- Core ----

def run_one(client, video_path, steps, fps_val, fmt_func, ts_format_label,
            ts_example, model="gemini-2.5-flash", max_retries=3,
            thinking_budget=None):
    """Upload video, prompt at given fps and format, return parsed predictions."""

    dur = get_duration(video_path)
    dur_str = fmt_func(dur)
    zero_str = fmt_func(0)
    step_list = "\n".join(f'{i+1}. "{s["description"]}"' for i, s in enumerate(steps))

    prompt = f"""You are analyzing a first-person egocentric cooking video. The video is exactly {dur_str} long ({zero_str} to {dur_str}).

For EACH step below, identify the start and end timestamps in {ts_format_label} format by carefully watching the video.

Rules:
- Use {ts_format_label} format (e.g., {ts_example}).
- All timestamps must be between {zero_str} and {dur_str}.
- If a step is not visible in the video, set both to "not_found".
- Watch the ENTIRE video before answering.

Steps:
{step_list}

Return JSON: {{"predictions": [{{"step_index": 1, "start_time": "{ts_example}", "end_time": "{ts_example}", "confidence": 0.9}}, ...]}}"""

    # Upload with retry
    video_file = None
    for attempt in range(max_retries):
        try:
            video_file = client.files.upload(
                file=str(video_path),
                config=types.UploadFileConfig(mime_type="video/mp4"),
            )
            while video_file.state == "PROCESSING":
                time.sleep(3)
                video_file = client.files.get(name=video_file.name)
            break
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"      Upload retry {attempt+1}: {e} (waiting {wait}s)")
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                raise

    # Generate with retry
    try:
        response = None
        for attempt in range(max_retries):
            try:
                gen_config_kwargs = {
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                }
                if thinking_budget is not None:
                    gen_config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=thinking_budget
                    )
                response = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_text(text=prompt),
                        types.Part(
                            file_data=types.FileData(
                                file_uri=video_file.uri, mime_type="video/mp4"
                            ),
                            video_metadata=types.VideoMetadata(fps=float(fps_val)),
                        ),
                    ],
                    config=types.GenerateContentConfig(**gen_config_kwargs),
                )
                break
            except Exception as e:
                wait = 60 * (attempt + 1)  # 60s, 120s, 180s backoff
                print(f"      Generate retry {attempt+1}: {e} (waiting {wait}s)")
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    raise
    finally:
        try:
            client.files.delete(name=video_file.name)
        except Exception:
            pass

    raw = json.loads(response.text)
    preds_raw = raw.get("predictions", raw)

    # Parse timestamps to float seconds
    aligned = []
    for i in range(len(steps)):
        if i < len(preds_raw):
            p = preds_raw[i]
            raw_start = str(p.get("start_time", "-1"))
            raw_end = str(p.get("end_time", "-1"))
            st = parse_ts(raw_start)
            et = parse_ts(raw_end)
            # Clamp out-of-bounds
            if st > dur * 1.1:
                st = -1.0
            if et > dur * 1.1:
                et = dur
        else:
            st, et = -1.0, -1.0
            raw_start, raw_end = "not_found", "not_found"
        aligned.append({
            "start_time": float(st),
            "end_time": float(et),
            "raw_start": raw_start,
            "raw_end": raw_end,
        })

    metrics = compute_all_metrics(aligned, steps)
    return aligned, metrics


def run_config(client, samples, fps_tag, fps_val_func, fmt_func, ts_format_label,
               ts_example, format_tag):
    """Run one config across all videos."""
    tag = f"clean-{fps_tag}-{format_tag}"
    out_dir = RESULTS / "captaincook4d" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CONFIG: {fps_tag} + {ts_format_label}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    results = []
    for video_path, rec_id, steps in samples:
        dur = get_duration(video_path)
        fps_val = fps_val_func(dur)
        print(f"  {rec_id} ({dur:.0f}s, fps={fps_val:.1f})...", end=" ", flush=True)

        try:
            aligned, metrics = run_one(
                client, video_path, steps, fps_val,
                fmt_func, ts_format_label, ts_example,
            )
            m = metrics
            print(f"IoU={m['mean_iou']:.1%} R@1(.3)={m['recall_at_1_iou_0.3']:.1%} "
                  f"R@1(.5)={m['recall_at_1_iou_0.5']:.1%}")

            result = {
                "recording_id": rec_id,
                "video": video_path.name,
                "dataset": "captaincook4d",
                "model": "gemini-2.5-flash",
                "fps_tag": fps_tag,
                "fps_actual": fps_val,
                "ts_format": ts_format_label,
                "format_tag": format_tag,
                "ground_truth": steps,
                "predictions": aligned,
                "metrics": metrics,
            }
            with open(out_dir / f"{rec_id}.json", "w") as f:
                json.dump(result, f, indent=2)
            results.append(result)

        except Exception as e:
            print(f"FAILED: {e}")

        time.sleep(3)

    # Summary
    if results:
        n = len(results)
        agg = {
            k: sum(r["metrics"][k] for r in results) / n
            for k in ["mean_iou", "recall_at_1_iou_0.3", "recall_at_1_iou_0.5",
                       "recall_at_1_iou_0.7", "step_detection_rate", "ordering_compliance"]
        }
        print(f"  AGGREGATE ({n}v): IoU={agg['mean_iou']:.1%} "
              f"R@1(.3)={agg['recall_at_1_iou_0.3']:.1%} "
              f"R@1(.5)={agg['recall_at_1_iou_0.5']:.1%}")

        with open(out_dir / "_summary.json", "w") as f:
            json.dump({
                "config": f"{fps_tag}-{format_tag}",
                "fps_tag": fps_tag,
                "format_tag": format_tag,
                "ts_format": ts_format_label,
                "num_videos": n,
                "aggregate": agg,
                "per_video": [
                    {"recording_id": r["recording_id"], "metrics": r["metrics"]}
                    for r in results
                ],
            }, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Clean FPS × Format experiment")
    parser.add_argument("--fps", type=str, default="all",
                        help="FPS to test: 1, 2, 4, 8, max, or 'all'")
    parser.add_argument("--format", type=str, default="all",
                        choices=["mmss", "sub", "all"],
                        help="Output format: mmss, sub, or all")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    samples = load_samples()

    # Define all configs
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

    # Filter based on args
    if args.fps != "all":
        key = f"fps{args.fps}"
        fps_configs = {key: fps_configs[key]}
    if args.format != "all":
        format_configs = {args.format: format_configs[args.format]}

    all_summaries = {}
    for fps_tag, fps_func in fps_configs.items():
        for fmt_tag, (fmt_func, ts_label, ts_example) in format_configs.items():
            results = run_config(
                client, samples, fps_tag, fps_func,
                fmt_func, ts_label, ts_example, fmt_tag,
            )
            if results:
                n = len(results)
                all_summaries[f"{fps_tag}-{fmt_tag}"] = {
                    "mean_iou": sum(r["metrics"]["mean_iou"] for r in results) / n,
                    "r1_03": sum(r["metrics"]["recall_at_1_iou_0.3"] for r in results) / n,
                    "r1_05": sum(r["metrics"]["recall_at_1_iou_0.5"] for r in results) / n,
                    "n": n,
                }

    # Final comparison table
    if len(all_summaries) > 1:
        print(f"\n{'='*70}")
        print("FINAL COMPARISON TABLE")
        print(f"{'='*70}")
        print(f"{'Config':<20s} | {'Mean IoU':>8s} | {'R@1(.3)':>8s} | {'R@1(.5)':>8s} | {'N':>3s}")
        print("-" * 55)
        for config, s in sorted(all_summaries.items()):
            print(f"{config:<20s} | {s['mean_iou']:>7.1%} | {s['r1_03']:>7.1%} | "
                  f"{s['r1_05']:>7.1%} | {s['n']:>3d}")


if __name__ == "__main__":
    main()
