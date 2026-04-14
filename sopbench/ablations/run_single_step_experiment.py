"""Experiment 12: Single-Step Grounding (Variant A — pure single-step).

Each API call queries ONE step at a time, with no context about other steps.
This mirrors how standard temporal-grounding benchmarks (Charades-STA,
ActivityNet) work — one natural-language query per call.

Variants: fps={1, 2, 4} × format=MM:SS × 5 videos × N steps_per_video.

Totals:
    63 steps across 5 CC4D videos
    3 fps configs × 63 = 189 API calls
    ~$8.80 at Gemini 2.5 Flash pricing
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from sopbench.metrics import compute_all_metrics
from sopbench.run_experiment import (
    fmt_mmss, parse_ts, get_duration, load_samples,
)

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"


# ---- Helpers ----

def clean_step_description(desc: str) -> str:
    """Strip CC4D's duplicated verb-prefix.

    "Rinse-Rinse a tomato" -> "Rinse a tomato"
    "dry-gently dry it with a paper/tea towel" -> "gently dry it with a paper/tea towel"
    "Slice-Slice one tomato..." -> "Slice one tomato..."
    "Cut or tear-Cut or tear 1 slices..." -> "Cut or tear 1 slices..."  (multi-word verb)
    """
    # Allow the prefix to be multiple words (letters + spaces), ending at first dash
    m = re.match(r"^([A-Za-z][A-Za-z\s]*?)\s*-\s*(.+)$", desc)
    if m:
        return m.group(2).strip()
    return desc.strip()


def build_single_step_prompt(step_description: str, duration_str: str) -> str:
    """Prompt used for Variant A — no context about other steps.

    Format choice rationale:
    - MM:SS matches Gemini's internal timestamp tokens at fps=1 (Exp 8 finding).
    - Explicit duration bound prevents out-of-range predictions (Exp 2 finding).
    - "not_found" option lets the model decline rather than hallucinate.
    - JSON output with explicit schema minimizes parse errors.
    """
    return f"""You are analyzing a first-person egocentric cooking video. The video is exactly {duration_str} long (00:00 to {duration_str}).

Find the time window in the video when this specific action occurs:

"{step_description}"

Rules:
- Return start_time and end_time in MM:SS format (e.g., 01:23).
- Both timestamps must be between 00:00 and {duration_str}.
- If this action is not clearly visible in the video, return "not_found" for both.
- Watch the ENTIRE video before answering.

Return JSON: {{"start_time": "MM:SS", "end_time": "MM:SS", "confidence": 0.9}}"""


# ---- Core ----

def run_single_step(
    client: genai.Client,
    video_file: types.File,
    step_description: str,
    duration: float,
    fps_val: float,
    model: str = "gemini-2.5-flash",
    max_retries: int = 3,
) -> dict:
    """Query Gemini for ONE step's temporal boundaries.

    Returns {"start_time": float_seconds, "end_time": float_seconds,
             "raw_start": str, "raw_end": str, "confidence": float}
    """
    duration_str = fmt_mmss(duration)
    prompt = build_single_step_prompt(step_description, duration_str)

    for attempt in range(max_retries):
        try:
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
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            break
        except Exception as e:
            wait = 60 * (attempt + 1)
            print(f"      Retry {attempt+1}: {e} (waiting {wait}s)")
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                raise

    raw = json.loads(response.text)
    raw_start = str(raw.get("start_time", "not_found"))
    raw_end = str(raw.get("end_time", "not_found"))
    st = parse_ts(raw_start)
    et = parse_ts(raw_end)
    if st > duration * 1.1:
        st = -1.0
    if et > duration * 1.1:
        et = duration
    return {
        "start_time": float(st),
        "end_time": float(et),
        "raw_start": raw_start,
        "raw_end": raw_end,
        "confidence": float(raw.get("confidence", 0.0)),
    }


def run_video(
    client: genai.Client,
    video_path: Path,
    steps: list,
    fps_val: float,
    sample_delay: float = 3.0,
    model: str = "gemini-2.5-flash",
):
    """Upload video once, then query each step separately.

    Returns (aligned_predictions, metrics, duration).
    """
    duration = get_duration(video_path)

    # Upload once, reuse for all N steps
    video_file = client.files.upload(
        file=str(video_path),
        config=types.UploadFileConfig(mime_type="video/mp4"),
    )
    while video_file.state == "PROCESSING":
        time.sleep(3)
        video_file = client.files.get(name=video_file.name)

    predictions = []
    try:
        for i, step in enumerate(steps):
            clean_desc = clean_step_description(step["description"])
            print(
                f"      step {i+1}/{len(steps)} \"{clean_desc[:45]}\"...",
                end=" ", flush=True,
            )
            try:
                pred = run_single_step(
                    client, video_file, clean_desc, duration, fps_val, model,
                )
                pred["step_index"] = i + 1
                pred["description"] = clean_desc
                pred["original_description"] = step["description"]
                print(f"{pred['raw_start']}-{pred['raw_end']}")
            except Exception as e:
                print(f"FAILED: {str(e)[:80]}")
                pred = {
                    "step_index": i + 1,
                    "description": clean_desc,
                    "original_description": step["description"],
                    "start_time": -1.0, "end_time": -1.0,
                    "raw_start": "error", "raw_end": "error",
                    "confidence": 0.0,
                }
            predictions.append(pred)
            time.sleep(sample_delay)
    finally:
        try:
            client.files.delete(name=video_file.name)
        except Exception:
            pass

    metrics = compute_all_metrics(predictions, steps)
    return predictions, metrics, duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=str, default="1",
                        help="FPS: 1, 2, 4, or 'all' (runs 1,2,4 sequentially)")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--sample-delay", type=float, default=3.0)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    samples = load_samples()

    fps_list = [1.0, 2.0, 4.0] if args.fps == "all" else [float(args.fps)]

    for fps_val in fps_list:
        tag = f"singlestep-fps{int(fps_val)}"
        out_dir = RESULTS / "captaincook4d" / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"EXP 12: Single-step grounding | fps={fps_val} | {args.model}")
        print(f"Output: {out_dir}")
        print(f"{'='*70}")

        all_results = []
        for video_path, rec_id, steps in samples:
            print(f"\n  Video {rec_id} ({len(steps)} steps, {get_duration(video_path):.0f}s):")
            predictions, metrics, duration = run_video(
                client, video_path, steps, fps_val,
                sample_delay=args.sample_delay,
                model=args.model,
            )
            m = metrics
            print(f"    => IoU={m['mean_iou']:.1%} R@1(.3)={m['recall_at_1_iou_0.3']:.1%} "
                  f"R@1(.5)={m['recall_at_1_iou_0.5']:.1%} Det={m['step_detection_rate']:.1%}")

            result = {
                "recording_id": rec_id,
                "video": video_path.name,
                "dataset": "captaincook4d",
                "model": args.model,
                "fps_tag": f"fps{int(fps_val)}",
                "fps_actual": fps_val,
                "duration_seconds": duration,
                "ts_format": "MM:SS",
                "variant": "A_pure_single_step",
                "ground_truth": steps,
                "predictions": predictions,
                "metrics": metrics,
            }
            with open(out_dir / f"{rec_id}.json", "w") as f:
                json.dump(result, f, indent=2)
            all_results.append(result)

        # Aggregate
        if all_results:
            n = len(all_results)
            agg = {
                k: sum(r["metrics"][k] for r in all_results) / n
                for k in ["mean_iou", "recall_at_1_iou_0.3",
                          "recall_at_1_iou_0.5", "recall_at_1_iou_0.7",
                          "step_detection_rate", "ordering_compliance"]
            }
            print(f"\n  AGGREGATE ({n} videos, fps={fps_val}):")
            for k, v in agg.items():
                print(f"    {k:<25s}: {v:.1%}")

            with open(out_dir / "_summary.json", "w") as f:
                json.dump({
                    "config": tag,
                    "variant": "A_pure_single_step",
                    "fps": fps_val,
                    "model": args.model,
                    "num_videos": n,
                    "aggregate": agg,
                }, f, indent=2)


if __name__ == "__main__":
    main()
