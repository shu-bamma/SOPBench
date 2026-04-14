"""FPS comparison experiment: uniform vs adaptive frame selection.

Set 1 (uniform): Native video upload at 1/2/4/max fps
Set 2 (adaptive): Extract most-informative frames, send as images

Usage:
    python -m sopbench.run_fps_experiment --mode uniform --fps 1
    python -m sopbench.run_fps_experiment --mode uniform --fps 2
    python -m sopbench.run_fps_experiment --mode uniform --fps 4
    python -m sopbench.run_fps_experiment --mode uniform --fps max
    python -m sopbench.run_fps_experiment --mode adaptive --fps 1
    python -m sopbench.run_fps_experiment --mode adaptive --fps 4
    python -m sopbench.run_fps_experiment --mode adaptive --fps max
"""

import argparse
import cv2
import io
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

from sopbench.metrics import compute_all_metrics

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "videos"
RESULTS = ROOT / "results"

CONTEXT_LIMIT = 1_000_000
PROMPT_RESERVE = 5_000


def fmt_mmss(ts):
    m = int(ts // 60)
    s = int(ts % 60)
    return f"{m:02d}:{s:02d}"

def parse_ts(ts_str):
    ts_str = str(ts_str).strip().strip('"')
    if ts_str in ("not_found", "n/a", "-1", ""):
        return -1.0
    parts = ts_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(ts_str)


def compute_max_fps(duration_secs):
    """Max FPS that fits in context window."""
    available = CONTEXT_LIMIT - PROMPT_RESERVE
    # tokens_per_sec = fps * 265 + 32
    max_fps = (available / duration_secs - 32) / 265
    return min(max_fps, 24.0)  # API cap


def get_video_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total / fps if fps > 0 else 0


def select_adaptive_frames(video_path, n_frames):
    """Select the N most informative frames based on inter-frame difference.

    Strategy: compute pixel-level change for every frame, then greedily pick
    frames that are most different from their neighbors — these are the
    transition/action frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read all frames as small grayscale for fast diff computation
    small_frames = []
    full_indices = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
        small_frames.append(gray.astype(np.float32))
        full_indices.append(idx)
        idx += 1

    # Compute inter-frame differences
    diffs = np.zeros(len(small_frames))
    for i in range(1, len(small_frames)):
        diffs[i] = np.mean(np.abs(small_frames[i] - small_frames[i - 1]))

    # Always include first and last frame
    # Then pick frames with highest inter-frame difference
    selected_indices = set([0, len(small_frames) - 1])

    # Sort by diff, pick top ones
    sorted_by_diff = np.argsort(-diffs)
    for idx in sorted_by_diff:
        if len(selected_indices) >= n_frames:
            break
        # Ensure minimum spacing (at least 2 frames apart from any selected)
        too_close = any(abs(idx - s) < 2 for s in selected_indices)
        if not too_close:
            selected_indices.add(idx)

    # If still need more, fill uniformly
    if len(selected_indices) < n_frames:
        uniform = np.linspace(0, len(small_frames) - 1, n_frames, dtype=int)
        for u in uniform:
            selected_indices.add(int(u))
            if len(selected_indices) >= n_frames:
                break

    selected = sorted(selected_indices)[:n_frames]
    cap.release()

    # Now re-read selected frames at full resolution
    cap = cv2.VideoCapture(str(video_path))
    results = []
    frame_idx = 0
    selected_set = set(selected)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in selected_set:
            ts = frame_idx / orig_fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            w, h = img.size
            scale = 256 / max(w, h)
            if scale < 1:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            results.append((ts, buf.getvalue()))
        frame_idx += 1
    cap.release()

    results.sort(key=lambda x: x[0])
    return results


def build_prompt(steps, dur_str, ts_format="MM:SS"):
    step_list = "\n".join(f'{i+1}. "{s["description"]}"' for i, s in enumerate(steps))
    return f"""You are analyzing a first-person egocentric cooking video. The video is exactly {dur_str} long (00:00 to {dur_str}).

For EACH step, identify start and end timestamps in {ts_format} format.
Rules:
- {ts_format} format. All between 00:00 and {dur_str}.
- If not visible, use "not_found".
- Watch the ENTIRE video before answering.

Steps:
{step_list}

Return JSON: {{"predictions": [{{"step_index": 1, "start_time": "{ts_format}", "end_time": "{ts_format}", "confidence": 0.9}}, ...]}}"""


def run_uniform(client, video_path, steps, fps_val, model="gemini-2.5-flash"):
    """Native video upload at given FPS."""
    video_file = client.files.upload(
        file=str(video_path),
        config=types.UploadFileConfig(mime_type="video/mp4"),
    )
    while video_file.state == "PROCESSING":
        time.sleep(3)
        video_file = client.files.get(name=video_file.name)

    dur = get_video_duration(video_path)
    dur_str = fmt_mmss(dur)
    prompt = build_prompt(steps, dur_str)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part(
                file_data=types.FileData(file_uri=video_file.uri, mime_type="video/mp4"),
                video_metadata=types.VideoMetadata(fps=float(fps_val)),
            ),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    client.files.delete(name=video_file.name)
    return json.loads(response.text), dur, fps_val


def run_adaptive(client, video_path, steps, fps_val, model="gemini-2.5-flash"):
    """Send adaptively selected frames as images."""
    dur = get_video_duration(video_path)
    n_frames = int(dur * fps_val)
    # Cap at 1200 to avoid API payload limits
    n_frames = min(n_frames, 1200)

    print(f"    Extracting {n_frames} adaptive frames...")
    frames = select_adaptive_frames(video_path, n_frames)
    print(f"    Got {len(frames)} frames")

    dur_str = fmt_mmss(dur)
    prompt = build_prompt(steps, dur_str)

    parts = [types.Part.from_text(text=prompt)]
    for ts, jpeg in frames:
        parts.append(types.Part.from_text(text=f"[{fmt_mmss(ts)}]"))
        parts.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))

    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    return json.loads(response.text), dur, fps_val


def evaluate(raw_response, steps, duration):
    preds_raw = raw_response.get("predictions", raw_response)
    if isinstance(preds_raw, dict):
        preds_raw = preds_raw.get("predictions", [])

    aligned = []
    for i in range(len(steps)):
        if i < len(preds_raw):
            p = preds_raw[i]
            st = parse_ts(p.get("start_time", "-1"))
            et = parse_ts(p.get("end_time", "-1"))
            if st > duration * 1.1:
                st = -1.0
            if et > duration * 1.1:
                et = duration
        else:
            st, et = -1.0, -1.0
        aligned.append({"start_time": float(st), "end_time": float(et)})

    return aligned, compute_all_metrics(aligned, steps)


def load_captaincook4d():
    ann_path = VIDEOS / "captaincook4d_samples" / "annotations.json"
    with open(ann_path) as f:
        annotations = json.load(f)
    samples = []
    for rec_id, ann in annotations.items():
        matches = list((VIDEOS / "captaincook4d_samples").glob(f"{rec_id}_*.mp4"))
        if not matches:
            continue
        steps = [s for s in ann["steps"] if s.get("start_time", -1) >= 0]
        samples.append((matches[0], rec_id, steps))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["uniform", "adaptive"])
    parser.add_argument("--fps", required=True, help="1, 2, 4, or 'max'")
    parser.add_argument("--model", default="gemini-2.5-flash")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    samples = load_captaincook4d()
    tag = f"{args.model}-{args.mode}-fps{args.fps}"
    out_dir = RESULTS / "captaincook4d" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode} | FPS: {args.fps} | Videos: {len(samples)}")

    all_results = []
    for video_path, rec_id, steps in samples:
        dur = get_video_duration(video_path)
        fps_val = compute_max_fps(dur) if args.fps == "max" else float(args.fps)

        print(f"\n  {rec_id} ({dur:.0f}s, fps={fps_val:.1f}, ~{int(dur*fps_val)} frames)")
        try:
            if args.mode == "uniform":
                raw, dur, actual_fps = run_uniform(client, video_path, steps, fps_val, args.model)
            else:
                raw, dur, actual_fps = run_adaptive(client, video_path, steps, fps_val, args.model)

            aligned, metrics = evaluate(raw, steps, dur)
            result = {
                "recording_id": rec_id, "video": video_path.name,
                "dataset": "captaincook4d", "model": args.model,
                "mode": args.mode, "fps": actual_fps,
                "ground_truth": steps, "predictions": aligned, "metrics": metrics,
            }
            with open(out_dir / f"{rec_id}.json", "w") as f:
                json.dump(result, f, indent=2)

            m = metrics
            print(f"    IoU={m['mean_iou']:.1%} R@1(.5)={m['recall_at_1_iou_0.5']:.1%} Det={m['step_detection_rate']:.1%}")
            all_results.append(result)
        except Exception as e:
            print(f"    FAILED: {e}")

        time.sleep(2)

    if all_results:
        n = len(all_results)
        print(f"\n{'='*50}")
        print(f"AGGREGATE: {args.mode} fps={args.fps} ({n} videos)")
        for k in ["mean_iou", "recall_at_1_iou_0.3", "recall_at_1_iou_0.5",
                   "recall_at_1_iou_0.7", "step_detection_rate"]:
            avg = sum(r["metrics"][k] for r in all_results) / n
            print(f"  {k}: {avg:.1%}")

        with open(out_dir / "_summary.json", "w") as f:
            json.dump({"mode": args.mode, "fps": args.fps, "num_videos": n,
                        "aggregate": {k: sum(r["metrics"][k] for r in all_results) / n
                                       for k in ["mean_iou", "recall_at_1_iou_0.3",
                                                  "recall_at_1_iou_0.5", "recall_at_1_iou_0.7",
                                                  "step_detection_rate", "ordering_compliance"]}},
                       f, indent=2)


if __name__ == "__main__":
    main()
