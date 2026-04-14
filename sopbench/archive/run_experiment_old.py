"""Unified experiment runner for all three approaches.

Usage:
    python -m sopbench.run_experiment --approach native --dataset captaincook4d
    python -m sopbench.run_experiment --approach manual-subsec --dataset captaincook4d
    python -m sopbench.run_experiment --approach manual-intsec --dataset captaincook4d
"""

import argparse
import cv2
import io
import json
import os
import time
from pathlib import Path

from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

from sopbench.metrics import compute_all_metrics

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "videos"
RESULTS = ROOT / "results"


# ---------- Timestamp formatting ----------

def fmt_mmss(ts):
    m = int(ts // 60)
    s = int(ts % 60)
    return f"{m:02d}:{s:02d}"

def fmt_subsec(ts):
    m = int(ts // 60)
    s = ts % 60
    return f"{m:02d}:{s:05.2f}"

def fmt_intsec(ts):
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


# ---------- Frame extraction ----------

def extract_frames(video_path, fps=4, max_side=256, jpeg_quality=70):
    cap = cv2.VideoCapture(str(video_path))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, round(orig_fps / fps))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            ts = idx / orig_fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            w, h = img.size
            scale = max_side / max(w, h)
            if scale < 1:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality)
            frames.append((ts, buf.getvalue()))
        idx += 1
    cap.release()
    return frames


# ---------- Data loading ----------

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


# ---------- Approach: Native video + fps=4 ----------

def run_native(client, video_path, steps, model="gemini-2.5-flash"):
    video_file = client.files.upload(
        file=str(video_path),
        config=types.UploadFileConfig(mime_type="video/mp4"),
    )
    while video_file.state == "PROCESSING":
        time.sleep(3)
        video_file = client.files.get(name=video_file.name)

    # Duration from metadata or GT
    dur = -1.0
    try:
        vm = video_file.video_metadata
        if vm and "videoDuration" in vm:
            dur = float(str(vm["videoDuration"]).rstrip("s"))
    except Exception:
        pass
    if dur <= 0:
        dur = max(s["end_time"] for s in steps) + 10
    dur_str = fmt_mmss(dur)

    step_list = "\n".join(f'{i+1}. "{s["description"]}"' for i, s in enumerate(steps))
    prompt = f"""You are analyzing a first-person egocentric cooking video. The video is exactly {dur_str} long (00:00 to {dur_str}).

For EACH step, identify start and end timestamps in MM:SS format.
Rules:
- MM:SS format (e.g., 01:15). All between 00:00 and {dur_str}.
- If not visible, use "not_found".
- Watch the ENTIRE video before answering.

Steps:
{step_list}

Return JSON: {{"predictions": [{{"step_index": 1, "start_time": "MM:SS", "end_time": "MM:SS", "confidence": 0.9}}, ...]}}"""

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part(
                file_data=types.FileData(file_uri=video_file.uri, mime_type="video/mp4"),
                video_metadata=types.VideoMetadata(fps=4.0),
            ),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    client.files.delete(name=video_file.name)
    return json.loads(response.text), dur


# ---------- Approach: Manual frames ----------

def run_manual_frames(client, video_path, steps, fmt_func, ts_label,
                      ts_example, model="gemini-2.5-flash"):
    frames = extract_frames(str(video_path), fps=4)
    dur = frames[-1][0] + 0.25
    dur_str = fmt_func(dur)

    step_list = "\n".join(f'{i+1}. "{s["description"]}"' for i, s in enumerate(steps))
    prompt = f"""You are analyzing frames from a first-person egocentric cooking video at 4 FPS.
Video duration: {dur_str}. Each frame is labeled with its timestamp in {ts_label} format.

For EACH step, find start and end timestamps from the frames.
Use {ts_label} format. All between {fmt_func(0)} and {dur_str}.
If not visible, use "not_found".

Steps:
{step_list}

Return JSON: {{"predictions": [{{"step_index": 1, "start_time": "{ts_example}", "end_time": "{ts_example}", "confidence": 0.9}}, ...]}}"""

    parts = [types.Part.from_text(text=prompt)]
    for ts, jpeg in frames:
        parts.append(types.Part.from_text(text=f"[{fmt_func(ts)}]"))
        parts.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))

    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    return json.loads(response.text), dur


# ---------- Common evaluation ----------

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

    metrics = compute_all_metrics(aligned, steps)
    return aligned, metrics


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", required=True,
                        choices=["native", "manual-subsec", "manual-intsec"])
    parser.add_argument("--dataset", default="captaincook4d")
    parser.add_argument("--model", default="gemini-2.5-flash")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    samples = load_captaincook4d()
    out_dir = RESULTS / args.dataset / f"{args.model}-{args.approach}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Approach:  {args.approach}")
    print(f"Model:     {args.model}")
    print(f"Videos:    {len(samples)}")
    print(f"Output:    {out_dir}")

    all_results = []
    for video_path, rec_id, steps in samples:
        print(f"\n--- {rec_id} ({video_path.name}) ---")

        try:
            if args.approach == "native":
                raw, dur = run_native(client, video_path, steps, args.model)
            elif args.approach == "manual-subsec":
                raw, dur = run_manual_frames(
                    client, video_path, steps,
                    fmt_subsec, "MM:SS.ss", "01:23.50", args.model,
                )
            else:
                raw, dur = run_manual_frames(
                    client, video_path, steps,
                    fmt_intsec, "MM:SS", "01:23", args.model,
                )

            aligned, metrics = evaluate(raw, steps, dur)

            result = {
                "recording_id": rec_id,
                "video": video_path.name,
                "dataset": args.dataset,
                "model": args.model,
                "approach": args.approach,
                "ground_truth": steps,
                "predictions": aligned,
                "metrics": metrics,
            }
            with open(out_dir / f"{rec_id}.json", "w") as f:
                json.dump(result, f, indent=2)

            m = metrics
            print(f"  IoU={m['mean_iou']:.1%}  R@1(.3)={m['recall_at_1_iou_0.3']:.1%}"
                  f"  R@1(.5)={m['recall_at_1_iou_0.5']:.1%}"
                  f"  R@1(.7)={m['recall_at_1_iou_0.7']:.1%}"
                  f"  Det={m['step_detection_rate']:.1%}")
            all_results.append(result)

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

        time.sleep(2)

    # Summary
    if all_results:
        n = len(all_results)
        print(f"\n{'='*60}")
        print(f"AGGREGATE ({args.approach}, {n} videos)")
        print(f"{'='*60}")
        for key in ["mean_iou", "recall_at_1_iou_0.3", "recall_at_1_iou_0.5",
                     "recall_at_1_iou_0.7", "step_detection_rate", "ordering_compliance"]:
            avg = sum(r["metrics"][key] for r in all_results) / n
            print(f"  {key:<25s}: {avg:.1%}")

        summary = {
            "approach": args.approach,
            "model": args.model,
            "num_videos": n,
            "aggregate": {
                k: sum(r["metrics"][k] for r in all_results) / n
                for k in ["mean_iou", "recall_at_1_iou_0.3", "recall_at_1_iou_0.5",
                           "recall_at_1_iou_0.7", "step_detection_rate", "ordering_compliance"]
            },
            "per_video": [
                {"recording_id": r["recording_id"], "metrics": r["metrics"]}
                for r in all_results
            ],
        }
        with open(out_dir / "_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
