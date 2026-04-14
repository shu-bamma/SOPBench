"""Test whether sub-second output format (MM:SS.ss) improves results at fps>1.

Compares:
  fps=1 + MM:SS      (matched — model sees 1s tokens, outputs 1s)
  fps=2 + MM:SS      (mismatched — model sees 0.5s tokens, outputs 1s)
  fps=2 + MM:SS.ss   (matched — model sees 0.5s tokens, outputs 0.5s)
  fps=4 + MM:SS.ss   (matched — model sees 0.25s tokens, outputs 0.25s)
"""

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


def fmt_mmss(ts):
    m = int(ts // 60)
    s = int(ts % 60)
    return f"{m:02d}:{s:02d}"


def fmt_mmss_sub(ts):
    m = int(ts // 60)
    s = ts % 60
    return f"{m:02d}:{s:05.2f}"


def get_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    dur = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return dur


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


def run_one(client, video_path, steps, fps_val, fmt_func, ts_format, ts_example):
    video_file = client.files.upload(
        file=str(video_path),
        config=types.UploadFileConfig(mime_type="video/mp4"),
    )
    while video_file.state == "PROCESSING":
        time.sleep(3)
        video_file = client.files.get(name=video_file.name)

    dur = get_duration(video_path)
    dur_str = fmt_func(dur)
    step_list = "\n".join(f'{i+1}. "{s["description"]}"' for i, s in enumerate(steps))

    prompt = f"""You are analyzing a first-person egocentric cooking video. The video is exactly {dur_str} long.

For EACH step, identify start and end timestamps in {ts_format} format.
Rules:
- Use {ts_format} format (e.g., {ts_example}). All between {fmt_func(0)} and {dur_str}.
- If not visible, use "not_found".

Steps:
{step_list}

Return JSON: {{"predictions": [{{"step_index": 1, "start_time": "{ts_example}", "end_time": "{ts_example}", "confidence": 0.9}}, ...]}}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
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

    raw = json.loads(response.text)
    preds_raw = raw.get("predictions", raw)

    # Check if model returned sub-second
    sample_ts = str(preds_raw[0].get("start_time", "")) if preds_raw else ""
    has_subsec = "." in sample_ts

    aligned = []
    for i in range(len(steps)):
        if i < len(preds_raw):
            p = preds_raw[i]
            st = parse_ts(p.get("start_time", "-1"))
            et = parse_ts(p.get("end_time", "-1"))
            if st > dur * 1.1:
                st = -1.0
            if et > dur * 1.1:
                et = dur
        else:
            st, et = -1.0, -1.0
        aligned.append({"start_time": float(st), "end_time": float(et)})

    metrics = compute_all_metrics(aligned, steps)
    return metrics, has_subsec, aligned


def main():
    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    samples = load_samples()

    configs = [
        (1, fmt_mmss, "MM:SS", "01:23"),
        (2, fmt_mmss, "MM:SS", "01:23"),
        (2, fmt_mmss_sub, "MM:SS.ss", "01:23.50"),
        (4, fmt_mmss_sub, "MM:SS.ss", "01:23.25"),
    ]

    all_results = {}
    for fps_val, fmt_func, ts_format, ts_example in configs:
        tag = f"fps{fps_val}_{ts_format.replace(':','').replace('.','')}"
        print(f"\n{'='*50}")
        print(f"Config: fps={fps_val}, format={ts_format}")
        print(f"{'='*50}")

        out_dir = RESULTS / "captaincook4d" / f"gemini-2.5-flash-subsec-{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for video_path, rec_id, steps in samples:
            print(f"  {rec_id}...", end=" ", flush=True)
            try:
                metrics, has_subsec, aligned = run_one(
                    client, video_path, steps, fps_val, fmt_func, ts_format, ts_example
                )
                m = metrics
                print(f"IoU={m['mean_iou']:.1%} R@1(.5)={m['recall_at_1_iou_0.5']:.1%} subsec={has_subsec}")

                result = {
                    "recording_id": rec_id, "video": video_path.name,
                    "dataset": "captaincook4d", "model": "gemini-2.5-flash",
                    "fps": fps_val, "ts_format": ts_format,
                    "has_subsec_output": has_subsec,
                    "ground_truth": steps, "predictions": aligned, "metrics": metrics,
                }
                with open(out_dir / f"{rec_id}.json", "w") as f:
                    json.dump(result, f, indent=2)
                results.append(result)
            except Exception as e:
                print(f"FAILED: {e}")

            time.sleep(2)

        if results:
            n = len(results)
            avg_iou = sum(r["metrics"]["mean_iou"] for r in results) / n
            avg_r15 = sum(r["metrics"]["recall_at_1_iou_0.5"] for r in results) / n
            print(f"  AGGREGATE ({n}v): IoU={avg_iou:.1%} R@1(.5)={avg_r15:.1%}")
            all_results[tag] = {"avg_iou": avg_iou, "avg_r15": avg_r15, "n": n}

    print(f"\n{'='*60}")
    print("FINAL COMPARISON: Sub-second output format")
    print(f"{'='*60}")
    print(f"{'Config':<25s} | {'IoU':>8s} | {'R@1(.5)':>8s}")
    print("-" * 47)
    for tag, r in all_results.items():
        print(f"{tag:<25s} | {r['avg_iou']:>7.1%} | {r['avg_r15']:>7.1%}")


if __name__ == "__main__":
    main()
