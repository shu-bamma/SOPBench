"""Synthetic probe video for testing Gemini's video perception pipeline.

Creates a video with known content at every frame, then queries Gemini
to verify what it actually sees at different fps settings.

Usage:
    # Generate the probe video
    python -m sopbench.probe_video generate

    # Run all probe tests
    python -m sopbench.probe_video test

    # Run specific test
    python -m sopbench.probe_video test --fps 2
"""

import argparse
import json
import os
import random
import string
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent
PROBE_DIR = ROOT / "videos" / "probe"

# ---- Video generation ----

def generate_probe_video(
    duration_sec: int = 30,
    video_fps: int = 30,
    width: int = 640,
    height: int = 480,
    seed: int = 42,
):
    """Generate a probe video with random codes on each frame.

    Every frame at each 0.25-second boundary gets a unique random 4-char code
    displayed as large white text on black background. The frame also shows
    the exact timestamp.

    Returns the ground truth mapping: {timestamp_sec: {"code": "AB3K", "frame_idx": N}}
    """
    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    total_frames = duration_sec * video_fps

    # Generate codes for every 0.25-second boundary (4 per second)
    # This gives us ground truth at sub-second resolution
    ground_truth = {}
    for quarter in range(duration_sec * 4):
        ts = quarter * 0.25
        code = ''.join(rng.choices(string.ascii_uppercase + string.digits, k=4))
        ground_truth[f"{ts:.2f}"] = {
            "code": code,
            "timestamp": ts,
            "frame_idx": int(ts * video_fps),
        }

    # Write video
    video_path = PROBE_DIR / "probe_30s.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, video_fps, (width, height))

    for frame_idx in range(total_frames):
        ts = frame_idx / video_fps
        # Find the nearest 0.25s boundary
        quarter = round(ts * 4) / 4
        quarter_key = f"{quarter:.2f}"

        if quarter_key in ground_truth:
            code = ground_truth[quarter_key]["code"]
        else:
            code = "????"

        # Create black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Large code in center
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, code, (width // 2 - 120, height // 2 + 30),
                    font, 3.5, (255, 255, 255), 8, cv2.LINE_AA)

        # Timestamp in top-left
        ts_str = f"{int(ts // 60):02d}:{ts % 60:05.2f}"
        cv2.putText(frame, ts_str, (20, 40),
                    font, 1.0, (180, 180, 180), 2, cv2.LINE_AA)

        # Frame index in top-right
        cv2.putText(frame, f"F{frame_idx}", (width - 150, 40),
                    font, 0.8, (100, 100, 100), 2, cv2.LINE_AA)

        out.write(frame)

    out.release()

    # Save ground truth
    gt_path = PROBE_DIR / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump({
            "video": "probe_30s.mp4",
            "duration_sec": duration_sec,
            "video_fps": video_fps,
            "width": width,
            "height": height,
            "seed": seed,
            "num_codes": len(ground_truth),
            "codes_per_second": 4,
            "codes": ground_truth,
        }, f, indent=2)

    print(f"Generated: {video_path} ({duration_sec}s, {video_fps}fps, {total_frames} frames)")
    print(f"Ground truth: {gt_path} ({len(ground_truth)} codes)")
    print(f"Sample codes:")
    for ts_key in sorted(ground_truth.keys(), key=float)[:12]:
        gt = ground_truth[ts_key]
        print(f"  t={gt['timestamp']:05.2f}s -> {gt['code']}")

    return video_path, ground_truth


# ---- Probe tests ----

def run_probe_test(client, video_path, ground_truth, fps_val, query_timestamps):
    """Upload video at given fps, ask Gemini what code it sees at specific timestamps."""

    video_file = client.files.upload(
        file=str(video_path),
        config=types.UploadFileConfig(mime_type="video/mp4"),
    )
    while video_file.state == "PROCESSING":
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)

    # Build query
    ts_list = "\n".join(f"- {t}" for t in query_timestamps)
    prompt = f"""This is a 30-second test video. Each frame shows a black background with a large 4-character alphanumeric code in white text (e.g., "AB3K", "9XM2").

The code changes every 0.25 seconds. There is also a timestamp shown in the top-left corner.

For EACH of the following timestamps, tell me the EXACT 4-character code you see displayed on screen:

{ts_list}

Return ONLY valid JSON:
{{"readings": [{{"timestamp": "00:05", "code": "AB3K"}}, ...]}}

Be precise — read the exact characters. Do not guess."""

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
            temperature=0.0,
        ),
    )

    client.files.delete(name=video_file.name)

    raw = json.loads(response.text)
    readings = raw.get("readings", raw)
    return readings


def evaluate_probe(readings, ground_truth, label=""):
    """Compare Gemini's readings against ground truth."""
    correct = 0
    total = len(readings)
    results = []

    for r in readings:
        ts_str = str(r.get("timestamp", ""))
        predicted_code = str(r.get("code", "")).strip().upper()

        # Parse timestamp to find nearest ground truth
        try:
            # Handle MM:SS, MM:SS.ss, etc
            parts = ts_str.split(":")
            if len(parts) == 2:
                secs = int(parts[0]) * 60 + float(parts[1])
            else:
                secs = float(ts_str)
        except (ValueError, IndexError):
            secs = -1

        # Find nearest 0.25s boundary
        quarter = round(secs * 4) / 4
        gt_key = f"{quarter:.2f}"
        gt_entry = ground_truth.get(gt_key, {})
        expected_code = gt_entry.get("code", "????")

        match = predicted_code == expected_code
        if match:
            correct += 1

        results.append({
            "query_ts": ts_str,
            "resolved_sec": secs,
            "gt_key": gt_key,
            "expected": expected_code,
            "predicted": predicted_code,
            "match": match,
        })

    accuracy = correct / total if total > 0 else 0
    print(f"\n  {label}: {correct}/{total} correct ({accuracy:.0%})")
    for r in results:
        icon = "Y" if r["match"] else "X"
        print(f"    [{icon}] t={r['query_ts']:>10s} -> expected={r['expected']} got={r['predicted']}")

    return {"label": label, "accuracy": accuracy, "correct": correct,
            "total": total, "details": results}


def run_all_tests(client, video_path, ground_truth):
    """Run comprehensive probe tests."""
    all_results = []

    # Test 1: Integer-second queries at different fps
    int_timestamps = ["00:02", "00:05", "00:08", "00:12", "00:15",
                      "00:18", "00:21", "00:25", "00:28"]

    for fps_val in [1, 2, 4, 8]:
        print(f"\n{'='*50}")
        print(f"TEST: fps={fps_val}, querying integer seconds")
        readings = run_probe_test(client, video_path, ground_truth,
                                  fps_val, int_timestamps)
        result = evaluate_probe(readings, ground_truth, f"fps={fps_val} int-sec")
        result["fps"] = fps_val
        result["query_type"] = "integer_seconds"
        all_results.append(result)
        time.sleep(3)

    # Test 2: Sub-second queries at fps=4 (can it see 0.25s boundaries?)
    sub_timestamps = ["00:02.00", "00:02.25", "00:02.50", "00:02.75",
                      "00:10.00", "00:10.25", "00:10.50", "00:10.75",
                      "00:20.00", "00:20.25", "00:20.50", "00:20.75"]

    for fps_val in [1, 2, 4]:
        print(f"\n{'='*50}")
        print(f"TEST: fps={fps_val}, querying sub-second (0.25s intervals)")
        readings = run_probe_test(client, video_path, ground_truth,
                                  fps_val, sub_timestamps)
        result = evaluate_probe(readings, ground_truth, f"fps={fps_val} sub-sec")
        result["fps"] = fps_val
        result["query_type"] = "sub_second"
        all_results.append(result)
        time.sleep(3)

    # Test 3: Ask model to list ALL codes it sees in first 5 seconds
    for fps_val in [1, 2, 4]:
        print(f"\n{'='*50}")
        print(f"TEST: fps={fps_val}, list all codes in first 5 seconds")

        video_file = client.files.upload(
            file=str(video_path),
            config=types.UploadFileConfig(mime_type="video/mp4"),
        )
        while video_file.state == "PROCESSING":
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)

        prompt = """This is a test video with 4-character codes on black background. The code changes every 0.25 seconds.

List EVERY distinct code you can see in the first 5 seconds (00:00 to 00:05) of the video, along with the exact timestamp when you see it.

Return JSON: {"codes_seen": [{"timestamp": "00:00.00", "code": "AB3K"}, ...]}"""

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
                temperature=0.0,
            ),
        )
        client.files.delete(name=video_file.name)

        raw = json.loads(response.text)
        codes_seen = raw.get("codes_seen", raw)

        # Count how many of the 20 codes (5 sec * 4/sec) it found
        gt_codes_0_5 = {v["code"] for k, v in ground_truth.items()
                        if v["timestamp"] < 5.0}
        seen_codes = {str(c.get("code", "")).strip().upper() for c in codes_seen}
        correct_codes = gt_codes_0_5 & seen_codes

        n_seen = len(codes_seen)
        n_correct = len(correct_codes)
        expected_at_fps = fps_val * 5  # frames in 5 seconds at this fps

        print(f"\n  fps={fps_val}: reported {n_seen} codes, {n_correct}/{len(gt_codes_0_5)} match GT")
        print(f"  Expected ~{expected_at_fps} frames in 5s at {fps_val}fps")
        for c in codes_seen[:15]:
            ts = c.get("timestamp", "?")
            code = c.get("code", "?")
            in_gt = code.upper() in gt_codes_0_5
            print(f"    t={ts} code={code} {'(correct)' if in_gt else '(wrong)'}")
        if len(codes_seen) > 15:
            print(f"    ... +{len(codes_seen)-15} more")

        all_results.append({
            "label": f"fps={fps_val} list-codes-5s",
            "fps": fps_val,
            "query_type": "list_all_5s",
            "codes_reported": n_seen,
            "codes_correct": n_correct,
            "expected_frames": expected_at_fps,
            "gt_total": len(gt_codes_0_5),
        })
        time.sleep(3)

    # Save all results
    results_path = PROBE_DIR / "probe_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {results_path}")

    return all_results


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("generate", help="Generate probe video")

    test_p = sub.add_parser("test", help="Run probe tests")
    test_p.add_argument("--fps", type=int, help="Test single fps only")

    args = parser.parse_args()

    if args.cmd == "generate":
        generate_probe_video()

    elif args.cmd == "test":
        load_dotenv(ROOT / ".env")
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        video_path = PROBE_DIR / "probe_30s.mp4"
        gt_path = PROBE_DIR / "ground_truth.json"

        if not video_path.exists():
            print("Generating probe video first...")
            generate_probe_video()

        with open(gt_path) as f:
            gt_data = json.load(f)

        run_all_tests(client, video_path, gt_data["codes"])

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
