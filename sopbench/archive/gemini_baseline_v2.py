"""Gemini VLM baseline v2 for step boundary detection.

Changes from v1:
- Uses fps=4 via VideoMetadata for higher temporal resolution
- Prompts model with video duration in MM:SS format
- Model returns timestamps in MM:SS format; we parse back to seconds
"""

import json
import re
import time
from pathlib import Path
from pydantic import BaseModel

from google import genai
from google.genai import types

from sopbench.metrics import compute_all_metrics


# --- Pydantic schemas for structured output ---

class StepPredictionV2(BaseModel):
    step_index: int
    description: str
    start_time: str  # MM:SS string or "not_found"
    end_time: str    # MM:SS string or "not_found"
    confidence: float


class GeminiResponseV2(BaseModel):
    predictions: list[StepPredictionV2]


# --- Helpers ---

def seconds_to_mmss(seconds: float) -> str:
    """Convert seconds to MM:SS string."""
    total_secs = max(0, int(round(seconds)))
    mins = total_secs // 60
    secs = total_secs % 60
    return f"{mins:02d}:{secs:02d}"


def mmss_to_seconds(mmss: str) -> float:
    """Convert MM:SS string to seconds. Returns -1.0 for 'not_found' or invalid."""
    if not mmss or mmss.strip().lower() in ("not_found", "n/a", "-1", ""):
        return -1.0
    # Accept H:MM:SS or MM:SS
    mmss = mmss.strip()
    match = re.match(r'^(\d+):(\d{2})$', mmss)
    if match:
        mins, secs = int(match.group(1)), int(match.group(2))
        return float(mins * 60 + secs)
    # Try H:MM:SS
    match = re.match(r'^(\d+):(\d{2}):(\d{2})$', mmss)
    if match:
        hours, mins, secs = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return float(hours * 3600 + mins * 60 + secs)
    # Try bare seconds
    try:
        return float(mmss)
    except ValueError:
        return -1.0


def parse_video_duration(video_file: types.File) -> float:
    """Extract video duration in seconds from file metadata."""
    try:
        meta = video_file.video_metadata
        if meta and hasattr(meta, 'video_duration'):
            # video_duration may be a Duration protobuf or a string like "231.5s"
            vd = meta.video_duration
            if isinstance(vd, str):
                vd = vd.rstrip('s')
                return float(vd)
            # protobuf Duration has .seconds and .nanos
            if hasattr(vd, 'seconds'):
                return float(vd.seconds) + float(getattr(vd, 'nanos', 0)) / 1e9
            return float(vd)
    except Exception:
        pass
    return -1.0


# --- Core functions ---

def create_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def upload_video(client: genai.Client, video_path: str | Path) -> types.File:
    """Upload a video file to Gemini Files API and wait until it's ready."""
    video_path = Path(video_path)
    print(f"  Uploading {video_path.name} ({video_path.stat().st_size / 1e6:.1f} MB)...")
    video_file = client.files.upload(
        file=str(video_path),
        config=types.UploadFileConfig(
            display_name=video_path.stem,
            mime_type="video/mp4",
        ),
    )
    # Poll until file is ACTIVE
    while video_file.state == "PROCESSING":
        print("  Waiting for video processing...")
        time.sleep(3)
        video_file = client.files.get(name=video_file.name)
    if video_file.state != "ACTIVE":
        raise RuntimeError(f"Video upload failed with state: {video_file.state}")
    print(f"  Upload complete: {video_file.uri}")
    return video_file


def build_prompt_v2(steps: list[dict], duration_mmss: str) -> str:
    """Build the prompt with explicit video duration and MM:SS timestamps."""
    step_list = "\n".join(
        f'{i + 1}. "{s["description"]}"'
        for i, s in enumerate(steps)
    )
    return f"""You are analyzing a first-person egocentric cooking video. The video is exactly {duration_mmss} long (00:00 to {duration_mmss}).

Below is a checklist of procedural steps. For EACH step, identify the start and end timestamps in MM:SS format by carefully watching the video.

Rules:
- Use MM:SS format (e.g., 01:15).
- All timestamps must be between 00:00 and {duration_mmss}.
- If a step is not visible, set both to "not_found".
- Watch the ENTIRE video before answering.

Steps:
{step_list}

Return JSON: {{"predictions": [{{"step_index": 1, "description": "...", "start_time": "MM:SS", "end_time": "MM:SS", "confidence": 0.9}}, ...]}}"""


def run_gemini_v2(
    client: genai.Client,
    video_file: types.File,
    prompt: str,
    model: str = "gemini-2.5-flash",
) -> list[dict]:
    """Send video + prompt to Gemini with fps=4 and parse MM:SS response."""
    print(f"  Sending to {model} (fps=4)...")
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part(
                file_data=types.FileData(
                    file_uri=video_file.uri,
                    mime_type="video/mp4",
                ),
                video_metadata=types.VideoMetadata(fps=4.0),
            ),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiResponseV2,
            temperature=0.1,
        ),
    )

    if response.parsed:
        raw_preds = [p.model_dump() for p in response.parsed.predictions]
    else:
        # Fallback: parse from text
        try:
            data = json.loads(response.text)
            raw_preds = data.get("predictions", data) if isinstance(data, dict) else data
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"  WARNING: Could not parse response: {e}")
            print(f"  Raw response: {response.text[:500]}")
            return []

    # Convert MM:SS strings back to float seconds
    converted = []
    for p in raw_preds:
        start_str = p.get("start_time", "not_found")
        end_str = p.get("end_time", "not_found")
        converted.append({
            "step_index": p.get("step_index"),
            "description": p.get("description", ""),
            "start_time": mmss_to_seconds(str(start_str)),
            "end_time": mmss_to_seconds(str(end_str)),
            "confidence": p.get("confidence", 0.0),
            # Keep originals for inspection
            "start_time_mmss": start_str,
            "end_time_mmss": end_str,
        })
    return converted


def cleanup_file(client: genai.Client, video_file: types.File):
    """Delete the uploaded video file from Gemini."""
    try:
        client.files.delete(name=video_file.name)
    except Exception:
        pass


def run_evaluation_v2(
    client: genai.Client,
    video_path: str | Path,
    ground_truth_steps: list[dict],
    model: str = "gemini-2.5-flash",
) -> dict:
    """Full evaluation pipeline v2: upload → fps=4 → MM:SS prompt → predict → metrics → cleanup."""
    video_path = Path(video_path)
    print(f"\nEvaluating: {video_path.name}")

    # Upload
    video_file = upload_video(client, video_path)

    try:
        # Get duration from metadata, fall back to GT max end time
        duration_secs = parse_video_duration(video_file)
        if duration_secs <= 0:
            # Estimate from GT annotations
            gt_valid = [s for s in ground_truth_steps if s.get("end_time", -1) > 0]
            duration_secs = max((s["end_time"] for s in gt_valid), default=600.0) + 10.0
            print(f"  Duration not in metadata; estimated from GT: {duration_secs:.1f}s")
        else:
            print(f"  Video duration: {duration_secs:.1f}s")

        duration_mmss = seconds_to_mmss(duration_secs)
        print(f"  Duration (MM:SS): {duration_mmss}")

        # Build prompt and run
        prompt = build_prompt_v2(ground_truth_steps, duration_mmss)
        predictions = run_gemini_v2(client, video_file, prompt, model)

        # Align predictions to GT by step_index
        pred_by_idx = {p["step_index"]: p for p in predictions}
        aligned_preds = []
        for i, gt in enumerate(ground_truth_steps):
            pred = pred_by_idx.get(i + 1, {
                "step_index": i + 1,
                "description": gt["description"],
                "start_time": -1.0,
                "end_time": -1.0,
                "confidence": 0.0,
                "start_time_mmss": "not_found",
                "end_time_mmss": "not_found",
            })
            aligned_preds.append(pred)

        # Filter GT to valid steps only
        valid_gt = [s for s in ground_truth_steps if s.get("start_time", -1) >= 0]
        valid_preds = aligned_preds[:len(valid_gt)]

        # Compute metrics
        metrics = compute_all_metrics(valid_preds, valid_gt)

        return {
            "video": video_path.name,
            "model": model,
            "fps": 4.0,
            "duration_secs": duration_secs,
            "duration_mmss": duration_mmss,
            "ground_truth": ground_truth_steps,
            "predictions": aligned_preds,
            "metrics": metrics,
        }
    finally:
        cleanup_file(client, video_file)
