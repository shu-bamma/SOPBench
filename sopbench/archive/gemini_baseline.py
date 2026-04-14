"""Gemini VLM baseline for step boundary detection."""

import json
import time
from pathlib import Path
from pydantic import BaseModel

from google import genai
from google.genai import types

from sopbench.metrics import compute_all_metrics


# --- Pydantic schemas for structured output ---

class StepPrediction(BaseModel):
    step_index: int
    description: str
    start_time: float
    end_time: float
    confidence: float


class GeminiResponse(BaseModel):
    predictions: list[StepPrediction]


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


def build_prompt(steps: list[dict]) -> str:
    """Build the prompt that asks Gemini to predict step timestamps."""
    step_list = "\n".join(
        f'{i + 1}. "{s["description"]}"'
        for i, s in enumerate(steps)
    )
    return f"""You are analyzing a first-person (egocentric) cooking video recorded from a head-mounted camera.

Below is a checklist of procedural steps that should occur in this video. For EACH step, predict the exact start_time and end_time (in seconds) where that step occurs in the video.

Watch the entire video carefully and identify when each step begins and ends based on the visual actions you observe.

Rules:
- Times must be in seconds (e.g., 17.5, not "0:17").
- If a step is clearly not visible or not performed in the video, set start_time and end_time both to -1.
- Provide a confidence score between 0.0 and 1.0 for each prediction.
- Steps may not be in chronological order in the video — predict the actual times you observe.

Steps:
{step_list}

Return your predictions as JSON matching this exact schema:
{{"predictions": [{{"step_index": 1, "description": "...", "start_time": 17.0, "end_time": 23.5, "confidence": 0.9}}, ...]}}"""


def run_gemini(
    client: genai.Client,
    video_file: types.File,
    prompt: str,
    model: str = "gemini-2.5-flash",
) -> list[dict]:
    """Send video + prompt to Gemini and parse the structured response."""
    print(f"  Sending to {model}...")
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_uri(file_uri=video_file.uri, mime_type="video/mp4"),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiResponse,
            temperature=0.1,
        ),
    )

    if response.parsed:
        return [p.model_dump() for p in response.parsed.predictions]

    # Fallback: parse from text
    try:
        data = json.loads(response.text)
        return data.get("predictions", data) if isinstance(data, dict) else data
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"  WARNING: Could not parse response: {e}")
        print(f"  Raw response: {response.text[:500]}")
        return []


def cleanup_file(client: genai.Client, video_file: types.File):
    """Delete the uploaded video file from Gemini."""
    try:
        client.files.delete(name=video_file.name)
    except Exception:
        pass


def run_evaluation(
    client: genai.Client,
    video_path: str | Path,
    ground_truth_steps: list[dict],
    model: str = "gemini-2.5-flash",
) -> dict:
    """Full evaluation pipeline: upload → prompt → predict → metrics → cleanup."""
    video_path = Path(video_path)
    print(f"\nEvaluating: {video_path.name}")

    # Upload
    video_file = upload_video(client, video_path)

    try:
        # Build prompt and run
        prompt = build_prompt(ground_truth_steps)
        predictions = run_gemini(client, video_file, prompt, model)

        # Align predictions to GT by step_index
        pred_by_idx = {p["step_index"]: p for p in predictions}
        aligned_preds = []
        for i, gt in enumerate(ground_truth_steps):
            pred = pred_by_idx.get(i + 1, {
                "step_index": i + 1,
                "description": gt["description"],
                "start_time": -1,
                "end_time": -1,
                "confidence": 0.0,
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
            "ground_truth": ground_truth_steps,
            "predictions": aligned_preds,
            "metrics": metrics,
        }
    finally:
        cleanup_file(client, video_file)
