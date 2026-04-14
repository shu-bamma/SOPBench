"""Gemini frame-based baseline for step boundary detection.

Two-pass approach:
  Pass 1 — Dense captioning: extract frames, send batches to Gemini, collect
            per-frame captions into a temporal transcript.
  Pass 2 — Step matching: send transcript + step list to Gemini (text only),
            get back JSON with start/end timestamps.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

from sopbench.frame_extractor import (
    extract_frames,
    format_timestamp,
    image_to_jpeg_bytes,
    parse_timestamp,
)
from sopbench.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

def create_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Pass 1: Dense captioning
# ---------------------------------------------------------------------------

_CAPTION_PROMPT_TEMPLATE = """\
These are frames from an egocentric (first-person) cooking video, extracted at 2 FPS.
Frame timestamps: [{timestamps}]

For each frame, write one short sentence describing what the person's hands are doing.
Focus on concrete actions: picking up, cutting, spreading, stirring, placing, etc.

Return ONLY valid JSON — no markdown, no explanation — in this exact format:
{{"captions": [{{"timestamp": "MM:SS", "description": "short action description"}}, ...]}}

Include one entry per frame, in the same order as the timestamps given above."""


def _build_caption_request(
    frames: list[tuple[float, bytes]],  # (timestamp_seconds, jpeg_bytes)
) -> tuple[str, list[types.Part]]:
    """Build the prompt string and image Parts for a caption batch."""
    ts_labels = [format_timestamp(t) for t, _ in frames]
    prompt = _CAPTION_PROMPT_TEMPLATE.format(timestamps=", ".join(ts_labels))

    parts: list[types.Part] = []
    parts.append(types.Part.from_text(text=prompt))
    for _, jpeg_bytes in frames:
        parts.append(types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"))

    return prompt, parts


def _call_with_retry(
    client: genai.Client,
    model: str,
    contents: list,
    config: types.GenerateContentConfig,
    max_retries: int = 3,
    initial_delay: float = 5.0,
) -> object:
    """Call generate_content with simple exponential-backoff retry."""
    delay = initial_delay
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            last_err = exc
            print(f"    API error (attempt {attempt + 1}/{max_retries}): {exc}")
            if attempt < max_retries - 1:
                print(f"    Retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"All {max_retries} attempts failed: {last_err}") from last_err


def dense_caption_frames(
    client: genai.Client,
    frames: list[tuple[float, bytes]],
    model: str = "gemini-2.5-flash",
    batch_size: int = 8,
) -> list[dict]:
    """Pass 1: caption every frame in batches of *batch_size*.

    Returns a flat list of {"timestamp": float, "description": str} dicts,
    sorted by timestamp.
    """
    all_captions: list[dict] = []

    for batch_start in range(0, len(frames), batch_size):
        batch = frames[batch_start: batch_start + batch_size]
        batch_labels = [format_timestamp(t) for t, _ in batch]
        print(f"    Captioning frames {batch_start + 1}–{batch_start + len(batch)} "
              f"({batch_labels[0]} → {batch_labels[-1]})...")

        _, parts = _build_caption_request(batch)

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        )

        response = _call_with_retry(client, model, parts, config)

        # Parse JSON from response
        raw = response.text if hasattr(response, "text") else ""
        captions = _parse_caption_response(raw, batch)
        all_captions.extend(captions)

        # Be polite to the API
        time.sleep(1.0)

    # Sort by timestamp just in case
    all_captions.sort(key=lambda c: c["timestamp"])
    return all_captions


def _parse_caption_response(
    raw: str,
    batch: list[tuple[float, bytes]],
) -> list[dict]:
    """Parse the caption JSON from Gemini, falling back gracefully."""
    # Strip markdown code fences if present
    clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(clean)
        if isinstance(data, dict) and "captions" in data:
            items = data["captions"]
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError(f"Unexpected structure: {type(data)}")

        result = []
        for item in items:
            ts_raw = item.get("timestamp", "")
            try:
                ts_sec = parse_timestamp(str(ts_raw))
            except (ValueError, AttributeError):
                ts_sec = -1.0
            result.append({
                "timestamp": ts_sec,
                "description": item.get("description", ""),
            })
        return result

    except Exception as exc:
        print(f"      WARNING: Caption parse failed ({exc}); using fallback timestamps")
        # Fall back: create entries with the ground-truth timestamps from the batch
        return [
            {"timestamp": t, "description": "(caption unavailable)"}
            for t, _ in batch
        ]


# ---------------------------------------------------------------------------
# Pass 2: Step matching
# ---------------------------------------------------------------------------

_STEP_MATCH_PROMPT_TEMPLATE = """\
Below is a temporal transcript of a first-person cooking video. \
Each line is a frame timestamp followed by a one-sentence description of what \
the cook's hands are doing.

TRANSCRIPT:
{transcript}

Your task is to match each procedural step in the checklist below to the \
timestamps in the transcript and return the START and END time (in seconds) \
of each step.

STEP CHECKLIST:
{steps}

Rules:
- Return start_time and end_time as plain decimal seconds (e.g. 17.0, not "0:17").
- If a step does not appear in the transcript at all, set both to -1.
- Steps should be matched in the order they most plausibly occur in the video.
- Do NOT invent times that are not supported by the transcript.

Return ONLY valid JSON — no markdown, no explanation:
{{"predictions": [{{"step_index": 1, "description": "...", "start_time": 17.0, "end_time": 23.5, "confidence": 0.85}}, ...]}}"""


def match_steps_to_transcript(
    client: genai.Client,
    captions: list[dict],
    steps: list[dict],
    video_duration: float,
    model: str = "gemini-2.5-flash",
) -> list[dict]:
    """Pass 2: text-only step matching using the caption transcript.

    Parameters
    ----------
    captions:
        Output of ``dense_caption_frames``.
    steps:
        Ground-truth step dicts (need ``description`` key).
    video_duration:
        Total video length in seconds (used to clip predictions).
    model:
        Gemini model name.

    Returns
    -------
    List of prediction dicts with keys:
    step_index, description, start_time, end_time, confidence.
    """
    # Build transcript string
    transcript_lines = [
        f"[{format_timestamp(c['timestamp'])}] {c['description']}"
        for c in captions
        if c.get("timestamp", -1) >= 0
    ]
    transcript = "\n".join(transcript_lines)

    step_list_str = "\n".join(
        f'{i + 1}. "{s["description"]}"'
        for i, s in enumerate(steps)
    )

    prompt = _STEP_MATCH_PROMPT_TEMPLATE.format(
        transcript=transcript,
        steps=step_list_str,
    )

    print(f"    Step matching ({len(steps)} steps, transcript {len(transcript_lines)} lines)...")

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1,
    )

    response = _call_with_retry(
        client,
        model,
        [types.Part.from_text(text=prompt)],
        config,
    )

    raw = response.text if hasattr(response, "text") else ""
    predictions = _parse_step_predictions(raw, steps, video_duration)
    return predictions


def _parse_step_predictions(
    raw: str,
    steps: list[dict],
    video_duration: float,
) -> list[dict]:
    """Parse step predictions from Gemini JSON response."""
    clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(clean)
        if isinstance(data, dict) and "predictions" in data:
            preds = data["predictions"]
        elif isinstance(data, list):
            preds = data
        else:
            raise ValueError(f"Unexpected structure: {type(data)}")

        result = []
        for p in preds:
            st = float(p.get("start_time", -1))
            et = float(p.get("end_time", -1))
            # Clip to valid range
            if st > video_duration:
                st = -1.0
            if et > video_duration:
                et = video_duration
            result.append({
                "step_index": int(p.get("step_index", 0)),
                "description": str(p.get("description", "")),
                "start_time": st,
                "end_time": et,
                "confidence": float(p.get("confidence", 0.5)),
            })
        return result

    except Exception as exc:
        print(f"      WARNING: Step prediction parse failed ({exc}); returning empty")
        return [
            {
                "step_index": i + 1,
                "description": s["description"],
                "start_time": -1.0,
                "end_time": -1.0,
                "confidence": 0.0,
            }
            for i, s in enumerate(steps)
        ]


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    client: genai.Client,
    video_path: str | Path,
    ground_truth_steps: list[dict],
    model: str = "gemini-2.5-flash",
    fps: float = 2.0,
    batch_size: int = 8,
) -> dict:
    """Full two-pass frame-based evaluation pipeline.

    1. Extract frames at *fps*.
    2. Dense-caption each batch of frames (Pass 1).
    3. Match steps to transcript (Pass 2).
    4. Compute metrics.

    Returns a result dict compatible with run_eval.py output format.
    """
    video_path = Path(video_path)
    print(f"\nEvaluating (frames): {video_path.name}")

    # --- Frame extraction ---
    print(f"  Extracting frames at {fps} FPS...")
    raw_frames = extract_frames(video_path, fps=fps, max_frames=300)
    print(f"  Extracted {len(raw_frames)} frames")

    if not raw_frames:
        raise RuntimeError("No frames extracted from video")

    video_duration = raw_frames[-1][0] + (1.0 / fps)

    # Convert PIL Images to JPEG bytes once (reused across passes)
    frames_bytes: list[tuple[float, bytes]] = [
        (ts, image_to_jpeg_bytes(img))
        for ts, img in raw_frames
    ]

    # --- Pass 1: Dense captioning ---
    print("  Pass 1: Dense captioning...")
    captions = dense_caption_frames(client, frames_bytes, model=model, batch_size=batch_size)
    print(f"  Got {len(captions)} captions")

    # --- Pass 2: Step matching ---
    print("  Pass 2: Step matching...")
    predictions_raw = match_steps_to_transcript(
        client, captions, ground_truth_steps,
        video_duration=video_duration, model=model,
    )

    # Align predictions to GT by step_index
    pred_by_idx = {p["step_index"]: p for p in predictions_raw}
    aligned_preds = []
    for i, gt in enumerate(ground_truth_steps):
        pred = pred_by_idx.get(i + 1, {
            "step_index": i + 1,
            "description": gt["description"],
            "start_time": -1.0,
            "end_time": -1.0,
            "confidence": 0.0,
        })
        aligned_preds.append(pred)

    # Filter GT to valid steps only (same logic as original run_eval.py)
    valid_gt = [s for s in ground_truth_steps if s.get("start_time", -1) >= 0]
    valid_preds = aligned_preds[:len(valid_gt)]

    # Compute metrics
    metrics = compute_all_metrics(valid_preds, valid_gt)

    return {
        "video": video_path.name,
        "model": model,
        "method": "gemini-frames-2pass",
        "fps": fps,
        "num_frames": len(raw_frames),
        "captions": captions,
        "ground_truth": ground_truth_steps,
        "predictions": aligned_preds,
        "metrics": metrics,
    }
