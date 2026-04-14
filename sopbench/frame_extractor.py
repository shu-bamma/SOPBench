"""Frame extraction utilities for the shiKai-style Gemini pipeline.

Extracts frames from a video at a fixed FPS, resizes to a max dimension,
and returns (timestamp_seconds, PIL.Image) pairs.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
from PIL import Image


def extract_frames(
    video_path: str | Path,
    fps: float = 2.0,
    max_frames: int = 300,
) -> list[tuple[float, Image.Image]]:
    """Extract frames at *fps* from *video_path*.

    Parameters
    ----------
    video_path:
        Path to the video file.
    fps:
        Target extraction rate (frames per second).  2 FPS is recommended to
        stay within Gemini image-per-request limits.
    max_frames:
        Hard cap on the number of frames returned.  Frames are sampled
        uniformly across the video if the extracted count would exceed this.

    Returns
    -------
    list of (timestamp_seconds, PIL.Image)
        Timestamps are computed deterministically as ``frame_number / original_fps``
        so they are always consistent with the video clock.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        original_fps: float = cap.get(cv2.CAP_PROP_FPS)
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if original_fps <= 0:
            raise RuntimeError(f"Invalid FPS from video: {original_fps}")

        # How many source frames to skip between each extracted frame
        frame_interval = original_fps / fps  # e.g. 30 / 2 = 15

        frames: list[tuple[float, Image.Image]] = []
        frame_number = 0  # running source-frame counter

        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            # Should we keep this frame?
            if frame_number % max(1, round(frame_interval)) == 0:
                timestamp = frame_number / original_fps
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img = _resize_to_max(img, 512)
                frames.append((timestamp, img))

            frame_number += 1

        # Uniformly subsample if over the cap
        if len(frames) > max_frames:
            step = len(frames) / max_frames
            frames = [frames[int(i * step)] for i in range(max_frames)]

        return frames
    finally:
        cap.release()


def _resize_to_max(img: Image.Image, max_side: int) -> Image.Image:
    """Resize *img* so its longest side is at most *max_side* pixels."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def image_to_jpeg_bytes(img: Image.Image, quality: int = 85) -> bytes:
    """Convert a PIL Image to JPEG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.s string (e.g. 01:23.5)."""
    minutes = int(seconds) // 60
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:04.1f}"


def parse_timestamp(ts: str) -> float:
    """Parse MM:SS or MM:SS.s back to seconds.

    Accepts:
    - "01:23"    -> 83.0
    - "01:23.5"  -> 83.5
    - "83.5"     -> 83.5  (plain seconds, pass-through)
    """
    ts = ts.strip()
    if ":" in ts:
        parts = ts.split(":", 1)
        minutes = int(parts[0])
        secs = float(parts[1])
        return minutes * 60 + secs
    return float(ts)
