"""CLI runner for Gemini v2 step boundary detection evaluation.

v2 changes: fps=4, MM:SS timestamps, duration-aware prompt.
Results saved to results/captaincook4d/gemini-2.5-flash-v2/
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from sopbench.gemini_baseline_v2 import create_client, run_evaluation_v2

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "videos"
RESULTS = ROOT / "results"


def load_captaincook4d() -> list[tuple[Path, str, list[dict]]]:
    """Load CaptainCook4D sample videos and annotations."""
    ann_path = VIDEOS / "captaincook4d_samples" / "annotations.json"
    with open(ann_path) as f:
        annotations = json.load(f)

    samples = []
    for rec_id, ann in annotations.items():
        video_dir = VIDEOS / "captaincook4d_samples"
        matches = list(video_dir.glob(f"{rec_id}_*.mp4"))
        if not matches:
            print(f"  WARNING: No video found for {rec_id}, skipping")
            continue
        video_path = matches[0]
        steps = [s for s in ann["steps"] if s.get("start_time", -1) >= 0]
        samples.append((video_path, rec_id, steps))
    return samples


def load_coin() -> list[tuple[Path, str, list[dict]]]:
    """Load COIN sample videos and annotations."""
    ann_path = VIDEOS / "coin_samples" / "annotations.json"
    with open(ann_path) as f:
        annotations = json.load(f)

    samples = []
    for yt_id, ann in annotations.items():
        video_path = VIDEOS / "coin_samples" / f"{yt_id}.mp4"
        if not video_path.exists():
            video_path = VIDEOS / "coin_samples" / f"{yt_id}.webm"
        if not video_path.exists():
            print(f"  WARNING: No video found for {yt_id}, skipping")
            continue
        steps = [
            {
                "step_id": int(s.get("id", i + 1)),
                "start_time": s["segment"][0],
                "end_time": s["segment"][1],
                "description": s["label"],
                "has_errors": False,
            }
            for i, s in enumerate(ann.get("annotation", []))
        ]
        samples.append((video_path, yt_id, steps))
    return samples


def print_summary(all_results: list[dict]) -> None:
    """Print per-video and aggregate metrics summary."""
    if not all_results:
        print("No results to summarize.")
        return

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (v2: fps=4, MM:SS timestamps)")
    print("=" * 60)

    for r in all_results:
        m = r["metrics"]
        print(f"\n  {r['recording_id']}  ({r.get('duration_mmss', '?')})")
        print(f"    Mean IoU:        {m['mean_iou']:.3f}")
        print(f"    R@1 (IoU>=0.3):  {m['recall_at_1_iou_0.3']:.3f}")
        print(f"    R@1 (IoU>=0.5):  {m['recall_at_1_iou_0.5']:.3f}")
        print(f"    R@1 (IoU>=0.7):  {m['recall_at_1_iou_0.7']:.3f}")
        print(f"    Detection Rate:  {m['step_detection_rate']:.3f}")
        print(f"    Ordering:        {m['ordering_compliance']:.3f}")
        print(f"    Steps (GT/Pred): {m['num_gt_steps']} / {m['num_predictions']}")

    n = len(all_results)
    avg_iou    = sum(r["metrics"]["mean_iou"] for r in all_results) / n
    avg_r1_03  = sum(r["metrics"]["recall_at_1_iou_0.3"] for r in all_results) / n
    avg_r1_05  = sum(r["metrics"]["recall_at_1_iou_0.5"] for r in all_results) / n
    avg_r1_07  = sum(r["metrics"]["recall_at_1_iou_0.7"] for r in all_results) / n
    avg_det    = sum(r["metrics"]["step_detection_rate"] for r in all_results) / n
    avg_order  = sum(r["metrics"]["ordering_compliance"] for r in all_results) / n

    print(f"\n  AGGREGATE ({n} videos):")
    print(f"    Avg Mean IoU:     {avg_iou:.3f}")
    print(f"    Avg R@1 (>=0.3):  {avg_r1_03:.3f}")
    print(f"    Avg R@1 (>=0.5):  {avg_r1_05:.3f}")
    print(f"    Avg R@1 (>=0.7):  {avg_r1_07:.3f}")
    print(f"    Avg Det. Rate:    {avg_det:.3f}")
    print(f"    Avg Ordering:     {avg_order:.3f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run Gemini v2 step boundary evaluation (fps=4, MM:SS timestamps)"
    )
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--recording-id", type=str, help="Recording ID (for single video mode)")
    parser.add_argument("--dataset", type=str, choices=["captaincook4d", "coin"],
                        help="Run on all videos in a dataset")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        help="Gemini model to use (default: gemini-2.5-flash)")
    args = parser.parse_args()

    if not args.video and not args.dataset:
        parser.error("Must specify either --video or --dataset")

    # Load API key
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env or environment")

    client = create_client(api_key)

    # Collect samples
    if args.video:
        video_path = Path(args.video)
        rec_id = args.recording_id or video_path.stem
        dataset_name = "custom"
        for ds, loader in [("captaincook4d", load_captaincook4d), ("coin", load_coin)]:
            for vp, rid, steps in loader():
                if rid == rec_id or vp == video_path:
                    samples = [(video_path, rec_id, steps)]
                    dataset_name = ds
                    break
            else:
                continue
            break
        else:
            parser.error(f"Could not find annotations for {rec_id}. "
                         "Use --dataset instead or ensure annotations exist.")
    else:
        dataset_name = args.dataset
        if args.dataset == "captaincook4d":
            samples = load_captaincook4d()
        else:
            samples = load_coin()

    # v2 output directory
    model_tag = args.model.replace("/", "_") + "-v2"
    out_dir = RESULTS / dataset_name / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:    {dataset_name}")
    print(f"Model:      {args.model}")
    print(f"FPS:        4")
    print(f"Videos:     {len(samples)}")
    print(f"Output dir: {out_dir}")

    # Run evaluations
    all_results = []
    for video_path, rec_id, steps in samples:
        result = run_evaluation_v2(client, video_path, steps, args.model)
        result["recording_id"] = rec_id
        result["dataset"] = dataset_name

        # Save individual result
        out_path = out_dir / f"{rec_id}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

        all_results.append(result)

    # Print full summary
    print_summary(all_results)

    # Save aggregate summary
    n = len(all_results)
    summary = {
        "dataset": dataset_name,
        "model": args.model,
        "fps": 4,
        "timestamp_format": "MM:SS",
        "num_videos": n,
        "aggregate": {
            "mean_iou": sum(r["metrics"]["mean_iou"] for r in all_results) / n if n else 0,
            "recall_at_1_iou_0.3": sum(r["metrics"]["recall_at_1_iou_0.3"] for r in all_results) / n if n else 0,
            "recall_at_1_iou_0.5": sum(r["metrics"]["recall_at_1_iou_0.5"] for r in all_results) / n if n else 0,
            "recall_at_1_iou_0.7": sum(r["metrics"]["recall_at_1_iou_0.7"] for r in all_results) / n if n else 0,
            "step_detection_rate": sum(r["metrics"]["step_detection_rate"] for r in all_results) / n if n else 0,
            "ordering_compliance": sum(r["metrics"]["ordering_compliance"] for r in all_results) / n if n else 0,
        },
        "results": [
            {
                "recording_id": r["recording_id"],
                "duration_mmss": r.get("duration_mmss"),
                "metrics": r["metrics"],
            }
            for r in all_results
        ],
    }
    summary_path = out_dir / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
