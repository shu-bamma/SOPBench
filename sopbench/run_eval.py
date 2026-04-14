"""CLI runner for Gemini step boundary detection evaluation."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from sopbench.gemini_baseline import create_client, run_evaluation

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
        # Find the video file matching this recording id
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
            # Try .webm
            video_path = VIDEOS / "coin_samples" / f"{yt_id}.webm"
        if not video_path.exists():
            print(f"  WARNING: No video found for {yt_id}, skipping")
            continue
        # Convert COIN format to common format
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


def main():
    parser = argparse.ArgumentParser(description="Run Gemini step boundary evaluation")
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
        # Try to find annotations
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

    print(f"Dataset: {dataset_name}")
    print(f"Model: {args.model}")
    print(f"Videos: {len(samples)}")

    # Output directory
    out_dir = RESULTS / dataset_name / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluations
    all_results = []
    for video_path, rec_id, steps in samples:
        result = run_evaluation(client, video_path, steps, args.model)
        result["recording_id"] = rec_id
        result["dataset"] = dataset_name

        # Save individual result
        out_path = out_dir / f"{rec_id}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

        all_results.append(result)

    # Print summary
    if all_results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for r in all_results:
            m = r["metrics"]
            print(f"\n  {r['recording_id']}:")
            print(f"    Mean IoU:        {m['mean_iou']:.3f}")
            print(f"    R@1 (IoU≥0.3):   {m['recall_at_1_iou_0.3']:.3f}")
            print(f"    R@1 (IoU≥0.5):   {m['recall_at_1_iou_0.5']:.3f}")
            print(f"    R@1 (IoU≥0.7):   {m['recall_at_1_iou_0.7']:.3f}")
            print(f"    Detection Rate:  {m['step_detection_rate']:.3f}")
            print(f"    Ordering:        {m['ordering_compliance']:.3f}")

        # Aggregate
        avg_iou = sum(r["metrics"]["mean_iou"] for r in all_results) / len(all_results)
        avg_r1_03 = sum(r["metrics"]["recall_at_1_iou_0.3"] for r in all_results) / len(all_results)
        avg_r1_05 = sum(r["metrics"]["recall_at_1_iou_0.5"] for r in all_results) / len(all_results)
        print(f"\n  AGGREGATE ({len(all_results)} videos):")
        print(f"    Avg Mean IoU:    {avg_iou:.3f}")
        print(f"    Avg R@1 (≥0.3):  {avg_r1_03:.3f}")
        print(f"    Avg R@1 (≥0.5):  {avg_r1_05:.3f}")

    # Save aggregate
    summary_path = out_dir / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "model": args.model,
            "num_videos": len(all_results),
            "results": [
                {"recording_id": r["recording_id"], "metrics": r["metrics"]}
                for r in all_results
            ],
        }, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
