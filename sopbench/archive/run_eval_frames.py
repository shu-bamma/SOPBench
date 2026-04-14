"""CLI runner for the frame-based Gemini step boundary evaluation.

Uses the two-pass shiKai-style approach:
  Pass 1: extract frames locally at 2 FPS, send batches to Gemini for dense captioning.
  Pass 2: send caption transcript + step checklist to Gemini (text only) for timestamps.

Results are saved to results/captaincook4d/gemini-2.5-flash-frames/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from sopbench.gemini_baseline_frames import create_client, run_evaluation

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "videos"
RESULTS = ROOT / "results"

# Recording ID → video filename stem mapping
CC4D_FILENAME_MAP = {
    "12_51": "12_51_tomatomozzarellasalad",
    "10_50": "10_50_pinwheels",
    "10_6":  "10_6_pinwheels_correct",
    "9_45":  "9_45_mugcake",
    "1_34":  "1_34_microwaveeggsandwich",
}


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
        # Keep all steps (including -1 timestamps — the baseline will predict for them)
        steps = ann["steps"]
        samples.append((video_path, rec_id, steps))
    return samples


def print_result(r: dict) -> None:
    m = r["metrics"]
    print(f"\n  {r['recording_id']}  ({r['video']}):")
    print(f"    Frames extracted:  {r.get('num_frames', '?')}")
    print(f"    GT steps:          {m['num_gt_steps']}")
    print(f"    Predictions:       {m['num_predictions']}")
    print(f"    Mean IoU:          {m['mean_iou']:.3f}")
    print(f"    R@1 (IoU>=0.3):    {m['recall_at_1_iou_0.3']:.3f}")
    print(f"    R@1 (IoU>=0.5):    {m['recall_at_1_iou_0.5']:.3f}")
    print(f"    R@1 (IoU>=0.7):    {m['recall_at_1_iou_0.7']:.3f}")
    print(f"    Detection Rate:    {m['step_detection_rate']:.3f}")
    print(f"    Ordering:          {m['ordering_compliance']:.3f}")

    # Per-step breakdown
    print(f"    Per-step IoUs:")
    preds = r["predictions"]
    gts   = r["ground_truth"]
    valid_gt = [s for s in gts if s.get("start_time", -1) >= 0]
    ious  = m.get("per_step_iou", [])
    for i, (gt, iou) in enumerate(zip(valid_gt, ious)):
        pred = preds[i] if i < len(preds) else {}
        ps = pred.get("start_time", -1)
        pe = pred.get("end_time", -1)
        gs = gt["start_time"]
        ge = gt["end_time"]
        flag = "OK" if iou >= 0.5 else ("MISS" if ps < 0 else "LOW")
        desc = gt["description"][:50]
        print(f"      [{i+1:2d}] IoU={iou:.2f} {flag:4s}  "
              f"GT={gs:.1f}-{ge:.1f}s  Pred={ps:.1f}-{pe:.1f}s  \"{desc}\"")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frame-based Gemini step boundary evaluation"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["captaincook4d"],
        default="captaincook4d",
        help="Dataset to evaluate (default: captaincook4d)",
    )
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-flash",
        help="Gemini model to use",
    )
    parser.add_argument(
        "--fps", type=float, default=2.0,
        help="Frame extraction rate (default: 2.0)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Frames per captioning batch (default: 8)",
    )
    parser.add_argument(
        "--recording-id", type=str, default=None,
        help="Run only this recording ID (e.g. 12_51)",
    )
    args = parser.parse_args()

    # Load API key
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env or environment")

    client = create_client(api_key)

    # Load samples
    if args.dataset == "captaincook4d":
        samples = load_captaincook4d()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Optional filter
    if args.recording_id:
        samples = [(vp, rid, steps) for vp, rid, steps in samples
                   if rid == args.recording_id]
        if not samples:
            raise ValueError(f"Recording ID not found: {args.recording_id}")

    method_tag = f"{args.model}-frames"
    out_dir = RESULTS / args.dataset / method_tag.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:    {args.dataset}")
    print(f"Model:      {args.model}")
    print(f"Method:     frame-based 2-pass (Pass1=dense captions, Pass2=step matching)")
    print(f"FPS:        {args.fps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Videos:     {len(samples)}")
    print(f"Output dir: {out_dir}")

    all_results = []
    for video_path, rec_id, steps in samples:
        try:
            result = run_evaluation(
                client,
                video_path,
                steps,
                model=args.model,
                fps=args.fps,
                batch_size=args.batch_size,
            )
        except Exception as exc:
            print(f"  ERROR on {rec_id}: {exc}")
            import traceback
            traceback.print_exc()
            continue

        result["recording_id"] = rec_id
        result["dataset"] = args.dataset

        # Save individual result
        out_path = out_dir / f"{rec_id}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved: {out_path}")

        all_results.append(result)

    # Print detailed summary
    if all_results:
        print("\n" + "=" * 70)
        print("DETAILED RESULTS")
        print("=" * 70)
        for r in all_results:
            print_result(r)

        # Aggregate
        print("\n" + "=" * 70)
        print(f"AGGREGATE ({len(all_results)} videos)")
        print("=" * 70)
        metrics_keys = [
            "mean_iou",
            "recall_at_1_iou_0.3",
            "recall_at_1_iou_0.5",
            "recall_at_1_iou_0.7",
            "step_detection_rate",
            "ordering_compliance",
        ]
        for key in metrics_keys:
            values = [r["metrics"][key] for r in all_results]
            avg = sum(values) / len(values)
            print(f"  {key:<30s}: {avg:.3f}")

        # Save summary
        summary = {
            "dataset": args.dataset,
            "model": args.model,
            "method": "gemini-frames-2pass",
            "fps": args.fps,
            "num_videos": len(all_results),
            "aggregate": {
                key: sum(r["metrics"][key] for r in all_results) / len(all_results)
                for key in metrics_keys
            },
            "results": [
                {
                    "recording_id": r["recording_id"],
                    "video": r["video"],
                    "num_frames": r.get("num_frames"),
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
