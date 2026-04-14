"""Temporal grounding metrics for step boundary detection."""


def temporal_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Compute Intersection over Union between two temporal segments."""
    if pred_start < 0 or pred_end < 0 or gt_start < 0 or gt_end < 0:
        return 0.0
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0.0, intersection_end - intersection_start)
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union <= 0:
        return 0.0
    return intersection / union


def mean_iou(predictions: list[dict], ground_truth: list[dict]) -> float:
    """Mean IoU across all matched step pairs.

    Each prediction and GT entry must have 'start_time' and 'end_time' keys.
    Lists are matched by index (step order).
    """
    if not predictions or not ground_truth:
        return 0.0
    n = min(len(predictions), len(ground_truth))
    ious = []
    for i in range(n):
        iou = temporal_iou(
            predictions[i]["start_time"], predictions[i]["end_time"],
            ground_truth[i]["start_time"], ground_truth[i]["end_time"],
        )
        ious.append(iou)
    return sum(ious) / len(ious) if ious else 0.0


def per_step_iou(predictions: list[dict], ground_truth: list[dict]) -> list[float]:
    """IoU for each step pair, matched by index."""
    n = min(len(predictions), len(ground_truth))
    return [
        temporal_iou(
            predictions[i]["start_time"], predictions[i]["end_time"],
            ground_truth[i]["start_time"], ground_truth[i]["end_time"],
        )
        for i in range(n)
    ]


def recall_at_k(predictions: list[dict], ground_truth: list[dict],
                iou_threshold: float = 0.5, k: int = 1) -> float:
    """Recall@k: fraction of GT steps with at least one prediction above IoU threshold.

    For step grounding, k=1 means we check the single matched prediction per GT step.
    """
    if not ground_truth:
        return 0.0
    n = min(len(predictions), len(ground_truth))
    hits = 0
    for i in range(n):
        iou = temporal_iou(
            predictions[i]["start_time"], predictions[i]["end_time"],
            ground_truth[i]["start_time"], ground_truth[i]["end_time"],
        )
        if iou >= iou_threshold:
            hits += 1
    return hits / len(ground_truth)


def step_detection_rate(predictions: list[dict]) -> float:
    """Fraction of steps where the model returned a valid prediction (not -1)."""
    if not predictions:
        return 0.0
    detected = sum(
        1 for p in predictions
        if p.get("start_time", -1) >= 0 and p.get("end_time", -1) >= 0
    )
    return detected / len(predictions)


def ordering_compliance(predictions: list[dict]) -> float:
    """Fraction of consecutive prediction pairs that are in correct temporal order.

    Checks that pred[i].start_time <= pred[i+1].start_time for valid predictions.
    """
    valid = [p for p in predictions if p.get("start_time", -1) >= 0]
    if len(valid) <= 1:
        return 1.0
    correct = sum(
        1 for i in range(len(valid) - 1)
        if valid[i]["start_time"] <= valid[i + 1]["start_time"]
    )
    return correct / (len(valid) - 1)


def compute_all_metrics(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """Compute all metrics and return as a dict."""
    step_ious = per_step_iou(predictions, ground_truth)
    return {
        "mean_iou": mean_iou(predictions, ground_truth),
        "per_step_iou": step_ious,
        "recall_at_1_iou_0.3": recall_at_k(predictions, ground_truth, 0.3, 1),
        "recall_at_1_iou_0.5": recall_at_k(predictions, ground_truth, 0.5, 1),
        "recall_at_1_iou_0.7": recall_at_k(predictions, ground_truth, 0.7, 1),
        "step_detection_rate": step_detection_rate(predictions),
        "ordering_compliance": ordering_compliance(predictions),
        "num_gt_steps": len(ground_truth),
        "num_predictions": len(predictions),
    }
