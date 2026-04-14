# SOPBench

Zero-shot procedural compliance verification from egocentric video. Given a text SOP and an untrimmed egocentric video, evaluate step completion, temporal grounding, ordering compliance, and error detection — without fine-tuning on target activities.

## Project structure

- `data/` — Ego4D Goal-Step annotations (train/val/test JSON). Hierarchical: Goal → Step → Substep.
- `methods/` — Baseline methods and new approaches.
- `step_grounding/` — VSLNet baseline for step grounding via NaQ codebase. Requires Ego4D CLI for video features.
- `SOPBench_goal.md` — Full research plan, dataset landscape, related work.
- `procedural_datasets.jsx` — Interactive React table of procedural video datasets.

## Technical context

- **Goal**: Benchmark frozen visual encoders on procedural step boundary detection, step completion, ordering compliance, and error detection — all zero-shot.
- **Key hypothesis**: V-JEPA 2.1 ViT-B (80M params) outperforms larger image-only encoders on step boundary detection — temporal self-supervised objectives matter more than model size.
- **Scope**: Egocentric video only.
- **Primary dataset**: Ego4D Goal-Step (430h, 7353 videos, 48K step segments, 319 goals, 514 steps).
- **Encoders to benchmark**: V-JEPA 2 / 2.1, VL-JEPA, SigLIP, DINOv2, CLIP ViT-L, InternVideo2, VideoMAE ViT-H, VLM encoders (Qwen2.5-VL, InternVL3).

## Commands

```bash
# Download Ego4D data
ego4d --datasets full_scale annotations --benchmark goalstep -o <out-dir>

# Step grounding baseline
cd step_grounding
bash train.sh experiments/vslnet/goalstep/
bash infer.sh experiments/vslnet/goalstep/
```

## Conventions

- Python 3.9+, PyTorch 2.x
- Experiment logging via W&B
