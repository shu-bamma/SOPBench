# SOPBench

**Zero-shot procedural step boundary detection in egocentric video using Gemini 2.5.**

Given a text checklist of procedural steps (an SOP) and an untrimmed first-person video, predict the start/end timestamps for each step — without fine-tuning on the target activity.

## Headline result

On 5 CaptainCook4D egocentric cooking videos (3–6 minutes each, 9–19 procedural steps per video), zero-shot Gemini 2.5 Flash achieves:

| Metric | SOPBench (Gemini 2.5 Flash) | VSLNet baseline (Ego4D NLQ, supervised) |
|--------|----------------------------:|----------------------------------------:|
| R@1 (IoU≥0.5) | **51.9%** | ~7% |
| R@1 (IoU≥0.3) | **68.1%** | ~12% |
| Mean IoU | **44.4%** | — |

That's **~7× the supervised baseline, zero training**. See [`docs/exp_log.md`](docs/exp_log.md) for the full experiment log.

## Repository layout

```
SOPBench/
├── sopbench/                   # main scripts (the SOPBench framework)
│   ├── run_experiment.py       #   main entry: 10 configs × 5 videos
│   ├── run_eval.py             #   load & evaluate
│   ├── metrics.py              #   IoU, R@K computation
│   ├── visualizer.py           #   local HTML results viewer
│   ├── frame_extractor.py      #   utility
│   ├── ablations/              #   focused ablation experiments
│   └── archive/                #   superseded scripts (kept for posterity)
│
├── benchmarks/
│   └── ego4d-goalstep/         # upstream facebookresearch/ego4d-goalstep repo
│       ├── data/               #   train/valid/test JSON annotations
│       ├── step_grounding/     #   VSLNet baseline starter code
│       ├── asset/              #   thumbnail
│       └── README.md           #   upstream README
│
├── methods/UVD/                # external baseline (Universal Visual Decomposer)
│
├── videos/                     # sample videos (Git LFS)
│   ├── captaincook4d_samples/  #   5 ego cooking videos with audio
│   ├── captaincook4d_samples_noaudio/  # audio-stripped (for ablation)
│   ├── coin_samples/           #   8 third-person YouTube cooking
│   └── probe/                  #   synthetic probe videos
│
├── results/                    # all experiment outputs (~3MB JSON)
│   └── captaincook4d/          #   55 result subdirs grouped by experiment
│
├── docs/
│   ├── exp_log.md              # full experiment log (read this!)
│   ├── SOPBench_goal.md        # research plan & dataset landscape
│   └── CLAUDE.md               # context for AI coding assistants
│
├── procedural_datasets.jsx     # interactive React table of procedural datasets
├── pyproject.toml
├── .env.example                # copy to .env, add GEMINI_API_KEY
└── .gitignore
```

## Quick start

```bash
# 1. Clone (with LFS for videos)
git lfs install
git clone https://github.com/<your-user>/SOPBench.git
cd SOPBench

# 2. Set up Python environment
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. Set your Gemini API key
cp .env.example .env
# Edit .env and add: GEMINI_API_KEY=AQ.your_key_here

# 4. Smoke test on one video
python -m sopbench.run_experiment --fps 1 --format mmss

# 5. View results in the browser
python -m sopbench.visualizer
# Open http://localhost:8080
```

## Reproducing the headline result

The best config from our experiments — `gemini-2.5-flash` at `fps=1` with `MM:SS.ss` output:

```bash
python -m sopbench.run_experiment --fps 1 --format sub
```

This runs 5 API calls (~$0.10 in Gemini Flash credits) and saves results to `results/captaincook4d/clean-fps1-sub/`.

## Reproducing all experiments

The full sweep documented in `docs/exp_log.md` (10 configs × 5 videos = 50 calls per run, ~$3 each):

```bash
# Main FPS × Format experiment (run twice to measure variance)
python -m sopbench.run_experiment   # all 10 configs

# Ablations
python -m sopbench.ablations.run_variance_verification   # sequential re-run for variance
python -m sopbench.ablations.run_noaudio_experiment      # audio-off variant
python -m sopbench.ablations.run_subsec_experiment       # MM:SS vs MM:SS.ss
python -m sopbench.ablations.run_pro_top_configs         # gemini-2.5-pro on top configs
python -m sopbench.ablations.probe_balanced              # synthetic perception probe
```

## Key findings

1. **Format matters:** Asking for `MM:SS` (matching Gemini's native timestamp tokens) and constraining to the video duration eliminates out-of-bounds predictions. **3.4× IoU improvement** over naive prompting (12.5% → 42.8%).

2. **Less is more (FPS):** `fps=1` consistently wins across all metrics. More frames = more tokens = degraded attention for long-range temporal reasoning. `fps=8` and `fps=max` are worse than `fps=1`.

3. **Native > manual:** Native video upload via `VideoMetadata(fps=N)` outperforms manual frame extraction by 10–20 IoU points and avoids API payload limits (~2,600 inline parts max).

4. **Audio helps:** Removing the audio track hurts performance in 7/10 configs — the cooking narration provides real step cues.

5. **Pro doesn't help:** `gemini-2.5-pro` with thinking-budget reasoning is **tied** with Flash on `fps=1` configs and **loses** by 11% on `fps=2`. Step boundary detection is bottlenecked by perception, not reasoning. **Use Flash.**

6. **Variance is huge:** Run-to-run deltas span -7.3% to +16.1% across configs at `temperature=0.1`. Multi-run averaging is required for confident ranking.

See [`docs/exp_log.md`](docs/exp_log.md) for the verified numbers from all ~320 API calls.

## Datasets

- **CaptainCook4D** ([github.com/CaptainCook4D](https://github.com/CaptainCook4D)) — 384 egocentric cooking recordings, GoPro 360p, with step + error annotations. 5 samples used here are downloaded directly from their public Box CDN. Licence: Apache 2.0.
- **Ego4D Goal-Step** ([facebookresearch/ego4d-goalstep](https://github.com/facebookresearch/ego4d-goalstep)) — annotation files only (videos require Ego4D license + AWS credentials). Used for benchmark methodology + R@1 metric definitions.
- **COIN** ([coin-dataset.github.io](https://coin-dataset.github.io/)) — 11.8K third-person YouTube how-to videos. 8 samples used as third-person comparison set.

## Citation

If you use this code or our findings, please cite the underlying datasets:

```bibtex
@inproceedings{song2024ego4d,
  title={Ego4D Goal-Step: Toward Hierarchical Understanding of Procedural Activities},
  author={Song, Yale and Byrne, Eugene and Nagarajan, Tushar and Wang, Huiyu and Martin, Miguel and Torresani, Lorenzo},
  booktitle={NeurIPS}, year={2024}
}

@article{peddi2024captaincook4d,
  title={CaptainCook4D: A Dataset for Understanding Errors in Procedural Activities},
  author={Peddi, Rohith and others},
  booktitle={NeurIPS Datasets and Benchmarks}, year={2024}
}
```

## License

MIT (see `LICENSE`). The upstream `ego4d-goalstep` files in `benchmarks/ego4d-goalstep/` retain their original MIT license (`benchmarks/ego4d-goalstep/LICENSE_upstream`).
