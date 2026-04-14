# SOPBench: Zero-Shot Procedural Compliance Verification from Video

## Project Goal

Build a benchmark and evaluation framework that takes a **text SOP (Standard Operating Procedure)** as input — with or without reference video — and measures **task progress, step completion, ordering compliance, and error detection** over unseen procedural videos, without fine-tuning on target activities.

The core algorithm is adapted from **UVD (Universal Visual Decomposer)** [1], which discovers procedural subgoals by detecting phase shifts in the embedding space of frozen visual encoders. We extend this from robotics to human procedural activity understanding, and benchmark temporal video encoders (V-JEPA family) against image encoders and VLMs for SOP progress tracking.

---

## The Gap This Fills

No benchmark currently exists that evaluates the following end-to-end:

> **Input**: A text SOP document (ordered steps with descriptions) + an untrimmed video of someone performing a task.  
> **Output**: For each SOP step — (a) was it performed, (b) when did it start/end, (c) was it performed correctly, (d) were steps done in correct order — all without fine-tuning on this specific SOP or activity.

This is a compound task combining temporal action localization + step ordering verification + error detection + text-conditioned grounding. No single benchmark evaluates all of these together in a zero-shot setting.

### Closest existing work and their limitations

| Task | Text-conditioned? | Sequential? | Error detection? | Zero-shot? |
|---|---|---|---|---|
| ZS-TAL (THUMOS14) [2][3] | Yes | No (atomic) | No | Yes |
| CrossTask step localization [4] | Yes | Yes | No | Partially |
| PREGO mistake detection [5] | Partially | Yes | Yes | Recognition branch: No |
| Ego4D-PMD [6] | Yes | Partially | Yes | Yes (VLM-based) |
| Assembly101 mistake detection [7] | No | Yes | Yes | No |
| EgoOops [8] | Yes (required) | Yes | Yes | Small scale (50 videos) |

None checks all four boxes simultaneously.

---

## Core Algorithm: UVD for SOP Progress Tracking

**UVD** [1] (ICRA 2024, Best Paper Finalist in Robot Vision) discovers subgoals by computing embedding distances between video frames and a goal state using a frozen visual encoder, then finding where monotonicity breaks in the distance curve. Each break = a phase transition = a step boundary.

Key properties making it suitable for SOP verification:
- **Zero additional training** — uses frozen pretrained visual encoders
- **Encoder agnostic** — works with any visual backbone
- **Pure visual** — no task-specific knowledge needed
- **Embedding distance = task progress signal** — monotonically decreasing distance to goal means procedure is progressing

The adaptation from robotics to SOP verification:

| UVD (robotics) | SOPBench adaptation |
|---|---|
| Long-horizon manipulation task | Multi-step SOP procedure |
| Subgoal discovery | Step boundary detection |
| Embedding distance to goal frame | Task progress / step completion |
| Phase shift in embedding space | Step transition |
| Goal frame conditioning | Text SOP step conditioning (novel) |

**Paper**: Zhang et al., "Universal Visual Decomposer: Long-Horizon Manipulation Made Easy", ICRA 2024  
**Code**: https://github.com/zcczhang/UVD/

---

## Video Encoders to Benchmark

### V-JEPA Family (Primary — temporal encoders)

V-JEPA predicts in **learned representation space** (not pixels), making it architecturally superior for temporal activity understanding. Predicts what masked video regions *mean*, not what they *look like*.

**V-JEPA 2** [9] — 1.2B params. 77.3% on SSv2 (temporal reasoning benchmark). 39.7 R@5 on EPIC-Kitchens-100 action anticipation. Self-supervised on 1M+ hours of video.

**V-JEPA 2.1** [10] — Fixes dense feature weakness of V-JEPA 2. New SOTA on SSv2. Available as:
- ViT-G (2B) and ViT-g (1B) — full models
- **ViT-L (300M)** — distilled
- **ViT-B (80M)** — distilled, edge-deployable

Key results:
- 40.8 R@5 on EPIC-Kitchens-100 action anticipation
- 7.71 mAP on Ego4D short-term object interaction anticipation
- +20% success rate over V-JEPA 2 on robot grasping tasks
- SOTA on SSv2 (temporal reasoning)

**VL-JEPA** [11] — adds language to JEPA. 1.6B params. Surpasses CLIP/SigLIP2 on video classification and retrieval with 43× less training data. 2.85× faster inference via selective decoding. Enables text-conditioned UVD where distance is measured to a text embedding of the SOP step rather than a goal frame.

**Papers**:
- [9] Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning", 2025 — https://arxiv.org/abs/2506.09985
- [10] "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning", 2026 — https://arxiv.org/html/2603.14482v1
- [11] VL-JEPA, 2026 — https://arxiv.org/html/2512.10942v1

### Comparison encoders

- **SigLIP / Perception Encoder** — image-text contrastive (spatial, no temporal)
- **DINOv2 / DINOv3** — image self-supervised (dense spatial features)
- **CLIP ViT-L** — image-text contrastive baseline
- **InternVideo2** — hybrid video encoder (1B-6B)
- **VideoMAE ViT-H** — pixel-space prediction (contrast with V-JEPA's representation-space prediction)
- **Qwen2.5-VL / InternVL3** — VLM encoders (intermediate features)

The hypothesis: V-JEPA 2.1 ViT-B (80M) outperforms larger image-only encoders on procedural step boundary detection because temporal self-supervised objectives matter more than model size or language alignment for SOP progress tracking.

---

## Domains and Real-World Use Cases

### Where SOPs exist and video compliance checking has commercial value

**1. QSR / Commercial Kitchen** — Every dish in a chain (McDonald's, Domino's) has a defined assembly sequence. SOP compliance affects food safety, consistency, speed. Existing players: Wobot AI, VuFindr.

**2. Manufacturing / Assembly Lines** — Step-by-step procedures for product assembly, soldering, wiring, packaging. Highest commercial value. Existing players: Retrocausal (Honda, Siemens), Drishti ($50M+, DENSO/Flex), ADLINK.

**3. Surgery / Medical Procedures** — Strictest SOPs. Phase recognition is mature. Existing players: Theator (acquired by J&J), Caresyntax ($180M+ raised), Surgical Safety Technologies.

**4. Warehouse / Logistics** — Pick-pack-ship procedures, safety compliance. Existing players: Voxel ($64M), Spot AI ($62M).

**5. Construction / Heavy Industry** — Safety SOP compliance (PPE, scaffolding procedures, lockout/tagout). Existing players: Intenseye ($29M+), viAct.

**6. Pharmaceutical / Lab** — GMP compliance, lab protocols. Rigid SOPs with regulatory backing. No significant AI video player yet.

**7. Field Maintenance / Repair** — Equipment maintenance checklists, technician procedures. Covered partially by COIN/CrossTask instructional domains.

**8. Agriculture** — Harvesting procedures, crop handling SOPs. Emerging: Claru AI egocentric agricultural dataset.

---

## Dataset Landscape

### Tier 1 — Hierarchical Steps + Error Annotations (Gold Standard for SOPBench)

**Ego4D Goal-Step** [12] — NeurIPS 2023 Spotlight. 430 hours, 7,353 videos, 48K step segments. Hierarchical: Goal → Step → Substep (319 goals, 514 steps). Text descriptions at every level. Egocentric. Active CVPR 2026 challenge. **Primary evaluation dataset.**
- https://ego4d-data.org/

**CaptainCook4D** [13] — NeurIPS 2024 D&B. 384 recordings, 94.5 hours. Recipe-as-SOP with task graph representations. 6 error categories (technique, measurement, ordering, timing, temperature, preparation). 164 correct + 220 error videos. **Best error annotation dataset.**
- https://captaincook4d.github.io/
- https://github.com/CaptainCook4D

**EgoOops** [8] — ICCV-W 2025. 50 videos, 6.8 hours, 5 domains (circuits, chemistry, crafts, toy building, color mixing). **Only dataset requiring reference to procedural TEXT to detect mistakes.** 6 mistake classes + natural language explanations.
- https://github.com/Y-Haneji/EgoOops-annotations

**IndustReal** [14] — WACV 2024. 84 egocentric videos, 5.8 hours. Industrial assembly setting. 38 error types, 48 valid execution orders. Defines the Procedure Step Recognition (PSR) task. **Closest to manufacturing SOP verification.**
- https://github.com/TimSchoonbeek/IndustReal

### Tier 2 — Step Annotations + Mistake Detection

**Assembly101** [7] — CVPR 2022. 4,321 videos, 513 hours. Toy vehicle assembly. 1M+ fine-grained action annotations. 328 mistake sequences. Multi-view + egocentric. Assembly101-O online benchmark used by PREGO.
- https://assembly-101.github.io/
- HuggingFace sample: https://huggingface.co/datasets/pablovela5620/Assembly101-Sample

**EgoPER** [15] — CVPR 2024. 385 videos, 28 hours. Kitchen domain. 5 error types (omission, correction, modification, slip, addition). Trains only on correct videos (one-class classification). Multi-modal: RGB + depth + audio + gaze + hand tracking.
- https://www.khoury.northeastern.edu/home/eelhami/egoper.htm

**HoloAssist** [16] — ICCV 2023. 2,221 videos, 166 hours. Mixed tasks with instructor-performer interaction. Segment-level mistake labels. HoloLens multi-modal. Active CVPR 2025 challenge.
- https://holoassist.github.io/

**Ego4D-PMD** [6] — EMNLP 2025. 12,500 examples from Ego4D. Pairs video frames with procedural text narrations. Binary mistake labels + mistake types. Tests VLMs with iterative self-dialog reasoning.
- Paper: "Transparent and Coherent Procedural Mistake Detection"

### Tier 3 — Procedural Step Annotations (No Errors)

**COIN** [17] — CVPR 2019. 11,827 YouTube videos, 476 hours. **180 diverse tasks** (car repair, cooking, crafts, gardening). 46K temporal step segments with text labels. Most diverse procedural dataset.
- https://coin-dataset.github.io/

**CrossTask** [4] — CVPR 2019. 4,700 YouTube videos, 83 instructional tasks. Explicitly ordered step annotations with text descriptions. Canonical weakly-supervised step localization benchmark.
- https://github.com/DmZhukov/CrossTask

**YouCook2** [18] — AAAI 2018. 2,000 YouTube videos, 176 hours, 89 recipes. Step-level temporal boundaries + human-written step captions. Gold standard for text-conditioned step grounding.
- http://youcook2.eecs.umich.edu/

**EPIC-Kitchens-100** [19] — IJCV 2022. 700 videos, 100 hours egocentric. 90K action segments, 97 verbs, 300 nouns. Dense temporal annotations. Canonical temporal encoder benchmark (V-JEPA evaluates here).
- https://epic-kitchens.github.io/2024

**Ego-Exo4D** [20] — CVPR 2024. 5,035 videos, 740 hours. Synchronized egocentric + exocentric views. Keystep recognition + skill assessment. Diverse activities (cooking, sports, repair).
- https://ego-exo4d-data.org/

**EgoExoLearn** [21] — CVPR 2024. Ego-exo procedural activities with skill assessment. 95 verbs, 254 nouns. Gaze signals + left/right hand attribution.
- https://github.com/OpenGVLab/EgoExoLearn

### Tier 4 — Industrial / Assembly Specific

**HA-ViD** [22] — NeurIPS 2024. Human assembly video with industrial scenarios. Human-robot shared annotations.
- https://HA-ViD.github.io/

**IKEA Manuals at Work** [23] — NeurIPS 2024 D&B. Grounds instruction manual images to video with 6DoF part poses. Closest to "SOP document → video alignment" paradigm.
- https://github.com/yunongLiu1/IKEA-Manuals-at-Work

**IKEA ASM** [24] — WACV 2021. 371 furniture assemblies. Multi-view + depth + pose. 33 action classes.
- https://ikeaasm.github.io/

**MECCANO** [25] — ECCV 2020. 20 egocentric videos, 7 hours. Toy motorcycle assembly.
- https://iplab.dmi.unict.it/MECCANO/

**HA4M** [26] — Sci Data 2022. 41 subjects, epicyclic gear train assembly. 6 modalities (RGB + Depth + IR + Skeleton).
- https://zenodo.org/record/7213301

### Tier 5 — Surgical Procedures

**Cholec80** [27] — TMI 2016. 80 cholecystectomy videos, 40 hours. 7 surgical phases. 13 surgeons. CC-BY-NC-SA 4.0.
- https://camma.unistra.fr/datasets/

**CholecT50** [28] — MIA 2022. 50 videos. Fine-grained action triplets: <instrument, verb, target>. 100 triplet classes.
- https://camma.unistra.fr/datasets/

**MultiBypass140** [29] — 2023. 140 gastric bypass videos. Multi-level: Phase → Step → Intraoperative Adverse Events. Error annotations.
- https://camma.unistra.fr/datasets/

**PhaKIR** [30] — MICCAI 2024 Challenge. 8 multi-institutional cholecystectomy videos. Phase + keypoint + segmentation.
- https://doi.org/10.5281/zenodo.15740619

### Tier 6 — Self-Supervised Pretraining (Unlabeled)

**Egocentric-100K** [31] — Build AI. 100,405 hours, 14,228 workers, 10.8B frames. Real SE Asian factories. Assembly, packing, line work. Apache 2.0. **Completely unlabeled.**
- https://huggingface.co/datasets/builddotai/Egocentric-100K

**Egocentric-10K** [32] — Build AI. 10,000 hours, 1080p version.
- https://huggingface.co/datasets/builddotai/Egocentric-10K

**HowTo100M** [33] — ICCV 2019. 1.22M YouTube instructional videos, 134K hours. ASR narrations as weak step labels. 23K different tasks.
- https://www.di.ens.fr/willow/research/howto100m/

### PPE / Safety (Spatial, complementary)

**keremberke/protective-equipment-detection** — 11,978 images, COCO format.
- https://huggingface.co/datasets/keremberke/protective-equipment-detection

**Construction Site Safety (Roboflow)** — pretrained models + annotated images.
- https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety

### Auto-Annotation Attempts

**Kriya-Egocentric-100K** — VLM-generated action annotations on 5 videos from Egocentric-100K. Action100M-compatible. MIT license.
- https://huggingface.co/datasets/ankk98/Kriya-Egocentric-100K (search HF)

---

## Key Publications by Category

### Online Mistake Detection

- [5] **PREGO** — Flaborea et al., "Online Mistake Detection in PRocedural EGOcentric Videos", CVPR 2024. Dual-branch: recognition + LLM anticipation. Assembly101-O and Epic-tent-O benchmarks.
  - https://github.com/aleflabo/PREGO

- [34] **TI-PREGO** — "Chain of Thought and In-Context Learning for Online Mistake Detection in PRocedural EGOcentric Videos", arXiv 2024. Extends PREGO with Auto-CoT + ICL. SOTA on Assembly101-O.
  - https://arxiv.org/html/2411.02570v1

- [35] **Differentiable Task Graph Learning** — Seminara et al., NeurIPS 2024 Spotlight. +16.7% F1 over PREGO on CaptainCook4D. Task Graph Transformer predicts graphs from text alone.
  - https://github.com/fpv-iplab/Differentiable-Task-Graph-Learning

- [36] **AMNAR** — Huang et al., CVPR 2025. Multiple normal action representations for error detection.

- [37] **Gazing Into Missteps** — Mazzamuto et al., CVPR 2025. Unsupervised mistake detection via gaze prediction.

- [6] **Transparent PMD** — Storks et al., EMNLP 2025. VLMs with iterative self-dialog for mistake detection.

### Zero-Shot Temporal Action Localization

- [2] **STALE** — Nag et al., "Zero-Shot Temporal Action Detection via Vision-Language Prompting", ECCV 2022.
  - https://github.com/sauradip/STALE

- [3] **T3AL** — Liberatori et al., "Test-Time Zero-Shot Temporal Action Localization", CVPR 2024.
  - https://github.com/benedettaliberatori/T3AL

- [38] **DeTAL** — "Open-Vocabulary Temporal Action Localization With Decoupled Networks", TPAMI 2024.

- [39] **OVFormer** — "Open-Vocabulary Temporal Action Localization using Multimodal Guidance", BMVC 2024.

### Procedure Planning

- [40] **PDPP** — Wang et al., "Projected Diffusion for Procedure Planning in Instructional Videos", CVPR 2023 Highlight.
  - https://arxiv.org/html/2303.14676

- [41] **SCHEMA** — Niu et al., "State CHangEs MAtter for Procedure Planning", ICLR 2024. LLM chain-of-thought for state changes.
  - https://arxiv.org/abs/2403.01599

- [42] **KEPP** — Nagasinghe et al., "Why Not Use Your Textbook? Knowledge-Enhanced Procedure Planning", CVPR 2024. Probabilistic Procedural Knowledge Graph.
  - https://github.com/Ravindu-Yasas-Nagasinghe/KEPP

- [43] **VEDIT** — ICLR 2025. Flow matching for latent-space procedure planning.

- [44] **CLAD** — "Constrained Latent Action Diffusion for Vision-Language Procedure Planning", 2025.
  - https://arxiv.org/html/2503.06637

### Procedural Knowledge Graphs

- [45] **Paprika** — Zhou et al., "Procedure-Aware Pretraining for Instructional Video Understanding", CVPR 2023. wikiHow + HowTo100M knowledge graph. +11% on step recognition.
  - https://arxiv.org/abs/2303.18230

- [46] **PKR-QA/KML** — "Neuro Symbolic Knowledge Reasoning for Procedural Video QA", arXiv 2025. Multi-hop + causal reasoning over procedural video.
  - https://arxiv.org/html/2503.14957v5

- [47] **EgoGraph** — "Temporal Knowledge Graph for Egocentric Video Understanding", arXiv 2025. Dynamic knowledge graphs for ultra-long egocentric video.
  - https://arxiv.org/abs/2602.23709

### Temporal Action Segmentation (Procedural)

- [48] **ProTAS** — Shen & Elhamifar, "Progress-Aware Online Action Segmentation for Egocentric Procedural Task Videos", CVPR 2024. Task graphs + progress prediction.
  - https://github.com/Yuhan-Shen/ProTAS

- [49] **DTL** — Xu et al., "Don't Pour Cereal into Coffee: Differentiable Temporal Logic for Temporal Action Segmentation", NeurIPS 2022. Encodes ordering rules as differentiable Linear Temporal Logic.
  - https://github.com/ZiweiXU/DTL-action-segmentation

### Task Verification

- [50] **SVIP** — "Sequence Verification for Procedures in Videos", CVPR 2022. Step-level video sequence matching.

- [51] **EgoTV** — "Egocentric Task Verification from Natural Language Task Descriptions", ICCV 2023. Binary: was the text-described task completed?
  - https://arxiv.org/html/2303.16975

- [52] **EgoCross** — "Procedural Heterogeneous Graph Completion for Text-Conditioned Task Verification", CVPR 2025.
  - https://xunchn.github.io/EgoCross/

### Tiny VLM / Edge Deployment

- [53] **Moondream** — M87 Labs. 2B params (SigLIP + Phi-1.5). Image-only, no temporal. Moondream 3 uses MoE (9B/2B active).
  - https://moondream.ai/

- [54] **SmolVLM2** — HuggingFace. 256M-2.2B. Processes video natively.

- [55] **Florence-2** — Microsoft. 0.23B-0.77B. Unified sequence-to-sequence vision.

---

## Commercial Landscape

### Step-Level Procedure Tracking (Direct competitors to SOPBench applications)

- **Retrocausal** — Assembly Copilot. Tracks individual steps against Bill of Process. Honda, Siemens. retrocausal.ai
- **Drishti** — $50M+ raised. Cycle-time analysis + deviation detection. DENSO, Flex, top auto OEMs. drishti.com
- **Theator** — Acquired by J&J MedTech. Surgical video analysis across 180+ procedure types.
- **Caresyntax** — $180M+ Series C. 3,500+ operating rooms.

### Safety/Behavior Monitoring (Adjacent)

- **Voxel** — $64M raised (incl. $44M Series B, June 2025). PPE + unsafe behavior detection. Dick's, Americold. voxelai.com
- **Spot AI** — $62M raised. Video intelligence platform. "Weighs every action against an SOP." spot.ai
- **Intenseye** — $29M+ raised. 35+ unsafe scenario types. 15+ countries. intenseye.com
- **viAct** — Hong Kong. Construction SOP compliance. Claims 90% accident reduction. viact.ai
- **ADLINK** — Taiwan, publicly traded. NEON AI smart cameras for SOP monitoring. adlinktech.com

### Food Safety

- **Wobot AI** — Kitchen SOP monitoring (handwashing, PPE, cross-contamination).
- **VuFindr** — Restaurant food safety video analytics. vufindr.com

---

## Key Findings Summary

1. **No unified SOP verification benchmark exists.** The field is fragmented across mistake detection (PREGO), temporal localization (T3AL), step recognition (IndustReal), and task verification (EgoTV). SOPBench unifies these.

2. **V-JEPA is architecturally superior for temporal tasks.** 77.3% SSv2 (V-JEPA 2) vs 75.4% (VideoMAE-H) vs 59.5% (TimeSformer). Representation-space prediction > pixel-space prediction for understanding *what happened* vs *what it looked like*.

3. **V-JEPA 2.1 ViT-B at 80M params is edge-deployable.** 25× smaller than Moondream (2B). Can serve as the temporal backbone for real-time SOP monitoring.

4. **UVD's phase-shift detection maps directly to step boundary detection.** The algorithm needs zero training and works with any frozen encoder — ideal for benchmarking encoders on procedural tasks.

5. **Text-conditioned UVD (via VL-JEPA) enables zero-shot SOP verification.** Measure video frame distance to text step embeddings instead of goal frames. No reference video needed.

6. **Kitchen is the richest domain (7+ datasets, ~900h, error annotations).** Industrial manufacturing is the most commercially valuable but has only ~15 hours of labeled data.

7. **Egocentric-100K (100K hours, Apache 2.0) is unlabeled but could be auto-annotated** using VLM pipelines (Kriya approach) to create the largest factory SOP pretraining dataset.

8. **The commercial market has bifurcated**: step-level procedure tracking (Retrocausal, Drishti) vs safety behavior monitoring (Voxel, Intenseye). SOPBench targets the former — technically harder but more defensible.

---

*Interactive dataset table available as a separate file: `procedural_datasets.jsx`*
