# methods/ — baseline methods

External baselines / reference implementations used for context. **Not** part of the SOPBench pipeline — these are kept here for reference and future comparison.

## UVD/ — Universal Visual Decomposer

Source: https://github.com/zcczhang/UVD (paper: [arXiv:2310.08581](https://arxiv.org/abs/2310.08581))

Self-supervised decomposition of long-horizon manipulation videos into subgoals using frozen visual encoders (R3M, VC-1, etc.). Originally applied to robot manipulation in the Franka Kitchen environment.

**Why it's here:** UVD's core idea — using frozen visual encoders to detect phase shifts in egocentric video — is conceptually related to the SOPBench task (procedural step boundary detection). It serves as a methodological reference for visual encoder–based step boundary detection that doesn't use a VLM. We did not run UVD ourselves on CC4D for the experiments documented in `docs/exp_log.md`; this folder is preserved for anyone wanting to compare a non-VLM baseline.

**License:** MIT (per upstream repo).
