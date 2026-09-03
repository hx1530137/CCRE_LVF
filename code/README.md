# Core Code

This directory contains only the core method and the final experiment scripts needed to reproduce the paper's principal analyses.

| Script | Role |
| --- | --- |
| `lvf_algorithm.py` | Core LVF formula, normalization, margin, bigram verification, and baselines |
| `train_encoder.py` | Encoder LoRA fine-tuning |
| `retrieval_experiment.py` | Main retrieval experiment |
| `ablation_experiment.py` | Fixed-alpha, adaptive-alpha, lexical, suspicion, and full LVF ablations |
| `supplementary_analysis.py` | Metrics, significance, efficiency, sensitivity, RRF-k, and case-study analyses |
| `evaluation.py` | Standalone Recall@K, MRR, and nDCG@K evaluation |

The experiment scripts use placeholder paths under `CCRE_LVF_ROOT`. Set that environment variable or replace the placeholder paths before running them.
