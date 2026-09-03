# Core Code

This directory contains only the core method and the final experiment scripts needed to reproduce the paper's principal analyses.

| Script | Role |
| --- | --- |
| `lvf_algorithm.py` | Core LVF formula, normalization, margin, bigram verification, and baselines |
| `train_finetune.py` | Qwen3-Embedding-4B LoRA fine-tuning |
| `run_v3_experiments.py` | Main test-set and cross-dataset experiments used for the v3 results |
| `run_ablation.py` | Fixed-alpha, adaptive-alpha, lexical, suspicion, and full LVF ablations |
| `run_supplementary.py` | Metrics, significance, efficiency, sensitivity, RRF-k, and case-study analyses |

The experiment scripts currently contain absolute path constants inherited from the original experiment environment. Update those constants before running them on another machine.
