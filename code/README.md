# Code Map

The scripts in this directory are copied from the reproducibility workspace without changing the experiment logic.

| Script | Role |
| --- | --- |
| `lvf_algorithm.py` | Core LVF formula, normalization, margin, bigram verification, and baselines |
| `train_finetune.py` | Qwen3-Embedding-4B LoRA fine-tuning |
| `train_both_full.py` | Fine-tune and dual-view training modes |
| `run_v3_experiments.py` | Main test-set and cross-dataset experiments used for the v3 results |
| `run_ablation.py` | Fixed-alpha, adaptive-alpha, lexical, suspicion, and full LVF ablations |
| `run_supplementary.py` | Metrics, significance, efficiency, sensitivity, RRF-k, and case-study analyses |
| `run_supplementary_batches.py` | Additional supplementary comparisons and leakage-safe checks |
| `run_fixes.py` | Leakage-safe logistic-regression and reranking checks |
| `run_all_models.py` | Multi-embedding-model comparison |
| `run_missing_models.py` | Additional model evaluations |
| `run_experiments.py` | Earlier complete experiment pipeline retained for provenance |
| `run_c3bench_experiment.py` | C3bench cross-language evaluation |
| `build_c3bench_crosslang.py` | Seeded C3bench sampling and duplicate-query check |
| `merge_all_models.py` | Result aggregation helper |

The experiment scripts currently contain absolute path constants inherited from the original experiment environment. Update those constants before running them on another machine.
