# Data Access

Raw datasets are not included in this code release. Replace the placeholder URL below with the final Hugging Face repository before making the project public:

**Hugging Face dataset:** <https://huggingface.co/datasets/YOUR-ORG/CCRE-LVF-Dataset>

The expected records are described in [`dataset_description.md`](dataset_description.md). The main fields are:

- `instruction`: modern-Chinese query;
- `input`: historical evidence passage, generally classical Chinese plus an optional modern translation;
- `output`: answer or reference response when supplied by the dataset.

The training JSONL format uses `messages` and `positive_messages` fields for contrastive fine-tuning. Keep the original train/validation/test separation and the paper's sampling seed (`42`).
