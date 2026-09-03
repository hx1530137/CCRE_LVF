# Dataset Format

The code expects task data to be supplied separately through the configured Hugging Face dataset or local files. No research dataset is redistributed in this repository.

## Retrieval Records

JSON records should contain a query field and a corresponding evidence field:

```json
{
  "instruction": "Modern-language query",
  "input": "Evidence passage",
  "output": "Optional reference answer"
}
```

The retrieval scripts deduplicate the `input` field to form the document collection and use the matching `instruction` record to identify the relevant document index.

## Contrastive Training Records

The encoder training script expects JSONL records in the following form:

```json
{
  "messages": [{"content": "Query text"}],
  "positive_messages": [[{"content": "Relevant evidence passage"}]]
}
```

Keep training, validation, and test records in separate files. The default paths in the scripts are illustrative and should be replaced with local paths or command-line arguments.

## Sampling

When a sampling ratio is used, the scripts apply a fixed random seed (`42`) so that the selected records can be reproduced. Record the dataset revision, split, sampling ratio, and seed alongside reported results.
