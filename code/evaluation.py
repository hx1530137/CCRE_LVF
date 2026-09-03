"""Ranking metrics used in the CCRE-LVF experiments."""

import argparse
import json
from pathlib import Path

import numpy as np


def _relevant_sets(relevant_indices):
    """Convert relevance labels to one set per query."""
    sets = []
    for item in relevant_indices:
        if np.isscalar(item):
            sets.append({int(item)})
        else:
            sets.append({int(value) for value in item})
    return sets


def recall_at_k(rankings, relevant_indices, k):
    """Compute macro-averaged Recall@K."""
    relevant = _relevant_sets(relevant_indices)
    values = []
    for ranked, target in zip(rankings, relevant):
        retrieved = set(map(int, ranked[:k]))
        values.append(len(retrieved & target) / len(target) if target else 0.0)
    return float(np.mean(values)) if values else 0.0


def mean_reciprocal_rank(rankings, relevant_indices):
    """Compute mean reciprocal rank."""
    relevant = _relevant_sets(relevant_indices)
    values = []
    for ranked, target in zip(rankings, relevant):
        reciprocal_rank = next(
            (1.0 / rank for rank, doc_id in enumerate(ranked, start=1) if int(doc_id) in target),
            0.0,
        )
        values.append(reciprocal_rank)
    return float(np.mean(values)) if values else 0.0


def ndcg_at_k(rankings, relevant_indices, k):
    """Compute macro-averaged binary nDCG@K."""
    relevant = _relevant_sets(relevant_indices)
    values = []
    for ranked, target in zip(rankings, relevant):
        dcg = sum(
            1.0 / np.log2(rank + 1)
            for rank, doc_id in enumerate(ranked[:k], start=1)
            if int(doc_id) in target
        )
        ideal_hits = min(len(target), k)
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        values.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(values)) if values else 0.0


def evaluate(rankings, relevant_indices, k_values=(1, 5, 10)):
    """Evaluate rankings with the paper's retrieval metrics."""
    if len(rankings) != len(relevant_indices):
        raise ValueError("Rankings and relevance labels must contain the same number of queries.")
    metrics = {f"R@{k}": recall_at_k(rankings, relevant_indices, k) for k in k_values}
    metrics["MRR"] = mean_reciprocal_rank(rankings, relevant_indices)
    for k in k_values:
        metrics[f"nDCG@{k}"] = ndcg_at_k(rankings, relevant_indices, k)
    return metrics


def _load(path):
    """Load a JSON or NumPy array."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=True)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ranked document indices.")
    parser.add_argument("--rankings", required=True, help="JSON or NPY ranked document indices")
    parser.add_argument("--relevant", required=True, help="JSON or NPY relevant document indices")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args()

    metrics = evaluate(_load(args.rankings), _load(args.relevant), tuple(args.k))
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
