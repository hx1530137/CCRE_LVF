''
import numpy as np


def minmax_normalize(candidates, score_map):
    ''
    n = len(candidates)
    present = [(j, score_map[c]) for j, c in enumerate(candidates) if c in score_map]
    if not present:
        return np.zeros(n)
    vals = np.array([v for _, v in present])
    vmin, vmax = vals.min(), vals.max()
    norm = np.zeros(n)
    if vmax > vmin:
        for j, v in present:
            norm[j] = (v - vmin) / (vmax - vmin)
    else:
        for j, _ in present:
            norm[j] = 1.0
    return norm


def compute_margin(norm_scores, top_n=5):
    ''
    if len(norm_scores) < 2:
        return 0
    sorted_s = np.sort(norm_scores)[::-1]
    top1 = sorted_s[0]
    rest = sorted_s[1:min(top_n + 1, len(sorted_s))]
    return top1 - rest.mean() if len(rest) > 0 else 0


def precompute_bigrams(doc_list):
    ''
    doc_bigrams = []
    for doc in doc_list:
        bigrams = set(doc[i:i + 2] for i in range(len(doc) - 1))
        doc_bigrams.append(bigrams)
    return doc_bigrams


def compute_query_bigram_overlap(query, doc_bigrams):
    ''
    q_bigrams = set(query[i:i + 2] for i in range(len(query) - 1))
    n_docs = len(doc_bigrams)
    overlap = np.zeros(n_docs)
    if not q_bigrams:
        return overlap
    for d in range(n_docs):
        overlap[d] = len(q_bigrams & doc_bigrams[d]) / len(q_bigrams)
    return overlap


def lvf_fusion(bm25_indices, bm25_scores, vec_indices, vec_scores, queries, doc_bigrams,
               alpha_base=0.4, alpha_range=0.10, alpha_scale=20,
               gamma=0.3, delta=0.2, top_k=100):
    ''
    n = len(bm25_indices)
    result = np.zeros((n, top_k), dtype=np.int32)

    for i in range(n):

        bm25_map = dict(zip(bm25_indices[i].tolist(), bm25_scores[i].tolist()))
        vec_map = dict(zip(vec_indices[i].tolist(), vec_scores[i].tolist()))
        candidates = list(dict.fromkeys(list(bm25_map.keys()) + list(vec_map.keys())))
        b_norm = minmax_normalize(candidates, bm25_map)
        v_norm = minmax_normalize(candidates, vec_map)





        b_margin = compute_margin(b_norm, 5)
        v_margin = compute_margin(v_norm, 5)
        diff = b_margin - v_margin
        alpha = alpha_base + alpha_range * (1 / (1 + np.exp(-diff * alpha_scale)))




        bigram_overlap = compute_query_bigram_overlap(queries[i], doc_bigrams)
        L = np.array([bigram_overlap[c] if c < len(bigram_overlap) else 0 for c in candidates])





        P = b_norm * v_norm


        final = alpha * b_norm + (1 - alpha) * v_norm + gamma * L - delta * P
        sorted_idx = np.argsort(-final)
        result[i] = [candidates[j] for j in sorted_idx[:top_k]]

    return result



def was_hybrid(bm25_idx, bm25_sc, vec_idx, vec_sc, alpha=0.4, top_k=100):
    ''
    n = len(bm25_idx)
    result = np.zeros((n, top_k), dtype=np.int32)
    for i in range(n):
        bm25_map = dict(zip(bm25_idx[i].tolist(), bm25_sc[i].tolist()))
        vec_map = dict(zip(vec_idx[i].tolist(), vec_sc[i].tolist()))
        candidates = list(dict.fromkeys(list(bm25_map.keys()) + list(vec_map.keys())))
        b_norm = minmax_normalize(candidates, bm25_map)
        v_norm = minmax_normalize(candidates, vec_map)
        final = alpha * b_norm + (1 - alpha) * v_norm
        sorted_idx = np.argsort(-final)
        result[i] = [candidates[j] for j in sorted_idx[:top_k]]
    return result


def rrf_fusion(bm25_idx, vec_idx, k=60, top_k=100):
    ''
    n = len(bm25_idx)
    result = np.zeros((n, top_k), dtype=np.int32)
    for i in range(n):
        scores = {}
        for rank, idx in enumerate(bm25_idx[i]):
            scores[int(idx)] = scores.get(int(idx), 0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(vec_idx[i]):
            scores[int(idx)] = scores.get(int(idx), 0) + 1.0 / (k + rank + 1)
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])
        result[i] = [idx for idx, _ in sorted_items[:top_k]]
    return result
