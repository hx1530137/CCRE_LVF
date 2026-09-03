import json, os, time, gc, math
import torch
import numpy as np

os.environ['TRITON_DISABLE_CUDA_KERNEL'] = '1'

ROOT_DIR = os.environ.get('CCRE_LVF_ROOT', 'path/to/project')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CACHE_DIR = os.path.join(ROOT_DIR, 'cache')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
MODEL_ID = 'adapted_model'
DATASETS = [
    ('dataset_main', os.path.join(DATA_DIR, 'dataset_main.json'), 1.0),
    ('dataset_auxiliary_1', os.path.join(DATA_DIR, 'dataset_auxiliary_1.json'), 0.1),
    ('dataset_auxiliary_2', os.path.join(DATA_DIR, 'dataset_auxiliary_2.json'), 0.1),
]



ABLATION_CONFIGS = [
    ('WAS(baseline)',    0.4,  0.0,  0.0,  0.0),
    ('+Adaptive-α',      0.4,  0.10, 0.0,  0.0),
    ('+Lexical-L',       0.4,  0.10, 0.3,  0.0),
    ('+Suspicion-P',     0.4,  0.10, 0.0,  0.2),
    ('LVF(full)',        0.4,  0.10, 0.3,  0.2),
]



def load_dataset(data_path, test_ratio=1.0, seed=42):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if test_ratio < 1.0:
        np.random.seed(seed)
        indices = np.random.permutation(len(data))
        test_size = int(len(data) * test_ratio)
        data = [data[i] for i in indices[:test_size]]
    doc_id_map, doc_list = {}, []
    for item in data:
        if item['input'] not in doc_id_map:
            doc_id_map[item['input']] = len(doc_list)
            doc_list.append(item['input'])
    queries = [item['instruction'] for item in data]
    positive_indices = [doc_id_map[item['input']] for item in data]
    return queries, doc_list, positive_indices


def compute_metrics(topk_indices, positive_indices, k_values=[1, 3, 5]):
    n = len(topk_indices)
    m = {}
    for k in k_values:
        hit = sum(1 for i in range(n) if positive_indices[i] in topk_indices[i][:k])
        m[f'R@{k}'] = hit / n
    m['MRR'] = sum(1.0 / (list(topk_indices[i]).index(positive_indices[i]) + 1)
                   for i in range(n) if positive_indices[i] in topk_indices[i]) / n
    return m



def build_bm25(documents):
    from rank_bm25 import BM25Okapi
    import jieba
    tokenized = [list(jieba.cut(doc)) for doc in documents]
    bm25 = BM25Okapi(tokenized)
    def search_batch(queries, top_k=100):
        indices, scores = [], []
        for q in queries:
            sc = bm25.get_scores(list(jieba.cut(q)))
            idx = np.argsort(-sc)[:top_k]
            indices.append(idx)
            scores.append(sc[idx])
        return np.array(indices), np.array(scores)
    return search_batch



def _minmax_normalize(candidates, score_map):
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


def _compute_margin(norm_scores, top_n=5):
    if len(norm_scores) < 2:
        return 0
    sorted_s = np.sort(norm_scores)[::-1]
    top1 = sorted_s[0]
    rest = sorted_s[1:min(top_n + 1, len(sorted_s))]
    return top1 - rest.mean() if len(rest) > 0 else 0


def precompute_bigrams(doc_list):
    doc_bigrams = []
    for doc in doc_list:
        bigrams = set(doc[i:i + 2] for i in range(len(doc) - 1))
        doc_bigrams.append(bigrams)
    return doc_bigrams


def compute_query_bigram_overlap(query, doc_bigrams):
    q_bigrams = set(query[i:i + 2] for i in range(len(query) - 1))
    n_docs = len(doc_bigrams)
    overlap = np.zeros(n_docs)
    if not q_bigrams:
        return overlap
    for d in range(n_docs):
        overlap[d] = len(q_bigrams & doc_bigrams[d]) / len(q_bigrams)
    return overlap



def ablation_fusion(bm25_indices, bm25_scores, vec_indices, vec_scores, queries, doc_bigrams,
                    alpha_base=0.4, alpha_range=0.10, alpha_scale=20,
                    gamma=0.3, delta=0.2, top_k=100):
    n = len(bm25_indices)
    result = np.zeros((n, top_k), dtype=np.int32)
    for i in range(n):
        bm25_map = dict(zip(bm25_indices[i].tolist(), bm25_scores[i].tolist()))
        vec_map = dict(zip(vec_indices[i].tolist(), vec_scores[i].tolist()))
        candidates = list(dict.fromkeys(list(bm25_map.keys()) + list(vec_map.keys())))
        b_norm = _minmax_normalize(candidates, bm25_map)
        v_norm = _minmax_normalize(candidates, vec_map)


        if alpha_range > 0:
            b_margin = _compute_margin(b_norm, 5)
            v_margin = _compute_margin(v_norm, 5)
            diff = b_margin - v_margin
            alpha = alpha_base + alpha_range * (1 / (1 + np.exp(-diff * alpha_scale)))
        else:
            alpha = alpha_base


        if gamma > 0:
            bigram_overlap = compute_query_bigram_overlap(queries[i], doc_bigrams)
            L = np.array([bigram_overlap[c] if c < len(bigram_overlap) else 0 for c in candidates])
        else:
            L = np.zeros(len(candidates))


        if delta > 0:
            P = b_norm * v_norm
        else:
            P = np.zeros(len(candidates))

        final = alpha * b_norm + (1 - alpha) * v_norm + gamma * L - delta * P
        sorted_idx = np.argsort(-final)
        result[i] = [candidates[j] for j in sorted_idx[:top_k]]
    return result



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}
    for ds_name, ds_path, test_ratio in DATASETS:
        queries, doc_list, pos_idx = load_dataset(ds_path, test_ratio=test_ratio, seed=SEED)
        bm25_search = build_bm25(doc_list)
        bm25_idx, bm25_sc = bm25_search(queries)
        doc_bigrams = precompute_bigrams(doc_list)
        safe_name = MODEL_ID.replace('/', '-').replace(' ', '_')
        doc_cache = f'{CACHE_DIR}/run_{ds_name}_{safe_name}_docs.npy'
        q_cache = f'{CACHE_DIR}/run_{ds_name}_{safe_name}_queries.npy'
        if not (os.path.exists(doc_cache) and os.path.exists(q_cache)):
            continue
        doc_embs, q_embs = np.load(doc_cache), np.load(q_cache)
        import faiss
        index = faiss.IndexFlatIP(doc_embs.shape[1])
        index.add(doc_embs.astype(np.float32))
        vec_scores, vec_indices = index.search(q_embs.astype(np.float32), 100)
        ds_results = {'BM25': compute_metrics(bm25_idx, pos_idx), 'Vector': compute_metrics(vec_indices, pos_idx)}
        for config_name, a_base, a_range, gamma, delta in ABLATION_CONFIGS:
            ranked = ablation_fusion(bm25_idx, bm25_sc, vec_indices, vec_scores, queries, doc_bigrams,
                                     alpha_base=a_base, alpha_range=a_range, gamma=gamma, delta=delta)
            ds_results[config_name] = compute_metrics(ranked, pos_idx)
        all_results[ds_name] = ds_results
    with open(os.path.join(OUTPUT_DIR, 'ablation_results.json'), 'w', encoding='utf-8') as handle:
        json.dump({'results': all_results}, handle, ensure_ascii=False, indent=2)
    print(json.dumps(all_results, ensure_ascii=False))


if __name__ == '__main__':
    main()
