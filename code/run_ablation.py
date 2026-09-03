"""
LVF 组件消融实验
=================
LVF(d) = α(q)·ŝ_b(d) + (1-α(q))·ŝ_v(d) + γ·L(d) - δ·P(d)

消融配置:
  1. WAS(baseline)      : 固定α=0.4, γ=0, δ=0
  2. +Adaptive          : 自适应α(q), γ=0, δ=0
  3. +Lexical           : 自适应α(q) + L(d), γ=0.3, δ=0
  4. +Suspicion         : 自适应α(q) - P(d), γ=0, δ=0.2
  5. LVF(full)          : 自适应α(q) + L(d) - P(d), γ=0.3, δ=0.2

在4个数据集上对Finetune4B-full-v3做消融, 复用已有向量编码缓存
"""
import json, os, time, gc, math
import torch
import numpy as np

os.environ['TRITON_DISABLE_CUDA_KERNEL'] = '1'

CACHE_DIR    = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/cache'
OUTPUT_DIR   = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/results'
SEED = 42

# 数据集配置 (与run_v3_experiments.py一致)
DATASETS = [
    ('sanguo_test',  '/home/huxin/Documents/trae_projects/sikuBERT/sanguo_test_filtered_final.json',         1.0),
    ('shiji',        '/home/huxin/Documents/trae_projects/sikuBERT/史记合并-with-prompt-api-response-extracted-train.json', 0.1),
    ('hanshu',       '/home/huxin/Documents/trae_projects/sikuBERT/汉书合并mini-with-prompt-api-response-extracted-train.json', 0.1),
    ('history-10k',  '/home/huxin/Documents/trae_projects/sikuBERT/classical_history_retrieval_10k.json',   0.1),
]

MODEL_NAME = 'Finetune4B-full-v3'

# 消融配置: (名称, alpha_base, alpha_range, gamma, delta)
# alpha_range=0 表示固定alpha=alpha_base
ABLATION_CONFIGS = [
    ('WAS(baseline)',    0.4,  0.0,  0.0,  0.0),
    ('+Adaptive-α',      0.4,  0.10, 0.0,  0.0),
    ('+Lexical-L',       0.4,  0.10, 0.3,  0.0),
    ('+Suspicion-P',     0.4,  0.10, 0.0,  0.2),
    ('LVF(full)',        0.4,  0.10, 0.3,  0.2),
]


# ==================== 数据加载 ====================
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


# ==================== BM25 ====================
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


# ==================== 融合工具 ====================
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


# ==================== 消融融合函数 ====================
def ablation_fusion(bm25_indices, bm25_scores, vec_indices, vec_scores, queries, doc_bigrams,
                    alpha_base=0.4, alpha_range=0.10, alpha_scale=20,
                    gamma=0.3, delta=0.2, top_k=100):
    """
    统一消融融合函数:
    score(d) = α(q)·ŝ_b(d) + (1-α(q))·ŝ_v(d) + γ·L(d) - δ·P(d)

    α(q) = alpha_base + alpha_range·σ((margin_b - margin_v)·alpha_scale)  [alpha_range=0则固定α]
    L(d) = bigram_overlap(q, d)                                            [gamma=0则关闭]
    P(d) = ŝ_b(d) × ŝ_v(d)                                                [delta=0则关闭]
    """
    n = len(bm25_indices)
    result = np.zeros((n, top_k), dtype=np.int32)
    for i in range(n):
        bm25_map = dict(zip(bm25_indices[i].tolist(), bm25_scores[i].tolist()))
        vec_map = dict(zip(vec_indices[i].tolist(), vec_scores[i].tolist()))
        candidates = list(dict.fromkeys(list(bm25_map.keys()) + list(vec_map.keys())))
        b_norm = _minmax_normalize(candidates, bm25_map)
        v_norm = _minmax_normalize(candidates, vec_map)

        # α(q): 查询自适应权重
        if alpha_range > 0:
            b_margin = _compute_margin(b_norm, 5)
            v_margin = _compute_margin(v_norm, 5)
            diff = b_margin - v_margin
            alpha = alpha_base + alpha_range * (1 / (1 + np.exp(-diff * alpha_scale)))
        else:
            alpha = alpha_base

        # L(d): 词汇验证 (字符bigram重叠)
        if gamma > 0:
            bigram_overlap = compute_query_bigram_overlap(queries[i], doc_bigrams)
            L = np.array([bigram_overlap[c] if c < len(bigram_overlap) else 0 for c in candidates])
        else:
            L = np.zeros(len(candidates))

        # P(d): 跨模态怀疑度
        if delta > 0:
            P = b_norm * v_norm
        else:
            P = np.zeros(len(candidates))

        final = alpha * b_norm + (1 - alpha) * v_norm + gamma * L - delta * P
        sorted_idx = np.argsort(-final)
        result[i] = [candidates[j] for j in sorted_idx[:top_k]]
    return result


# ==================== 主流程 ====================
def main():
    t0_all = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 100)
    print('LVF 组件消融实验')
    print(f'模型: {MODEL_NAME}')
    print(f'数据集: {[d[0] for d in DATASETS]}')
    print(f'消融配置: {len(ABLATION_CONFIGS)}种')
    print('=' * 100, flush=True)

    all_results = {}

    for ds_name, ds_path, test_ratio in DATASETS:
        print(f'\n==== 数据集: {ds_name} ====', flush=True)
        queries, doc_list, pos_idx = load_dataset(ds_path, test_ratio=test_ratio, seed=SEED)
        print(f'  数据: {len(queries)} 查询, {len(doc_list)} 文档', flush=True)

        # BM25
        print(f'  构建BM25 ...', end=' ', flush=True)
        t0 = time.time()
        bm25_search = build_bm25(doc_list)
        bm25_idx, bm25_sc = bm25_search(queries)
        print(f'{time.time()-t0:.1f}s', flush=True)

        # bigram
        print(f'  预计算bigram ...', end=' ', flush=True)
        t0 = time.time()
        doc_bigrams = precompute_bigrams(doc_list)
        print(f'{time.time()-t0:.1f}s', flush=True)

        # 加载向量编码缓存
        safe_name = MODEL_NAME.replace('/', '-').replace(' ', '_').replace('(', '').replace(')', '')
        doc_cache = f'{CACHE_DIR}/v3_{ds_name}_{safe_name}_docs.npy'
        q_cache   = f'{CACHE_DIR}/v3_{ds_name}_{safe_name}_queries.npy'

        if os.path.exists(doc_cache) and os.path.exists(q_cache):
            doc_embs = np.load(doc_cache)
            q_embs = np.load(q_cache)
            print(f'  加载向量缓存: {safe_name}', flush=True)
        else:
            print(f'  [错误] 缓存不存在: {doc_cache}', flush=True)
            continue

        # 向量检索
        import faiss
        index = faiss.IndexFlatIP(doc_embs.shape[1])
        index.add(doc_embs.astype(np.float32))
        vec_scores, vec_indices = index.search(q_embs.astype(np.float32), 100)

        # 也计算BM25和Vector的单独指标
        ds_results = {}
        m_bm25 = compute_metrics(bm25_idx, pos_idx)
        ds_results['BM25'] = m_bm25
        m_vec = compute_metrics(vec_indices, pos_idx)
        ds_results['Vector'] = m_vec
        print(f'  BM25    R@1={m_bm25["R@1"]:.4f}', flush=True)
        print(f'  Vector  R@1={m_vec["R@1"]:.4f}', flush=True)

        # 消融实验
        was_r1 = None
        for config_name, a_base, a_range, g, d in ABLATION_CONFIGS:
            idx = ablation_fusion(bm25_idx, bm25_sc, vec_indices, vec_scores, queries, doc_bigrams,
                                  alpha_base=a_base, alpha_range=a_range,
                                  gamma=g, delta=d)
            m = compute_metrics(idx, pos_idx)
            ds_results[config_name] = m

            if config_name == 'WAS(baseline)':
                was_r1 = m['R@1']
                print(f'  {config_name:<20} R@1={m["R@1"]:.4f} R@5={m["R@5"]:.4f}  (baseline)', flush=True)
            else:
                gain = (m['R@1'] - was_r1) * 100 if was_r1 else 0
                print(f'  {config_name:<20} R@1={m["R@1"]:.4f} R@5={m["R@5"]:.4f}  ΔR@1(WAS)={gain:+.2f}%', flush=True)

        all_results[ds_name] = ds_results

    # 保存结果
    out_path = f'{OUTPUT_DIR}/v3_ablation_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'description': 'LVF组件消融实验: 自适应α / 词汇验证L / 跨模态怀疑度P',
            'model': MODEL_NAME,
            'lvf_formula': 'LVF(d) = α(q)·ŝ_b(d) + (1-α(q))·ŝ_v(d) + γ·L(d) - δ·P(d)',
            'ablation_configs': {
                name: {'alpha_base': a, 'alpha_range': r, 'gamma': g, 'delta': d}
                for name, a, r, g, d in ABLATION_CONFIGS
            },
            'results': all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f'\n消融结果已保存: {out_path}', flush=True)

    # 打印汇总表
    print('\n' + '=' * 100)
    print('LVF 组件消融实验汇总 (R@1)')
    print('=' * 100)
    configs = ['WAS(baseline)', '+Adaptive-α', '+Lexical-L', '+Suspicion-P', 'LVF(full)']
    print(f'{"数据集":<16}', end='')
    for c in configs:
        print(f' {c:>16}', end='')
    print()
    print('-' * 100)
    for ds_name in all_results:
        was_r1 = all_results[ds_name]['WAS(baseline)']['R@1']
        print(f'{ds_name:<16}', end='')
        for c in configs:
            r1 = all_results[ds_name][c]['R@1']
            if c == 'WAS(baseline)':
                print(f' {r1:>16.4f}', end='')
            else:
                gain = (r1 - was_r1) * 100
                print(f' {r1:>10.4f}({gain:+.1f}%)', end='')
        print()
    print('-' * 100)

    print(f'\n总耗时: {(time.time()-t0_all)/60:.1f} 分钟', flush=True)


if __name__ == '__main__':
    main()
