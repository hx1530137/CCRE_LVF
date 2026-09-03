'Supplementary evaluation and analysis routines.'
import json, os, time, math
import numpy as np
from scipy import stats

CACHE_DIR    = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/cache'
OUTPUT_DIR   = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/results'
SEED = 42

DATASETS = [
    ('sanguo_test',  '/home/huxin/Documents/trae_projects/sikuBERT/sanguo_test_filtered_final.json',         1.0),
    ('shiji',        '/home/huxin/Documents/trae_projects/sikuBERT/史记合并-with-prompt-api-response-extracted-train.json', 0.1),
    ('hanshu',       '/home/huxin/Documents/trae_projects/sikuBERT/汉书合并mini-with-prompt-api-response-extracted-train.json', 0.1),
]

MODEL_NAME = 'Finetune4B-full-v3'


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


def compute_recall_at_k(topk_indices, positive_indices, k_values=[1, 3, 5, 10]):
    n = len(topk_indices)
    m = {}
    for k in k_values:
        hit = sum(1 for i in range(n) if positive_indices[i] in topk_indices[i][:k])
        m[f'R@{k}'] = hit / n
    return m


def compute_mrr(topk_indices, positive_indices):
    n = len(topk_indices)
    total = 0
    for i in range(n):
        if positive_indices[i] in topk_indices[i]:
            rank = list(topk_indices[i]).index(positive_indices[i]) + 1
            total += 1.0 / rank
    return total / n


def compute_ndcg_at_k(topk_indices, positive_indices, k=10):
    'nDCG@k (二元相关性: 正确文档rel=1, 其他rel=0)'
    n = len(topk_indices)
    total = 0
    for i in range(n):
        topk = topk_indices[i][:k]
        dcg = 0
        for rank, doc_idx in enumerate(topk):
            if doc_idx == positive_indices[i]:
                dcg += 1.0 / math.log2(rank + 2)
        idcg = 1.0
        total += dcg / idcg
    return total / n


def compute_per_query_hit(topk_indices, positive_indices, k=1):
    '返回每个查询是否命中(top-k), 用于统计显著性检验'
    return np.array([1 if positive_indices[i] in topk_indices[i][:k] else 0
                     for i in range(len(topk_indices))])


def paired_t_test(method_a_hits, method_b_hits, method_a_name, method_b_name):
    '配对t检验'
    diff = method_a_hits - method_b_hits
    if np.std(diff) == 0:
        return {'t_statistic': 0, 'p_value': 1.0, 'mean_diff': 0, 'significant': False}
    t_stat, p_value = stats.ttest_rel(method_a_hits, method_b_hits)
    cohen_d = diff.mean() / diff.std() if diff.std() > 0 else 0
    return {
        'comparison': f'{method_a_name} vs {method_b_name}',
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'mean_diff': float(diff.mean()),
        'cohen_d': float(cohen_d),
        'significant': bool(p_value < 0.05),
        'n_queries': len(diff),
    }


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


def was_hybrid(bm25_idx, bm25_sc, vec_idx, vec_sc, alpha=0.4, top_k=100):
    n = len(bm25_idx)
    result = np.zeros((n, top_k), dtype=np.int32)
    for i in range(n):
        bm25_map = dict(zip(bm25_idx[i].tolist(), bm25_sc[i].tolist()))
        vec_map = dict(zip(vec_idx[i].tolist(), vec_sc[i].tolist()))
        candidates = list(dict.fromkeys(list(bm25_map.keys()) + list(vec_map.keys())))
        b_norm = _minmax_normalize(candidates, bm25_map)
        v_norm = _minmax_normalize(candidates, vec_map)
        final = alpha * b_norm + (1 - alpha) * v_norm
        sorted_idx = np.argsort(-final)
        result[i] = [candidates[j] for j in sorted_idx[:top_k]]
    return result


def rrf_fusion(bm25_idx, vec_idx, k=60, top_k=100):
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


def lvf_fusion(bm25_indices, bm25_scores, vec_indices, vec_scores, queries, doc_bigrams,
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


def evaluate_all_metrics(topk_indices, positive_indices):
    '计算全套指标: R@1/3/5/10, MRR, nDCG@10'
    m = compute_recall_at_k(topk_indices, positive_indices)
    m['MRR'] = compute_mrr(topk_indices, positive_indices)
    m['nDCG@10'] = compute_ndcg_at_k(topk_indices, positive_indices, 10)
    return m


def run_metrics_and_significance(all_data):
    print('\n' + '=' * 100)
    print('实验1: nDCG@10指标 + 统计显著性检验')
    print('=' * 100, flush=True)

    results = {}
    for ds_name, queries, doc_list, pos_idx, doc_bigrams, bm25_idx, bm25_sc, vec_idx, vec_sc in all_data:
        print(f'\n  ---- {ds_name} ----', flush=True)
        ds_result = {}


        methods = {
            'BM25': bm25_idx,
            'Vector': vec_idx,
            'RRF': rrf_fusion(bm25_idx, vec_idx, k=60),
            'WAS': was_hybrid(bm25_idx, bm25_sc, vec_idx, vec_sc, alpha=0.4),
            'LVF': lvf_fusion(bm25_idx, bm25_sc, vec_idx, vec_sc, queries, doc_bigrams),
        }

        for name, idx in methods.items():
            m = evaluate_all_metrics(idx, pos_idx)
            ds_result[name] = m
            print(f'    {name:<10} R@1={m["R@1"]:.4f} R@5={m["R@5"]:.4f} R@10={m["R@10"]:.4f} '
                  f'MRR={m["MRR"]:.4f} nDCG@10={m["nDCG@10"]:.4f}', flush=True)


        lvf_hits_1 = compute_per_query_hit(methods['LVF'], pos_idx, k=1)
        was_hits_1 = compute_per_query_hit(methods['WAS'], pos_idx, k=1)
        rrf_hits_1 = compute_per_query_hit(methods['RRF'], pos_idx, k=1)
        vec_hits_1 = compute_per_query_hit(methods['Vector'], pos_idx, k=1)

        sig_tests = []
        for name, hits in [('WAS', was_hits_1), ('RRF', rrf_hits_1), ('Vector', vec_hits_1)]:
            test = paired_t_test(lvf_hits_1, hits, 'LVF', name)
            sig_tests.append(test)
            sig_str = '***' if test['p_value'] < 0.001 else '**' if test['p_value'] < 0.01 else '*' if test['p_value'] < 0.05 else 'ns'
            print(f'    显著性: LVF vs {name:<8} t={test["t_statistic"]:+.3f} p={test["p_value"]:.4f} '
                  f'd={test["cohen_d"]:+.3f} {sig_str}', flush=True)

        ds_result['significance_tests'] = sig_tests
        results[ds_name] = ds_result

    return results


def run_efficiency_analysis(all_data):
    print('\n' + '=' * 100)
    print('实验2: 效率/延迟分析')
    print('=' * 100, flush=True)

    results = {}
    for ds_name, queries, doc_list, pos_idx, doc_bigrams, bm25_idx, bm25_sc, vec_idx, vec_sc in all_data:
        print(f'\n  ---- {ds_name} ({len(queries)} 查询) ----', flush=True)
        n_subset = min(200, len(queries))
        idx_subset = list(range(n_subset))

        timings = {}


        from rank_bm25 import BM25Okapi
        import jieba
        tokenized = [list(jieba.cut(doc)) for doc in doc_list]
        bm25 = BM25Okapi(tokenized)
        t0 = time.time()
        for i in idx_subset:
            sc = bm25.get_scores(list(jieba.cut(queries[i])))
        timings['BM25_search'] = (time.time() - t0) / n_subset * 1000


        import faiss
        safe_name = MODEL_NAME.replace('/', '-').replace(' ', '_').replace('(', '').replace(')', '')
        doc_embs = np.load(f'{CACHE_DIR}/v3_{ds_name}_{safe_name}_docs.npy')
        q_embs = np.load(f'{CACHE_DIR}/v3_{ds_name}_{safe_name}_queries.npy')
        index = faiss.IndexFlatIP(doc_embs.shape[1])
        index.add(doc_embs.astype(np.float32))
        t0 = time.time()
        for i in idx_subset:
            index.search(q_embs[i:i+1].astype(np.float32), 100)
        timings['Vector_search'] = (time.time() - t0) / n_subset * 1000


        bm25_idx_sub = bm25_idx[idx_subset]
        bm25_sc_sub = bm25_sc[idx_subset]
        vec_idx_sub = vec_idx[idx_subset]
        vec_sc_sub = vec_sc[idx_subset]
        queries_sub = [queries[i] for i in idx_subset]
        doc_bigrams_sub = doc_bigrams

        t0 = time.time()
        was_hybrid(bm25_idx_sub, bm25_sc_sub, vec_idx_sub, vec_sc_sub, alpha=0.4)
        timings['WAS_fusion'] = (time.time() - t0) / n_subset * 1000

        t0 = time.time()
        rrf_fusion(bm25_idx_sub, vec_idx_sub, k=60)
        timings['RRF_fusion'] = (time.time() - t0) / n_subset * 1000

        t0 = time.time()
        lvf_fusion(bm25_idx_sub, bm25_sc_sub, vec_idx_sub, vec_sc_sub, queries_sub, doc_bigrams_sub)
        timings['LVF_fusion'] = (time.time() - t0) / n_subset * 1000

        timings['LVF_total'] = timings['BM25_search'] + timings['Vector_search'] + timings['LVF_fusion']
        timings['WAS_total'] = timings['BM25_search'] + timings['Vector_search'] + timings['WAS_fusion']

        for k, v in timings.items():
            print(f'    {k:<20} {v:.3f} ms/query', flush=True)

        results[ds_name] = timings

    return results


def run_hyperparam_sensitivity(all_data):
    print('\n' + '=' * 100)
    print('实验3: 超参数敏感性分析')
    print('=' * 100, flush=True)


    ds_name, queries, doc_list, pos_idx, doc_bigrams, bm25_idx, bm25_sc, vec_idx, vec_sc = all_data[0]
    print(f'\n  数据集: {ds_name}', flush=True)

    results = {'gamma_sweep': {}, 'delta_sweep': {}, 'alpha_range_sweep': {}}


    print(f'\n  --- γ (词汇验证权重) 扫描 ---', flush=True)
    for gamma in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        idx = lvf_fusion(bm25_idx, bm25_sc, vec_idx, vec_sc, queries, doc_bigrams,
                         gamma=gamma, delta=0.2)
        m = compute_recall_at_k(idx, pos_idx)
        results['gamma_sweep'][gamma] = m['R@1']
        print(f'    γ={gamma:.1f}  R@1={m["R@1"]:.4f}', flush=True)


    print(f'\n  --- δ (跨模态怀疑度) 扫描 ---', flush=True)
    for delta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        idx = lvf_fusion(bm25_idx, bm25_sc, vec_idx, vec_sc, queries, doc_bigrams,
                         gamma=0.3, delta=delta)
        m = compute_recall_at_k(idx, pos_idx)
        results['delta_sweep'][delta] = m['R@1']
        print(f'    δ={delta:.1f}  R@1={m["R@1"]:.4f}', flush=True)


    print(f'\n  --- α_range (自适应范围) 扫描 ---', flush=True)
    for alpha_range in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        idx = lvf_fusion(bm25_idx, bm25_sc, vec_idx, vec_sc, queries, doc_bigrams,
                         alpha_range=alpha_range, gamma=0.3, delta=0.2)
        m = compute_recall_at_k(idx, pos_idx)
        results['alpha_range_sweep'][alpha_range] = m['R@1']
        print(f'    α_range={alpha_range:.2f}  R@1={m["R@1"]:.4f}', flush=True)


    print(f'\n  --- γ × δ 二维网格 ---', flush=True)
    grid = {}
    for gamma in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for delta in [0.1, 0.2, 0.3, 0.4]:
            idx = lvf_fusion(bm25_idx, bm25_sc, vec_idx, vec_sc, queries, doc_bigrams,
                             gamma=gamma, delta=delta)
            m = compute_recall_at_k(idx, pos_idx)
            grid[f'g{gamma}_d{delta}'] = m['R@1']
    results['gamma_delta_grid'] = grid

    best_key = max(grid, key=grid.get)
    print(f'    最优组合: {best_key} R@1={grid[best_key]:.4f}', flush=True)
    print(f'    默认组合(g0.3_d0.2) R@1={grid["g0.3_d0.2"]:.4f}', flush=True)

    return results


def run_rrf_k_comparison(all_data):
    print('\n' + '=' * 100)
    print('实验4: RRF不同k值对比')
    print('=' * 100, flush=True)

    results = {}
    for ds_name, queries, doc_list, pos_idx, doc_bigrams, bm25_idx, bm25_sc, vec_idx, vec_sc in all_data:
        print(f'\n  ---- {ds_name} ----', flush=True)
        ds_result = {}
        for k in [1, 10, 30, 60, 100, 200]:
            idx = rrf_fusion(bm25_idx, vec_idx, k=k)
            m = compute_recall_at_k(idx, pos_idx)
            ds_result[f'k={k}'] = m['R@1']
            print(f'    RRF k={k:<4}  R@1={m["R@1"]:.4f}', flush=True)


        idx = lvf_fusion(bm25_idx, bm25_sc, vec_idx, vec_sc, queries, doc_bigrams)
        m = compute_recall_at_k(idx, pos_idx)
        ds_result['LVF'] = m['R@1']
        print(f'    LVF         R@1={m["R@1"]:.4f}  (对比)', flush=True)

        results[ds_name] = ds_result

    return results


def run_case_study(all_data):
    print('\n' + '=' * 100)
    print('实验5: Case Study (成功/失败案例)')
    print('=' * 100, flush=True)

    ds_name, queries, doc_list, pos_idx, doc_bigrams, bm25_idx, bm25_sc, vec_idx, vec_sc = all_data[0]

    was_idx = was_hybrid(bm25_idx, bm25_sc, vec_idx, vec_sc, alpha=0.4)
    lvf_idx = lvf_fusion(bm25_idx, bm25_sc, vec_idx, vec_sc, queries, doc_bigrams)

    was_hits = compute_per_query_hit(was_idx, pos_idx, k=1)
    lvf_hits = compute_per_query_hit(lvf_idx, pos_idx, k=1)


    lvf_win = np.where((lvf_hits == 1) & (was_hits == 0))[0]

    was_win = np.where((lvf_hits == 0) & (was_hits == 1))[0]

    both_fail = np.where((lvf_hits == 0) & (was_hits == 0))[0]

    cases = {'lvf_wins': [], 'was_wins': [], 'both_fail': []}

    for i in lvf_win[:3]:
        cases['lvf_wins'].append({
            'query': queries[i][:100],
            'correct_doc_preview': doc_list[pos_idx[i]][:100],
            'was_top1_preview': doc_list[was_idx[i][0]][:100],
            'lvf_top1_preview': doc_list[lvf_idx[i][0]][:100],
            'description': 'LVF通过词汇验证/怀疑度惩罚将正确文档提升到top1'
        })

    for i in was_win[:3]:
        cases['was_wins'].append({
            'query': queries[i][:100],
            'correct_doc_preview': doc_list[pos_idx[i]][:100],
            'was_top1_preview': doc_list[was_idx[i][0]][:100],
            'lvf_top1_preview': doc_list[lvf_idx[i][0]][:100],
            'description': 'WAS在此案例中表现更好, LVF可能过度惩罚'
        })

    for i in both_fail[:3]:
        cases['both_fail'].append({
            'query': queries[i][:100],
            'correct_doc_preview': doc_list[pos_idx[i]][:100],
            'was_top1_preview': doc_list[was_idx[i][0]][:100],
            'lvf_top1_preview': doc_list[lvf_idx[i][0]][:100],
            'description': '两者均未命中, 可能是向量检索和BM25都无法找到正确文档'
        })

    print(f'\n  LVF胜出案例: {len(lvf_win)}个 (展示前3个)', flush=True)
    for c in cases['lvf_wins']:
        print(f'    Q: {c["query"][:60]}...', flush=True)
        print(f'    → 正确文档排在LVF top1', flush=True)

    print(f'\n  WAS胜出案例: {len(was_win)}个', flush=True)
    print(f'  两者都失败: {len(both_fail)}个', flush=True)

    cases['summary'] = {
        'lvf_wins_count': int(len(lvf_win)),
        'was_wins_count': int(len(was_win)),
        'both_fail_count': int(len(both_fail)),
        'both_success_count': int(np.sum((lvf_hits == 1) & (was_hits == 1))),
    }

    return cases


def main():
    t0_all = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 100)
    print('论文补充实验: nDCG@10 + 显著性检验 + 效率分析 + 超参数敏感性 + RRF k值 + Case Study')
    print(f'模型: {MODEL_NAME}')
    print(f'数据集: {[d[0] for d in DATASETS]}')
    print('=' * 100, flush=True)


    all_data = []
    for ds_name, ds_path, test_ratio in DATASETS:
        print(f'\n加载 {ds_name} ...', end=' ', flush=True)
        queries, doc_list, pos_idx = load_dataset(ds_path, test_ratio=test_ratio, seed=SEED)
        bm25_search = build_bm25(doc_list)
        bm25_idx, bm25_sc = bm25_search(queries)
        doc_bigrams = precompute_bigrams(doc_list)

        safe_name = MODEL_NAME.replace('/', '-').replace(' ', '_').replace('(', '').replace(')', '')
        doc_cache = f'{CACHE_DIR}/v3_{ds_name}_{safe_name}_docs.npy'
        q_cache = f'{CACHE_DIR}/v3_{ds_name}_{safe_name}_queries.npy'
        doc_embs = np.load(doc_cache)
        q_embs = np.load(q_cache)

        import faiss
        index = faiss.IndexFlatIP(doc_embs.shape[1])
        index.add(doc_embs.astype(np.float32))
        vec_scores, vec_indices = index.search(q_embs.astype(np.float32), 100)

        all_data.append((ds_name, queries, doc_list, pos_idx, doc_bigrams, bm25_idx, bm25_sc, vec_indices, vec_scores))
        print(f'{len(queries)}查询, {len(doc_list)}文档', flush=True)


    metrics_results = run_metrics_and_significance(all_data)
    with open(f'{OUTPUT_DIR}/supplementary_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'description': 'nDCG@10指标 + 配对t检验', 'results': metrics_results},
                  f, ensure_ascii=False, indent=2)


    efficiency_results = run_efficiency_analysis(all_data)
    with open(f'{OUTPUT_DIR}/supplementary_efficiency.json', 'w', encoding='utf-8') as f:
        json.dump({'description': '效率/延迟分析 (ms/query)', 'results': efficiency_results},
                  f, ensure_ascii=False, indent=2)


    hyperparam_results = run_hyperparam_sensitivity(all_data)
    with open(f'{OUTPUT_DIR}/supplementary_hyperparam.json', 'w', encoding='utf-8') as f:
        json.dump({'description': '超参数敏感性分析', 'results': hyperparam_results},
                  f, ensure_ascii=False, indent=2)


    rrf_results = run_rrf_k_comparison(all_data)
    with open(f'{OUTPUT_DIR}/supplementary_rrf_k.json', 'w', encoding='utf-8') as f:
        json.dump({'description': 'RRF不同k值对比', 'results': rrf_results},
                  f, ensure_ascii=False, indent=2)


    case_results = run_case_study(all_data)
    with open(f'{OUTPUT_DIR}/supplementary_case_study.json', 'w', encoding='utf-8') as f:
        json.dump({'description': 'Case Study: 成功/失败案例', 'results': case_results},
                  f, ensure_ascii=False, indent=2)


    print('\n' + '=' * 100)
    print('补充实验汇总')
    print('=' * 100)

    print('\n表S1: 全套指标 (含nDCG@10)')
    print(f'{"数据集":<14} {"方法":<10} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"MRR":>8} {"nDCG@10":>10}')
    print('-' * 70)
    for ds_name in metrics_results:
        for method in ['BM25', 'Vector', 'RRF', 'WAS', 'LVF']:
            m = metrics_results[ds_name][method]
            mark = '★' if method == 'LVF' else ' '
            print(f'{ds_name:<14} {method:<10} {m["R@1"]:>8.4f} {m["R@5"]:>8.4f} {m["R@10"]:>8.4f} {m["MRR"]:>8.4f} {m["nDCG@10"]:>10.4f} {mark}')
        print('-' * 70)

    print('\n表S2: 统计显著性检验 (配对t检验, LVF vs baselines)')
    print(f'{"数据集":<14} {"对比":<16} {"t值":>8} {"p值":>10} {"Cohen_d":>10} {"显著":>6}')
    print('-' * 70)
    for ds_name in metrics_results:
        for test in metrics_results[ds_name]['significance_tests']:
            sig = '是' if test['significant'] else '否'
            print(f'{ds_name:<14} {test["comparison"]:<16} {test["t_statistic"]:>+8.3f} {test["p_value"]:>10.4f} {test["cohen_d"]:>+10.3f} {sig:>6}')
        print('-' * 70)

    print(f'\n总耗时: {(time.time()-t0_all)/60:.1f} 分钟', flush=True)
    print('\n所有补充实验结果已保存到:', OUTPUT_DIR, flush=True)


if __name__ == '__main__':
    main()
