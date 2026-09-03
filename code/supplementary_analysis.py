import json, os, time, math
import numpy as np
from scipy import stats

ROOT_DIR = os.environ.get('CCRE_LVF_ROOT', 'path/to/project')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CACHE_DIR = os.path.join(ROOT_DIR, 'cache')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
SEED = 42
DATASETS = [
    ('dataset_main', os.path.join(DATA_DIR, 'dataset_main.json'), 1.0),
    ('dataset_auxiliary_1', os.path.join(DATA_DIR, 'dataset_auxiliary_1.json'), 0.1),
    ('dataset_auxiliary_2', os.path.join(DATA_DIR, 'dataset_auxiliary_2.json'), 0.1),
]
MODEL_NAME = 'adapted_model'



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
    return np.array([1 if positive_indices[i] in topk_indices[i][:k] else 0
                     for i in range(len(topk_indices))])


def paired_t_test(method_a_hits, method_b_hits, method_a_name, method_b_name):
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
    m = compute_recall_at_k(topk_indices, positive_indices)
    m['MRR'] = compute_mrr(topk_indices, positive_indices)
    m['nDCG@10'] = compute_ndcg_at_k(topk_indices, positive_indices, 10)
    return m



def run_metrics_and_significance(all_data):




    results = {}
    for ds_name, queries, doc_list, pos_idx, doc_bigrams, bm25_idx, bm25_sc, vec_idx, vec_sc in all_data:

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



        
        lvf_hits_1 = compute_per_query_hit(methods['LVF'], pos_idx, k=1)
        was_hits_1 = compute_per_query_hit(methods['WAS'], pos_idx, k=1)
        rrf_hits_1 = compute_per_query_hit(methods['RRF'], pos_idx, k=1)
        vec_hits_1 = compute_per_query_hit(methods['Vector'], pos_idx, k=1)

        sig_tests = []
        for name, hits in [('WAS', was_hits_1), ('RRF', rrf_hits_1), ('Vector', vec_hits_1)]:
            test = paired_t_test(lvf_hits_1, hits, 'LVF', name)
            sig_tests.append(test)
            sig_str = '***' if test['p_value'] < 0.001 else '**' if test['p_value'] < 0.01 else '*' if test['p_value'] < 0.05 else 'ns'



        ds_result['significance_tests'] = sig_tests
        results[ds_name] = ds_result

    return results



def run_efficiency_analysis(all_data):




    results = {}
    for ds_name, queries, doc_list, pos_idx, doc_bigrams, bm25_idx, bm25_sc, vec_idx, vec_sc in all_data:

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
        doc_embs = np.load(f'{CACHE_DIR}/run_{ds_name}_{safe_name}_docs.npy')
        q_embs = np.load(f'{CACHE_DIR}/run_{ds_name}_{safe_name}_queries.npy')
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
            pass

        results[ds_name] = timings

    return results



def run_hyperparam_sensitivity(all_data):




    
    ds_name, queries, doc_list, pos_idx, doc_bigrams, bm25_idx, bm25_sc, vec_idx, vec_sc = all_data[0]


    results = {'gamma_sweep': {}, 'delta_sweep': {}, 'alpha_range_sweep': {}}

    

    for gamma in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        idx = lvf_fusion(bm25_idx, bm25_sc, vec_idx, vec_sc, queries, doc_bigrams,
                         gamma=gamma, delta=0.2)
        m = compute_recall_at_k(idx, pos_idx)
        results['gamma_sweep'][gamma] = m['R@1']


    

    for delta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        idx = lvf_fusion(bm25_idx, bm25_sc, vec_idx, vec_sc, queries, doc_bigrams,
                         gamma=0.3, delta=delta)
        m = compute_recall_at_k(idx, pos_idx)
        results['delta_sweep'][delta] = m['R@1']


    

    for alpha_range in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        idx = lvf_fusion(bm25_idx, bm25_sc, vec_idx, vec_sc, queries, doc_bigrams,
                         alpha_range=alpha_range, gamma=0.3, delta=0.2)
        m = compute_recall_at_k(idx, pos_idx)
        results['alpha_range_sweep'][alpha_range] = m['R@1']


    

    grid = {}
    for gamma in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for delta in [0.1, 0.2, 0.3, 0.4]:
            idx = lvf_fusion(bm25_idx, bm25_sc, vec_idx, vec_sc, queries, doc_bigrams,
                             gamma=gamma, delta=delta)
            m = compute_recall_at_k(idx, pos_idx)
            grid[f'g{gamma}_d{delta}'] = m['R@1']
    results['gamma_delta_grid'] = grid
    
    best_key = max(grid, key=grid.get)



    return results



def run_rrf_k_comparison(all_data):




    results = {}
    for ds_name, queries, doc_list, pos_idx, doc_bigrams, bm25_idx, bm25_sc, vec_idx, vec_sc in all_data:

        ds_result = {}
        for k in [1, 10, 30, 60, 100, 200]:
            idx = rrf_fusion(bm25_idx, vec_idx, k=k)
            m = compute_recall_at_k(idx, pos_idx)
            ds_result[f'k={k}'] = m['R@1']


        
        idx = lvf_fusion(bm25_idx, bm25_sc, vec_idx, vec_sc, queries, doc_bigrams)
        m = compute_recall_at_k(idx, pos_idx)
        ds_result['LVF'] = m['R@1']


        results[ds_name] = ds_result

    return results



def run_case_study(all_data):




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
            'description': 'LVF ranks the relevant document first'
        })

    for i in was_win[:3]:
        cases['was_wins'].append({
            'query': queries[i][:100],
            'correct_doc_preview': doc_list[pos_idx[i]][:100],
            'was_top1_preview': doc_list[was_idx[i][0]][:100],
            'lvf_top1_preview': doc_list[lvf_idx[i][0]][:100],
            'description': 'The fixed-weight baseline ranks the relevant document first'
        })

    for i in both_fail[:3]:
        cases['both_fail'].append({
            'query': queries[i][:100],
            'correct_doc_preview': doc_list[pos_idx[i]][:100],
            'was_top1_preview': doc_list[was_idx[i][0]][:100],
            'lvf_top1_preview': doc_list[lvf_idx[i][0]][:100],
            'description': 'Neither method ranks the relevant document first'
        })


    for c in cases['lvf_wins']:
        pass





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







    
    all_data = []
    for ds_name, ds_path, test_ratio in DATASETS:

        queries, doc_list, pos_idx = load_dataset(ds_path, test_ratio=test_ratio, seed=SEED)
        bm25_search = build_bm25(doc_list)
        bm25_idx, bm25_sc = bm25_search(queries)
        doc_bigrams = precompute_bigrams(doc_list)

        safe_name = MODEL_NAME.replace('/', '-').replace(' ', '_').replace('(', '').replace(')', '')
        doc_cache = f'{CACHE_DIR}/run_{ds_name}_{safe_name}_docs.npy'
        q_cache = f'{CACHE_DIR}/run_{ds_name}_{safe_name}_queries.npy'
        doc_embs = np.load(doc_cache)
        q_embs = np.load(q_cache)

        import faiss
        index = faiss.IndexFlatIP(doc_embs.shape[1])
        index.add(doc_embs.astype(np.float32))
        vec_scores, vec_indices = index.search(q_embs.astype(np.float32), 100)

        all_data.append((ds_name, queries, doc_list, pos_idx, doc_bigrams, bm25_idx, bm25_sc, vec_indices, vec_scores))


    
    metrics_results = run_metrics_and_significance(all_data)
    with open(f'{OUTPUT_DIR}/supplementary_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'description': 'ranking metrics and significance', 'results': metrics_results},
                  f, ensure_ascii=False, indent=2)

    
    efficiency_results = run_efficiency_analysis(all_data)
    with open(f'{OUTPUT_DIR}/supplementary_efficiency.json', 'w', encoding='utf-8') as f:
        json.dump({'description': 'efficiency analysis', 'results': efficiency_results},
                  f, ensure_ascii=False, indent=2)

    
    hyperparam_results = run_hyperparam_sensitivity(all_data)
    with open(f'{OUTPUT_DIR}/supplementary_hyperparam.json', 'w', encoding='utf-8') as f:
        json.dump({'description': 'hyperparameter sensitivity', 'results': hyperparam_results},
                  f, ensure_ascii=False, indent=2)

    
    rrf_results = run_rrf_k_comparison(all_data)
    with open(f'{OUTPUT_DIR}/supplementary_rrf_k.json', 'w', encoding='utf-8') as f:
        json.dump({'description': 'RRF parameter comparison', 'results': rrf_results},
                  f, ensure_ascii=False, indent=2)

    
    case_results = run_case_study(all_data)
    with open(f'{OUTPUT_DIR}/supplementary_case_study.json', 'w', encoding='utf-8') as f:
        json.dump({'description': 'case study', 'results': case_results},
                  f, ensure_ascii=False, indent=2)

    







    for ds_name in metrics_results:
        for method in ['BM25', 'Vector', 'RRF', 'WAS', 'LVF']:
            m = metrics_results[ds_name][method]
            mark = '★' if method == 'LVF' else ' '






    for ds_name in metrics_results:
        for test in metrics_results[ds_name]['significance_tests']:
            sig = '是' if test['significant'] else '否'






    print(json.dumps({'metrics': metrics_results, 'efficiency': efficiency_results, 'sensitivity': hyperparam_results, 'rrf': rrf_results, 'cases': case_results}, ensure_ascii=False))


if __name__ == '__main__':
    main()
