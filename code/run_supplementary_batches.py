"""
补充实验: 3批完整实验 (E2/E4/E6/E1/E3/E7/E8)
================================================
E2: Dev/Test分离 (80/20), 超参数在dev调优
E4: LVF-L与BM25互补性分析
E6: 数据集详细统计
E1: 标准融合基线 (CombSUM/CombMNZ/Interleave/LR)
E3: Cross-Encoder Reranking
E7: 补充R@100/MAP指标
E8: 吞吐量和显存分析
"""
import json, os, time, gc, sys
import numpy as np
import torch
import faiss
from rank_bm25 import BM25Okapi
import jieba
from scipy import stats

os.environ['TRITON_DISABLE_CUDA_KERNEL'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

MODELS_DIR = '/home/huxin/Documents/trae_projects/sikuBERT/models'
BASE_4B = f'{MODELS_DIR}/Qwen3-Embedding-4B'
CACHE_DIR = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/cache'
OUTPUT_DIR = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/results'
DATASET_PATH = '/home/huxin/Documents/trae_projects/sikuBERT/sanguo_test_filtered_final.json'
CROSS_ENC = f'{MODELS_DIR}/cross_encoder_sanguo'
SEED = 42
TOP_K = 100
MAIN_DOC_CACHE = f'{CACHE_DIR}/v3_main_Finetune4B-full-v3_docs.npy'
MAIN_Q_CACHE = f'{CACHE_DIR}/v3_main_Finetune4B-full-v3_queries.npy'
LVF_PARAMS = dict(alpha_base=0.4, alpha_range=0.10, alpha_scale=20, gamma=0.3, delta=0.2)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

def load_dataset():
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    doc_id_map, doc_list = {}, []
    for item in data:
        if item['input'] not in doc_id_map:
            doc_id_map[item['input']] = len(doc_list)
            doc_list.append(item['input'])
    queries = [item['instruction'] for item in data]
    pos_idx = [doc_id_map[item['input']] for item in data]
    return queries, doc_list, pos_idx, data

def split_dev_test(queries, pos_idx, dev_ratio=0.8, seed=42):
    rng = np.random.RandomState(seed)
    n = len(queries)
    indices = rng.permutation(n)
    dev_size = int(n * dev_ratio)
    dev_idx = indices[:dev_size]
    test_idx = indices[dev_size:]
    def subset(idx):
        return [queries[i] for i in idx], [pos_idx[i] for i in idx]
    return subset(dev_idx), subset(test_idx), list(test_idx)

def build_bm25(doc_list, queries):
    tokenized = [list(jieba.cut(doc)) for doc in doc_list]
    bm25 = BM25Okapi(tokenized)
    idx, sc = [], []
    for q in queries:
        s = bm25.get_scores(list(jieba.cut(q)))
        i = np.argsort(-s)[:TOP_K]
        idx.append(i)
        sc.append(s[i])
    return np.array(idx), np.array(sc), bm25

def precompute_bigrams(doc_list):
    return [set(doc[i:i+2] for i in range(len(doc)-1)) for doc in doc_list]

def precompute_query_doc_overlaps(queries, doc_bigrams):
    print('  预计算 query-doc 二元组重叠矩阵...', flush=True)
    overlaps = []
    for q in queries:
        q_bigrams = set(q[i:i+2] for i in range(len(q)-1))
        if not q_bigrams:
            overlaps.append(np.zeros(len(doc_bigrams)))
            continue
        row = np.zeros(len(doc_bigrams))
        for d in range(len(doc_bigrams)):
            row[d] = len(q_bigrams & doc_bigrams[d]) / len(q_bigrams)
        overlaps.append(row)
    print(f'  完成: {len(overlaps)}x{len(doc_bigrams)}', flush=True)
    return overlaps

def minmax_normalize(candidates, score_map):
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
    if len(norm_scores) < 2:
        return 0
    sorted_s = np.sort(norm_scores)[::-1]
    top1 = sorted_s[0]
    rest = sorted_s[1:min(top_n + 1, len(sorted_s))]
    return top1 - rest.mean() if len(rest) > 0 else 0

def rrf_fusion(bm25_idx, vec_idx, k=60):
    n = len(bm25_idx)
    result = np.zeros((n, TOP_K), dtype=np.int32)
    for i in range(n):
        scores = {}
        for rank, idx in enumerate(bm25_idx[i]):
            scores[int(idx)] = scores.get(int(idx), 0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(vec_idx[i]):
            scores[int(idx)] = scores.get(int(idx), 0) + 1.0 / (k + rank + 1)
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])
        result[i] = [idx for idx, _ in sorted_items[:TOP_K]]
    return result

def lvf_fusion(bm25_indices, bm25_scores, vec_indices, vec_scores, queries, doc_bigrams, overlaps_matrix,
               alpha_base=0.4, alpha_range=0.10, alpha_scale=20, gamma=0.3, delta=0.2):
    n = len(bm25_indices)
    result = np.zeros((n, TOP_K), dtype=np.int32)
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
        L = np.array([overlaps_matrix[i][c] if c < len(overlaps_matrix[i]) else 0 for c in candidates])
        P = b_norm * v_norm
        final = alpha * b_norm + (1 - alpha) * v_norm + gamma * L - delta * P
        sorted_idx = np.argsort(-final)
        result[i] = [candidates[j] for j in sorted_idx[:TOP_K]]
    return result

def recall_at_k(topk_indices, pos_idx, k=1):
    n = len(topk_indices)
    return sum(1 for i in range(n) if pos_idx[i] in topk_indices[i][:k]) / n

def ndcg_at_k(topk_indices, pos_idx, k=10):
    n = len(topk_indices)
    total_dcg = 0.0
    total_idcg = 0.0
    for i in range(n):
        pred = topk_indices[i][:k]
        pos = pos_idx[i]
        dcg = 0.0
        if pos in pred:
            rank = list(pred).index(pos) + 1
            dcg = 1.0 / np.log2(rank + 1)
        idcg = 1.0 / np.log2(1 + 1)
        total_dcg += dcg
        total_idcg += idcg
    return total_dcg / total_idcg if total_idcg > 0 else 0.0

def map_at_k(topk_indices, pos_idx, k=100):
    n = len(topk_indices)
    total_ap = 0.0
    for i in range(n):
        pred = topk_indices[i][:k]
        pos = pos_idx[i]
        if pos not in pred:
            continue
        rank = list(pred).index(pos) + 1
        total_ap += 1.0 / rank
    return total_ap / n

# ==================== 主流程 ====================
def main():
    t0_all = time.time()
    print('=' * 100)
    print('补充实验: E2/E4/E6/E1/E3/E7/E8 (3批全部)')
    print('=' * 100, flush=True)

    queries, doc_list, pos_idx, _ = load_dataset()
    print(f'数据集: sanguo_test ({len(queries)}查询 / {len(doc_list)}文档)\n', flush=True)

    # 加载缓存向量
    doc_embs = np.load(MAIN_DOC_CACHE)
    q_embs_full = np.load(MAIN_Q_CACHE)
    print(f'向量加载完成: doc={doc_embs.shape}, query={q_embs_full.shape}\n', flush=True)

    # BM25
    bm25_idx, bm25_sc, bm25_obj = build_bm25(doc_list, queries)
    bm25_r1 = recall_at_k(bm25_idx, pos_idx, k=1)
    print(f'BM25 R@1 = {bm25_r1:.4f}\n', flush=True)

    # 向量检索
    index = faiss.IndexFlatIP(doc_embs.shape[1])
    index.add(doc_embs.astype(np.float32))
    vec_scores, vec_idx = index.search(q_embs_full.astype(np.float32), TOP_K)
    vec_r1 = recall_at_k(vec_idx, pos_idx, k=1)
    print(f'Vector R@1 = {vec_r1:.4f}\n', flush=True)

    doc_bigrams = precompute_bigrams(doc_list)
    print('预计算 query-doc overlaps...', flush=True)
    all_overlaps = precompute_query_doc_overlaps(queries, doc_bigrams)

    # ==================== E6: 数据集统计 ====================
    print('\n' + '=' * 80)
    print('E6: 数据集详细统计')
    print('=' * 80)
    q_lens = [len(q) for q in queries]
    d_lens = [len(d) for d in doc_list]
    print(f'  查询数: {len(queries)}')
    print(f'  文档数: {len(doc_list)}')
    print(f'  查询长度: min={min(q_lens)}, max={max(q_lens)}, mean={np.mean(q_lens):.1f}, median={np.median(q_lens):.0f}')
    print(f'  文档长度: min={min(d_lens)}, max={max(d_lens)}, mean={np.mean(d_lens):.0f}, median={np.median(d_lens):.0f}')
    print(f'  唯一正文档数: {len(set(pos_idx))}/{len(doc_list)} ({len(set(pos_idx))/len(doc_list)*100:.1f}%)')

    # ==================== E2: Dev/Test分离 ====================
    print('\n' + '=' * 80)
    print('E2: Dev/Test 分离 (80/20), 超参数在dev调优')
    print('=' * 80)

    (dev_q, dev_p), (test_q, test_p), test_orig_idx = split_dev_test(queries, pos_idx, 0.8, SEED)
    print(f'Dev: {len(dev_q)} queries, Test: {len(test_q)} queries')

    test_embs = q_embs_full[test_orig_idx]
    dev_mask = np.ones(len(queries), dtype=bool)
    dev_mask[test_orig_idx] = False
    dev_embs = q_embs_full[dev_mask]

    # Dev/Test BM25 (use pre-built bm25_obj)
    dev_bm25_idx, dev_bm25_sc = [], []
    for q in dev_q:
        s = bm25_obj.get_scores(list(jieba.cut(q)))
        i = np.argsort(-s)[:TOP_K]
        dev_bm25_idx.append(i)
        dev_bm25_sc.append(s[i])
    dev_bm25_idx = np.array(dev_bm25_idx)
    dev_bm25_sc = np.array(dev_bm25_sc)

    test_bm25_idx, test_bm25_sc = [], []
    for q in test_q:
        s = bm25_obj.get_scores(list(jieba.cut(q)))
        i = np.argsort(-s)[:TOP_K]
        test_bm25_idx.append(i)
        test_bm25_sc.append(s[i])
    test_bm25_idx = np.array(test_bm25_idx)
    test_bm25_sc = np.array(test_bm25_sc)

    # Dev/Test 向量
    dev_index = faiss.IndexFlatIP(doc_embs.shape[1])
    dev_index.add(doc_embs.astype(np.float32))
    dev_vec_sc, dev_vec_idx = dev_index.search(dev_embs.astype(np.float32), TOP_K)
    test_vec_sc, test_vec_idx = dev_index.search(test_embs.astype(np.float32), TOP_K)

    # Dev/Test overlaps
    dev_overlaps = [all_overlaps[i] for i in range(len(queries)) if dev_mask[i]]
    test_overlaps = [all_overlaps[i] for i in test_orig_idx]

    # Dev 超参数搜索
    print('  在 Dev 集上搜索最优 LVF 超参数...')
    best_r1 = 0
    best_params = None
    for ab in [0.3, 0.4, 0.5]:
        for ar in [0.05, 0.10, 0.15]:
            for g in [0.1, 0.3, 0.5]:
                for d in [0.1, 0.2, 0.3]:
                    p = dict(alpha_base=ab, alpha_range=ar, alpha_scale=20, gamma=g, delta=d)
                    idx = lvf_fusion(dev_bm25_idx, dev_bm25_sc, dev_vec_idx, dev_vec_sc, dev_q, doc_bigrams, dev_overlaps, **p)
                    r1 = recall_at_k(idx, dev_p, k=1)
                    if r1 > best_r1:
                        best_r1 = r1
                        best_params = p

    print(f'  Dev 最优: R@1={best_r1:.4f}, params=base={best_params["alpha_base"]},range={best_params["alpha_range"]},gamma={best_params["gamma"]},delta={best_params["delta"]}')

    # Test 集报告
    test_lvf_dev = lvf_fusion(test_bm25_idx, test_bm25_sc, test_vec_idx, test_vec_sc, test_q, doc_bigrams, test_overlaps, **best_params)
    test_lvf_def = lvf_fusion(test_bm25_idx, test_bm25_sc, test_vec_idx, test_vec_sc, test_q, doc_bigrams, test_overlaps, **LVF_PARAMS)
    test_rrf = rrf_fusion(test_bm25_idx, test_vec_idx)

    print(f'  Test 集结果:')
    print(f'    Vector R@1  = {recall_at_k(test_vec_idx, test_p, k=1):.4f}')
    print(f'    RRF R@1     = {recall_at_k(test_rrf, test_p, k=1):.4f}')
    print(f'    LVF(默认)   = {recall_at_k(test_lvf_def, test_p, k=1):.4f}')
    print(f'    LVF(dev调优)= {recall_at_k(test_lvf_dev, test_p, k=1):.4f}  (ΔRRF={recall_at_k(test_lvf_dev, test_p, k=1)-recall_at_k(test_rrf, test_p, k=1):+.4f})')
    print(f'    dev调优 vs 默认: {recall_at_k(test_lvf_dev, test_p, k=1)-recall_at_k(test_lvf_def, test_p, k=1):+.4f}')

    # ==================== E4: L与BM25互补性 ====================
    print('\n' + '=' * 80)
    print('E4: LVF-L与BM25互补性分析')
    print('=' * 80)

    bm25_map = [{int(bm25_idx[i][j]): bm25_sc[i][j] for j in range(TOP_K)} for i in range(len(queries))]

    l_top1 = [all_overlaps[i][int(bm25_idx[i][0])] for i in range(len(queries))]
    bm25_top1 = [bm25_sc[i][0] for i in range(len(queries))]

    pearson_r, pearson_p = stats.pearsonr(bm25_top1, l_top1)
    spearman_r, spearman_p = stats.spearmanr(bm25_top1, l_top1)
    comp = '互补(弱相关)' if abs(pearson_r) < 0.5 else '部分重叠'

    print(f'  BM25 vs L 相关性:')
    print(f'    Pearson  r={pearson_r:.4f}, p={pearson_p:.2e}')
    print(f'    Spearman r={spearman_r:.4f}, p={spearman_p:.2e}')
    print(f'    结论: {comp}')

    # 消融
    bm25_only_r1 = recall_at_k(bm25_idx, pos_idx, k=1)
    lvf_idx = lvf_fusion(bm25_idx, bm25_sc, vec_idx, vec_scores, queries, doc_bigrams, all_overlaps, **LVF_PARAMS)
    lvf_r1 = recall_at_k(lvf_idx, pos_idx, k=1)

    # 仅L+Vector (无BM25)
    def l_only():
        result = np.zeros((len(queries), TOP_K), dtype=np.int32)
        for i in range(len(queries)):
            vec_map = {int(vec_idx[i][j]): vec_scores[i][j] for j in range(TOP_K)}
            candidates = list(vec_map.keys())
            v_norm = minmax_normalize(candidates, vec_map)
            L = np.array([all_overlaps[i][c] for c in candidates])
            combined = 0.5 * v_norm + 0.5 * L
            sorted_idx = np.argsort(-combined)
            result[i] = [candidates[j] for j in sorted_idx[:TOP_K]]
        return result

    l_only_r1 = recall_at_k(l_only(), pos_idx, k=1)
    print(f'\n  消融:')
    print(f'    仅BM25 R@1     = {bm25_only_r1:.4f}')
    print(f'    仅L+Vector R@1 = {l_only_r1:.4f}')
    print(f'    LVF (BM25+L+P) = {lvf_r1:.4f}')

    # BM25错误分析
    bm25_wrong = [i for i in range(len(queries)) if pos_idx[i] not in bm25_idx[i][:1]]
    lvf_corrected = sum(1 for i in bm25_wrong if pos_idx[i] in lvf_idx[i][:1])
    print(f'\n  BM25错误={len(bm25_wrong)}, 被LVF纠正={lvf_corrected} ({lvf_corrected/max(len(bm25_wrong),1)*100:.1f}%)')

    # ==================== E1: 标准融合基线 ====================
    print('\n' + '=' * 80)
    print('E1: 标准融合基线对比')
    print('=' * 80)

    rrf_idx = rrf_fusion(bm25_idx, vec_idx, k=60)
    rrf_r1 = recall_at_k(rrf_idx, pos_idx, k=1)

    # CombSUM
    def combsum():
        result = np.zeros((len(queries), TOP_K), dtype=np.int32)
        for i in range(len(queries)):
            bm = {int(bm25_idx[i][j]): bm25_sc[i][j] for j in range(TOP_K)}
            vm = {int(vec_idx[i][j]): vec_scores[i][j] for j in range(TOP_K)}
            cands = list(dict.fromkeys(list(bm.keys()) + list(vm.keys())))
            bn = minmax_normalize(cands, bm)
            vn = minmax_normalize(cands, vm)
            combined = bn + vn
            result[i] = [cands[j] for j in np.argsort(-combined)[:TOP_K]]
        return result

    # CombMNZ
    def combmnz():
        result = np.zeros((len(queries), TOP_K), dtype=np.int32)
        for i in range(len(queries)):
            bm = {int(bm25_idx[i][j]): bm25_sc[i][j] for j in range(TOP_K)}
            vm = {int(vec_idx[i][j]): vec_scores[i][j] for j in range(TOP_K)}
            cands = list(dict.fromkeys(list(bm.keys()) + list(vm.keys())))
            bn = minmax_normalize(cands, bm)
            vn = minmax_normalize(cands, vm)
            nb = np.array([1.0 if c in bm else 0.0 for c in cands])
            nv = np.array([1.0 if c in vm else 0.0 for c in cands])
            combined = (bn + vn) * (nb + nv)
            result[i] = [cands[j] for j in np.argsort(-combined)[:TOP_K]]
        return result

    # Interleave
    def interleave():
        result = np.zeros((len(queries), TOP_K), dtype=np.int32)
        for i in range(len(queries)):
            bl = list(bm25_idx[i])
            vl = list(vec_idx[i])
            merged, bi, vi = [], 0, 0
            while len(merged) < TOP_K and (bi < len(bl) or vi < len(vl)):
                if bi < len(bl) and (len(merged) % 2 == 0 or vi >= len(vl)):
                    if bl[bi] not in merged: merged.append(bl[bi])
                    bi += 1
                if vi < len(vl) and (len(merged) % 2 == 1 or bi >= len(bl)):
                    if vl[vi] not in merged: merged.append(vl[vi])
                    vi += 1
            result[i] = merged[:TOP_K]
        return result

    # Fixed-α
    def fixed_alpha(alpha=0.4):
        result = np.zeros((len(queries), TOP_K), dtype=np.int32)
        for i in range(len(queries)):
            bm = {int(bm25_idx[i][j]): bm25_sc[i][j] for j in range(TOP_K)}
            vm = {int(vec_idx[i][j]): vec_scores[i][j] for j in range(TOP_K)}
            cands = list(dict.fromkeys(list(bm.keys()) + list(vm.keys())))
            bn = minmax_normalize(cands, bm)
            vn = minmax_normalize(cands, vm)
            combined = alpha * bn + (1 - alpha) * vn
            result[i] = [cands[j] for j in np.argsort(-combined)[:TOP_K]]
        return result

    # LR 学习权重
    print('  训练 LR 学习权重融合...')
    from sklearn.linear_model import LogisticRegression
    X_train, y_train = [], []
    for i in range(len(queries)):
        bm = {int(bm25_idx[i][j]): bm25_sc[i][j] for j in range(TOP_K)}
        vm = {int(vec_idx[i][j]): vec_scores[i][j] for j in range(TOP_K)}
        cands = list(dict.fromkeys(list(bm.keys()) + list(vm.keys())))
        bn = minmax_normalize(cands, bm)
        vn = minmax_normalize(cands, vm)
        for j, c in enumerate(cands):
            X_train.append([bn[j], vn[j], bn[j] * vn[j]])
            y_train.append(1 if c == pos_idx[i] else 0)
    lr_model = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    lr_model.fit(np.array(X_train), np.array(y_train))
    print(f'    LR系数: b={lr_model.coef_[0][0]:.3f}, v={lr_model.coef_[0][1]:.3f}, cross={lr_model.coef_[0][2]:.3f}')

    def lr_fusion():
        result = np.zeros((len(queries), TOP_K), dtype=np.int32)
        for i in range(len(queries)):
            bm = {int(bm25_idx[i][j]): bm25_sc[i][j] for j in range(TOP_K)}
            vm = {int(vec_idx[i][j]): vec_scores[i][j] for j in range(TOP_K)}
            cands = list(dict.fromkeys(list(bm.keys()) + list(vm.keys())))
            bn = minmax_normalize(cands, bm)
            vn = minmax_normalize(cands, vm)
            X = np.array([[bn[j], vn[j], bn[j] * vn[j]] for j in range(len(cands))])
            scores = lr_model.predict_proba(X)[:, 1]
            result[i] = [cands[j] for j in np.argsort(-scores)[:TOP_K]]
        return result

    results_e1 = {
        'BM25': recall_at_k(bm25_idx, pos_idx, k=1),
        'Vector': recall_at_k(vec_idx, pos_idx, k=1),
        'RRF': rrf_r1,
        'CombSUM': recall_at_k(combsum(), pos_idx, k=1),
        'CombMNZ': recall_at_k(combmnz(), pos_idx, k=1),
        'Interleave': recall_at_k(interleave(), pos_idx, k=1),
        'Fixed-α': recall_at_k(fixed_alpha(0.4), pos_idx, k=1),
        'LR': recall_at_k(lr_fusion(), pos_idx, k=1),
        'LVF': lvf_r1,
    }

    print(f'\n  汇总 (按R@1降序):')
    for name, r1 in sorted(results_e1.items(), key=lambda x: -x[1]):
        marker = ' ← Ours' if name == 'LVF' else ''
        delta = f' (+{r1-rrf_r1:.4f} vs RRF)' if name != 'LVF' else f' (+{r1-rrf_r1:.4f} vs RRF)'
        print(f'    {name:<20s} R@1 = {r1:.4f}{marker}{delta}')

    # ==================== E3: Cross-Encoder ====================
    print('\n' + '=' * 80)
    print('E3: Cross-Encoder Reranking')
    print('=' * 80)

    try:
        from sentence_transformers import CrossEncoder
        print(f'  加载 Cross-Encoder: {CROSS_ENC}')
        ce_model = CrossEncoder(CROSS_ENC, max_length=512)
        print('  加载成功')
    except Exception as e:
        print(f'  本地Cross-Encoder加载失败: {e}')
        print('  尝试使用BAAI/bge-reranker-large...')
        try:
            ce_model = CrossEncoder('BAAI/bge-reranker-large', max_length=512)
        except Exception as e2:
            print(f'  失败: {e2}, 跳过Cross-Encoder实验')
            ce_model = None

    if ce_model is not None:
        print(f'  对 {len(queries)} 个查询做重排序...', flush=True)
        t0 = time.time()
        rerank_results = []
        for i in range(len(queries)):
            cands = list(dict.fromkeys(list(bm25_idx[i].tolist()) + list(vec_idx[i].tolist())))[ :200]
            pairs = [[queries[i], doc_list[c]] for c in cands]
            scores = []
            for j in range(0, len(pairs), 32):
                bs = pairs[j:j+32]
                ss = ce_model.predict(bs, show_progress_bar=False)
                if isinstance(ss, np.ndarray):
                    scores.extend(ss.tolist())
                else:
                    scores.extend(ss)
            ranked = [cands[j] for j in np.argsort(-np.array(scores))[:TOP_K]]
            rerank_results.append(ranked)
            if (i+1) % 200 == 0:
                print(f'    {i+1}/{len(queries)} ({time.time()-t0:.0f}s)', flush=True)
        rerank_idx = np.array(rerank_results)
        ce_r1 = recall_at_k(rerank_idx, pos_idx, k=1)
        ce_time = time.time() - t0
        print(f'\n  Cross-Encoder结果:')
        print(f'    Cross-Enc R@1 = {ce_r1:.4f}  ({ce_time/len(queries)*1000:.1f}ms/query)')
        print(f'    LVF R@1       = {lvf_r1:.4f}  (ΔCE={lvf_r1-ce_r1:+.4f})')
        print(f'    RRF R@1       = {rrf_r1:.4f}')
        del ce_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        ce_r1 = None
        ce_time = None

    # ==================== E7: 补充指标 ====================
    print('\n' + '=' * 80)
    print('E7: 补充 R@100 和 MAP 指标')
    print('=' * 80)

    methods = {'BM25': bm25_idx, 'Vector': vec_idx, 'RRF': rrf_idx, 'LVF': lvf_idx}
    print(f'\n  完整指标:')
    header = f'  {"Method":<12s} {"R@1":>8s} {"R@5":>8s} {"R@10":>8s} {"R@20":>8s} {"R@50":>8s} {"R@100":>8s} {"nDCG@10":>8s} {"MAP@100":>8s}'
    print(header)
    print('  ' + '-' * (len(header) - 2))
    metrics_all = {}
    for name, idx in methods.items():
        m = {
            'R@1': recall_at_k(idx, pos_idx, 1),
            'R@5': recall_at_k(idx, pos_idx, 5),
            'R@10': recall_at_k(idx, pos_idx, 10),
            'R@20': recall_at_k(idx, pos_idx, 20),
            'R@50': recall_at_k(idx, pos_idx, 50),
            'R@100': recall_at_k(idx, pos_idx, 100),
            'nDCG@10': ndcg_at_k(idx, pos_idx, 10),
            'MAP@100': map_at_k(idx, pos_idx, 100),
        }
        metrics_all[name] = m
        print(f'  {name:<12s} {m["R@1"]:8.4f} {m["R@5"]:8.4f} {m["R@10"]:8.4f} {m["R@20"]:8.4f} {m["R@50"]:8.4f} {m["R@100"]:8.4f} {m["nDCG@10"]:8.4f} {m["MAP@100"]:8.4f}')

    # ==================== E8: 吞吐量 ====================
    print('\n' + '=' * 80)
    print('E8: 吞吐量和显存分析')
    print('=' * 80)

    t0 = time.time()
    for q in queries:
        s = bm25_obj.get_scores(list(jieba.cut(q)))
        _ = np.argsort(-s)[:TOP_K]
    bm25_time = time.time() - t0

    t0 = time.time()
    _ = index.search(q_embs_full.astype(np.float32), TOP_K)
    vec_time = time.time() - t0

    t0 = time.time()
    _ = rrf_fusion(bm25_idx, vec_idx, k=60)
    rrf_time = time.time() - t0

    t0 = time.time()
    _ = lvf_fusion(bm25_idx, bm25_sc, vec_idx, vec_scores, queries, doc_bigrams, all_overlaps, **LVF_PARAMS)
    lvf_time = time.time() - t0

    total_rrf = bm25_time + vec_time + rrf_time
    total_lvf = bm25_time + vec_time + lvf_time
    gpu_mem = torch.cuda.memory_allocated() / 1024**2

    print(f'\n  吞吐量 ({len(queries)} queries):')
    print(f'  {"组件":<16s} {"时间(s)":>10s} {"QPS":>10s} {"ms/query":>10s}')
    print(f'  {"-"*50}')
    print(f'  {"BM25检索":<16s} {bm25_time:10.3f} {len(queries)/bm25_time:10.1f} {bm25_time/len(queries)*1000:10.2f}')
    print(f'  {"向量检索":<16s} {vec_time:10.3f} {len(queries)/vec_time:10.1f} {vec_time/len(queries)*1000:10.2f}')
    print(f'  {"RRF融合":<16s} {rrf_time:10.3f} {len(queries)/rrf_time:10.1f} {rrf_time/len(queries)*1000:10.2f}')
    print(f'  {"LVF融合":<16s} {lvf_time:10.3f} {len(queries)/lvf_time:10.1f} {lvf_time/len(queries)*1000:10.2f}')
    print(f'  {"RRF总计":<16s} {total_rrf:10.3f} {len(queries)/total_rrf:10.1f} {total_rrf/len(queries)*1000:10.2f}')
    print(f'  {"LVF总计":<16s} {total_lvf:10.3f} {len(queries)/total_lvf:10.1f} {total_lvf/len(queries)*1000:10.2f}')
    print(f'\n  GPU已分配: {gpu_mem:.1f} MB')

    # ============ 保存结果 ============
    all_results = {
        'E6_dataset_stats': {
            'num_queries': len(queries), 'num_docs': len(doc_list),
            'query_len': {'min': min(q_lens), 'max': max(q_lens), 'mean': round(float(np.mean(q_lens)), 1), 'median': float(np.median(q_lens))},
            'doc_len': {'min': min(d_lens), 'max': max(d_lens), 'mean': round(float(np.mean(d_lens)), 0), 'median': float(np.median(d_lens))},
            'unique_pos_docs': len(set(pos_idx)),
        },
        'E2_dev_test': {
            'split': {'dev': len(dev_q), 'test': len(test_q)},
            'best_dev_params': best_params, 'best_dev_r1': best_r1,
            'test': {
                'Vector': recall_at_k(test_vec_idx, test_p, 1),
                'RRF': recall_at_k(test_rrf, test_p, 1),
                'LVF_default': recall_at_k(test_lvf_def, test_p, 1),
                'LVF_dev_tuned': recall_at_k(test_lvf_dev, test_p, 1),
            }
        },
        'E4_complementarity': {
            'correlation': {'pearson_r': pearson_r, 'spearman_r': spearman_r, 'conclusion': comp},
            'ablation': {'bm25_only': bm25_only_r1, 'l_only': l_only_r1, 'lvf_full': lvf_r1},
            'error_analysis': {'bm25_wrong': len(bm25_wrong), 'lvf_corrected': lvf_corrected, 'rate': lvf_corrected/max(len(bm25_wrong),1)},
        },
        'E1_fusion_baselines': results_e1,
        'E3_cross_encoder': {'R1': ce_r1, 'latency': ce_time} if ce_r1 else {'error': 'failed'},
        'E7_extended_metrics': metrics_all,
        'E8_throughput': {
            'bm25_qps': len(queries)/bm25_time, 'vec_qps': len(queries)/vec_time,
            'rrf_qps': len(queries)/rrf_time, 'lvf_qps': len(queries)/lvf_time,
            'total_rrf_qps': len(queries)/total_rrf, 'total_lvf_qps': len(queries)/total_lvf,
            'gpu_mem_mb': gpu_mem,
        }
    }

    output_path = f'{OUTPUT_DIR}/supplementary_full_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    elapsed = (time.time() - t0_all) / 60
    print(f'\n{"="*100}')
    print(f'全部补充实验完成! 总耗时: {elapsed:.1f}分钟')
    print(f'结果已保存: {output_path}')
    print('=' * 100)

if __name__ == '__main__':
    main()
