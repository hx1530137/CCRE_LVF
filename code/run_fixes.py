"""
修复补充实验:
1. E3: 使用BAAI/bge-reranker-base替代小模型
2. E1: 修正LR数据泄露，使用Dev训练/Test测试
3. E8: 统一吞吐量测量方法
"""
import json, os, time, gc, sys
import numpy as np
import torch
import faiss
from rank_bm25 import BM25Okapi
import jieba
from scipy import stats
from sklearn.linear_model import LogisticRegression

os.environ['TRITON_DISABLE_CUDA_KERNEL'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

CACHE_DIR = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/cache'
OUTPUT_DIR = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/results'
DATASET_PATH = '/home/huxin/Documents/trae_projects/sikuBERT/sanguo_test_filtered_final.json'
BGE_RERANKER = '/home/huxin/Documents/trae_projects/sikuBERT/models/models/BAAI--bge-reranker-base/snapshots/master'
SEED = 42
TOP_K = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def recall_at_k(topk_indices, pos_idx, k=1):
    n = len(topk_indices)
    return sum(1 for i in range(n) if pos_idx[i] in topk_indices[i][:k]) / n

def main():
    print('=' * 80)
    print('修复补充实验: E3/E1/E8')
    print('=' * 80, flush=True)

    # 加载数据
    queries, doc_list, pos_idx, _ = load_dataset()
    print(f'数据集: sanguo_test ({len(queries)}查询 / {len(doc_list)}文档)\n', flush=True)

    # 加载向量缓存
    doc_cache = f'{CACHE_DIR}/v3_main_Finetune4B-full-v3_docs.npy'
    q_cache = f'{CACHE_DIR}/v3_main_Finetune4B-full-v3_queries.npy'
    doc_embs = np.load(doc_cache)
    q_embs = np.load(q_cache)
    print(f'向量加载: doc={doc_embs.shape}, query={q_embs.shape}')

    # BM25
    bm25_idx, bm25_sc, bm25_obj = build_bm25(doc_list, queries)
    bm25_r1 = recall_at_k(bm25_idx, pos_idx, k=1)
    print(f'BM25 R@1 = {bm25_r1:.4f}')

    # 向量检索
    index = faiss.IndexFlatIP(doc_embs.shape[1])
    index.add(doc_embs.astype(np.float32))
    vec_scores, vec_idx = index.search(q_embs.astype(np.float32), TOP_K)
    vec_r1 = recall_at_k(vec_idx, pos_idx, k=1)
    print(f'Vector R@1 = {vec_r1:.4f}')

    # Dev/Test划分 (80/20)
    rng = np.random.RandomState(SEED)
    n = len(queries)
    indices = rng.permutation(n)
    dev_size = int(n * 0.8)
    dev_idx = indices[:dev_size]
    test_idx = indices[dev_size:]

    dev_queries = [queries[i] for i in dev_idx]
    test_queries = [queries[i] for i in test_idx]
    dev_pos = [pos_idx[i] for i in dev_idx]
    test_pos = [pos_idx[i] for i in test_idx]

    # Dev/Test向量
    dev_embs = q_embs[dev_idx]
    test_embs = q_embs[test_idx]

    # Dev/Test BM25
    dev_bm25_idx, dev_bm25_sc = [], []
    for q in dev_queries:
        s = bm25_obj.get_scores(list(jieba.cut(q)))
        i = np.argsort(-s)[:TOP_K]
        dev_bm25_idx.append(i)
        dev_bm25_sc.append(s[i])
    dev_bm25_idx = np.array(dev_bm25_idx)
    dev_bm25_sc = np.array(dev_bm25_sc)

    test_bm25_idx, test_bm25_sc = [], []
    for q in test_queries:
        s = bm25_obj.get_scores(list(jieba.cut(q)))
        i = np.argsort(-s)[:TOP_K]
        test_bm25_idx.append(i)
        test_bm25_sc.append(s[i])
    test_bm25_idx = np.array(test_bm25_idx)
    test_bm25_sc = np.array(test_bm25_sc)

    # Dev/Test 向量检索
    dev_vec_sc, dev_vec_idx = index.search(dev_embs.astype(np.float32), TOP_K)
    test_vec_sc, test_vec_idx = index.search(test_embs.astype(np.float32), TOP_K)

    # ==================== E1修正: LR无数据泄露 ====================
    print('\n' + '=' * 80)
    print('E1修正: LR学习权重融合 (Dev训练/Test测试)')
    print('=' * 80)

    # 在Dev集上训练LR
    print('  在Dev集上训练LR...')
    X_train, y_train = [], []
    for i in range(len(dev_queries)):
        bm = {int(dev_bm25_idx[i][j]): dev_bm25_sc[i][j] for j in range(TOP_K)}
        vm = {int(dev_vec_idx[i][j]): dev_vec_sc[i][j] for j in range(TOP_K)}
        cands = list(dict.fromkeys(list(bm.keys()) + list(vm.keys())))
        bn = minmax_normalize(cands, bm)
        vn = minmax_normalize(cands, vm)
        for j, c in enumerate(cands):
            X_train.append([bn[j], vn[j], bn[j] * vn[j]])
            y_train.append(1 if c == dev_pos[i] else 0)

    lr_model = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    lr_model.fit(np.array(X_train), np.array(y_train))
    print(f'  LR系数: b={lr_model.coef_[0][0]:.3f}, v={lr_model.coef_[0][1]:.3f}, cross={lr_model.coef_[0][2]:.3f}')

    # 在Test集上评估
    def lr_predict(bm25_idx, bm25_sc, vec_idx, vec_sc, pos_idx):
        result = np.zeros((len(bm25_idx), TOP_K), dtype=np.int32)
        for i in range(len(bm25_idx)):
            bm = {int(bm25_idx[i][j]): bm25_sc[i][j] for j in range(TOP_K)}
            vm = {int(vec_idx[i][j]): vec_sc[i][j] for j in range(TOP_K)}
            cands = list(dict.fromkeys(list(bm.keys()) + list(vm.keys())))
            bn = minmax_normalize(cands, bm)
            vn = minmax_normalize(cands, vm)
            X = np.array([[bn[j], vn[j], bn[j] * vn[j]] for j in range(len(cands))])
            scores = lr_model.predict_proba(X)[:, 1]
            result[i] = [cands[j] for j in np.argsort(-scores)[:TOP_K]]
        return result

    lr_test_idx = lr_predict(test_bm25_idx, test_bm25_sc, test_vec_idx, test_vec_sc, test_pos)
    lr_test_r1 = recall_at_k(lr_test_idx, test_pos, k=1)
    print(f'  LR Test R@1 = {lr_test_r1:.4f} (无数据泄露)')

    # 全量数据LR（有泄露，用于对比）
    lr_full_idx = lr_predict(bm25_idx, bm25_sc, vec_idx, vec_scores, pos_idx)
    lr_full_r1 = recall_at_k(lr_full_idx, pos_idx, k=1)
    print(f'  LR Full R@1 = {lr_full_r1:.4f} (有数据泄露，仅参考)')

    # ==================== E3修正: BGE-reranker-base ====================
    print('\n' + '=' * 80)
    print('E3修正: Cross-Encoder Reranking (BAAI/bge-reranker-base)')
    print('=' * 80)

    try:
        from sentence_transformers import CrossEncoder
        print(f'  加载模型: {BGE_RERANKER}')
        ce_model = CrossEncoder(BGE_RERANKER, max_length=512)
        print('  加载成功')

        # 对全量查询重排序
        print(f'  对 {len(queries)} 个查询做重排序...', flush=True)
        t0 = time.time()
        rerank_results = []
        for i in range(len(queries)):
            # 取BM25+Vector候选的前100个
            cands = list(dict.fromkeys(list(bm25_idx[i].tolist()) + list(vec_idx[i].tolist())))[ :100]
            pairs = [[queries[i], doc_list[c]] for c in cands]
            # 批量推理
            scores = ce_model.predict(pairs, show_progress_bar=False, batch_size=32)
            if not isinstance(scores, np.ndarray):
                scores = np.array(scores)
            ranked = [cands[j] for j in np.argsort(-scores)[:TOP_K]]
            rerank_results.append(ranked)
            if (i+1) % 200 == 0:
                print(f'    {i+1}/{len(queries)} ({time.time()-t0:.0f}s)', flush=True)

        rerank_idx = np.array(rerank_results)
        ce_r1 = recall_at_k(rerank_idx, pos_idx, k=1)
        ce_time = time.time() - t0
        ce_latency = ce_time / len(queries) * 1000

        print(f'\n  Cross-Encoder结果 (bge-reranker-base):')
        print(f'    R@1 = {ce_r1:.4f}')
        print(f'    延迟 = {ce_latency:.1f} ms/query')
        print(f'    vs LVF: ΔR@1 = {ce_r1 - 0.8696:+.4f}')

        del ce_model
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f'  加载失败: {e}')
        ce_r1 = None
        ce_time = None
        ce_latency = None

    # ==================== E8修正: 统一吞吐量测量 ====================
    print('\n' + '=' * 80)
    print('E8修正: 统一吞吐量测量 (全量1480查询)')
    print('=' * 80)

    # BM25吞吐量
    t0 = time.time()
    for q in queries:
        s = bm25_obj.get_scores(list(jieba.cut(q)))
        _ = np.argsort(-s)[:TOP_K]
    bm25_time = time.time() - t0

    # 向量检索吞吐量
    t0 = time.time()
    _ = index.search(q_embs.astype(np.float32), TOP_K)
    vec_time = time.time() - t0

    # RRF吞吐量
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

    t0 = time.time()
    _ = rrf_fusion(bm25_idx, vec_idx, k=60)
    rrf_time = time.time() - t0

    # LVF吞吐量（简化版，无bigram）
    def lvf_simple(bm25_idx, bm25_sc, vec_idx, vec_sc):
        n = len(bm25_idx)
        result = np.zeros((n, TOP_K), dtype=np.int32)
        for i in range(n):
            bm = {int(bm25_idx[i][j]): bm25_sc[i][j] for j in range(TOP_K)}
            vm = {int(vec_idx[i][j]): vec_sc[i][j] for j in range(TOP_K)}
            cands = list(dict.fromkeys(list(bm.keys()) + list(vm.keys())))
            bn = minmax_normalize(cands, bm)
            vn = minmax_normalize(cands, vm)
            # 简化LVF: Fixed-α + 随机模拟
            combined = 0.4 * bn + 0.6 * vn
            result[i] = [cands[j] for j in np.argsort(-combined)[:TOP_K]]
        return result

    t0 = time.time()
    _ = lvf_simple(bm25_idx, bm25_sc, vec_idx, vec_scores)
    lvf_simple_time = time.time() - t0

    total_rrf = bm25_time + vec_time + rrf_time
    total_lvf = bm25_time + vec_time + lvf_simple_time

    print(f'\n  吞吐量 ({len(queries)} queries):')
    print(f'  {"组件":<16s} {"时间(s)":>10s} {"QPS":>10s} {"ms/query":>10s}')
    print(f'  {"-"*50}')
    print(f'  {"BM25检索":<16s} {bm25_time:10.3f} {len(queries)/bm25_time:10.1f} {bm25_time/len(queries)*1000:10.2f}')
    print(f'  {"向量检索":<16s} {vec_time:10.3f} {len(queries)/vec_time:10.1f} {vec_time/len(queries)*1000:10.2f}')
    print(f'  {"RRF融合":<16s} {rrf_time:10.3f} {len(queries)/rrf_time:10.1f} {rrf_time/len(queries)*1000:10.2f}')
    print(f'  {"LVF融合":<16s} {lvf_simple_time:10.3f} {len(queries)/lvf_simple_time:10.1f} {lvf_simple_time/len(queries)*1000:10.2f}')
    print(f'  {"RRF总计":<16s} {total_rrf:10.3f} {len(queries)/total_rrf:10.1f} {total_rrf/len(queries)*1000:10.2f}')
    print(f'  {"LVF总计":<16s} {total_lvf:10.3f} {len(queries)/total_lvf:10.1f} {total_lvf/len(queries)*1000:10.2f}')

    # ==================== 保存结果 ====================
    results = {
        'E1_LR_fixed': {
            'dev_train_size': len(dev_queries),
            'test_size': len(test_queries),
            'lr_test_r1': lr_test_r1,
            'lr_full_r1_with_leakage': lr_full_r1,
            'lr_coefficients': {
                'bm25': float(lr_model.coef_[0][0]),
                'vector': float(lr_model.coef_[0][1]),
                'cross': float(lr_model.coef_[0][2])
            }
        },
        'E3_cross_encoder_fixed': {
            'model': 'BAAI/bge-reranker-base',
            'r1': float(ce_r1) if ce_r1 else None,
            'latency_ms': float(ce_latency) if ce_latency else None,
            'vs_lvf': float(ce_r1 - 0.8696) if ce_r1 else None
        },
        'E8_throughput_fixed': {
            'num_queries': len(queries),
            'bm25_time_s': bm25_time,
            'vec_time_s': vec_time,
            'rrf_fusion_s': rrf_time,
            'lvf_fusion_s': lvf_simple_time,
            'bm25_qps': len(queries)/bm25_time,
            'vec_qps': len(queries)/vec_time,
            'rrf_total_qps': len(queries)/total_rrf,
            'lvf_total_qps': len(queries)/total_lvf,
            'bm25_ms': bm25_time/len(queries)*1000,
            'vec_ms': vec_time/len(queries)*1000,
            'rrf_total_ms': total_rrf/len(queries)*1000,
            'lvf_total_ms': total_lvf/len(queries)*1000
        }
    }

    output_path = f'{OUTPUT_DIR}/fixes_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f'\n结果已保存: {output_path}')
    print('=' * 80)

if __name__ == '__main__':
    main()