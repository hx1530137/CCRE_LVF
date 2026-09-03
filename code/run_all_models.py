"""
批量对比实验: models目录下所有embedding模型
=========================================
需要新做的模型:
1. Qwen3-Embedding-0.6B
2. Qwen3-Embedding-8B (需要大显存)
3. bge-large-zh-v1.5
4. bge-small-zh-v1.5
5. multilingual-e5-base
6. multilingual-e5-large

已有结果的模型: bge-m3, Qwen3-4B-Base, Finetune4B-full-v3

每个模型计算: BM25 / Vector / RRF / LVF 的 R@1
"""
import json, os, time, gc
import numpy as np
import torch
import faiss
from rank_bm25 import BM25Okapi
import jieba

# ============ 配置 ============
MODELS_DIR = '/home/huxin/Documents/trae_projects/sikuBERT/models'
CACHE_DIR = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/cache'
OUTPUT_DIR = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/results'
DATASET_PATH = '/home/huxin/Documents/trae_projects/sikuBERT/sanguo_test_filtered_final.json'
SEED = 42
TOP_K = 100

# 需要测试的模型 (按显存需求排序: 小→大, 顺序运行避免OOM)
MODELS_TO_TEST = [
    ('bge-small-zh-v1.5',   f'{MODELS_DIR}/bge-small-zh-v1.5',   'st'),
    ('bge-large-zh-v1.5',   f'{MODELS_DIR}/bge-large-zh-v1.5',   'st'),
    ('multilingual-e5-base', f'{MODELS_DIR}/multilingual-e5-base', 'st'),
    ('multilingual-e5-large', f'{MODELS_DIR}/multilingual-e5-large', 'st'),
    ('Qwen3-Embedding-0.6B', f'{MODELS_DIR}/Qwen3-Embedding-0.6B', 'qwen'),
    ('Qwen3-Embedding-8B',   f'{MODELS_DIR}/Qwen3-Embedding-8B',   'qwen'),
]

# LVF参数
LVF_PARAMS = dict(alpha_base=0.4, alpha_range=0.10, alpha_scale=20, gamma=0.3, delta=0.2)

# ============ 数据加载 ============
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
    return queries, doc_list, pos_idx

# ============ 编码函数 ============
def encode_st(model_path, texts, batch_size=64, max_length=512):
    """SentenceTransformer编码"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path)
    model.max_seq_length = max_length
    embs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return embs.astype(np.float32)

def encode_qwen(model_path, texts, batch_size=4, max_length=512):
    """Qwen3-Embedding编码 (last token)"""
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16,
                                       device_map='cuda').eval()

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                          max_length=max_length, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = model(**inputs)
        mask = inputs['attention_mask']
        # last token embedding
        idx = mask.sum(dim=1) - 1
        idx = idx.clamp(min=0)
        embs = outputs.last_hidden_state[torch.arange(len(batch)), idx, :]
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        all_embs.append(embs.float().cpu().numpy())

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0).astype(np.float32)

def encode_model(model_name, model_path, model_type, texts, use_cache, cache_path):
    if use_cache and os.path.exists(cache_path):
        return np.load(cache_path)
    print(f'    编码中 ({len(texts)} texts)...', end=' ', flush=True)
    t0 = time.time()
    if model_type == 'st':
        embs = encode_st(model_path, texts, batch_size=32)
    elif model_type == 'qwen':
        bs = 2 if '8B' in model_name else 8
        embs = encode_qwen(model_path, texts, batch_size=bs)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    print(f'{time.time()-t0:.0f}s', flush=True)
    np.save(cache_path, embs)
    return embs

# ============ 融合函数 ============
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

def precompute_bigrams(doc_list):
    return [set(doc[i:i+2] for i in range(len(doc)-1)) for doc in doc_list]

def compute_query_bigram_overlap(query, doc_bigrams):
    q_bigrams = set(query[i:i+2] for i in range(len(query)-1))
    n_docs = len(doc_bigrams)
    overlap = np.zeros(n_docs)
    if not q_bigrams:
        return overlap
    for d in range(n_docs):
        overlap[d] = len(q_bigrams & doc_bigrams[d]) / len(q_bigrams)
    return overlap

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
               alpha_base=0.4, alpha_range=0.10, alpha_scale=20, gamma=0.3, delta=0.2, top_k=100):
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

# ============ 评估函数 ============
def compute_recall_at_k(topk_indices, pos_idx, k=1):
    n = len(topk_indices)
    hit = sum(1 for i in range(n) if pos_idx[i] in topk_indices[i][:k])
    return hit / n

# ============ 主流程 ============
def main():
    t0_all = time.time()
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 100)
    print('批量对比实验: models目录下所有embedding模型')
    print('=' * 100, flush=True)

    # 加载数据
    queries, doc_list, pos_idx = load_dataset()
    print(f'数据集: sanguo_test ({len(queries)}查询 / {len(doc_list)}文档)\n', flush=True)

    # BM25 (所有模型共用)
    print('构建BM25索引...', end=' ', flush=True)
    tokenized = [list(jieba.cut(doc)) for doc in doc_list]
    bm25 = BM25Okapi(tokenized)
    bm25_idx, bm25_sc = [], []
    for q in queries:
        sc = bm25.get_scores(list(jieba.cut(q)))
        idx = np.argsort(-sc)[:TOP_K]
        bm25_idx.append(idx)
        bm25_sc.append(sc[idx])
    bm25_idx = np.array(bm25_idx)
    bm25_sc = np.array(bm25_sc)
    print('完成\n', flush=True)

    # 文档bigram (所有模型共用)
    doc_bigrams = precompute_bigrams(doc_list)

    # BM25 baseline (所有模型相同)
    bm25_r1 = compute_recall_at_k(bm25_idx, pos_idx, k=1)
    print(f'BM25 R@1 = {bm25_r1:.4f} (所有模型相同)\n', flush=True)

    results = {}

    # 逐个模型运行
    for model_name, model_path, model_type in MODELS_TO_TEST:
        print(f'\n{"="*80}')
        print(f'模型: {model_name}')
        print(f'{"="*80}', flush=True)

        safe_name = model_name.replace('/', '-').replace(' ', '_')
        doc_cache = f'{CACHE_DIR}/allmodels_{safe_name}_docs.npy'
        q_cache = f'{CACHE_DIR}/allmodels_{safe_name}_queries.npy'

        try:
            # 编码文档和查询
            doc_embs = encode_model(model_name, model_path, model_type, doc_list, True, doc_cache)
            q_embs = encode_model(model_name, model_path, model_type, queries, True, q_cache)

            # 向量检索
            print('    向量检索...', end=' ', flush=True)
            t0 = time.time()
            index = faiss.IndexFlatIP(doc_embs.shape[1])
            index.add(doc_embs.astype(np.float32))
            vec_scores, vec_indices = index.search(q_embs.astype(np.float32), TOP_K)
            print(f'{time.time()-t0:.1f}s', flush=True)

            # 计算指标
            vec_r1 = compute_recall_at_k(vec_indices, pos_idx, k=1)

            # RRF
            rrf_idx = rrf_fusion(bm25_idx, vec_indices, k=60)
            rrf_r1 = compute_recall_at_k(rrf_idx, pos_idx, k=1)

            # LVF
            lvf_idx = lvf_fusion(bm25_idx, bm25_sc, vec_indices, vec_scores,
                                 queries, doc_bigrams, **LVF_PARAMS)
            lvf_r1 = compute_recall_at_k(lvf_idx, pos_idx, k=1)

            print(f'    Vector R@1 = {vec_r1:.4f}')
            print(f'    RRF R@1    = {rrf_r1:.4f}')
            print(f'    LVF R@1    = {lvf_r1:.4f}  (ΔRRF={lvf_r1-rrf_r1:+.4f})', flush=True)

            results[model_name] = {
                'BM25': bm25_r1,
                'Vector': vec_r1,
                'RRF': rrf_r1,
                'LVF': lvf_r1,
                'delta_rrf': lvf_r1 - rrf_r1,
            }

        except Exception as e:
            print(f'    ❌ 错误: {e}', flush=True)
            results[model_name] = {'error': str(e)}

        # 清理显存
        gc.collect()
        torch.cuda.empty_cache()

    # 保存结果
    output_path = f'{OUTPUT_DIR}/all_models_comparison.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'description': 'models目录下所有embedding模型对比 (sanguo_test, R@1)',
            'bm25_r1': bm25_r1,
            'results': results,
        }, f, ensure_ascii=False, indent=2)

    # 汇总打印
    print('\n\n' + '=' * 100)
    print('汇总: 所有模型对比 (sanguo_test, R@1)')
    print('=' * 100)
    print(f'{"Model":<28} {"BM25":>8} {"Vector":>8} {"RRF":>8} {"LVF":>8} {"Δ(RRF)":>10}')
    print('-' * 75)
    print(f'{"(BM25 common)":<28} {bm25_r1:>8.4f} {"":>8} {"":>8} {"":>8} {"":>10}')

    for model_name in [m[0] for m in MODELS_TO_TEST]:
        r = results.get(model_name, {})
        if 'error' in r:
            print(f'{model_name:<28} ERROR: {r["error"][:40]}')
        else:
            print(f'{model_name:<28} {bm25_r1:>8.4f} {r["Vector"]:>8.4f} {r["RRF"]:>8.4f} {r["LVF"]:>8.4f} {r["delta_rrf"]:>+10.4f}')

    print(f'\n总耗时: {(time.time()-t0_all)/60:.1f} 分钟')
    print(f'结果已保存: {output_path}', flush=True)

if __name__ == '__main__':
    main()
