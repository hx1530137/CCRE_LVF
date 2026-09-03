"""
补充对比实验: models目录下未测试的LoRA模型
==========================================
补充测试:
  1. qwen3-emb-finetune-4b-full      (Finetune4B-2k, 7月训练, 2k数据)
  2. qwen3-emb-dualview-4b-full      (DualView4B, 双视图微调)

两个均为 Qwen3-Embedding-4B 的 LoRA 适配器, 使用手动LoRA加载
(参考 lessons learned: PeftModel.from_pretrained + merge_and_unload 存在 key prefix 不匹配问题)

完成后合并到 all_models_comparison.json
"""
import json, os, time, gc, sys
import numpy as np
import torch
import faiss
from rank_bm25 import BM25Okapi
import jieba

os.environ['TRITON_DISABLE_CUDA_KERNEL'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ============ 路径配置 ============
MODELS_DIR   = '/home/huxin/Documents/trae_projects/sikuBERT/models'
BASE_4B      = f'{MODELS_DIR}/Qwen3-Embedding-4B'
FINETUNE_2K  = f'{MODELS_DIR}/qwen3-emb-finetune-4b-full'      # Finetune4B-2k
DUALVIEW     = f'{MODELS_DIR}/qwen3-emb-dualview-4b-full'      # DualView4B

DATASET_PATH = '/home/huxin/Documents/trae_projects/sikuBERT/sanguo_test_filtered_final.json'
CACHE_DIR    = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/cache'
OUTPUT_DIR   = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/results'
ALL_MODELS_JSON = f'{OUTPUT_DIR}/all_models_comparison.json'

TOP_K = 100
MAX_LENGTH = 512
BATCH_SIZE = 4

# 待测试的 LoRA 模型
MODELS_TO_TEST = [
    ('Finetune4B-2k',  FINETUNE_2K),
    ('DualView4B',     DUALVIEW),
]

# LVF参数 (与 run_all_models.py 一致)
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


# ============ BM25 ============
def build_bm25(doc_list, queries):
    print('构建BM25索引...', end=' ', flush=True)
    t0 = time.time()
    tokenized = [list(jieba.cut(doc)) for doc in doc_list]
    bm25 = BM25Okapi(tokenized)
    bm25_idx, bm25_sc = [], []
    for q in queries:
        sc = bm25.get_scores(list(jieba.cut(q)))
        idx = np.argsort(-sc)[:TOP_K]
        bm25_idx.append(idx)
        bm25_sc.append(sc[idx])
    print(f'{time.time()-t0:.1f}s', flush=True)
    return np.array(bm25_idx), np.array(bm25_sc)


# ============ LoRA 编码 (手动加载) ============
def encode_lora(lora_path, texts, cache_path):
    """加载 Qwen3-Embedding-4B + LoRA 适配器, 手动合并后编码"""
    if os.path.exists(cache_path):
        print(f'    使用缓存: {os.path.basename(cache_path)}', flush=True)
        return np.load(cache_path)

    from transformers import AutoModel, AutoTokenizer
    import safetensors.torch

    print(f'    加载base模型...', end=' ', flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(BASE_4B, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        BASE_4B, trust_remote_code=True, dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation='eager').to('cuda')
    model.eval()
    print(f'{time.time()-t0:.1f}s', flush=True)

    # 手动加载LoRA (避免 PeftModel key prefix 问题)
    print(f'    手动加载LoRA: {os.path.basename(lora_path)}...', end=' ', flush=True)
    t0 = time.time()
    lora_state = safetensors.torch.load_file(f'{lora_path}/adapter_model.safetensors')
    with open(f'{lora_path}/adapter_config.json') as f:
        lora_cfg = json.load(f)
    r = lora_cfg.get('r', 8)
    alpha = lora_cfg.get('lora_alpha', 32)
    scale = alpha / r

    loaded_count = 0
    for name, param in model.named_parameters():
        a_key = None
        for prefix in ['base_model.model.', 'model.', '']:
            cand_a = prefix + name.replace('.weight', '') + '.lora_A.weight'
            cand_b = prefix + name.replace('.weight', '') + '.lora_B.weight'
            if cand_a in lora_state and cand_b in lora_state:
                a_key = cand_a
                b_key = cand_b
                break
        if a_key is not None:
            A = lora_state[a_key].float()
            B = lora_state[b_key].float()
            delta = (B @ A) * scale
            param.data = param.data.float() + delta.to(param.data.device)
            param.data = param.data.to(torch.float16)
            loaded_count += 1
    print(f'{time.time()-t0:.1f}s (加载{loaded_count}个LoRA层, r={r}, alpha={alpha})', flush=True)

    # 编码
    print(f'    编码中 ({len(texts)} texts)...', end=' ', flush=True)
    t0 = time.time()
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=MAX_LENGTH, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model(**inputs)
        mask = inputs['attention_mask']
        idx = mask.sum(dim=1) - 1
        idx = idx.clamp(min=0)
        embs = out.last_hidden_state[torch.arange(len(batch)), idx, :]
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        all_embs.append(embs.float().cpu().numpy())
    embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    print(f'{time.time()-t0:.0f}s', flush=True)

    np.save(cache_path, embs)

    # 释放显存
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
    return embs


# ============ 融合函数 (与 run_all_models.py 一致) ============
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
    return [set(doc[i:i + 2] for i in range(len(doc) - 1)) for doc in doc_list]


def compute_query_bigram_overlap(query, doc_bigrams):
    q_bigrams = set(query[i:i + 2] for i in range(len(query) - 1))
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


# ============ 评估 ============
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
    print('补充对比实验: 未测试的LoRA模型 (Finetune4B-2k, DualView4B)')
    print('=' * 100, flush=True)

    # 加载数据
    queries, doc_list, pos_idx = load_dataset()
    print(f'数据集: sanguo_test ({len(queries)}查询 / {len(doc_list)}文档)\n', flush=True)

    # BM25 (共用)
    bm25_idx, bm25_sc = build_bm25(doc_list, queries)
    doc_bigrams = precompute_bigrams(doc_list)
    bm25_r1 = compute_recall_at_k(bm25_idx, pos_idx, k=1)
    print(f'BM25 R@1 = {bm25_r1:.4f}\n', flush=True)

    # 加载已有结果
    if os.path.exists(ALL_MODELS_JSON):
        with open(ALL_MODELS_JSON, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        results = all_results.get('results', {})
        print(f'已加载现有结果: {len(results)}个模型\n', flush=True)
    else:
        all_results = {'description': '所有embedding模型完整对比 (sanguo_test, R@1)',
                       'bm25_r1': bm25_r1, 'results': {}}
        results = {}

    # 逐个测试 LoRA 模型
    for model_name, lora_path in MODELS_TO_TEST:
        print(f'\n{"=" * 80}')
        print(f'模型: {model_name}')
        print(f'LoRA路径: {lora_path}')
        print(f'{"=" * 80}', flush=True)

        safe_name = model_name.replace('/', '-').replace(' ', '_')
        doc_cache = f'{CACHE_DIR}/allmodels_{safe_name}_docs.npy'
        q_cache = f'{CACHE_DIR}/allmodels_{safe_name}_queries.npy'

        try:
            # 编码
            doc_embs = encode_lora(lora_path, doc_list, doc_cache)
            q_embs = encode_lora(lora_path, queries, q_cache)

            # 向量检索
            print('    向量检索...', end=' ', flush=True)
            t0 = time.time()
            index = faiss.IndexFlatIP(doc_embs.shape[1])
            index.add(doc_embs.astype(np.float32))
            vec_scores, vec_indices = index.search(q_embs.astype(np.float32), TOP_K)
            print(f'{time.time() - t0:.1f}s', flush=True)

            # 指标
            vec_r1 = compute_recall_at_k(vec_indices, pos_idx, k=1)
            rrf_idx = rrf_fusion(bm25_idx, vec_indices, k=60)
            rrf_r1 = compute_recall_at_k(rrf_idx, pos_idx, k=1)
            lvf_idx = lvf_fusion(bm25_idx, bm25_sc, vec_indices, vec_scores,
                                 queries, doc_bigrams, **LVF_PARAMS)
            lvf_r1 = compute_recall_at_k(lvf_idx, pos_idx, k=1)

            print(f'    Vector R@1 = {vec_r1:.4f}')
            print(f'    RRF R@1    = {rrf_r1:.4f}')
            print(f'    LVF R@1    = {lvf_r1:.4f}  (ΔRRF={lvf_r1 - rrf_r1:+.4f})', flush=True)

            results[model_name] = {
                'BM25': bm25_r1,
                'Vector': vec_r1,
                'RRF': rrf_r1,
                'LVF': lvf_r1,
                'delta_rrf': lvf_r1 - rrf_r1,
            }

        except Exception as e:
            import traceback
            print(f'    ❌ 错误: {e}', flush=True)
            traceback.print_exc()
            results[model_name] = {'error': str(e)}

        # 清理
        gc.collect()
        torch.cuda.empty_cache()

    # 保存合并结果 (按 LVF R@1 降序)
    all_results['results'] = results
    all_results['bm25_r1'] = bm25_r1
    sorted_results = dict(sorted(
        [(k, v) for k, v in results.items() if 'error' not in v],
        key=lambda x: -x[1]['LVF']
    ))
    # 错误的放最后
    err_results = {k: v for k, v in results.items() if 'error' in v}
    all_results['results'] = {**sorted_results, **err_results}

    with open(ALL_MODELS_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 汇总打印
    print('\n\n' + '=' * 100)
    print(f'汇总: 所有模型对比 (sanguo_test, R@1) - 共{len(results)}个模型')
    print('=' * 100)
    print(f'{"Model":<28} {"BM25":>8} {"Vector":>8} {"RRF":>8} {"LVF":>8} {"Δ(RRF)":>10}')
    print('-' * 80)
    print(f'{"(BM25 common)":<28} {bm25_r1:>8.4f} {"":>8} {"":>8} {"":>8} {"":>10}')
    for name, r in results.items():
        if 'error' in r:
            print(f'{name:<28} ERROR: {r["error"][:40]}')
        else:
            print(f'{name:<28} {r["BM25"]:>8.4f} {r["Vector"]:>8.4f} {r["RRF"]:>8.4f} {r["LVF"]:>8.4f} {r["delta_rrf"]:>+10.4f}')

    print(f'\n总耗时: {(time.time() - t0_all) / 60:.1f} 分钟')
    print(f'结果已保存: {ALL_MODELS_JSON}', flush=True)


if __name__ == '__main__':
    main()
