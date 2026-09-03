'Main and cross-dataset retrieval experiments.'
import json, os, time, gc, math
import torch
import numpy as np

os.environ['TRITON_DISABLE_CUDA_KERNEL'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


MODELS_DIR   = '/home/huxin/Documents/trae_projects/sikuBERT/models'
BASE_4B      = f'{MODELS_DIR}/Qwen3-Embedding-4B'
FINETUNE_LORA_2K   = f'{MODELS_DIR}/qwen3-emb-finetune-4b-full'
FINETUNE_LORA_V3   = f'{MODELS_DIR}/qwen3-emb-finetune-4b-full-v3'
BGE_M3       = f'{MODELS_DIR}/bge-m3'


TEST_DATA    = '/home/huxin/Documents/trae_projects/sikuBERT/sanguo_test_filtered_final.json'
SHIJI_DATA   = '/home/huxin/Documents/trae_projects/sikuBERT/史记合并-with-prompt-api-response-extracted-train.json'
HANSHU_DATA  = '/home/huxin/Documents/trae_projects/sikuBERT/汉书合并mini-with-prompt-api-response-extracted-train.json'
HISTORY10K   = '/home/huxin/Documents/trae_projects/sikuBERT/classical_history_retrieval_10k.json'

OUTPUT_DIR   = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/results'
CACHE_DIR    = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/cache'
SEED = 42


MAIN_MODELS = [
    ('Qwen3-4B-Base',       BASE_4B,           'hf',   512),
    ('Finetune4B-2k',       BASE_4B,           'lora2k', 512),
    ('Finetune4B-full-v3',  BASE_4B,           'lorav3', 512),
]


CROSS_MODELS = [
    ('bge-m3',              BGE_M3,            'st',    8192),
    ('Qwen3-4B-Base',       BASE_4B,           'hf',    512),
    ('Finetune4B-full-v3',  BASE_4B,           'lorav3', 512),
]

CROSS_DATASETS = [
    ('sanguo_test',  TEST_DATA,   1.0),
    ('shiji',        SHIJI_DATA,   0.1),
    ('hanshu',       HANSHU_DATA,  0.1),
    ('history-10k',  HISTORY10K,   0.1),
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


def encode_st(model_path, max_len):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path, trust_remote_code=True, device='cuda')
    model.max_seq_length = max_len
    def encode(texts, batch_size=8):
        return model.encode(texts, batch_size=batch_size,
                            normalize_embeddings=True, show_progress_bar=False)
    return encode, model, None


def encode_hf_base(base_path, max_len):
    from transformers import AutoModel, AutoTokenizer
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(base_path, trust_remote_code=True,
                                      dtype=torch.float16, low_cpu_mem_usage=True,
                                      attn_implementation='eager').to(device)
    model.eval()
    def encode(texts, batch_size=4):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=max_len, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inputs)
            mask = inputs['attention_mask']
            idx = mask.sum(dim=1) - 1
            embs = torch.stack([out.last_hidden_state[j, idx[j], :] for j in range(len(batch))])
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().float())
        return torch.cat(all_embs, dim=0).numpy()
    return encode, model, tokenizer


def encode_hf_lora(base_path, lora_path, max_len):
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(base_path, trust_remote_code=True,
                                      dtype=torch.float16, low_cpu_mem_usage=True,
                                      attn_implementation='eager').to(device)
    try:
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print(f'    LoRA已加载并合并: {os.path.basename(lora_path)}', flush=True)
    except Exception as e:
        print(f'    [WARN] PeftModel加载失败({e}), 尝试手动加载LoRA', flush=True)
        import safetensors.torch
        lora_state = safetensors.torch.load_file(f'{lora_path}/adapter_model.safetensors')
        with open(f'{lora_path}/adapter_config.json') as f:
            lora_cfg = json.load(f)
        r = lora_cfg.get('r', 8)
        alpha = lora_cfg.get('lora_alpha', 32)
        scale = alpha / r
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
        print(f'    LoRA手动加载完成 (r={r}, alpha={alpha})', flush=True)
    model.eval()
    def encode(texts, batch_size=4):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=max_len, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inputs)
            mask = inputs['attention_mask']
            idx = mask.sum(dim=1) - 1
            embs = torch.stack([out.last_hidden_state[j, idx[j], :] for j in range(len(batch))])
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().float())
        return torch.cat(all_embs, dim=0).numpy()
    return encode, model, tokenizer


def encode_factory(name, model_path, encode_type, max_len):
    if encode_type == 'st':
        return encode_st(model_path, max_len)
    elif encode_type == 'hf':
        return encode_hf_base(model_path, max_len)
    elif encode_type == 'lora2k':
        return encode_hf_lora(model_path, FINETUNE_LORA_2K, max_len)
    elif encode_type == 'lorav3':
        return encode_hf_lora(model_path, FINETUNE_LORA_V3, max_len)
    else:
        raise ValueError(f'Unknown encode_type: {encode_type}')


def free_mem(model, tok=None):
    try:
        if hasattr(model, 'to'):
            model.to('cpu')
    except Exception:
        pass
    del model
    if tok is not None:
        del tok
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def vector_search(query_embs, doc_embs, top_k=100):
    import faiss
    index = faiss.IndexFlatIP(doc_embs.shape[1])
    index.add(doc_embs.astype(np.float32))
    scores, indices = index.search(query_embs.astype(np.float32), top_k)
    return indices, scores


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
    'LVF: Lexical-Verified Fusion'
    n = len(bm25_indices)
    result = np.zeros((n, top_k), dtype=np.int32)
    for i in range(n):
        bm25_map = dict(zip(bm25_indices[i].tolist(), bm25_scores[i].tolist()))
        vec_map = dict(zip(vec_indices[i].tolist(), vec_scores[i].tolist()))
        candidates = list(dict.fromkeys(list(bm25_map.keys()) + list(vec_map.keys())))
        b_norm = _minmax_normalize(candidates, bm25_map)
        v_norm = _minmax_normalize(candidates, vec_map)

        b_margin = _compute_margin(b_norm, 5)
        v_margin = _compute_margin(v_norm, 5)
        diff = b_margin - v_margin
        alpha = alpha_base + alpha_range * (1 / (1 + np.exp(-diff * alpha_scale)))

        bigram_overlap = compute_query_bigram_overlap(queries[i], doc_bigrams)
        L = np.array([bigram_overlap[c] if c < len(bigram_overlap) else 0 for c in candidates])
        P = b_norm * v_norm

        final = alpha * b_norm + (1 - alpha) * v_norm + gamma * L - delta * P
        sorted_idx = np.argsort(-final)
        result[i] = [candidates[j] for j in sorted_idx[:top_k]]
    return result


def evaluate_one(queries, doc_list, positive_indices, doc_bigrams,
                 bm25_idx, bm25_sc, model_name, encode_type, model_path, max_len,
                 cache_suffix):
    safe_name = model_name.replace('/', '-').replace(' ', '_').replace('(', '').replace(')', '')
    doc_cache = f'{CACHE_DIR}/{cache_suffix}_{safe_name}_docs.npy'
    q_cache   = f'{CACHE_DIR}/{cache_suffix}_{safe_name}_queries.npy'

    print(f'    加载编码器 ...', end=' ', flush=True)
    t0 = time.time()
    encode_fn, model, tok = encode_factory(model_name, model_path, encode_type, max_len)
    print(f'{time.time()-t0:.1f}s', flush=True)

    if os.path.exists(doc_cache) and os.path.exists(q_cache):
        print(f'    加载编码缓存: {safe_name}', flush=True)
        doc_embs = np.load(doc_cache)
        q_embs = np.load(q_cache)
    else:
        print(f'    编码文档 ({len(doc_list)}) ...', end=' ', flush=True)
        t0 = time.time()
        doc_embs = encode_fn(doc_list)
        print(f'{time.time()-t0:.1f}s', flush=True)
        print(f'    编码查询 ({len(queries)}) ...', end=' ', flush=True)
        t0 = time.time()
        q_embs = encode_fn(queries)
        print(f'{time.time()-t0:.1f}s', flush=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(doc_cache, doc_embs)
        np.save(q_cache, q_embs)

    print(f'    向量检索 ...', end=' ', flush=True)
    vec_idx, vec_sc = vector_search(q_embs, doc_embs)
    print('done', flush=True)

    results = {}
    m = compute_metrics(bm25_idx, positive_indices)
    results['BM25'] = m
    print(f'    BM25           R@1={m["R@1"]:.4f} R@5={m["R@5"]:.4f} MRR={m["MRR"]:.4f}', flush=True)

    m = compute_metrics(vec_idx, positive_indices)
    results['Vector'] = m
    print(f'    Vector         R@1={m["R@1"]:.4f} R@5={m["R@5"]:.4f} MRR={m["MRR"]:.4f}', flush=True)

    idx = rrf_fusion(bm25_idx, vec_idx)
    m = compute_metrics(idx, positive_indices)
    results['RRF'] = m
    print(f'    RRF            R@1={m["R@1"]:.4f} R@5={m["R@5"]:.4f} MRR={m["MRR"]:.4f}', flush=True)

    idx = was_hybrid(bm25_idx, bm25_sc, vec_idx, vec_sc, alpha=0.4)
    m = compute_metrics(idx, positive_indices)
    results['WAS'] = m
    print(f'    WAS(α=0.4)     R@1={m["R@1"]:.4f} R@5={m["R@5"]:.4f} MRR={m["MRR"]:.4f}', flush=True)

    idx = lvf_fusion(bm25_idx, bm25_sc, vec_idx, vec_sc, queries, doc_bigrams,
                     alpha_base=0.4, alpha_range=0.10, gamma=0.3, delta=0.2)
    m = compute_metrics(idx, positive_indices)
    results['LVF(Ours)'] = m
    lvf_gain = (results['LVF(Ours)']['R@1'] - results['WAS']['R@1']) * 100
    print(f'    LVF(Ours)      R@1={m["R@1"]:.4f} R@5={m["R@5"]:.4f} MRR={m["MRR"]:.4f}  ΔR@1(WAS)={lvf_gain:+.2f}%', flush=True)

    free_mem(model, tok)
    time.sleep(3)
    return results


def run_main_experiment():
    print('\n' + '=' * 100)
    print('实验1: 主测试集对比 (Base vs Finetune4B-2k vs Finetune4B-full-v3)')
    print('=' * 100, flush=True)

    queries, doc_list, pos_idx = load_dataset(TEST_DATA, test_ratio=1.0, seed=SEED)
    print(f'  数据: {len(queries)} 查询, {len(doc_list)} 文档', flush=True)

    print(f'  构建BM25 ...', end=' ', flush=True)
    t0 = time.time()
    bm25_search = build_bm25(doc_list)
    bm25_idx, bm25_sc = bm25_search(queries)
    print(f'{time.time()-t0:.1f}s', flush=True)

    print(f'  预计算bigram ...', end=' ', flush=True)
    t0 = time.time()
    doc_bigrams = precompute_bigrams(doc_list)
    print(f'{time.time()-t0:.1f}s', flush=True)

    all_results = {}
    for model_name, model_path, encode_type, max_len in MAIN_MODELS:
        print(f'\n  ---- {model_name} ----', flush=True)
        res = evaluate_one(queries, doc_list, pos_idx, doc_bigrams,
                           bm25_idx, bm25_sc, model_name, encode_type, model_path, max_len,
                           cache_suffix='v3_main')
        all_results[model_name] = res

    return all_results


def run_cross_dataset_experiment():
    print('\n' + '=' * 100)
    print('实验2: 跨数据集泛化性 (Finetune4B-full-v3)')
    print('=' * 100, flush=True)

    all_results = {}
    for ds_name, ds_path, test_ratio in CROSS_DATASETS:
        print(f'\n  ==== 数据集: {ds_name} ====', flush=True)
        queries, doc_list, pos_idx = load_dataset(ds_path, test_ratio=test_ratio, seed=SEED)
        print(f'  数据: {len(queries)} 查询, {len(doc_list)} 文档', flush=True)

        print(f'  构建BM25 ...', end=' ', flush=True)
        t0 = time.time()
        bm25_search = build_bm25(doc_list)
        bm25_idx, bm25_sc = bm25_search(queries)
        print(f'{time.time()-t0:.1f}s', flush=True)

        print(f'  预计算bigram ...', end=' ', flush=True)
        t0 = time.time()
        doc_bigrams = precompute_bigrams(doc_list)
        print(f'{time.time()-t0:.1f}s', flush=True)

        ds_results = {}
        for model_name, model_path, encode_type, max_len in CROSS_MODELS:
            print(f'\n  ---- {model_name} ----', flush=True)
            res = evaluate_one(queries, doc_list, pos_idx, doc_bigrams,
                               bm25_idx, bm25_sc, model_name, encode_type, model_path, max_len,
                               cache_suffix=f'v3_{ds_name}')
            ds_results[model_name] = res

        all_results[ds_name] = ds_results

    return all_results


def print_summary(main_results, cross_results):
    lines = []
    lines.append('=' * 100)
    lines.append('experiment_v3 汇总: 4B模型全量微调实验结果')
    lines.append('=' * 100)


    lines.append('\n表1: 主测试集 (sanguo_test, 1480查询/1023文档) - 数据量消融')
    lines.append(f'{"模型":<22} {"方法":<12} {"R@1":>8} {"R@5":>8} {"MRR":>8} {"ΔR@1(WAS)":>12}')
    lines.append('-' * 75)
    methods = ['BM25', 'Vector', 'RRF', 'WAS', 'LVF(Ours)']
    for model in main_results:
        was_r1 = main_results[model]['WAS']['R@1']
        for mi, method in enumerate(methods):
            m = main_results[model][method]
            gain = f'{(m["R@1"]-was_r1)*100:+.2f}%' if method != 'WAS' else 'baseline'
            mark = '★' if method == 'LVF(Ours)' else ' '
            lines.append(f'{model if mi==0 else "":<22} {method:<12} {m["R@1"]:>8.4f} {m["R@5"]:>8.4f} {m["MRR"]:>8.4f} {gain:>12} {mark}')
        lines.append('-' * 75)


    lines.append('\n数据量消融: Finetune4B-2k vs Finetune4B-full-v3 (Vector R@1)')
    if 'Finetune4B-2k' in main_results and 'Finetune4B-full-v3' in main_results:
        v2k = main_results['Finetune4B-2k']['Vector']['R@1']
        vv3 = main_results['Finetune4B-full-v3']['Vector']['R@1']
        base = main_results['Qwen3-4B-Base']['Vector']['R@1']
        lines.append(f'  Base:              {base:.4f}')
        lines.append(f'  Finetune4B-2k:     {v2k:.4f} (ΔBase={((v2k-base)*100):+.2f}%)')
        lines.append(f'  Finetune4B-full-v3:{vv3:.4f} (ΔBase={((vv3-base)*100):+.2f}%, Δ2k={((vv3-v2k)*100):+.2f}%)')


    lines.append('\n\n表2: 跨数据集泛化性 (LVF vs WAS, R@1)')
    lines.append(f'{"数据集":<16} {"模型":<22} {"WAS":>8} {"LVF":>8} {"ΔR@1":>10}')
    lines.append('-' * 70)
    for ds_name in cross_results:
        for model in cross_results[ds_name]:
            was_r1 = cross_results[ds_name][model]['WAS']['R@1']
            lvf_r1 = cross_results[ds_name][model]['LVF(Ours)']['R@1']
            gain = (lvf_r1 - was_r1) * 100
            lines.append(f'{ds_name:<16} {model:<22} {was_r1:>8.4f} {lvf_r1:>8.4f} {gain:>+9.2f}%')
        lines.append('-' * 70)

    summary_text = '\n'.join(lines)
    print(summary_text, flush=True)
    return summary_text


def main():
    t0_all = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print('=' * 100)
    print('experiment_v3: 4B模型全量微调完整实验')
    print(f'训练数据: 12654条 (之前2000条)')
    print(f'新模型: Finetune4B-full-v3 (LoRA r=8, alpha=32)')
    print(f'测试集: sanguo_test(1480) / 史记(1/10) / 汉书(1/10) / history-10k(1/10)')
    print('=' * 100, flush=True)


    main_results = run_main_experiment()


    with open(f'{OUTPUT_DIR}/v3_main_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'description': '主测试集对比: Base vs Finetune4B-2k vs Finetune4B-full-v3',
            'dataset': 'sanguo_test (1480查询/1023文档)',
            'results': main_results,
        }, f, ensure_ascii=False, indent=2)
    print(f'\n主测试集结果已保存: {OUTPUT_DIR}/v3_main_results.json', flush=True)


    cross_results = run_cross_dataset_experiment()


    with open(f'{OUTPUT_DIR}/v3_cross_dataset_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'description': '跨数据集泛化性: bge-m3 / Base / Finetune4B-full-v3',
            'datasets': {'sanguo_test': '1480查询', 'shiji': '1/10', 'hanshu': '1/10', 'history-10k': '1/10'},
            'results': cross_results,
        }, f, ensure_ascii=False, indent=2)
    print(f'\n跨数据集结果已保存: {OUTPUT_DIR}/v3_cross_dataset_results.json', flush=True)


    summary_text = print_summary(main_results, cross_results)
    with open(f'{OUTPUT_DIR}/v3_summary_tables.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f'\n汇总表格已保存: {OUTPUT_DIR}/v3_summary_tables.txt', flush=True)

    print(f'\n总耗时: {(time.time()-t0_all)/60:.1f} 分钟', flush=True)


if __name__ == '__main__':
    main()
