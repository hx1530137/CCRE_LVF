"""
C3bench 跨领域泛化性补充实验
==========================================
实验目的:
  在C3bench数据集上验证LVF的跨领域泛化性。
  C3bench覆盖儒/道/佛/医/兵/法/农/艺等10个领域，
  与现有史书数据集(三国志/史记/汉书)形成互补，
  验证LVF在不同古文风格上的鲁棒性。

数据集:
  - 文件: C3bench_crosslang_1000.json
  - 规模: 1000查询/1000文档（纯古文）
  - 类别: 儒/史/道/佛/医/诗/兵/法/农/艺
  - 来源: 孙膑兵法/韩非子/伤寒论/齐民要术/列子等
  - 任务: 现代文查询 → 古文文档（跨语言检索）

模型:
  - bge-m3 (568M, 基线)
  - Qwen3-4B-Base (4B, 基线)
  - Finetune4B-full-v3 (4B+LoRA, 本文微调模型)

方法:
  - BM25 (纯词汇)
  - Vector (纯向量)
  - RRF (k=60)
  - LVF (本文方法)

输出:
  - experiment_v3/results/c3bench_results.json
  - experiment_v3/results/c3bench_summary.txt
"""
import argparse
import json, os, time, sys, gc
import torch
import numpy as np

# 复用 run_v3_experiments.py 中的所有函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_v3_experiments import (
    load_dataset, compute_metrics, build_bm25, precompute_bigrams,
    rrf_fusion, was_hybrid, lvf_fusion, free_mem,
    encode_factory, BASE_4B, FINETUNE_LORA_V3, BGE_M3, SEED
)

# ============ 配置 ============
C3BENCH_DATA = '/home/huxin/Documents/trae_projects/sikuBERT/C3bench_crosslang_1000.json'
OUTPUT_DIR   = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/results'
CACHE_DIR    = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/cache'
SAMPLE_RATIO = 1.0   # 已预采样1000条，直接全量使用

# 与跨数据集实验一致的模型配置
C3BENCH_MODELS = [
    ('bge-m3',              BGE_M3,    'st',    8192),
    ('Qwen3-4B-Base',       BASE_4B,   'hf',    512),
    ('Finetune4B-full-v3',  BASE_4B,   'lorav3', 512),
]


def load_c3bench(data_path, test_ratio=1.0, seed=42):
    """加载C3bench数据集并按类别统计"""
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

    # 类别统计
    from collections import Counter
    cats = Counter(item.get('category', 'unknown') for item in data)
    sources = Counter(item.get('source', 'unknown') for item in data)

    return queries, doc_list, positive_indices, cats, sources


def evaluate_one(queries, doc_list, positive_indices, doc_bigrams,
                        bm25_idx, bm25_sc, model_name, encode_type, model_path, max_len,
                        cache_suffix):
    """评估单个模型，不包含WAS方法"""
    safe_name = model_name.replace('/', '-').replace(' ', '_').replace('(', '').replace(')', '')
    doc_cache = f'{CACHE_DIR}/{cache_suffix}_{safe_name}_docs.npy'
    q_cache   = f'{CACHE_DIR}/{cache_suffix}_{safe_name}_queries.npy'

    print(f'    加载编码器 ...', end=' ', flush=True)
    t0 = time.time()
    encode_fn, model, tok = encode_factory(model_name, model_path, encode_type, max_len)
    print(f'{time.time()-t0:.1f}s', flush=True)

    # 编码文档
    if os.path.exists(doc_cache):
        print(f'    加载文档向量缓存 ...', end=' ', flush=True)
        doc_embs = np.load(doc_cache)
        print(f'{doc_embs.shape}', flush=True)
    else:
        print(f'    编码文档 ({len(doc_list)}) ...', end=' ', flush=True)
        t0 = time.time()
        doc_embs = encode_fn(doc_list)
        print(f'{time.time()-t0:.1f}s', flush=True)
        np.save(doc_cache, doc_embs)

    # 编码查询
    if os.path.exists(q_cache):
        print(f'    加载查询向量缓存 ...', end=' ', flush=True)
        q_embs = np.load(q_cache)
        print(f'{q_embs.shape}', flush=True)
    else:
        print(f'    编码查询 ({len(queries)}) ...', end=' ', flush=True)
        t0 = time.time()
        q_embs = encode_fn(queries)
        print(f'{time.time()-t0:.1f}s', flush=True)
        np.save(q_cache, q_embs)

    # 向量检索
    import faiss
    print(f'    向量检索 ...', end=' ', flush=True)
    index = faiss.IndexFlatIP(doc_embs.shape[1])
    index.add(doc_embs.astype(np.float32))
    vec_scores, vec_idx = index.search(q_embs.astype(np.float32), 100)
    print('done', flush=True)

    results = {}
    
    # BM25
    m = compute_metrics(bm25_idx, positive_indices)
    results['BM25'] = m
    print(f'    BM25           R@1={m["R@1"]:.4f} R@5={m["R@5"]:.4f} MRR={m["MRR"]:.4f}', flush=True)

    # Vector
    m = compute_metrics(vec_idx, positive_indices)
    results['Vector'] = m
    print(f'    Vector         R@1={m["R@1"]:.4f} R@5={m["R@5"]:.4f} MRR={m["MRR"]:.4f}', flush=True)

    # RRF
    idx = rrf_fusion(bm25_idx, vec_idx)
    m = compute_metrics(idx, positive_indices)
    results['RRF'] = m
    print(f'    RRF            R@1={m["R@1"]:.4f} R@5={m["R@5"]:.4f} MRR={m["MRR"]:.4f}', flush=True)

    # Fixed score fusion: candidate-union min-max normalization, BM25/dense = 0.4/0.6.
    idx = was_hybrid(bm25_idx, bm25_sc, vec_idx, vec_scores, alpha=0.4)
    m = compute_metrics(idx, positive_indices)
    results['Fixed fusion'] = m
    print(f'    Fixed fusion   R@1={m["R@1"]:.4f} R@5={m["R@5"]:.4f} MRR={m["MRR"]:.4f}', flush=True)

    # LVF (本文方法)
    idx = lvf_fusion(bm25_idx, bm25_sc, vec_idx, vec_scores, queries, doc_bigrams,
                     alpha_base=0.4, alpha_range=0.10, gamma=0.3, delta=0.2)
    m = compute_metrics(idx, positive_indices)
    results['LVF(Ours)'] = m
    lvf_gain = (results['LVF(Ours)']['R@1'] - results['RRF']['R@1']) * 100
    print(f'    LVF(Ours)      R@1={m["R@1"]:.4f} R@5={m["R@5"]:.4f} MRR={m["MRR"]:.4f}  ΔR@1(RRF)={lvf_gain:+.2f}%', flush=True)

    free_mem(model, tok)
    time.sleep(3)
    return results


def run_c3bench_experiment(selected_models=None):
    print('\n' + '=' * 100)
    print('C3bench 跨领域泛化性补充实验')
    print('=' * 100, flush=True)

    # 加载数据
    queries, doc_list, pos_idx, cats, sources = load_c3bench(C3BENCH_DATA, SAMPLE_RATIO, SEED)
    print(f'\n  数据集: C3bench-crosslang (纯古文文档, 1000查询)', flush=True)
    print(f'  查询数: {len(queries)}, 文档数: {len(doc_list)}', flush=True)
    print(f'  类别分布: {dict(cats.most_common())}', flush=True)
    print(f'  来源前5: {dict(sources.most_common(5))}', flush=True)

    # BM25
    print(f'\n  构建BM25 ...', end=' ', flush=True)
    t0 = time.time()
    bm25_search = build_bm25(doc_list)
    bm25_idx, bm25_sc = bm25_search(queries)
    print(f'{time.time()-t0:.1f}s', flush=True)

    # bigram
    print(f'  预计算bigram ...', end=' ', flush=True)
    t0 = time.time()
    doc_bigrams = precompute_bigrams(doc_list)
    print(f'{time.time()-t0:.1f}s', flush=True)

    # 逐模型评估
    all_results = {}
    model_configs = C3BENCH_MODELS
    if selected_models:
        selected = set(selected_models)
        model_configs = [config for config in C3BENCH_MODELS if config[0] in selected]

    for model_name, model_path, encode_type, max_len in model_configs:
        print(f'\n  ---- {model_name} ----', flush=True)
        res = evaluate_one(queries, doc_list, pos_idx, doc_bigrams,
                           bm25_idx, bm25_sc, model_name, encode_type, model_path, max_len,
                           cache_suffix='c3bench')
        all_results[model_name] = res

    return all_results, cats, sources


def print_summary(results, cats, sources):
    """打印汇总表格"""
    lines = []
    lines.append('=' * 100)
    lines.append('C3bench 跨领域泛化性实验结果')
    lines.append('=' * 100)
    lines.append(f'数据集: C3bench-crosslang (纯古文文档, 1000查询)')
    lines.append(f'类别: {dict(cats.most_common())}')
    lines.append(f'来源前5: {dict(sources.most_common(5))}')
    lines.append('')

    # 表1: 各模型各方法 R@1
    lines.append('表1: C3bench 各方法 R@1 对比')
    lines.append(f'{"模型":<22} {"方法":<12} {"R@1":>8} {"R@5":>8} {"MRR":>8} {"ΔR@1(RRF)":>12}')
    lines.append('-' * 75)
    methods = ['BM25', 'Vector', 'RRF', 'Fixed fusion', 'LVF(Ours)']
    for model in results:
        rrf_r1 = results[model]['RRF']['R@1']
        for mi, method in enumerate(methods):
            m = results[model][method]
            gain = f'{(m["R@1"]-rrf_r1)*100:+.2f}%' if method != 'RRF' else 'baseline'
            mark = '★' if method == 'LVF(Ours)' else ' '
            lines.append(f'{model if mi==0 else "":<22} {method:<12} {m["R@1"]:>8.4f} {m["R@5"]:>8.4f} {m["MRR"]:>8.4f} {gain:>12} {mark}')
        lines.append('-' * 75)

    # 表2: LVF vs RRF 跨模型对比
    lines.append('\n表2: LVF vs RRF 跨模型对比 (R@1)')
    lines.append(f'{"模型":<22} {"BM25":>8} {"Vector":>8} {"RRF":>8} {"LVF":>8} {"Δ(RRF)":>10}')
    lines.append('-' * 60)
    for model in results:
        bm25_r1 = results[model]['BM25']['R@1']
        vec_r1 = results[model]['Vector']['R@1']
        rrf_r1 = results[model]['RRF']['R@1']
        fixed_r1 = results[model]['Fixed fusion']['R@1']
        lvf_r1 = results[model]['LVF(Ours)']['R@1']
        gain = (lvf_r1 - rrf_r1) * 100
        lines.append(f'{model:<22} {bm25_r1:>8.4f} {vec_r1:>8.4f} {rrf_r1:>8.4f} {fixed_r1:>8.4f} {lvf_r1:>8.4f} {gain:>+9.2f}%')

    # 表3: 与其他数据集对比 (LVF R@1)
    lines.append('\n表3: LVF跨数据集泛化性对比 (R@1, Finetune4B-full-v3)')
    lines.append(f'{"数据集":<16} {"类型":<10} {"BM25":>8} {"Vector":>8} {"RRF":>8} {"LVF":>8} {"Δ(RRF)":>10}')
    lines.append('-' * 75)

    # 加载已有跨数据集结果
    cross_path = f'{OUTPUT_DIR}/v3_cross_dataset_results.json'
    if os.path.exists(cross_path):
        with open(cross_path, 'r') as f:
            cross_raw = json.load(f)
        cross = cross_raw.get('results', cross_raw)
        for ds_name in ['sanguo_test', 'shiji', 'hanshu', 'history-10k']:
            if ds_name in cross and 'Finetune4B-full-v3' in cross[ds_name]:
                r = cross[ds_name]['Finetune4B-full-v3']
                ds_type = '问答' if ds_name != 'history-10k' else '翻译对'
                delta = (r['LVF(Ours)']['R@1'] - r['RRF']['R@1']) * 100
                lines.append(f'{ds_name:<16} {ds_type:<10} {r["BM25"]["R@1"]:>8.4f} {r["Vector"]["R@1"]:>8.4f} {r["RRF"]["R@1"]:>8.4f} {r["LVF(Ours)"]["R@1"]:>8.4f} {delta:>+9.2f}%')

    # C3bench结果
    if 'Finetune4B-full-v3' in results:
        r = results['Finetune4B-full-v3']
        delta = (r['LVF(Ours)']['R@1'] - r['RRF']['R@1']) * 100
        lines.append(f'{"C3bench":<16} {"跨语言":<10} {r["BM25"]["R@1"]:>8.4f} {r["Vector"]["R@1"]:>8.4f} {r["RRF"]["R@1"]:>8.4f} {r["LVF(Ours)"]["R@1"]:>8.4f} {delta:>+9.2f}%')

    summary_text = '\n'.join(lines)
    print('\n' + summary_text)

    # 保存汇总
    summary_path = f'{OUTPUT_DIR}/c3bench_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f'\n汇总已保存: {summary_path}')

    return summary_text


def main():
    parser = argparse.ArgumentParser(description='Run C3-derived retrieval evaluation.')
    parser.add_argument(
        '--models', nargs='+', choices=[config[0] for config in C3BENCH_MODELS],
        help='Evaluate only the selected encoder configurations (default: all).',
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 运行实验
    results, cats, sources = run_c3bench_experiment(args.models)

    # 打印汇总
    print_summary(results, cats, sources)

    # 保存JSON结果
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'C3bench-crosslang',
        'data_path': C3BENCH_DATA,
        'sample_ratio': SAMPLE_RATIO,
        'num_queries': 1000,
        'categories': dict(cats.most_common()),
        'top_sources': dict(sources.most_common(10)),
        'models_evaluated': list(results.keys()),
        'methods': ['BM25', 'Vector', 'RRF', 'Fixed fusion', 'LVF(Ours)'],
        'results': results,
    }

    output_path = f'{OUTPUT_DIR}/c3bench_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n结果已保存: {output_path}')
    print('=' * 100)


if __name__ == '__main__':
    main()
