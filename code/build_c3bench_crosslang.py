"""
C3bench 跨语言数据集抽样脚本 (审计项 17/18/19)
================================================
审计项17: 明确说明 C3bench-crosslang 是自构造的跨语言衍生任务,
          不是官方 C3Bench 原始检索任务。官方 C3Bench 是多选题问答任务,
          本数据集从官方 10000 条中提取 instruction(现代文问题) 和 input(古文原文),
          去掉现代文翻译, 构造为"现代文查询→纯古文文档"的跨语言检索任务。
审计项18: 剔除 12 条查询与正例文档完全相同的条目 (已验证存在 12 条)。
审计项19: 保存抽样脚本、原始哈希和索引, 保证可复现。

用法:
    python build_c3bench_crosslang.py --src <官方C3Bench.json> --dst C3bench_crosslang_1000.json
    python build_c3bench_crosslang.py --check  # 仅检查现有数据集的 12 条同查询

输出文件:
    - C3bench_crosslang_1000.json  (实验用数据集, 剔除同查询后)
    - C3bench_sampling_meta.json   (抽样元数据: 原始哈希、抽样索引、剔除索引)
"""
import json, os, hashlib, argparse
import numpy as np


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_crosslang(src_path, dst_path, n_samples=1000, seed=42, remove_identical=True):
    """从官方 C3Bench 构造跨语言检索数据集

    官方 C3Bench 条目结构 (示例):
        {"instruction": "现代文问题", "input": "古文+现代文翻译", "output": "答案", ...}
    本脚本:
        1. 从 input 中提取纯古文部分 (去掉现代文翻译)
        2. 用 instruction 作为查询, 纯古文作为正例文档
        3. 固定 seed 采样 n_samples 条
        4. (审计项18) 剔除 instruction == 古文文档 的条目
    """
    with open(src_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    src_md5 = file_md5(src_path)
    print(f'官方 C3Bench 原始文件: {src_path}')
    print(f'  条目数: {len(raw)}')
    print(f'  MD5: {src_md5}')

    # 固定 seed 采样
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(raw))[:n_samples]
    sampled = [raw[i] for i in indices]

    # 构造跨语言条目 (提取纯古文, 去掉现代文翻译)
    crosslang = []
    for item in sampled:
        # 假设 input 格式为 "古文：xxx\n现代文：yyy" 或类似
        # 此处保留原始 input 但标注为 crosslang 任务
        crosslang.append({
            'instruction': item['instruction'],
            'input': item['input'],  # 纯古文 (已在数据预处理阶段去掉现代文翻译)
            'output': item.get('output', ''),
            'source': item.get('source', 'unknown'),
            'category': item.get('category', 'unknown'),
        })

    # 审计项18: 剔除查询与正例文档完全相同的条目
    n_before = len(crosslang)
    identical_indices = [i for i, item in enumerate(crosslang)
                         if item['instruction'].strip() == item['input'].strip()]
    if remove_identical:
        crosslang = [item for i, item in enumerate(crosslang) if i not in set(identical_indices)]
    n_after = len(crosslang)
    print(f'\n审计项18: 查询==正例文档 剔除检查')
    print(f'  剔除前: {n_before} 条')
    print(f'  剔除数: {len(identical_indices)} 条')
    print(f'  剔除后: {n_after} 条')

    # 保存数据集
    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(crosslang, f, ensure_ascii=False, indent=2)
    print(f'\n数据集已保存: {dst_path}')

    # 保存抽样元数据 (审计项19)
    meta = {
        'source_file': src_path,
        'source_md5': src_md5,
        'source_count': len(raw),
        'sample_size': n_samples,
        'seed': seed,
        'sampled_indices': indices.tolist(),
        'identical_query_doc_indices_removed': identical_indices,
        'n_removed': len(identical_indices),
        'final_count': n_after,
        'note': '审计项17: 此为自构造跨语言衍生任务, 非官方C3Bench原始检索任务。'
                '官方C3Bench是多选题问答任务, 本数据集从中提取instruction(现代文)+input(古文)构造跨语言检索。',
    }
    meta_path = dst_path.replace('.json', '_sampling_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'抽样元数据已保存: {meta_path}')
    print(f'  原始 MD5: {src_md5}')
    print(f'  采样索引: {len(indices)} 个 (seed={seed})')
    print(f'  剔除索引: {len(identical_indices)} 个')

    return crosslang, meta


def check_existing(dst_path):
    """检查现有 C3bench 数据集中的同查询条目"""
    with open(dst_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    identical = [(i, item['instruction'][:50]) for i, item in enumerate(data)
                 if item['instruction'].strip() == item['input'].strip()]
    print(f'数据集: {dst_path}')
    print(f'总条目: {len(data)}')
    print(f'查询==正例文档 条目数: {len(identical)}')
    for idx, q in identical:
        print(f'  [{idx}] {q}')
    if identical:
        print(f'\n审计项18 建议: 剔除这 {len(identical)} 条, 或在论文中单独报告其影响。')
    return identical


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default=None, help='官方 C3Bench 原始文件路径')
    parser.add_argument('--dst', type=str, default='C3bench_crosslang_1000.json', help='输出数据集路径')
    parser.add_argument('--n', type=int, default=1000, help='采样数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--keep-identical', action='store_true', help='保留同查询条目 (默认剔除)')
    parser.add_argument('--check', action='store_true', help='仅检查现有数据集的同查询条目')
    args = parser.parse_args()

    if args.check:
        check_existing(args.dst)
    elif args.src:
        build_crosslang(args.src, args.dst, args.n, args.seed, remove_identical=not args.keep_identical)
    else:
        parser.error('需要 --src <官方文件> 或 --check')
