"""
合并所有模型对比结果
====================
将 paper_supplementary (9模型) + 新测试的2个LoRA模型 (Finetune4B-2k, DualView4B)
合并为完整的11模型对比, 保存到两个位置:
  - paper_supplementary/results/all_models_comparison.json
  - experiment_v3/results/all_models_comparison.json
"""
import json

PAPER_JSON = '/home/huxin/Documents/trae_projects/sikuBERT/paper_supplementary/results/all_models_comparison.json'
EXP_JSON   = '/home/huxin/Documents/trae_projects/sikuBERT/experiment_v3/results/all_models_comparison.json'

# 加载 paper_supplementary 版本 (9模型, 含 bge-m3/Qwen3-4B-Base/Finetune4B-full-v3)
with open(PAPER_JSON, 'r', encoding='utf-8') as f:
    paper_data = json.load(f)

# 加载 experiment_v3 版本 (含新测试的 Finetune4B-2k, DualView4B)
with open(EXP_JSON, 'r', encoding='utf-8') as f:
    exp_data = json.load(f)

# 合并: 以 paper 版本为基础, 加入 exp 中的新模型
merged = dict(paper_data)
for name, res in exp_data['results'].items():
    if name not in merged['results']:
        merged['results'][name] = res
        print(f'新增: {name} -> LVF={res["LVF"]:.4f}')

# 按 LVF R@1 降序排序
sorted_results = dict(sorted(
    merged['results'].items(),
    key=lambda x: -x[1].get('LVF', 0) if isinstance(x[1], dict) and 'LVF' in x[1] else 0
))
merged['results'] = sorted_results
merged['description'] = '所有embedding模型完整对比 (sanguo_test, R@1) - 共11个模型'

# 保存到两个位置
for path in [PAPER_JSON, EXP_JSON]:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

print(f'\n合并完成: {len(merged["results"])}个模型')
print(f'已保存到:')
print(f'  - {PAPER_JSON}')
print(f'  - {EXP_JSON}')

# 打印汇总
print('\n' + '=' * 90)
print(f'{"Model":<28} {"Params":>10} {"Vector":>8} {"RRF":>8} {"LVF":>8} {"Δ(RRF)":>10}')
print('-' * 90)

params_map = {
    'Finetune4B-full-v3': '4B+5.9M',
    'Finetune4B-2k':      '4B+5.9M',
    'DualView4B':         '4B+5.9M',
    'bge-m3':             '568M',
    'Qwen3-4B-Base':      '4B',
    'Qwen3-Embedding-8B': '8B',
    'bge-large-zh-v1.5':  '326M',
    'Qwen3-Embedding-0.6B':'0.6B',
    'multilingual-e5-large':'560M',
    'bge-small-zh-v1.5':  '24M',
    'multilingual-e5-base':'278M',
}

for name, r in merged['results'].items():
    if 'error' in r:
        print(f'{name:<28} ERROR')
    else:
        p = params_map.get(name, '?')
        print(f'{name:<28} {p:>10} {r["Vector"]:>8.4f} {r["RRF"]:>8.4f} {r["LVF"]:>8.4f} {r["delta_rrf"]:>+10.4f}')
