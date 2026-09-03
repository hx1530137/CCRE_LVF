"""
LVF: Lexical-Verified Fusion — 核心算法实现
=============================================

论文核心方法: 一种无需训练的混合检索融合算法, 专为古文领域检索设计。

算法公式:
    LVF(d) = α(q)·ŝ_b(d) + (1-α(q))·ŝ_v(d) + γ·L(d) - δ·P(d)

三个核心模块:
    1. 查询自适应权重 α(q)  — 根据BM25/向量检索分数margin动态调整融合权重
    2. 词汇验证 L(d)        — 字符bigram重叠验证候选文档的词汇可信度
    3. 跨模态怀疑度 P(d)     — 识别在双模态中均排名靠前但可能为难负例的候选

参数:
    α_base=0.4, α_range=0.10, α_scale=20  (查询自适应)
    γ=0.3                                  (词汇验证权重)
    δ=0.2                                  (跨模态怀疑度惩罚强度)
"""
import numpy as np


def minmax_normalize(candidates, score_map):
    """对候选文档的原始分数做min-max归一化到[0,1]"""
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
    """
    计算分数margin: top1与top2~top_n+1均值的差距
    margin越大 → 检索器对该查询越自信
    """
    if len(norm_scores) < 2:
        return 0
    sorted_s = np.sort(norm_scores)[::-1]
    top1 = sorted_s[0]
    rest = sorted_s[1:min(top_n + 1, len(sorted_s))]
    return top1 - rest.mean() if len(rest) > 0 else 0


def precompute_bigrams(doc_list):
    """
    预计算每个文档的字符bigram集合
    古文中字符bigram是核心语义单元(如"曹操"="曹"+"操"的bigram)
    """
    doc_bigrams = []
    for doc in doc_list:
        bigrams = set(doc[i:i + 2] for i in range(len(doc) - 1))
        doc_bigrams.append(bigrams)
    return doc_bigrams


def compute_query_bigram_overlap(query, doc_bigrams):
    """
    模块2: 词汇验证 L(d)
    计算查询与每个文档的字符bigram重叠率
    L(d) = |bigrams(q) ∩ bigrams(d)| / |bigrams(q)|

    古文检索中, 字符bigram重叠是强正信号(权重+9.05, 从LTR特征分析得出)
    """
    q_bigrams = set(query[i:i + 2] for i in range(len(query) - 1))
    n_docs = len(doc_bigrams)
    overlap = np.zeros(n_docs)
    if not q_bigrams:
        return overlap
    for d in range(n_docs):
        overlap[d] = len(q_bigrams & doc_bigrams[d]) / len(q_bigrams)
    return overlap


def lvf_fusion(bm25_indices, bm25_scores, vec_indices, vec_scores, queries, doc_bigrams,
               alpha_base=0.4, alpha_range=0.10, alpha_scale=20,
               gamma=0.3, delta=0.2, top_k=100):
    """
    LVF: Lexical-Verified Fusion (完整实现)

    输入:
        bm25_indices, bm25_scores : BM25检索的文档索引和原始分数 [n_queries, top_k]
        vec_indices, vec_scores   : 向量检索的文档索引和原始分数 [n_queries, top_k]
        queries                   : 查询文本列表 [n_queries]
        doc_bigrams               : 预计算的文档bigram集合列表 [n_docs]

    输出:
        result : 融合后的top_k文档索引 [n_queries, top_k]

    公式:
        LVF(d) = α(q)·ŝ_b(d) + (1-α(q))·ŝ_v(d) + γ·L(d) - δ·P(d)

        其中:
        α(q) = α_base + α_range·σ((margin_b - margin_v)·α_scale)   [模块1: 查询自适应]
        L(d) = bigram_overlap(q, d)                                  [模块2: 词汇验证]
        P(d) = ŝ_b(d) × ŝ_v(d)                                      [模块3: 跨模态怀疑度]
    """
    n = len(bm25_indices)
    result = np.zeros((n, top_k), dtype=np.int32)

    for i in range(n):
        # --- 合并候选文档, 计算归一化分数 ---
        bm25_map = dict(zip(bm25_indices[i].tolist(), bm25_scores[i].tolist()))
        vec_map = dict(zip(vec_indices[i].tolist(), vec_scores[i].tolist()))
        candidates = list(dict.fromkeys(list(bm25_map.keys()) + list(vec_map.keys())))
        b_norm = minmax_normalize(candidates, bm25_map)  # ŝ_b(d)
        v_norm = minmax_normalize(candidates, vec_map)   # ŝ_v(d)

        # --- 模块1: 查询自适应权重 α(q) ---
        # 根据BM25和向量检索的margin差异, 动态调整权重
        # margin_b > margin_v → BM25更自信 → 增大α(偏向BM25)
        # margin_b < margin_v → 向量更自信 → 减小α(偏向向量)
        b_margin = compute_margin(b_norm, 5)
        v_margin = compute_margin(v_norm, 5)
        diff = b_margin - v_margin
        alpha = alpha_base + alpha_range * (1 / (1 + np.exp(-diff * alpha_scale)))

        # --- 模块2: 词汇验证 L(d) ---
        # 字符bigram重叠率, 验证候选文档与查询的词汇匹配度
        # 古文中同一实体的不同表述(如"曹操"/"魏武")仍有字符重叠
        bigram_overlap = compute_query_bigram_overlap(queries[i], doc_bigrams)
        L = np.array([bigram_overlap[c] if c < len(bigram_overlap) else 0 for c in candidates])

        # --- 模块3: 跨模态怀疑度 P(d) ---
        # BM25和向量分数都高 → 疑似难负例 → 惩罚
        # 难负例: 在双模态中均排名靠前但非正确答案的候选
        # 这类候选最容易被误判, 通过乘积P(d)=ŝ_b×ŝ_v识别并惩罚
        P = b_norm * v_norm

        # --- 最终融合 ---
        final = alpha * b_norm + (1 - alpha) * v_norm + gamma * L - delta * P
        sorted_idx = np.argsort(-final)
        result[i] = [candidates[j] for j in sorted_idx[:top_k]]

    return result


# ==================== 消融变体 ====================
def was_hybrid(bm25_idx, bm25_sc, vec_idx, vec_sc, alpha=0.4, top_k=100):
    """WAS baseline: 固定α加权融合, 无L无P"""
    n = len(bm25_idx)
    result = np.zeros((n, top_k), dtype=np.int32)
    for i in range(n):
        bm25_map = dict(zip(bm25_idx[i].tolist(), bm25_sc[i].tolist()))
        vec_map = dict(zip(vec_idx[i].tolist(), vec_sc[i].tolist()))
        candidates = list(dict.fromkeys(list(bm25_map.keys()) + list(vec_map.keys())))
        b_norm = minmax_normalize(candidates, bm25_map)
        v_norm = minmax_normalize(candidates, vec_map)
        final = alpha * b_norm + (1 - alpha) * v_norm
        sorted_idx = np.argsort(-final)
        result[i] = [candidates[j] for j in sorted_idx[:top_k]]
    return result


def rrf_fusion(bm25_idx, vec_idx, k=60, top_k=100):
    """RRF baseline: Reciprocal Rank Fusion"""
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
