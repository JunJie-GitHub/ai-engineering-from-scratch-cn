---
name: recall@K 运行器
description: 编写一个规范的 K值召回率 (Recall@K) 评估框架，支持训练集/验证集/图库划分及明确的数据契约
version: 1.0.0
phase: 4
lesson: 20
tags: [检索, 评估, 召回率, faiss]
---

# K值召回率 (Recall@K) 运行器

将包含查询图像、图库图像及其标签的文件夹，转化为可复现的 K值召回率 (Recall@K) 指标。

## 适用场景

- 新主干网络 (Backbone) 的首个检索基准测试。
- 跟踪微调 (Fine-tuning) 各轮次中嵌入向量 (Embedding) 的质量变化。
- 在同一数据集上对比两种检索系统。

## 输入参数

- `query_images`：路径列表。
- `gallery_images`：路径列表（查询集与图库可能存在重叠）。
- `query_labels`、`gallery_labels`：类别或实例 ID。
- `encoder_fn`：可调用函数 `image -> embedding`（预计算或实时生成）。
- `ks`：列表，例如 `[1, 5, 10]`。

## 执行步骤

1. 对每张图库图像进行编码。保存为 NumPy 数组。
2. 对每张查询图像进行编码。
3. 对两组嵌入向量进行 L2 归一化 (L2-normalisation)。
4. 针对每个查询，计算其与所有图库项的相似度。
5. 按降序排序，取前 `max(ks)` 个结果。
6. 针对每个 K 值，检查 Top-K 图库项中是否存在与查询标签相同的项。
7. 输出结果：`recall@K = 在 Top K 中至少包含一个正确邻居的查询所占比例`。

## 输出模板

import numpy as np
from sklearn.preprocessing import normalize

def encode_all(images, encoder_fn, batch=32):
    out = []
    for i in range(0, len(images), batch):
        embs = encoder_fn(images[i:i + batch])
        out.append(embs)
    return np.concatenate(out)


def recall_at_k(query_emb, gallery_emb, q_labels, g_labels,
                ks=(1, 5, 10), query_ids=None, gallery_ids=None):
    if len(query_emb) == 0 or len(gallery_emb) == 0:
        return {f"recall@{k}": 0.0 for k in ks}

    g_label_set = set(g_labels.tolist())
    keep = np.array([lbl in g_label_set for lbl in q_labels])
    if not keep.any():
        return {f"recall@{k}": 0.0 for k in ks}

    q_emb_f = query_emb[keep]
    q_lab_f = q_labels[keep]
    q_id_f = query_ids[keep] if query_ids is not None else None

    q = normalize(q_emb_f)
    g = normalize(gallery_emb)
    sims = q @ g.T

    if q_id_f is not None and gallery_ids is not None:
        self_mask = q_id_f[:, None] == gallery_ids[None, :]
        sims = np.where(self_mask, -np.inf, sims)

    top_k_max = min(max(ks), g.shape[0])
    if top_k_max <= 0:
        return {f"recall@{k}": 0.0 for k in ks}

    top = np.argpartition(-sims, top_k_max - 1, axis=1)[:, :top_k_max]
    sorted_top = np.take_along_axis(
        top, np.argsort(-sims[np.arange(len(q))[:, None], top], axis=1), axis=1
    )
    out = {}
    for k in ks:
        k_eff = min(k, top_k_max)
        hits = np.any(g_labels[sorted_top[:, :k_eff]] == q_lab_f[:, None], axis=1)
        out[f"recall@{k}"] = float(hits.mean())
    return out


def evaluate(query_images, query_labels, gallery_images, gallery_labels, encoder_fn, ks=(1, 5, 10)):
    q_emb = encode_all(query_images, encoder_fn)
    g_emb = encode_all(gallery_images, encoder_fn)
    return recall_at_k(q_emb, g_emb, np.array(query_labels), np.array(gallery_labels), ks)

## 报告格式

[evaluation]
  num queries:   <int>
  num gallery:   <int>
  embedding_dim: <int>

[recall]
  recall@1:  <float>
  recall@5:  <float>
  recall@10: <float>

## 注意事项

- 计算相似度前需对嵌入向量进行归一化；在归一化向量上使用 FAISS 的 `IndexFlatIP` 等价于计算余弦相似度 (Cosine Similarity)。
- 若查询的真实标签 (Ground-truth) 未出现在图库中，应将其排除；否则召回率将被人为限制在 1 以下。
- 若查询集与图库存在重叠，需将查询自身从其 Top-K 结果中剔除，否则测量的是自相似度而非检索能力。
- 当 `num_queries > 10,000` 时，应对相似度矩阵乘法 (Matrix Multiplication) 进行分批处理，以避免内存溢出 (Out Of Memory)。