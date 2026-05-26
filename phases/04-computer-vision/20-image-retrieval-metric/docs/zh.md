# 图像检索 (Image Retrieval) 与度量学习 (Metric Learning)

> 检索系统通过嵌入空间 (Embedding Space) 中的距离对候选结果进行排序。度量学习是一门塑造该空间的学科，旨在让距离度量符合你的业务需求。

**类型：** 构建
**语言：** Python
**前置条件：** 第4阶段第14课（ViT）、第4阶段第18课（CLIP）
**时长：** 约45分钟

## 学习目标

- 解释三元组损失 (Triplet Loss)、对比损失 (Contrastive Loss) 和基于代理的度量学习损失 (Proxy-based Metric Learning Loss)，并为给定数据集选择合适的损失函数
- 正确实现 L2 归一化 (L2-normalisation) 与余弦相似度 (Cosine Similarity)，并审查“相同物品”与“相同类别”检索之间的差异
- 构建 FAISS 索引，支持通过文本和图像进行查询，并在预留查询集上报告 Recall@K 指标
- 将 DINOv2、CLIP 和 SigLIP 作为开箱即用的嵌入骨干网络 (Embedding Backbone) 使用，并了解各自的优势场景

## 问题背景

检索技术在工业级视觉应用中无处不在：重复检测、以图搜图、视觉搜索（“查找相似商品”）、人脸重识别、监控场景中的人员重识别 (Person Re-ID)，以及电商领域的实例级匹配。产品层面的核心问题始终如一：“给定这张查询图像，如何对我的商品目录进行排序？”

整个系统由两项设计决策主导：嵌入 (Embedding)——即使用何种模型生成向量；索引 (Index)——即如何大规模查找最近邻 (Nearest Neighbours)。到 2026 年，这两者都已成为标准化组件（DINOv2 用于嵌入，FAISS 用于索引），这也提高了技术门槛：真正的难点在于为你的应用定义*“何为相似”*，随后调整嵌入空间，使距离度量与之匹配。

这种空间塑造过程就是度量学习。它是一门体量虽小但杠杆效应极高的学科。

## 核心概念

### 检索 (Retrieval) 概览

flowchart LR
    Q["Query image<br/>or text"] --> ENC["Encoder"]
    ENC --> EMB["Query embedding"]
    EMB --> IDX["FAISS index"]
    CAT["Catalogue images"] --> ENC2["Encoder (same)"] --> IDX_BUILD["Build index"]
    IDX_BUILD --> IDX
    IDX --> RANK["Top-k nearest<br/>by cosine / L2"]
    RANK --> OUT["Ranked results"]

    style ENC fill:#dbeafe,stroke:#2563eb
    style IDX fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a

### 四大损失函数 (Loss) 家族

| 损失函数 (Loss) | 需要 (Requires) | 优点 (Pros) | 缺点 (Cons) |
|------|----------|------|------|
| **对比损失 (Contrastive Loss)** | (锚点 (anchor), 正样本 (positive)) + 负样本 (negative) | 简单，适用于任何成对标签 | 负样本不足时收敛缓慢 |
| **三元组损失 (Triplet Loss)** | (锚点, 正样本, 负样本) | 直观；可直接控制间隔 (margin) | 困难三元组挖掘 (hard-triplet mining) 计算成本高 |
| **NT-Xent / InfoNCE** | 样本对 + 批次内挖掘的负样本 | 易于扩展至大批次 | 需要大批次或动量队列 (momentum queue) |
| **基于代理的损失 (Proxy-based Loss, 如 ProxyNCA)** | 仅需类别标签 | 快速、稳定，无需挖掘 | 在小数据集上容易对代理过拟合 |

对于大多数生产环境用例，建议首先使用预训练骨干网络 (pretrained backbone)，仅当现成嵌入 (off-the-shelf embeddings) 在测试集上表现不佳时，再引入度量学习 (metric learning) 微调。

### 三元组损失 (Triplet Loss) 公式

L = max(0, ||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + margin)

该损失函数旨在将锚点 `a` 拉近至正样本 `p`，同时推离负样本 `n`，并通过 `margin`（间隔）确保两者之间存在安全距离。这种三样本结构可泛化至任意相似度排序任务。

样本挖掘 (mining) 策略至关重要：简单三元组（`n` 已远离 `a`）产生的损失为零，只有困难三元组才能有效训练网络。半困难挖掘 (semi-hard mining)（`n` 比 `p` 远但仍在 `margin` 内）是 2016 年 FaceNet 提出的经典策略，至今仍是主流。

### 余弦相似度 (Cosine Similarity) 与 L2 距离 (L2 Distance)

两种度量标准，对应两种惯例：

- **余弦相似度**：向量间的夹角。要求嵌入向量经过 L2 归一化 (L2-normalised)。
- **L2 距离**：欧几里得距离 (Euclidean distance)。适用于原始或归一化后的嵌入，但通常与 L2 归一化结合使用，并计算平方 L2 距离。

对于大多数现代网络，两者在数学上是等价的：当 `||a|| = ||b|| = 1` 时，`||a - b||^2 = 2 - 2 cos(a, b)`。请选择与嵌入训练阶段一致的度量方式；混用会悄无声息地改变“最近邻”的定义。

### 召回率@K (Recall@K)

检索任务的标准评估指标：

recall@K = fraction of queries where at least one correct match is in the top K results

建议并列报告 recall@1、@5 和 @10。若 recall@10 高于 0.95 但 recall@1 低于 0.5，表明嵌入空间 (embedding space) 的整体结构正确，但排序结果存在噪声——可尝试延长微调时间或引入重排序 (re-ranking) 步骤。

在重复项检测任务中，精确率@K (precision@K) 更为关键，因为每一个假阳性 (false positive) 都是用户可见的错误。而在视觉搜索场景中，recall@K 才是核心产品指标。

### 一段话了解 FAISS

Facebook AI Similarity Search。目前业界事实标准 (de-facto) 的最近邻搜索 (nearest-neighbour search) 库。提供三种索引选择：

- `IndexFlatIP` / `IndexFlatL2` —— 暴力搜索，结果精确，无需训练。适用于约 100 万以内的向量。
- `IndexIVFFlat` —— 将向量空间划分为 K 个单元 (cells)，仅搜索最近的几个单元。近似搜索，速度快，需要训练数据。
- `IndexHNSW` —— 基于图结构，在海量查询下速度最快，但索引体积较大。

对于 10 万级向量，通常建议在余弦相似度下使用 `IndexFlatIP`。对于 1000 万级向量，推荐使用 `IndexIVFFlat`。对于 1 亿级以上向量，则需结合乘积量化 (product quantisation) 使用 `IndexIVFPQ`。

### 实例级检索 (Instance-level Retrieval) 与类别级检索 (Category-level Retrieval)

名称相似但本质截然不同的两类问题：

- **类别级检索** —— “在我的商品目录中找出所有猫。” 属于类别条件相似度匹配；现成的 CLIP / DINOv2 嵌入通常表现良好。
- **实例级检索** —— “在我的商品目录中找出*这款确切的商品*。” 需要对同类别中视觉相似的对象进行细粒度区分 (fine-grained discrimination)；现成嵌入往往表现不佳，此时基于度量学习的微调至关重要。

在选择模型前，务必先明确你要解决的是哪一类问题。

## 构建模型

### 步骤 1：三元组损失（Triplet Loss）

import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=0.2):
    d_ap = F.pairwise_distance(anchor, positive, p=2)
    d_an = F.pairwise_distance(anchor, negative, p=2)
    return F.relu(d_ap - d_an + margin).mean()

仅需一行代码。适用于 L2 归一化（L2-normalised）或原始嵌入向量（raw embeddings）。

### 步骤 2：半难样本挖掘（Semi-hard Mining）

给定一批嵌入向量（embeddings）和标签，为每个锚点样本（anchor）找出最难的半难负样本（semi-hard negative）。

def semi_hard_negatives(emb, labels, margin=0.2):
    dist = torch.cdist(emb, emb)
    same_class = labels[:, None] == labels[None, :]
    diff_class = ~same_class
    N = emb.size(0)

    positives = dist.clone()
    positives[~same_class] = float("-inf")
    positives.fill_diagonal_(float("-inf"))
    pos_idx = positives.argmax(dim=1)

    semi_hard = dist.clone()
    semi_hard[same_class] = float("inf")
    d_ap = dist[torch.arange(N), pos_idx].unsqueeze(1)
    semi_hard[dist <= d_ap] = float("inf")
    neg_idx = semi_hard.argmin(dim=1)

    fallback_mask = semi_hard[torch.arange(N), neg_idx] == float("inf")
    if fallback_mask.any():
        hardest = dist.clone()
        hardest[same_class] = float("inf")
        neg_idx = torch.where(fallback_mask, hardest.argmin(dim=1), neg_idx)
    return pos_idx, neg_idx

每个锚点样本将匹配同类中最难的正样本（positive），以及一个距离大于正样本但仍在边界值（margin）内的半难负样本。

### 步骤 3：K 值召回率（Recall@K）

def recall_at_k(query_emb, gallery_emb, query_labels, gallery_labels, k=1):
    sim = query_emb @ gallery_emb.T
    _, top_k = sim.topk(k, dim=-1)
    matches = (gallery_labels[top_k] == query_labels[:, None]).any(dim=-1)
    return matches.float().mean().item()

在 L2 归一化嵌入向量上按内积（inner product）取 Top-K，等价于按余弦相似度（cosine similarity）取 Top-K。该指标报告的是至少包含一个正确邻居的查询样本（query）所占的平均比例。

### 步骤 4：整合代码

import torch
import torch.nn as nn
from torch.optim import Adam

class Encoder(nn.Module):
    def __init__(self, in_dim=128, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

torch.manual_seed(0)
num_classes = 6
protos = F.normalize(torch.randn(num_classes, 128), dim=-1)

def sample_batch(bs=32):
    labels = torch.randint(0, num_classes, (bs,))
    x = protos[labels] + 0.15 * torch.randn(bs, 128)
    return x, labels

enc = Encoder()
opt = Adam(enc.parameters(), lr=3e-3)

for step in range(200):
    x, y = sample_batch(32)
    emb = enc(x)
    pos_idx, neg_idx = semi_hard_negatives(emb, y)
    loss = triplet_loss(emb, emb[pos_idx], emb[neg_idx])
    opt.zero_grad(); loss.backward(); opt.step()

经过数百步训练后，嵌入向量将按类别各自聚集成簇。

## 使用模型

2026 年的生产级技术栈：

- **DINOv2 + FAISS** — 通用视觉检索（Visual Retrieval）。开箱即用。
- **CLIP + FAISS** — 适用于文本查询（Text Query）场景。
- **Fine-tuned DINOv2 + FAISS** — 实例级检索（Instance-level Retrieval）、人脸重识别（Face Re-ID）、时尚与电商领域。
- **Milvus / Weaviate / Qdrant** — 基于 FAISS 或 HNSW 的托管向量数据库（Vector DB）封装。

若要实现当前最优（State-of-the-Art, SOTA）的实例检索，标准流程为：使用 DINOv2 作为骨干网络（Backbone），添加嵌入头（Embedding Head），在实例标注的数据对上使用三元组损失（Triplet Loss）或 InfoNCE 损失进行微调，最后使用 FAISS 构建索引。

## 交付上线

本课时将产出：

- `outputs/prompt-retrieval-loss-picker.md` — 一个提示词（Prompt），用于针对特定检索问题自动选择三元组损失 / InfoNCE / ProxyNCA。
- `outputs/skill-recall-at-k-runner.md` — 一个技能模块，用于编写清晰的 Recall@K 评估框架（Evaluation Harness），包含训练集/验证集/图库（Gallery）划分及规范的数据契约（Data Contract）。

## 练习

1. **（简单）** 运行上述简易示例（Toy Example）。使用主成分分析（PCA）绘制训练前后的嵌入向量（Embeddings）分布图，观察六个聚类（Clusters）的形成过程。
2. **（中等）** 实现 ProxyNCA 损失：为每个类别学习一个“代理”（Proxy），基于余弦相似度计算标准交叉熵（Cross-Entropy）。在简易数据上对比其与三元组损失的收敛速度。
3. **（困难）** 选取 1,000 张 ImageNet 验证集图像，通过 HuggingFace 使用 DINOv2 提取嵌入向量，构建 FAISS 扁平索引（Flat Index）。分别以相同图像作为查询（预期 Recall 应为 1.0）以及以带有 ImageNet 标签作为真实值（Ground Truth）的独立划分集作为查询，报告 Recall@{1, 5, 10} 指标。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 度量学习（Metric Learning） | “塑造特征空间” | 训练编码器，使其输出空间中的距离能够反映目标相似度 |
| 三元组损失（Triplet Loss） | “拉近与推远” | L = max(0, d(a, p) - d(a, n) + margin)；度量学习的经典损失函数 |
| 半难样本挖掘（Semi-hard Mining） | “有用的负样本” | 距离锚点（Anchor）比正样本远，但仍在边界（Margin）内的负样本；经验上最具信息量 |
| 基于代理的损失（Proxy-based Loss） | “类别原型” | 每个类别学习一个代理；基于与代理的相似度计算交叉熵；无需样本对挖掘 |
| Recall@K | “Top-K 命中率” | 查询结果 Top-K 中至少包含一个正确结果的查询比例 |
| 实例检索（Instance Retrieval） | “精准查找该物体” | 细粒度匹配；开箱即用的特征通常表现不佳 |
| FAISS | “近邻搜索库” | Facebook 开发的最近邻（Nearest-Neighbour）搜索库；支持精确与近似索引 |
| HNSW | “图索引” | 分层可导航小世界（Hierarchical Navigable Small World）；内存开销小的快速近似最近邻搜索算法 |

## 延伸阅读

- [FaceNet: A Unified Embedding for Face Recognition (Schroff et al., 2015)](https://arxiv.org/abs/1503.03832) — 介绍三元组损失（Triplet Loss）与半难样本挖掘（Semi-hard Mining）的论文
- [In Defense of the Triplet Loss for Person Re-Identification (Hermans et al., 2017)](https://arxiv.org/abs/1703.07737) — 三元组损失微调（Triplet Fine-tuning）的实战指南
- [FAISS documentation](https://github.com/facebookresearch/faiss/wiki) — 全面解析各类索引（Index）与性能权衡（Trade-off）
- [SMoT: Metric Learning Taxonomy (Kim et al., 2021)](https://arxiv.org/abs/2010.06927) — 现代损失函数（Loss Functions）及其内在联系的综述