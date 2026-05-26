# 开放词汇视觉（Open-Vocabulary Vision）—— CLIP

> 联合训练图像编码器（Image Encoder）与文本编码器（Text Encoder），使匹配的（图像，文本描述）对在共享特征空间中映射到相同的位置。这就是其核心原理。

**类型：** 构建与使用
**编程语言：** Python
**前置知识：** 第4阶段第14课（ViT）、第4阶段第17课（自监督学习）
**预计耗时：** 约45分钟

## 学习目标

- 解释 CLIP 的双塔架构（Two-Tower Architecture）与对比训练目标（Contrastive Training Objective）
- 使用预训练的 CLIP（或 SigLIP）进行零样本分类（Zero-Shot Classification），无需任何针对特定任务的训练
- 从零实现零样本分类：编码类别提示词（Class Prompts）、计算余弦相似度（Cosine Similarity）并取最大值索引（Argmax）
- 区分 CLIP、SigLIP、OpenCLIP 以及 LLaVA/LLaMA-vision 模型——明确它们在 2026 年的各自定位与用途

## 问题背景

传统分类器属于闭词汇（Closed-Vocabulary）模型：例如一个包含 1000 个类别的 ImageNet 模型只能预测这 1000 个标签。每引入一个新类别，都需要标注数据并重新训练分类头（Classification Head）。

CLIP（Radford 等人，OpenAI 2021）的研究表明，利用从网络上抓取的 4 亿个（图像，文本描述）对进行训练，可以生成一个在推理阶段能够对任意类别集合进行分类的模型，且类别描述完全使用自然语言。你只需写一句话，就能为它定义一个新类别。

这种能力——零样本迁移（Zero-Shot Transfer）——正是所有现代视觉系统都以 CLIP 系列模型检查点（Checkpoint）作为起点的原因。目标检测（Grounding DINO、OWL-ViT）、图像分割（CLIPSeg、SAM）、图像检索、内容审核、视觉语言模型（VLMs）以及文生图（Text-to-Image Generation）等任务，均建立在 CLIP 风格的联合嵌入（Joint Embeddings）基础之上。

## 核心概念

### 双塔架构 (Two Towers)

flowchart LR
    IMG["Image"] --> IENC["Image encoder<br/>(ViT-L/14)"] --> IEMB["Image embedding<br/>(1024,)"]
    TXT["Caption"] --> TENC["Text encoder<br/>(transformer)"] --> TEMB["Text embedding<br/>(1024,)"]
    IEMB --> SIM["Cosine similarity"]
    TEMB --> SIM

    style IENC fill:#dbeafe,stroke:#2563eb
    style TENC fill:#fef3c7,stroke:#d97706
    style SIM fill:#dcfce7,stroke:#16a34a

两个编码器最终均通过线性投影层映射至相同的嵌入维度 (Embedding dimension)（CLIP-B/32 为 512，CLIP-L/14 为 1024）。随后对向量进行 L2 归一化 (L2-normalization) 并计算余弦相似度 (Cosine similarity)。

### 优化目标 (The Objective)

给定一个包含 N 个（图像，文本描述）对的批次 (Batch)，构建一个 N×N 的相似度矩阵。训练两个编码器，使对角线元素（匹配对）的相似度最大化，而非对角线元素（非匹配对）的相似度最小化。

sim_matrix = image_embeddings @ text_embeddings.T / tau

loss_i2t = cross_entropy(sim_matrix,       targets=arange(N))
loss_t2i = cross_entropy(sim_matrix.T,     targets=arange(N))
loss = (loss_i2t + loss_t2i) / 2

该设计是对称的，因为模型需同时支持图像到文本与文本到图像的检索。`tau`（温度参数，Temperature）通常作为可学习的标量参数进行优化，初始值设为 0.07。

### SigLIP：更优的损失函数 (SigLIP: A Better Loss)

SigLIP（Zhai 等人，2023）使用逐对 Sigmoid (Per-pair Sigmoid) 替代了 Softmax：

loss = mean over pairs of log(1 + exp(-y_ij * sim_ij))
y_ij = +1 if matching, -1 otherwise

逐对损失 (Per-pair loss) 消除了 CLIP 所依赖的批次级归一化 (Batch-level normalization)。SigLIP 在较小批次大小 (Batch size) 下训练效果更佳，且在同等数据规模下性能可匹敌甚至超越 CLIP。

### 零样本分类 (Zero-shot Classification)

给定一个训练好的 CLIP 模型：

1. 为每个类别构建提示词 (Prompt)：“a photo of a {class}”。
2. 使用文本编码器对所有类别提示词进行编码，得到张量 `T`，形状为 (C, d)。
3. 对测试图像进行编码，得到张量 `I`，形状为 (1, d)。
4. 计算相似度：`I @ T.T`，形状为 (1, C)。
5. 执行 Argmax 操作 -> 得到预测类别。

提示词工程 (Prompt engineering) 至关重要。OpenAI 针对 ImageNet 发布了 80 个提示词模板（如 "a photo of a {}"、"a blurry photo of a {}"、"a sketch of a {}" 等）。对每个类别的所有模板嵌入向量 (Embeddings) 取平均值，可额外带来 1%~3% 的 Top-1 准确率 (Top-1 accuracy) 提升。

### 2026 年 CLIP 类模型的应用场景

- **零样本分类 (Zero-shot classification)** — 直接应用。
- **图像检索 (Image retrieval)** — 预先对所有图像进行一次编码，推理时仅对查询进行嵌入。
- **文本条件目标检测 (Text-conditioned detection)** — Grounding DINO、OWL-ViT 等模型将 CLIP 文本塔 (Text tower) 与检测器结合。
- **文本条件图像分割 (Text-conditioned segmentation)** — CLIPSeg；SAM 通过 CLIP 接收文本提示输入。
- **视觉语言模型 (VLMs)** — LLaVA、Qwen-VL、InternVL 将 CLIP 家族的视觉编码器接入大语言模型 (LLM)。
- **文生图 (Text-to-image generation)** — Stable Diffusion、DALL-E 3 以 CLIP 文本嵌入作为生成条件。

一旦构建了共享的嵌入空间 (Shared embedding space)，所有视觉与语言结合的任务均可转化为距离计算问题。

## 动手构建 (Build It)

### 步骤 1：微型双塔模型 (Two-Tower Model)

真实的 CLIP 模型采用视觉 Transformer (Vision Transformer, ViT) 与 Transformer 架构。在本课中，双塔结构基于预提取特征 (Pre-extracted Features) 使用小型多层感知机 (Multi-Layer Perceptron, MLP) 构建，以便在 CPU 上也能清晰观察到训练信号。

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    def __init__(self, img_in=128, txt_in=64, emb=64):
        super().__init__()
        self.image_proj = nn.Sequential(nn.Linear(img_in, 128), nn.ReLU(), nn.Linear(128, emb))
        self.text_proj = nn.Sequential(nn.Linear(txt_in, 128), nn.ReLU(), nn.Linear(128, emb))
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(1/0.07)

    def forward(self, img_feats, txt_feats):
        i = F.normalize(self.image_proj(img_feats), dim=-1)
        t = F.normalize(self.text_proj(txt_feats), dim=-1)
        return i, t, self.logit_scale.exp()

包含两个投影层，输出共享维度，并采用可学习的温度参数 (Temperature)。其接口设计与真实的 CLIP API 保持一致。

### 步骤 2：对比损失 (Contrastive Loss)

def clip_loss(image_emb, text_emb, logit_scale):
    N = image_emb.size(0)
    sim = logit_scale * image_emb @ text_emb.T
    targets = torch.arange(N, device=sim.device)
    l_i = F.cross_entropy(sim, targets)
    l_t = F.cross_entropy(sim.T, targets)
    return (l_i + l_t) / 2

采用对称设计。较高的 `logit_scale` 值会使 Softmax 分布更尖锐，模型预测更自信，但也增加了训练不稳定的风险。

### 步骤 3：零样本分类器 (Zero-Shot Classifier)

@torch.no_grad()
def zero_shot_classify(model, image_feats, class_text_feats, class_names):
    """
    image_feats:      (N, img_in)
    class_text_feats: (C, txt_in)   one averaged embedding per class
    """
    i = F.normalize(model.image_proj(image_feats), dim=-1)
    t = F.normalize(model.text_proj(class_text_feats), dim=-1)
    sim = i @ t.T
    pred = sim.argmax(dim=-1)
    return [class_names[p] for p in pred.tolist()]

每个步骤仅需一行代码。这正是生产环境中使用 CLIP 检查点 (Checkpoint) 进行零样本推理的标准流程。

### 步骤 4：健全性检查 (Sanity Check)

torch.manual_seed(0)
model = TwoTower()

img = torch.randn(8, 128)
txt = torch.randn(8, 64)
i, t, scale = model(img, txt)
loss = clip_loss(i, t, scale)
print(f"batch size: {i.size(0)}   loss: {loss.item():.3f}")

对于随机初始化的模型，损失值应接近 `log(N) = log(8) = 2.08` —— 这是在模型尚未学习到任何有效结构时，对称交叉熵 (Symmetric Cross-Entropy) 的理论目标值。

## 实际应用

OpenCLIP 是 2026 年社区默认的选择：

import open_clip
import torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image = preprocess(Image.open("dog.jpg")).unsqueeze(0)
text = tokenizer(["a photo of a dog", "a photo of a cat", "a photo of a car"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print(probs)

SigLIP 是较新的模型，在小规模训练下表现更佳，是新项目的首选：`google/siglip-base-patch16-224`。Hugging Face 同时提供了这两个模型。

## 交付成果

本课时将产出：

- `outputs/prompt-zero-shot-class-picker.md` — 一个提示词（prompt），用于根据给定的类别列表和领域，为零样本（zero-shot）CLIP 设计类别模板。
- `outputs/skill-image-text-retriever.md` — 一项技能（skill），可使用任意 CLIP 检查点（checkpoint）构建图像嵌入（embedding）索引，支持文本查询和图像查询。

## 练习

1. **(简单)** 使用预训练的 OpenCLIP ViT-B/32 模型，在 CIFAR-10 数据集上利用包含 80 个模板的提示词集进行零样本分类。报告 Top-1 准确率（Top-1 accuracy）；预期应在 85%~90% 左右。
2. **(中等)** 在相同的 CIFAR-10 任务上，对比单模板（"a photo of a {}"）与 80 个模板平均嵌入的效果。量化两者差距，并解释模板为何能提升效果。
3. **(困难)** 构建一个零样本图像检索索引：使用 CLIP 对 1,000 张图像进行嵌入，构建 FAISS 索引，并使用自然语言描述进行查询。针对你手动编写的 20 个预留查询，报告检索的召回率@5（Recall@5）指标。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 双塔架构 (Two-tower) | “双编码器” | 独立的图像和文本编码器，末端连接共享维度的投影头（projection head） |
| 零样本 (Zero-shot) | “无需特定任务训练” | 推理时仅依据文本描述的类别进行分类；不接触任何标注数据 |
| 温度系数 / 对数缩放因子 (Temperature / logit_scale) | “tau” | 可学习的标量参数，在 softmax 操作前用于缩放相似度矩阵 |
| 提示词模板 (Prompt template) | “A photo of a {}” | 包裹类别名称的自然语言句式；对多个模板的嵌入取平均可提升零样本准确率 |
| CLIP | “图文模型” | OpenAI 于 2021 年发布的模型；2026 年已成为该领域的代名词 |
| SigLIP | “Sigmoid CLIP” | 将 softmax 替换为逐对 sigmoid 函数；在小批量（small batch）训练下收敛更佳 |
| OpenCLIP | “开源复现版” | 社区基于 LAION 数据集训练的 CLIP 变体；开源生产管线的默认选择 |
| 视觉语言模型 (VLM) | “Vision-language model” | 结合 CLIP 系列编码器与大语言模型（LLM），经训练用于回答图像相关问题 |

## 延伸阅读

- [CLIP：基于自然语言监督学习可迁移视觉模型 (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [SigLIP：用于语言-图像预训练 (Pre-Training) 的 Sigmoid 损失 (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343)
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — 社区代码库 (Community Codebase)
- [DINOv2、CLIP 与 MAE 特征对比](https://huggingface.co/blog/dinov2) — Hugging Face 指南，附带并列用例 (Use Cases) 对比