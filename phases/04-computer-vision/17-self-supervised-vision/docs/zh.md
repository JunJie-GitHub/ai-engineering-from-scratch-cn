# 自监督视觉（Self-Supervised Vision）—— SimCLR、DINO 与 MAE

> 标注数据是监督式视觉（Supervised Vision）的瓶颈。自监督预训练（Self-Supervised Pretraining）消除了这一限制：从 1 亿张无标注图像中学习视觉特征，随后在 1 万张标注图像上进行微调。

**类型：** 学习 + 实践
**编程语言：** Python
**前置知识：** 第 4 阶段第 04 课（图像分类），第 4 阶段第 14 课（ViT）
**预计耗时：** 约 75 分钟

## 学习目标

- 梳理三大主流自监督学习（Self-Supervised Learning）范式——对比学习（Contrastive Learning，如 SimCLR）、师生蒸馏（Teacher-Student Distillation，如 DINO）与掩码重建（Masked Reconstruction，如 MAE），并阐明各自的优化目标
- 从零实现 InfoNCE 损失函数（InfoNCE Loss），并解释为何批次大小（Batch Size）为 512 时有效，而为 32 时则失效
- 解释 MAE 采用 75% 掩码比例（Masking Ratio）并非随意设定，并说明其与文本模型 BERT 的 15% 掩码比例有何不同
- 使用 DINOv2 或 MAE 的 ImageNet 模型检查点（Checkpoints）进行线性探测（Linear Probing）与零样本检索（Zero-Shot Retrieval）

## 问题背景

监督式 ImageNet 数据集包含 130 万张标注图像，其标注成本估计高达 1000 万美元。医疗与工业领域的数据集规模更小，且标注成本更为高昂。每个视觉团队都在思考：我们能否利用廉价的无标注数据（如 YouTube 视频帧、网络爬虫抓取的图像、网络摄像头录像、卫星扫描图像等）进行预训练，随后仅在少量标注数据上进行微调？

自监督学习（Self-Supervised Learning）正是答案。在 LAION 或 JFT 数据集上训练的现代自监督视觉 Transformer（Vision Transformer, ViT），在微调后能够达到甚至超越监督式 ImageNet 的准确率。此外，相较于监督式预训练，它在下游任务（Downstream Tasks，如目标检测、图像分割、深度估计）上的迁移能力也更强。DINOv2（Meta, 2023）与 MAE（Meta, 2022）已成为当前工业界提取可迁移视觉特征的默认首选方案。

其核心理念的转变在于：代理任务（Pretext Task）——即模型训练时执行的任务——无需与最终的下游任务一致。关键在于该任务能否迫使模型学习到有用的特征。例如，预测灰度图像的色彩、旋转图像并要求模型分类旋转角度、掩码图像块并进行重建——这些方法均已被验证有效。目前能够成功扩展至大规模数据的三种主流方法分别为：对比学习（Contrastive Learning）、师生蒸馏（Teacher-Student Distillation）与掩码重建（Masked Reconstruction）。

## 核心概念

### 三大类

flowchart LR
    A["Contrastive<br/>SimCLR, MoCo, CLIP"] --> AT["positive pairs<br/>(same image, 2 augs)<br/>pulled together,<br/>negatives pushed apart"]
    B["Teacher-student<br/>DINO, BYOL, iBOT"] --> BT["student predicts<br/>teacher's output;<br/>teacher is EMA of student"]
    C["Masked reconstruction<br/>MAE, BEiT, SimMIM"] --> CT["mask 75% of patches;<br/>reconstruct pixel or<br/>token targets"]

    style A fill:#dbeafe,stroke:#2563eb
    style B fill:#fef3c7,stroke:#d97706
    style C fill:#dcfce7,stroke:#16a34a

### 对比学习（Contrastive Learning）(SimCLR)

选取一张图像，施加两次随机数据增强（Data Augmentation），得到两个视图（View）。将这两个视图输入同一个编码器（Encoder）及投影头（Projection Head）。最小化一个损失函数，其核心思想是：“这两个嵌入（Embedding）应当彼此靠近”，且“该嵌入应与当前批次（Batch）中其他所有图像的嵌入尽可能远离”。

Loss for positive pair (z_i, z_j) among 2N views per batch:

   L_ij = -log( exp(sim(z_i, z_j) / tau) / sum_k in batch \ {i} exp(sim(z_i, z_k) / tau) )

sim = cosine similarity
tau = temperature (0.1 standard)

这就是 InfoNCE 损失（InfoNCE Loss）。它要求每个正样本对应大量负样本，因此批次大小（Batch Size）至关重要——SimCLR 需要 512 到 8192 的批次大小。MoCo 引入了动量队列（Momentum Queue）来存储历史批次，从而将负样本数量与批次大小解耦。

### 师生架构（Teacher-Student）(DINO)

包含两个结构相同的网络：学生网络（Student）和教师网络（Teacher）。教师网络的权重是学生网络权重的指数移动平均（Exponential Moving Average, EMA）。两者均接收经过数据增强的图像视图。学生网络的输出被训练以匹配教师网络的输出——整个过程无需显式的负样本。

loss = CE( student_output(view_1),  teacher_output(view_2) )
     + CE( student_output(view_2),  teacher_output(view_1) )

teacher_weights = m * teacher_weights + (1 - m) * student_weights   (m ≈ 0.996)

为何模型不会退化为“预测一个常数”：教师网络的输出会进行中心化（减去各维度均值）和锐化（除以较小的温度系数）。中心化可防止单一维度主导输出；锐化则能避免输出退化为均匀分布。

DINOv2 正是在此基础上，使用 1.42 亿张精选图像进行规模扩展的产物。由此提取的特征在零样本视觉检索（Zero-Shot Visual Retrieval）和密集预测（Dense Prediction）任务中达到了当前最先进水平（State-of-the-Art, SOTA）。

### 掩码重建（Masked Reconstruction）(MAE)

对视觉变换器（Vision Transformer, ViT）输入的图像块（Patch）进行 75% 的掩码（Mask）处理。仅将可见的 25% 图像块输入编码器。一个小型解码器（Decoder）接收编码器的输出以及在掩码位置插入的掩码标记（Mask Token），并被训练以重建被掩码图像块的像素。

Encoder:  visible 25% of patches -> features
Decoder:  features + mask tokens at masked positions -> reconstructed pixels
Loss:     MSE between reconstructed and original pixels on masked patches only

使 MAE 奏效的关键设计选择包括：

- **75% 的掩码比例**——比例较高。这迫使编码器学习语义特征；若仅掩码 25%，重建任务将变得近乎平凡（相邻像素高度相关，卷积神经网络（CNN）即可轻松解决）。
- **非对称的编码器/解码器架构**——大型 ViT 编码器仅处理可见图像块；小型解码器（8 层，512 维）负责重建任务。其预训练速度比原始 BEiT 快 3 倍。
- **像素空间重建目标**——比 BEiT 的离散化标记（Tokenized）目标更简单，且在 ViT 上表现更优。

预训练完成后，丢弃解码器。编码器即作为特征提取器（Feature Extractor）使用。

### 为何选择 75% 而非 15%

BERT 掩码 15% 的标记（Token），而 MAE 掩码 75%。两者的核心差异在于信息密度（Information Density）。

- 自然语言中每个标记的信息熵（Entropy）较高。预测 15% 的标记依然具有挑战性，因为每个被掩码的位置都存在多种合理的补全可能。
- 图像块的信息熵较低——未被掩码的邻域通常能几乎精确地决定被掩码图像块的像素。若要使预测任务依赖语义理解，就必须采用更激进的掩码策略。

75% 的比例已足够高，使得简单的空间外推无法解决该任务；编码器必须真正表征图像的内容。

### 线性探测评估（Linear-Probe Evaluation）

自监督预训练（Self-Supervised Pretraining）完成后，标准的评估方法是**线性探测（Linear Probe）**：冻结编码器，仅在其顶部训练一个单层线性分类器以拟合 ImageNet 标签。最终报告 Top-1 准确率。

- SimCLR ResNet-50：~71%（2020 年）
- DINO ViT-S/16：~77%（2021 年）
- MAE ViT-L/16：~76%（2022 年）
- DINOv2 ViT-g/14：~86%（2023 年）

线性探测是衡量特征质量的纯粹指标；微调（Fine-Tuning）通常能提升 2-5 个百分点，但同时也混杂了分类头重新训练带来的影响。

## 动手构建

### 步骤 1：双视图数据增强流水线 (Two-view augmentation pipeline)

import torch
import torchvision.transforms as T

two_view_train = lambda: T.Compose([
    T.RandomResizedCrop(96, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
])


class TwoViewDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
        self.aug = two_view_train()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, _ = self.base[i]
        v1 = self.aug(img)
        v2 = self.aug(img)
        return v1, v2

每个 `__getitem__` 都会返回同一图像的两个增强视图；无需提供标签。

### 步骤 2：InfoNCE 损失函数 (InfoNCE loss)

import torch.nn.functional as F

def info_nce(z1, z2, tau=0.1):
    """
    z1, z2: (N, D) L2-normalised embeddings of paired views
    """
    N, D = z1.shape
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = z @ z.T / tau              # (2N, 2N)

    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)

在调用该函数前，需对嵌入向量 (embeddings) 进行 L2 归一化。`tau=0.1` 是 SimCLR 的默认值；降低该温度系数会使损失函数更尖锐，但需要更多的负样本 (negatives)。

### 步骤 3：InfoNCE 健全性检查 (Sanity check)

z1 = F.normalize(torch.randn(16, 32), dim=-1)
z2 = z1.clone()
loss_same = info_nce(z1, z2, tau=0.1).item()
z2_random = F.normalize(torch.randn(16, 32), dim=-1)
loss_random = info_nce(z1, z2_random, tau=0.1).item()
print(f"InfoNCE with identical pairs:  {loss_same:.3f}")
print(f"InfoNCE with random pairs:     {loss_random:.3f}")

相同样本对 (identical pairs) 应产生较低的损失值（在大批量 (batch) 和较低温度系数下接近 0）。对于包含 16 个样本对的批次，随机样本对 (random pairs) 的损失值应为 log(2N-1) ≈ log(31) ≈ 3.4。

### 步骤 4：MAE 风格掩码 (MAE-style masking)

def random_mask_indices(num_patches, mask_ratio=0.75, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_keep = int(num_patches * (1 - mask_ratio))
    perm = torch.randperm(num_patches, generator=g)
    visible = perm[:n_keep]
    masked = perm[n_keep:]
    return visible.sort().values, masked.sort().values


num_patches = 196
visible, masked = random_mask_indices(num_patches, mask_ratio=0.75)
print(f"visible: {len(visible)} / {num_patches}")
print(f"masked:  {len(masked)} / {num_patches}")

该方法简单、快速，且对于给定的随机种子 (seed) 具有确定性。实际的 MAE 实现会对其进行批处理，并保留逐样本掩码 (per-sample masks)。

## 使用它

DINOv2 是 2026 年的工业级标准：

import torch
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")
model.eval()

# Per-image embeddings for zero-shot retrieval
with torch.no_grad():
    inputs = processor(images=[pil_image], return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0]  # CLS token

生成的 768 维嵌入（embedding）是现代图像检索、密集对应（dense correspondence）和零样本迁移（zero-shot transfer）流水线的骨干网络。在下游任务上进行微调（fine-tuning）通常只需添加一个线性分类头（linear head）即可。

对于图像-文本嵌入，SigLIP 或 OpenCLIP 是等效选择；对于掩码自编码器（Masked Autoencoder, MAE）风格的微调，`timm` 仓库提供了所有 MAE 的检查点（checkpoint）。

## 交付上线

本节将产出：

- `outputs/prompt-ssl-pretraining-picker.md` — 一个提示词（prompt），可根据数据集规模、算力资源和下游任务自动选择 SimCLR / MAE / DINOv2。
- `outputs/skill-linear-probe-runner.md` — 一项技能（skill），可为任意冻结编码器（frozen encoder）与标注数据集编写线性探测（linear probe）评估代码。

## 练习

1. **(简单)** 验证当降低温度参数（temperature）时，对齐良好的嵌入的 InfoNCE 损失会下降，而随机嵌入的损失会上升。绘制 `tau in [0.05, 0.1, 0.2, 0.5]` 与损失值的对比图。
2. **(中等)** 实现 DINO 风格的中心缓冲区（centre buffer）。证明若不进行中心化（centring），学生网络（student）会在几个训练周期（epoch）内坍缩为常向量。
3. **(困难)** 使用第 10 课中的 TinyUNet 作为骨干网络，在 CIFAR-100 上训练 MAE。报告在第 10、50 和 200 个训练周期时的线性探测准确率。证明在相同的 1,000 张图像子集上，经过 MAE 预训练的线性探测模型优于从零开始训练的有监督线性探测模型。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 自监督学习（Self-supervised） | “无标签” | 一种前置任务（pretext task），可从无标签数据中提取有用的特征表示 |
| 前置任务（Pretext task） | “假任务” | 自监督学习（SSL）期间使用的优化目标（如重建图像块、匹配视图）；预训练完成后即被丢弃 |
| 线性探测（Linear probe） | “冻结编码器 + 线性头” | 标准的自监督学习评估方法：仅在冻结的特征之上训练一个线性分类器 |
| InfoNCE | “对比损失（contrastive loss）” | 基于余弦相似度的 softmax 计算；正样本对为目标类别，其余均为负样本 |
| EMA 教师网络（EMA teacher） | “移动平均教师” | 权重为学生网络权重的指数移动平均（exponential moving average）的教师网络；被 BYOL、MoCo、DINO 等模型采用 |
| 掩码比例（Mask ratio） | “隐藏图像块百分比” | MAE 训练期间被掩码的图像块比例；视觉任务通常为 75%，文本任务为 15% |
| 表示坍缩（Representation collapse） | “恒定输出” | 自监督学习的一种失败模式，编码器对所有输入均输出相同的常向量；可通过中心化、锐化或引入负样本来避免 |
| DINOv2 | “工业级自监督骨干网络” | Meta 于 2023 年发布的自监督视觉 Transformer（Vision Transformer, ViT）；截至 2026 年最强的通用图像特征提取模型 |

## 延伸阅读

- [SimCLR (Chen et al., 2020)](https://arxiv.org/abs/2002.05709) — 对比学习（Contrastive Learning）的基准参考
- [DINO (Caron et al., 2021)](https://arxiv.org/abs/2104.14294) — 结合动量（Momentum）、中心化（Centering）与锐化（Sharpening）的教师-学生（Teacher-Student）架构
- [MAE (He et al., 2022)](https://arxiv.org/abs/2111.06377) — 面向 ViT 的掩码自编码器（Masked Autoencoder）预训练（Pretraining）
- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193) — 将自监督（Self-Supervised）ViT 扩展至生产级特征（Production Features）