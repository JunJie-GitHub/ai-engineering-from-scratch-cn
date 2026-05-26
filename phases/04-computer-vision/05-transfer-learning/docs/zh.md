# 迁移学习 (Transfer Learning) 与微调 (Fine-Tuning)

> 别人已经花费了上百万个 GPU 小时来教神经网络识别边缘、纹理和物体部件。在训练自己的模型之前，你应该先借用这些特征。

**类型：** 构建
**语言：** Python
**前置条件：** 第 4 阶段 第 03 课（卷积神经网络 (CNNs)）、第 4 阶段 第 04 课（图像分类）
**时长：** 约 75 分钟

## 学习目标

- 区分特征提取 (Feature Extraction) 与微调，并根据数据集规模、领域差异和计算预算选择合适的方法
- 加载预训练骨干网络 (Pretrained Backbone)，替换其分类头 (Classifier Head)，并在 20 行代码内仅训练该分类头以达到可用的基线水平
- 使用差异化学习率 (Discriminative Learning Rates) 逐步解冻网络层，使早期通用特征获得的更新幅度小于后期特定任务特征
- 诊断三种常见故障：解冻模块学习率过高导致的特征漂移 (Feature Drift)、小数据集上批归一化 (Batch Normalization, BN) 统计量崩溃，以及灾难性遗忘 (Catastrophic Forgetting)

## 问题背景

在 ImageNet 上训练 ResNet-50 大约需要 2,000 个 GPU 小时。极少有团队能为其交付的每个任务都承担这样的预算。实际上，几乎所有团队交付的都是一个预训练骨干网络，其顶部连接着一个仅用几百或几千张任务特定图像训练过的新分类头。

这并非捷径。任何在 ImageNet 上训练过的卷积神经网络 (Convolutional Neural Network, CNN) 的第一个卷积块都会学习边缘和类 Gabor 滤波器 (Gabor-like Filters)。随后的几个块学习纹理和简单图案。中间块学习物体部件。最后的块学习组合特征，这些特征开始呈现出 1,000 个 ImageNet 类别的雏形。该层级结构的前 90% 几乎可以原封不动地迁移到医学影像、工业检测、卫星数据以及其他所有视觉任务中——因为自然界中边缘和纹理的“词汇表”是有限的。最后 10% 才是你真正需要训练的部分。

想要正确实现迁移，有三个常见的陷阱在等着你：学习率过高会破坏预训练特征；冻结过多层会导致模型信息匮乏；以及让批归一化 (Batch Normalization, BN) 的运行统计量漂向一个网络其余部分从未学习过的极小数据集。本课程将逐一深入剖析这些问题。

## 核心概念

### 特征提取（Feature Extraction）与微调（Fine-tuning）

这两种策略的选择，取决于你对预训练特征（Pretrained Features）的信任程度以及可用数据量的大小。

flowchart TB
    subgraph FE["Feature extraction — backbone frozen"]
        FE1["Pretrained backbone<br/>(no gradient)"] --> FE2["New head<br/>(trained)"]
    end
    subgraph FT["Fine-tuning — end-to-end"]
        FT1["Pretrained backbone<br/>(tiny LR)"] --> FT2["New head<br/>(normal LR)"]
    end

    style FE1 fill:#e5e7eb,stroke:#6b7280
    style FE2 fill:#dcfce7,stroke:#16a34a
    style FT1 fill:#fef3c7,stroke:#d97706
    style FT2 fill:#dcfce7,stroke:#16a34a

经验法则：

| 数据集规模 | 领域差异 | 推荐方案 |
|--------------|-----------------|--------|
| < 1k 张图像 | 接近 ImageNet | 冻结主干网络（Backbone），仅训练分类头（Head） |
| 1k-10k | 接近 | 冻结前 2-3 个阶段，微调其余部分 |
| 10k-100k | 任意 | 使用差异化学习率（Discriminative Learning Rate）进行端到端微调 |
| 100k+ | 差异较大 | 全量微调；若领域差异极大，可考虑从头训练 |

“接近 ImageNet”大致指包含物体内容的自然 RGB 照片。医学 CT 扫描、高空卫星图像和显微成像则属于差异较大的领域——预训练特征依然有效，但你需要允许更多网络层进行自适应调整。

### 为什么冻结网络依然有效

卷积神经网络（CNN）在 ImageNet 上学到的特征并非专门针对那 1,000 个类别，而是专门针对自然图像的统计规律：特定方向的边缘、纹理、对比度模式以及基础形状基元。这些统计规律在人类已知的几乎所有视觉领域中都保持稳定。这就是为什么在 ImageNet 上训练的模型，仅替换一个新的线性分类头（不微调主干网络）并在 CIFAR-10 上进行零样本（Zero-shot）评估时，准确率就能达到 80% 以上。分类头实际上是在学习如何为当前任务分配已学特征的权重。

### 差异化学习率（Discriminative Learning Rate）

当你解冻网络时，浅层网络的学习速度应慢于深层网络。浅层编码的是你希望保留的通用特征；而深层编码的是特定任务的结构，需要进行大幅调整。

Typical recipe:

  stage 0 (stem + first group): lr = base_lr / 100    (mostly fixed)
  stage 1:                       lr = base_lr / 10
  stage 2:                       lr = base_lr / 3
  stage 3 (last backbone group): lr = base_lr
  head:                          lr = base_lr  (or slightly higher)

在 PyTorch 中，这只需向优化器（Optimizer）传递一个参数组列表即可。同一个模型，五种学习率，无需编写额外代码。

### 批归一化（Batch Normalization）问题

BN 层包含在 ImageNet 上计算得出的 `running_mean` 和 `running_var` 缓冲区。如果你的任务具有不同的像素分布（例如不同的光照条件、传感器或色彩空间），这些缓冲区的数据将不再适用。按推荐优先级排序，有三种解决方案：

1. **在训练模式下微调 BN。** 让 BN 层与其他参数一起更新其运行统计量。当任务数据集规模中等（>= 5k 样本）时的默认选择。
2. **在评估模式下冻结 BN。** 保留 ImageNet 的统计量，仅训练权重。当数据集过小、BN 的移动平均（Moving Average）会产生较大噪声时适用。
3. **用组归一化（Group Normalization）替换 BN。** 彻底消除移动平均问题。常用于目标检测和图像分割的主干网络中，因为这类任务每张 GPU 的批次大小（Batch Size）通常很小。

如果处理不当，准确率会悄无声息地下降 5% 到 15%。

### 分类头设计（Head Design）

分类头通常由 1 到 3 个线性层（Linear Layers）及可选的 Dropout 组成。每个 `torchvision` 主干网络都附带一个默认的分类头，你需要将其替换：

backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)          # ResNet
backbone.classifier[1] = nn.Linear(..., num_classes)                    # EfficientNet, MobileNet
backbone.heads.head = nn.Linear(..., num_classes)                       # torchvision ViT

对于小型数据集，单个线性层通常就足够了。当任务分布与主干网络的训练分布差异较大时，添加一个隐藏层（Linear -> ReLU -> Dropout -> Linear）会有所帮助。

### 逐层学习率衰减（Layer-wise Learning Rate Decay）

这是现代微调方法（如 BEiT、DINOv2、ViT-B 微调）中使用的差异化学习率的平滑版本。它不再将网络层按阶段分组，而是让每一层的学习率都比其上一层略小：

lr_layer_k = base_lr * decay^(L - k)

当衰减系数 `decay = 0.75` 且 Transformer 块数量 `L = 12` 时，第一个块的学习率约为分类头学习率的 `0.75^11 ≈ 0.04x`。这种方法对 Transformer 微调的影响比对 CNN 更大，因为对于 CNN 而言，按阶段分组的学习率通常已经足够。

### 评估指标

迁移学习（Transfer Learning）实验需要跟踪两个在从头训练（Training from Scratch）时通常不会关注的指标：

- **仅预训练准确率（Pretrained-only Accuracy）**：冻结主干网络时分类头的准确率。这是你的性能底线。
- **微调后准确率（Fine-tuned Accuracy）**：同一模型经过端到端训练后的准确率。这是你的性能上限。

如果微调后的准确率低于仅预训练的准确率，说明你的学习率设置或 BN 处理存在缺陷。务必同时打印这两个指标。

## 构建

### 步骤 1：加载预训练主干网络（Backbone）并进行检查

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print(backbone)
print()
print("classifier head:", backbone.fc)
print("feature dim:", backbone.fc.in_features)

`ResNet18` 包含四个阶段（`layer1` 至 `layer4`），外加一个初始层（Stem）和一个全连接分类头（FC Head）。所有 `torchvision` 分类主干网络都具有类似的结构。

### 步骤 2：特征提取（Feature Extraction）—— 冻结所有参数并替换分类头

def make_feature_extractor(num_classes=10):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model = make_feature_extractor(num_classes=10)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"trainable: {trainable:>10,}")
print(f"frozen:    {frozen:>10,}")

只有 `model.fc` 是可训练的。此时，主干网络充当一个冻结的特征提取器（Feature Extractor）。

### 步骤 3：判别式微调（Discriminative Fine-tuning）

这是一个用于构建具有阶段特定学习率（Stage-specific Learning Rates）的参数组的工具函数。

def discriminative_param_groups(model, base_lr=1e-3, decay=0.3):
    stages = [
        ["conv1", "bn1"],
        ["layer1"],
        ["layer2"],
        ["layer3"],
        ["layer4"],
        ["fc"],
    ]
    groups = []
    for i, names in enumerate(stages):
        lr = base_lr * (decay ** (len(stages) - 1 - i))
        params = [p for n, p in model.named_parameters()
                  if any(n.startswith(k) for k in names)]
        if params:
            groups.append({"params": params, "lr": lr, "name": "_".join(names)})
    return groups

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 10)
for p in model.parameters():
    p.requires_grad = True

groups = discriminative_param_groups(model)
for g in groups:
    print(f"{g['name']:>10s}  lr={g['lr']:.2e}  params={sum(p.numel() for p in g['params']):>8,}")

`decay=0.3` 表示每个阶段的学习率是其后一阶段的 30%。`fc` 层获得 `base_lr`，`layer4` 获得 `0.3 * base_lr`，而 `conv1` 获得 `0.3^5 * base_lr ≈ 0.00243 * base_lr`。这听起来有些极端，但经验表明它非常有效。

### 步骤 4：批归一化（BatchNorm）处理

这是一个辅助函数，用于在不冻结权重（Weights）的情况下冻结 BN 的运行统计量（Running Statistics）。

def freeze_bn_stats(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
    return model

在每个 epoch 开始时调用 `model.train()` 之后执行此函数。`model.train()` 会将所有模块切换至训练模式；而该函数仅针对 BN 层将其恢复为评估模式。

### 步骤 5：最小化的端到端（End-to-End）微调循环

from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

def fine_tune(model, train_loader, val_loader, device, epochs=5, base_lr=1e-3, freeze_bn=False):
    model = model.to(device)
    groups = discriminative_param_groups(model, base_lr=base_lr)
    optimizer = SGD(groups, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        if freeze_bn:
            freeze_bn_stats(model)
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            tr_total += x.size(0)
            tr_correct += (logits.argmax(-1) == y).sum().item()
        scheduler.step()

        model.eval()
        va_total, va_correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(-1)
                va_total += x.size(0)
                va_correct += (pred == y).sum().item()
        print(f"epoch {epoch}  train {tr_loss/tr_total:.3f}/{tr_correct/tr_total:.3f}  "
              f"val {va_correct/va_total:.3f}")
    return model

在 CIFAR-10 数据集上应用上述流程训练 5 个 epoch，可使 `ResNet18-IMAGENET1K_V1` 的准确率从约 70% 的零样本线性探测（Zero-shot Linear-probe）准确率提升至约 93% 的微调准确率。如果仅训练分类头而不更新主干网络，准确率通常会停滞在 86% 左右。

### 步骤 6：渐进式解冻（Progressive Unfreezing）

这是一种调度策略，每个 epoch 从网络末端向起始端依次解冻一个阶段。它以增加少量训练 epoch 为代价，有效缓解了特征漂移（Feature Drift）问题。

def progressive_unfreeze_schedule(model):
    stages = ["layer4", "layer3", "layer2", "layer1"]
    yielded = set()

    def start():
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    def unfreeze(epoch):
        if epoch < len(stages):
            name = stages[epoch]
            yielded.add(name)
            for n, p in model.named_parameters():
                if n.startswith(name):
                    p.requires_grad = True
            return name
        return None

    return start, unfreeze

在第一个 epoch 开始前调用一次 `start()`。在每个 epoch 开始时调用 `unfreeze(epoch)`。每当可训练参数集合发生变化时，都需要重新构建优化器（Optimizer），否则已冻结的参数仍会保留缓存的动量（Moments），从而导致优化器行为异常。

## 使用方法

对于大多数实际任务，使用 `torchvision.models` 配合三行代码即可满足需求。上述更重量级的工具链仅在遇到库默认配置无法解决的问题时才派上用场。

from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, num_classes)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

另外两个达到生产级标准的默认方案：

- `timm` 提供了约 800 个预训练视觉骨干网络 (vision backbones)，并保持一致的 API（`timm.create_model("resnet50", pretrained=True, num_classes=10)`）。对于任何超出 `torchvision` 模型库的微调 (fine-tuning) 需求，它都是行业标准。
- 对于 Transformer 架构，使用 `transformers.AutoModelForImageClassification.from_pretrained(name, num_labels=N)` 即可加载 ViT / BEiT / DeiT 等模型，其加载语义与文本模型完全一致。

## 输出结果

本章节将生成以下文件：

- `outputs/prompt-fine-tune-planner.md` — 一个提示词 (prompt)，可根据数据集大小、领域差异 (domain distance) 和计算预算 (compute budget)，在特征提取 (feature extraction)、渐进式微调 (progressive fine-tuning) 与端到端微调 (end-to-end fine-tuning) 之间做出选择。
- `outputs/skill-freeze-inspector.md` — 一个技能脚本，接收 PyTorch 模型后，会报告哪些参数是可训练的 (trainable)、哪些批归一化层 (BatchNorm) 处于评估模式 (eval mode)，以及优化器是否真正接收到了可训练参数。

## 练习

1. **(简单)** 在同一个合成 CIFAR 数据集上，分别将 `ResNet18` 作为线性探针 (linear probe，骨干网络冻结) 和进行全量微调 (full fine-tuning) 进行训练。并列报告两者的准确率。解释哪种准确率差距表明特征迁移效果良好，哪种表明迁移效果不佳。
2. **(中等)** 故意引入一个 Bug：将骨干网络阶段的基础学习率 (`base_lr`) 设置为 `1e-1`，而不是分类头 (head)。展示训练损失 (training loss) 如何爆炸，然后通过应用 `discriminative_param_groups` 辅助函数使其恢复。记录每个阶段开始发散时的学习率 (LR)。
3. **(困难)** 选取一个医学影像数据集（例如 CheXpert-small、PatchCamelyon 或 HAM10000），并对比三种训练策略：(a) 基于 ImageNet 预训练的冻结骨干网络 + 线性分类头；(b) 基于 ImageNet 预训练的端到端微调；(c) 从零开始训练 (scratch training)。报告每种模式的准确率和计算开销。数据集规模达到多大时，从零开始训练才具备竞争力？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 特征提取 (Feature extraction) | “冻结主干，仅训练分类头” | 主干网络参数被冻结，仅新的分类头接收梯度更新 |
| 微调 (Fine-tuning) | “端到端重新训练” | 所有参数均可训练，通常使用比从头训练小得多的学习率 (Learning Rate, LR) |
| 差异化学习率 (Discriminative LR) | “浅层使用更小的学习率” | 优化器参数分组策略，其中浅层网络的学习率是深层网络学习率的一个比例 |
| 逐层学习率衰减 (Layer-wise LR decay) | “平滑的学习率梯度” | 每层的学习率乘以衰减系数^(L - k)；常见于 Transformer 模型的微调中 |
| 灾难性遗忘 (Catastrophic forgetting) | “模型把 ImageNet 的知识忘了” | 学习率过高，导致在新任务信号被学习之前，预训练特征就被覆盖 |
| 批归一化统计量漂移 (BN statistics drift) | “滑动均值计算有误” | BatchNorm 的滑动均值/方差基于与当前任务不同的数据分布计算，从而在暗中降低模型准确率 |
| 线性探针 (Linear probe) | “冻结主干 + 线性分类头” | 对预训练特征的评估——在冻结的特征表示之上训练最佳线性分类器所得到的准确率 |
| 灾难性崩溃 (Catastrophic collapse) | “所有样本都预测为同一类别” | 微调时学习率过高，在分类头的梯度稳定之前就破坏了特征表示，从而导致此现象 |

## 进一步阅读

- [深度神经网络中的特征可迁移性如何？(Yosinski 等, 2014)](https://arxiv.org/abs/1411.1792) —— 量化了特征在不同网络层之间可迁移性的论文
- [通用语言模型微调 (ULMFiT, Howard & Ruder, 2018)](https://arxiv.org/abs/1801.06146) —— 提出了最初的差异化学习率 / 渐进式解冻方案；其思想可直接迁移至计算机视觉领域
- [timm 文档](https://huggingface.co/docs/timm) —— 现代视觉主干网络的参考指南，以及它们训练时所使用的确切微调默认配置
- [线性探针评估的简单框架 (Kornblith 等, 2019)](https://arxiv.org/abs/1805.08974) —— 阐述了线性探针准确率的重要性以及如何正确报告该指标