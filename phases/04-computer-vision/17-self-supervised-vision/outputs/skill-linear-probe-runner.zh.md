---
name: skill-linear-probe-runner
description: 为任意冻结编码器（frozen encoder）和带标签数据集编写完整的线性探针（linear probe）评估流程
version: 1.0.0
phase: 4
lesson: 17
tags: [自监督学习 (self-supervised), 评估 (evaluation), 线性探针 (linear probe), PyTorch]
---

# 线性探针运行器 (Linear Probe Runner)

通过在顶部训练一个单层线性分类器来评估冻结编码器（frozen encoder）的特征。这是每篇自监督学习（self-supervised learning）论文的标准评估方法。

## 适用场景

- 比较自监督学习（self-supervised learning）的检查点（checkpoint）。
- 跟踪预训练（pretraining）周期（epoch）内的特征质量变化。
- 判断预训练编码器（pretrained encoder）是否足以直接用于下游任务（downstream task），而无需进行微调（fine-tuning）。

## 输入参数

- `encoder`：冻结的 `nn.Module`，为每张图像返回固定维度的特征。
- `feature_dim`：编码器输出的维度。
- `train_dataset`：带标签的数据集（图像，类别ID）。
- `val_dataset`：预留验证集（held-out set）。
- `num_classes`：任务类别数。
- `epochs`：训练轮数，ImageNet 规模通常为 100，较小数据集为 50。

## 操作步骤

1. 将编码器设置为评估模式（eval mode），并将所有参数的 `requires_grad` 设为 `False`。
2. 对训练集和验证集各进行一次特征提取（feature extraction）。将结果存储为 NumPy 数组或内存映射文件（memory-mapped file）。
3. 使用随机梯度下降（SGD）结合余弦调度（cosine schedule），在缓存的特征上训练一个 `nn.Linear(feature_dim, num_classes)` 分类头。
4. 标准超参数：`lr=0.1`、`momentum=0.9`、`weight_decay=0`、`batch_size=1024`。线性探针（linear probe）对学习率（`lr`）异常敏感——若准确率不佳，请进行超参数搜索（sweep）。
5. 在训练结束时报告验证集上的 Top-1 准确率（top-1 accuracy）。

## 输出模板

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def extract(encoder, loader, device="cpu"):
    encoder.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            f = encoder(x.to(device)).cpu()
            feats.append(f)
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def linear_probe(encoder, feature_dim, train_loader, val_loader,
                 num_classes, epochs=50, lr=0.1, device="cpu"):
    for p in encoder.parameters():
        p.requires_grad = False

    f_train, y_train = extract(encoder, train_loader, device)
    f_val, y_val = extract(encoder, val_loader, device)

    head = nn.Linear(feature_dim, num_classes).to(device)
    opt = SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    sched = CosineAnnealingLR(opt, T_max=epochs)

    ds = torch.utils.data.TensorDataset(f_train, y_train)
    train_iter = DataLoader(ds, batch_size=1024, shuffle=True)

    best_val = 0.0
    for ep in range(epochs):
        head.train()
        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(head(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            acc = (head(f_val.to(device)).argmax(-1).cpu() == y_val).float().mean().item()
        best_val = max(best_val, acc)
    return best_val

## 报告格式

[linear probe]
  encoder:     <name + pretrain checkpoint>
  feature_dim: <int>
  epochs:      <int>
  best_val_top1: <float>

## 注意事项

- 在线性探针（linear probe）过程中切勿更新编码器权重；否则将变成微调（fine-tuning），而非探针评估。
- 仅预计算一次特征；在每个 epoch 重新训练编码器会浪费 100 倍的计算资源。
- 使用带余弦调度（cosine schedule）且无权重衰减（weight decay）的 SGD；Adam 优化器在此场景下有时表现不佳。
- 针对每类编码器架构至少进行一次学习率搜索（sweep）；不同自监督学习（SSL）方法的最佳学习率差异较大。