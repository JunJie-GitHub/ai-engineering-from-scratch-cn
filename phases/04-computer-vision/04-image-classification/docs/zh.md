# 图像分类 (Image Classification)

> 分类器（Classifier）是一个将像素映射到类别概率分布的函数。其余部分都只是底层管道的搭建。

**类型：** 构建实践
**编程语言：** Python
**前置课程：** 第二阶段第09课（模型评估 (Model Evaluation)）、第三阶段第10课（迷你框架 (Mini Framework)）、第四阶段第03课（卷积神经网络 (CNNs)）
**预计时长：** 约75分钟

## 学习目标

- 在 CIFAR-10 上构建端到端的图像分类流水线：涵盖数据集、数据增强（Augmentation）、模型、训练循环与评估
- 阐明每个组件（数据加载器 (Dataloader)、损失函数 (Loss)、优化器 (Optimizer)、学习率调度器 (Scheduler)）的作用，并预测其中任一组件出错时会在损失曲线（Loss Curve）上如何体现
- 从零实现 Mixup、Cutout 和标签平滑（Label Smoothing），并论证在何种场景下值得引入它们
- 通过解读混淆矩阵（Confusion Matrix）和各类别的精确率/召回率（Precision/Recall）表格，诊断超出整体准确率（Accuracy）范畴的数据集与模型缺陷

## 问题背景

所有最终交付的视觉任务，在某种程度上都可以归结为图像分类。目标检测（Detection）是对区域进行分类，图像分割（Segmentation）是对像素进行分类，图像检索（Retrieval）则是根据与类别质心的相似度进行排序。掌握分类任务的核心——理顺数据集循环、制定数据增强策略、设计损失函数与评估流程——是一项能够迁移到本阶段所有其他任务的关键技能。

大多数分类任务的缺陷并不在模型本身，而是隐藏在流水线中：错误的归一化（Normalisation）、未打乱的训练集、扭曲标签的数据增强、被训练数据污染的验证集划分，或是第30个 Epoch 后学习率悄然发散。一个配置正确的卷积神经网络（CNN）在 CIFAR-10 上本可达到 93% 的准确率，但在流水线出错时通常只能拿到 70-75%，而整个过程中的损失曲线看起来却完全正常。

本课程将手动搭建整个流水线，确保每个环节都清晰可查。你将不会使用 `torchvision.datasets` 中任何可能掩盖潜在缺陷的封装模块。

## 核心概念

### 分类流水线 (Classification Pipeline)

flowchart LR
    A["Dataset<br/>(images + labels)"] --> B["Augment<br/>(random transforms)"]
    B --> C["Normalise<br/>(mean/std)"]
    C --> D["DataLoader<br/>(batch + shuffle)"]
    D --> E["Model<br/>(CNN)"]
    E --> F["Logits<br/>(N, C)"]
    F --> G["Cross-entropy loss"]
    F --> H["Argmax<br/>at eval"]
    G --> I["Backward"]
    I --> J["Optimizer step"]
    J --> K["Scheduler step"]
    K --> E

    style A fill:#dbeafe,stroke:#2563eb
    style E fill:#fef3c7,stroke:#d97706
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#dcfce7,stroke:#16a34a

这个循环中的每一行都可能潜藏 bug。交叉熵 (Cross-entropy) 接收的是原始 logits，而非 Softmax 输出，因此在计算损失前调用 `model(x).softmax()` 会悄无声息地导致梯度计算错误。数据增强 (Data augmentation) 仅作用于输入数据，不作用于标签——Mixup 除外，它会同时对两者进行混合。`optimizer.zero_grad()` 必须在每个训练步中执行一次；若遗漏此步骤，梯度将会累积，其表现会类似于学习率极度不稳定。上述任何一个 bug 都会导致学习曲线趋于平缓，且不会引发任何报错。

### 交叉熵、Logits 与 Softmax

分类器会为每张图像输出 `C` 个数值，称为 logits。应用 Softmax 可将其转换为概率分布：

softmax(z)_i = exp(z_i) / sum_j exp(z_j)

交叉熵用于衡量正确类别的负对数概率：

CE(z, y) = -log( softmax(z)_y )
        = -z_y + log( sum_j exp(z_j) )

右侧的表达式是数值稳定的形式（log-sum-exp）。PyTorch 的 `nn.CrossEntropyLoss` 将 Softmax 与负对数似然损失 (Negative Log-Likelihood, NLL) 融合在单一操作中，并直接接收原始 logits。自行先应用 Softmax 几乎总是一个 bug——这实际上是在计算 `log(softmax(softmax(z)))`，毫无意义。

### 为什么数据增强有效

卷积神经网络 (Convolutional Neural Network, CNN) 具备平移归纳偏置 (inductive bias)（源于权重共享），但并未内置对裁剪、翻转、颜色抖动或遮挡的不变性 (invariance)。教会模型这些不变性的唯一途径，就是向其展示能够激发这些特性的像素数据。训练过程中的每一次随机变换都在传递一个信号：“这两张图像属于同一类别；请学习那些能够忽略表面差异的特征。”

Original crop:  "dog facing left"
Flip:           "dog facing right"       <- same label, different pixels
Rotate(+15):    "dog, slight tilt"
Colour jitter:  "dog in warmer light"
RandomErasing:  "dog with patch missing"

核心原则：数据增强必须保持标签语义不变。例如，对数字图像进行 Cutout 或旋转可能会将“6”误变为“9”；针对此类数据集，应限制旋转角度范围，并选择符合数字特定不变性的增强策略。

### Mixup 与 Cutmix

常规的数据增强仅对像素进行变换，而标签仍保持为独热编码 (one-hot)。**Mixup** 与 **Cutmix** 则通过对输入和标签同时进行插值打破了这一限制。

Mixup:
  lambda ~ Beta(a, a)
  x = lambda * x_i + (1 - lambda) * x_j
  y = lambda * y_i + (1 - lambda) * y_j

Cutmix:
  paste a random rectangle of x_j into x_i
  y = area-weighted mix of y_i and y_j

其优势在于：模型不再死记硬背尖锐的独热目标，而是学会在类别边界之间进行平滑插值。这会导致训练损失上升，但测试准确率也会随之提高。它是为任意分类器提升鲁棒性 (robustness) 性价比最高的方法。

### 标签平滑 (Label Smoothing)

它是 Mixup 的“近亲”。训练时不再以 `[0, 0, 1, 0, 0]` 为目标，而是改用 `[eps/C, eps/C, 1-eps, eps/C, eps/C]`（`eps` 取较小值，如 0.1）。这能防止模型输出过于尖锐的 logits，并以几乎为零的额外开销提升模型校准度 (calibration)。自 PyTorch 1.10 起，该功能已直接内置于 `nn.CrossEntropyLoss(label_smoothing=0.1)` 中。

### 超越准确率的评估指标

整体准确率往往会掩盖数据分布的不平衡。例如，在一个正负样本比例为 90:10 的二分类任务中，若模型始终预测多数类，其准确率也能轻松达到 90%。真正能揭示模型实际表现的工具包括：

- **各类别准确率 (Per-class accuracy)** —— 为每个类别单独计算一个数值；能立即暴露出表现欠佳的类别。
- **混淆矩阵 (Confusion matrix)** —— C x C 的网格，其中第 i 行第 j 列的值为真实类别 i 被预测为类别 j 的样本数；对角线表示正确预测，非对角线则直观反映了模型的错误分布。
- **Top-1 / Top-5 准确率** —— 正确类别是否位列预测概率最高的前 1 或前 5 名；Top-5 对 ImageNet 等数据集至关重要，因为诸如“诺里奇梗犬”与“诺福克梗犬”这类类别在视觉上本就极易混淆。
- **校准度 (Calibration / ECE)** —— 当模型给出 0.8 的置信度时，其预测正确的概率是否真的接近 80%？现代神经网络普遍存在系统性过度自信 (over-confident) 的问题；可通过温度缩放 (temperature scaling) 或标签平滑进行修正。

## 构建

### 步骤 1：确定性合成数据集 (Deterministic Synthetic Dataset)

CIFAR-10 数据集存储在磁盘上。为了使本教程可复现且运行快速，我们构建了一个类似 CIFAR 的合成数据集 (Synthetic Dataset) —— 包含 32x32 的 RGB 图像，并具有模型必须学习的类别特定结构。完全相同的流水线 (Pipeline) 无需任何修改即可直接应用于真实的 CIFAR-10 数据集。

import numpy as np
import torch
from torch.utils.data import Dataset


def synthetic_cifar(num_per_class=1000, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    Y = []
    for c in range(num_classes):
        centre = rng.uniform(0, 1, (3,))
        freq = 2 + c
        for _ in range(num_per_class):
            yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
            r = np.sin(xx * freq) * 0.5 + centre[0]
            g = np.cos(yy * freq) * 0.5 + centre[1]
            b = (xx + yy) * 0.5 * centre[2]
            img = np.stack([r, g, b], axis=-1)
            img += rng.normal(0, 0.08, img.shape)
            img = np.clip(img, 0, 1)
            X.append(img.astype(np.float32))
            Y.append(c)
    X = np.stack(X)
    Y = np.array(Y)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


class ArrayDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, int(self.Y[i])

每个类别都拥有独立的调色板和频率模式，并添加了高斯噪声 (Gaussian Noise)，以迫使模型学习底层信号而非死记硬背像素。共十个类别，每类一千张图像，且已进行随机打乱。

### 步骤 2：归一化与数据增强 (Normalization and Augmentation)

这是每个计算机视觉 (Computer Vision) 流水线都必备的两个变换操作。

def standardize(mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    def _fn(img):
        return (img - mean) / std
    return _fn


def random_hflip(p=0.5):
    def _fn(img):
        if np.random.random() < p:
            return img[:, ::-1, :].copy()
        return img
    return _fn


def random_crop(pad=4):
    def _fn(img):
        h, w = img.shape[:2]
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = np.random.randint(0, 2 * pad)
        x = np.random.randint(0, 2 * pad)
        return padded[y:y + h, x:x + w, :]
    return _fn


def compose(*fns):
    def _fn(img):
        for fn in fns:
            img = fn(img)
        return img
    return _fn

在裁剪前使用反射填充 (Reflect Padding) 而非零填充 (Zero Padding)，因为黑色边框会成为一种信号，导致模型以无益的方式学会忽略它们。

### 步骤 3：Mixup

在训练步骤中混合两张图像及其对应的标签。它被实现为批次变换 (Batch Transform)，因此位于前向传播 (Forward Pass) 附近，而不是嵌入在数据集内部。

def mixup_batch(x, y, num_classes, alpha=0.2):
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, num_classes).float()
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_onehot = torch.nn.functional.one_hot(y, num_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mixed, y_mixed


def soft_cross_entropy(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()

`soft_cross_entropy` 是针对软标签分布 (Soft-label Distribution) 计算的交叉熵 (Cross-Entropy)。当目标标签恰好为独热编码 (One-hot) 时，它会退化为常规的交叉熵计算。

### 步骤 4：训练循环 (Training Loop)

完整的训练配方：遍历一次数据，每个批次 (Batch) 计算一次梯度，每个轮次 (Epoch) 更新一次学习率调度器 (Scheduler)。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, device, num_classes, use_mixup=True):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x_m, y_soft = mixup_batch(x, y, num_classes)
            logits = model(x_m)
            loss = soft_cross_entropy(logits, y_soft)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        # Training accuracy vs the un-mixed labels `y` is only an approximation
        # when mixup is on (the model saw soft targets, not y). Treat it as a
        # rough progress signal; rely on val accuracy for real performance.
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        for t, p in zip(y.cpu(), pred.cpu()):
            cm[t, p] += 1
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return loss_sum / total, correct / total, cm

每次编写训练循环时，都应检查的五个核心原则：

1. 训练前调用 `model.train()`，评估前调用 `model.eval()` —— 这会切换 Dropout 和 BatchNorm 的行为模式。
2. 在 `.backward()` 之前调用 `.zero_grad()`。
3. 累积指标时使用 `.item()`，以确保没有任何张量继续维持计算图 (Computation Graph) 的存活。
4. 评估期间使用 `@torch.no_grad()` —— 节省内存和时间，防止潜在的意外错误。
5. 直接对原始 logits 进行 Argmax 操作，而非先经过 Softmax —— 结果相同，但减少了一次运算操作。

### 步骤 5：整合运行

使用上一课中的 `TinyResNet`，训练几个轮次并进行评估。

from main import synthetic_cifar, ArrayDataset
from main import standardize, random_hflip, random_crop, compose
from main import mixup_batch, soft_cross_entropy
from main import train_one_epoch, evaluate
# TinyResNet comes from the previous lesson (03-cnns-lenet-to-resnet).
# Adjust the import path to wherever you stored the previous lesson's code.
from cnns_lenet_to_resnet import TinyResNet  # example placeholder

X, Y = synthetic_cifar(num_per_class=500)
split = int(0.9 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
train_tf = compose(random_hflip(), random_crop(pad=4), standardize(mean, std))
eval_tf = standardize(mean, std)

train_ds = ArrayDataset(X_train, Y_train, transform=train_tf)
val_ds = ArrayDataset(X_val, Y_val, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyResNet(num_classes=10).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, 10, use_mixup=True)
    va_loss, va_acc, _ = evaluate(model, val_loader, device, 10)
    scheduler.step()
    print(f"epoch {epoch:2d}  lr {scheduler.get_last_lr()[0]:.4f}  "
          f"train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")

在该合成数据集上，模型在五个轮次内即可达到接近完美的验证集准确率 (Validation Accuracy)。这正是本教程的目的：验证流水线正确无误，且模型能够学习其可学习的内容。若将数据集替换为真实的 CIFAR-10，无需任何修改，同一套训练循环即可达到约 90% 的准确率。

### 步骤 6：解读混淆矩阵 (Confusion Matrix)

仅凭准确率 (Accuracy) 永远无法告诉你模型在哪些地方表现不佳，而混淆矩阵可以。

def print_confusion(cm, labels=None):
    c = cm.shape[0]
    labels = labels or [str(i) for i in range(c)]
    print(f"{'':>6}" + "".join(f"{l:>5}" for l in labels))
    for i in range(c):
        row = cm[i].tolist()
        print(f"{labels[i]:>6}" + "".join(f"{v:>5}" for v in row))
    print()
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    prec = tp / (tp + fp).clamp_min(1)
    rec = tp / (tp + fn).clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
    for i in range(c):
        print(f"{labels[i]:>6}  prec {prec[i]:.3f}  rec {rec[i]:.3f}  f1 {f1[i]:.3f}")

_, _, cm = evaluate(model, val_loader, device, 10)
print_confusion(cm)

行代表真实类别，列代表预测类别。如果在类别 3 和 5 之间出现非对角线 (Off-diagonal) 计数的聚集，则意味着模型容易混淆这两个类别。这为你提供了针对性数据收集或类别特定数据增强 (Class-specific Augmentation) 的切入点。

## 使用它

`torchvision` 将上述所有内容封装为符合 Python 习惯的组件。对于真实的 CIFAR-10 数据集，完整的流水线（pipeline）仅需四行代码加上一个训练循环（training loop）。

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
train_tf = Compose([
    RandomCrop(32, padding=4, padding_mode="reflect"),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean, std),
])
eval_tf = Compose([ToTensor(), Normalize(mean, std)])

train_ds = CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
val_ds   = CIFAR10(root="./data", train=False, download=True, transform=eval_tf)

需要注意两点：均值和标准差（mean/std）是**特定于数据集的**（dataset-specific）——它们基于 CIFAR-10 训练集计算得出，而非 ImageNet；此外，反射填充（reflect pad）是社区默认的裁剪策略。若直接照搬 ImageNet 的统计参数，会造成约 1% 的准确率损失（accuracy leak），这一问题往往直到有人对模型进行性能分析（profile）时才会被察觉。

## 交付产出

本章节将生成以下文件：

- `outputs/prompt-classifier-pipeline-auditor.md` —— 一个提示词（prompt），用于根据上述五个不变量（invariants）审查训练脚本，并指出首个违规项。
- `outputs/skill-classification-diagnostics.md` —— 一项技能（skill），在给定混淆矩阵（confusion matrix）和类别名称列表的情况下，能够总结各类别的失败情况，并提出最具影响力的单一修复方案。

## 练习

1. **(简单)** 在合成数据集上，分别使用和不使用 Mixup 训练同一模型五个轮次（epoch）。绘制两者的训练损失（train loss）和验证损失（val loss）曲线。解释为何使用 Mixup 时训练损失更高，但验证准确率却相近或更好。
2. **(中等)** 实现 Cutout 数据增强——将每张训练图像中随机一个 8x8 的区域置零——并针对以下配置进行消融实验（ablation）：无数据增强、水平翻转+裁剪（hflip+crop）、水平翻转+裁剪+Cutout、水平翻转+裁剪+Mixup。报告每种配置的验证准确率。
3. **(困难)** 构建一个 CIFAR-100 流水线（100 个类别，输入尺寸相同），并复现 ResNet-34 的训练过程，使其准确率与已发表结果相差不超过 1%。附加任务：对三种学习率（learning rate）和两种权重衰减（weight decay）进行参数扫描（sweep），将日志记录到本地 CSV 文件中，并生成最终的混淆矩阵高频误分类表（confusion-matrix-top-confusions table）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 未归一化预测值 (Logits) | “原始输出” | 每张图像对应的 C 维向量（softmax 处理前）；交叉熵损失函数直接接收这些值，而非经过 softmax 归一化后的概率值 |
| 交叉熵 (Cross-entropy) | “损失函数” | 正确类别的负对数概率；将 log-softmax 与负对数似然 (NLL) 融合为一个数值稳定的算子 |
| 数据加载器 (DataLoader) | “批次打包器” | 封装数据集并提供打乱、分批以及（可选的）多进程加载功能；训练过程中一半的 Bug 都容易归咎于它 |
| 数据增强 (Augmentation) | “随机变换” | 训练阶段应用的任何保持标签不变的像素级变换；用于赋予卷积神经网络 (CNN) 原本不具备的不变性特征 |
| 混合增强 (Mixup / Cutmix) | “混合两张图像” | 同时混合输入数据与标签，促使分类器学习平滑的插值过渡，而非生硬的决策边界 |
| 标签平滑 (Label smoothing) | “软化目标” | 将独热编码 (one-hot) 替换为 `(1-eps, eps/(C-1), ...)` 的软标签分布；可改善模型校准度 (calibration) 并略微提升准确率 |
| Top-k 准确率 (Top-k accuracy) | “Top-5” | 正确类别包含在概率最高的 k 个预测结果中；常用于类别本身存在真实歧义的数据集 |
| 混淆矩阵 (Confusion matrix) | “错误分布图” | C x C 的表格，其中 `(i, j)` 位置的数值统计了真实类别为 i 却被预测为 j 的图像数量；对角线表示预测正确，非对角线元素则直接指出需要针对性优化的类别混淆情况 |

## 延伸阅读

- [CS231n: Training Neural Networks](https://cs231n.github.io/neural-networks-3/) — 依然是单页篇幅内对训练流程讲解最清晰的指南
- [Bag of Tricks for Image Classification (He et al., 2019)](https://arxiv.org/abs/1812.01187) — 汇总了各项实用技巧，组合使用后可在 ImageNet 上将 ResNet 的准确率提升 3-4%
- [mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)](https://arxiv.org/abs/1710.09412) — mixup 的原始论文；仅三页理论推导辅以极具说服力的实验验证
- [Why temperature scaling matters (Guo et al., 2017)](https://arxiv.org/abs/1706.04599) — 该论文证实了现代神经网络普遍存在校准偏差，并证明仅通过一个标量参数即可修复此问题