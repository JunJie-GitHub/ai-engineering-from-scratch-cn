# 正则化 (Regularization)

> 你的模型在训练数据上达到了 99% 的准确率，但在测试数据上只有 60%。它只是在死记硬背，而非真正学习。正则化是你为了强制模型具备泛化 (generalization) 能力而对模型复杂度征收的“税”。

**类型：** 构建实践
**语言：** Python
**前置课程：** 第 03.06 课（优化器 (Optimizers)）
**时长：** 约 75 分钟

## 学习目标

- 从零实现带反向缩放 (inverted scaling) 的随机失活 (Dropout)、L2 权重衰减 (L2 weight decay)、批归一化 (batch normalization)、层归一化 (layer normalization) 以及均方根归一化 (RMSNorm)
- 测量训练集与测试集的准确率差距，并通过正则化实验诊断过拟合 (overfitting)
- 解释为何 Transformer 使用层归一化 (LayerNorm) 而非批归一化 (BatchNorm)，以及为何现代大语言模型 (LLMs) 更偏好 RMSNorm
- 根据过拟合的严重程度，应用正确的正则化技术组合

## 问题所在

只要参数足够多，神经网络就能记住任何数据集。这并非假设——Zhang 等人（2017）通过在带有随机标签的 ImageNet 上训练标准网络证明了这一点。这些网络在完全随机的标签分配上实现了接近零的训练损失 (training loss)。它们死记硬背了一百万个毫无规律可循的随机输入输出对。训练损失堪称完美，但测试准确率却为零。

这就是过拟合问题，且随着模型规模的扩大，该问题会愈发严重。GPT-3 拥有 1750 亿个参数，而其训练集仅包含约 5000 亿个词元 (tokens)。拥有如此庞大的参数量，模型完全有能力逐字逐句地记住训练数据中的大量片段。如果没有正则化，模型只会机械地复述训练样本，而无法学习到可泛化的模式。

训练性能与测试性能之间的差距即为过拟合差距。本课介绍的每一项技术都会从不同角度来缩小这一差距。随机失活强制网络不依赖于任何一个单独的神经元。权重衰减防止单个权重变得过大。批归一化平滑了损失曲面 (loss landscape)，使优化器能够找到更平坦、泛化能力更强的极小值 (minima)。层归一化的作用相同，但它能在批归一化失效的场景（如小批量数据、变长序列）中正常工作。RMSNorm 通过省略均值计算，将速度提升了 10%。每一项技术本身都很简单，但将它们组合起来，就决定了模型是只会死记硬背，还是真正具备泛化能力。

## 核心概念

### 过拟合谱系 (Overfitting Spectrum)

每个模型都处于欠拟合（Underfitting，过于简单而无法捕捉数据模式）到过拟合（Overfitting，过于复杂以至于捕捉到了噪声）之间的某个位置。最佳状态位于两者之间，而正则化（Regularization）的作用正是将模型从过拟合一侧推向这一理想状态。

graph LR
    Under["Underfitting<br/>Train: 60%<br/>Test: 58%<br/>Model too simple"] --> Good["Good Fit<br/>Train: 95%<br/>Test: 92%<br/>Generalizes well"]
    Good --> Over["Overfitting<br/>Train: 99.9%<br/>Test: 65%<br/>Memorized noise"]

    Dropout["Dropout"] -->|"Pushes left"| Over
    WD["Weight Decay"] -->|"Pushes left"| Over
    BN["BatchNorm"] -->|"Pushes left"| Over
    Aug["Data Augmentation"] -->|"Pushes left"| Over

### 随机失活 (Dropout)

这是最简单且解释最为优雅的正则化技术。在训练过程中，以概率 $p$ 随机将每个神经元的输出置为零。

output = activation(z) * mask    where mask[i] ~ Bernoulli(1 - p)

当 $p = 0.5$ 时，每次前向传播（Forward Pass）都会有一半的神经元被置零。由于网络无法预测哪些神经元会被激活，它必须学习冗余的特征表示（Representations）。这有效防止了协同适应（Co-adaptation），即神经元过度依赖其他特定神经元的存在。

集成学习（Ensemble）视角的解释：一个包含 $N$ 个神经元并使用 Dropout 的网络，会生成 $2^N$ 种可能的子网络（每个神经元开启或关闭的所有组合）。使用 Dropout 进行训练，近似于同时在不同的 mini-batch（小批量数据）上训练所有 $2^N$ 个子网络。在测试时，你使用所有神经元（不启用 Dropout），并将输出乘以 $(1 - p)$ 以匹配训练期间的期望值。这等效于对 $2^N$ 个子网络的预测结果进行平均——仅用单个模型就实现了庞大的集成效果。

在实际应用中，缩放操作通常在训练阶段而非测试阶段进行（即反向 Dropout / Inverted Dropout）：

During training:  output = activation(z) * mask / (1 - p)
During testing:   output = activation(z)   (no change needed)

这种做法更为简洁，因为测试代码完全无需感知 Dropout 的存在。

默认丢弃率：Transformer 模型通常设为 $p = 0.1$，多层感知机（MLP）为 $p = 0.5$，卷积神经网络（CNN）为 $p = 0.2\text{-}0.3$。Dropout 率越高，正则化强度越大，但欠拟合的风险也随之增加。

### 权重衰减 (Weight Decay) 与 L2 正则化 (L2 Regularization)

将所有权重的平方幅值添加到损失（Loss）中：

total_loss = task_loss + (lambda / 2) * sum(w_i^2)

正则化项的梯度为 $\lambda \times w$。这意味着在每一步更新中，每个权重都会按其自身大小的比例向零收缩。较大的权重会受到更强的惩罚。模型会被推向一种没有任何单一权重占据主导地位的解。

为何这有助于提升泛化能力（Generalization）：过拟合模型往往具有较大的权重，这些权重会放大训练数据中的噪声。权重衰减通过保持权重较小，限制了模型的有效容量（Effective Capacity），并迫使模型依赖稳健且可泛化的特征，而非死记硬背数据中的特例。

超参数 $\lambda$ 控制正则化的强度。典型取值如下：

- Transformer 模型使用 AdamW 优化器时：0.01
- CNN 模型使用 SGD 优化器时：1e-4
- 严重过拟合的模型：0.1

如第 06 课所述：在 SGD 中，权重衰减与 L2 正则化是等效的，但在 Adam 中则不然。使用 Adam 训练时，务必采用 AdamW（解耦权重衰减 / Decoupled Weight Decay）。

### 批归一化 (Batch Normalization)

在将每一层的输出传递给下一层之前，基于 mini-batch 对其进行归一化。

对于某一层的一个 mini-batch 激活值：

mu = (1/B) * sum(x_i)           (batch mean)
sigma^2 = (1/B) * sum((x_i - mu)^2)   (batch variance)
x_hat = (x_i - mu) / sqrt(sigma^2 + eps)   (normalize)
y = gamma * x_hat + beta        (scale and shift)

$\gamma$ 和 $\beta$ 是可学习参数，允许网络在必要时撤销归一化操作。如果没有它们，你将强制每一层的输出都保持零均值和单位方差，而这可能并非网络所需。

**训练与推理的差异**：训练期间，$\mu$ 和 $\sigma$ 来自当前 mini-batch。推理期间，则使用训练过程中累积的滑动平均值（指数移动平均，动量 momentum = 0.1，即 90% 旧值 + 10% 新值）。

BatchNorm 为何有效仍存在争议。原始论文声称它减少了“内部协变量偏移”（Internal Covariate Shift，即随着浅层网络更新，深层网络输入分布发生变化的现象）。Santurkar 等人（2018）证明这一解释是错误的。真正的原因是：BatchNorm 使损失曲面（Loss Landscape）更加平滑。梯度更具预测性，利普希茨常数（Lipschitz Constants）更小，优化器可以安全地采取更大的步长。这就是为什么 BatchNorm 允许你使用更高的学习率（Learning Rate）并实现更快的收敛。

BatchNorm 存在一个根本性局限：它依赖于批次统计量。当 batch size（批量大小）为 1 时，均值和方差毫无意义。当批次较小（< 32）时，统计量会充满噪声并损害模型性能。这对于目标检测（受内存限制导致 batch size 较小）和语言建模（序列长度可变）等任务尤为重要。

### 层归一化 (Layer Normalization)

沿特征维度而非批次维度进行归一化。对于单个样本：

mu = (1/D) * sum(x_j)           (feature mean)
sigma^2 = (1/D) * sum((x_j - mu)^2)   (feature variance)
x_hat = (x_j - mu) / sqrt(sigma^2 + eps)
y = gamma * x_hat + beta

$D$ 为特征维度。每个样本独立进行归一化，完全不依赖于 batch size。这就是 Transformer 使用 LayerNorm 而非 BatchNorm 的原因。序列长度可变，batch size 通常较小（或在生成阶段为 1），且训练与推理的计算过程完全一致。

在 Transformer 中，LayerNorm 应用于每个自注意力块（Self-Attention Block）和前馈网络块（Feed-Forward Block）之后（Post-LN），或应用于它们之前（Pre-LN，后者训练更稳定）。

### RMS 归一化 (RMSNorm)

去除了均值减法的 LayerNorm。由 Zhang & Sennrich（2019）提出。

rms = sqrt((1/D) * sum(x_j^2))
y = gamma * x / rms

仅此而已。无需计算均值，也无需 $\beta$ 参数。研究发现：LayerNorm 中的重新中心化（均值减法）对模型性能贡献甚微，却增加了计算开销。移除该操作可在保持相同准确率的同时，减少约 10% 的计算开销。

LLaMA、LLaMA 2、LLaMA 3、Mistral 以及大多数现代大语言模型（LLM）均采用 RMSNorm 替代 LayerNorm。在数十亿参数和数万亿 token 的规模下，这 10% 的节省意义重大。

### 归一化方法对比

graph TD
    subgraph "Batch Normalization"
        BN_D["Normalize across BATCH<br/>for each feature"]
        BN_S["Batch: [x1, x2, x3, x4]<br/>Feature 1: normalize [x1f1, x2f1, x3f1, x4f1]"]
        BN_P["Needs batch > 32<br/>Different train vs eval<br/>Used in CNNs"]
    end
    subgraph "Layer Normalization"
        LN_D["Normalize across FEATURES<br/>for each sample"]
        LN_S["Sample x1: normalize [f1, f2, f3, f4]"]
        LN_P["Batch-independent<br/>Same train vs eval<br/>Used in Transformers"]
    end
    subgraph "RMS Normalization"
        RN_D["Like LayerNorm<br/>but skip mean subtraction"]
        RN_S["Just divide by RMS<br/>No centering"]
        RN_P["10% faster than LayerNorm<br/>Same accuracy<br/>Used in LLaMA, Mistral"]
    end

### 数据增强 (Data Augmentation) 作为正则化手段

这不是对模型的修改，而是对数据的修改。在保留标签不变的前提下，对训练输入进行变换：

- 图像：随机裁剪、翻转、旋转、颜色抖动、随机遮挡（Cutout）
- 文本：同义词替换、回译、随机删除
- 音频：时间拉伸、音高偏移、添加噪声

其效果与正则化完全一致：它增加了训练集的有效规模，使模型更难死记硬背特定样本。如果模型只看到每张图像的原始形态一次，它很容易将其记住。但如果模型看到每张图像的 50 种增强版本，它就被迫去学习数据中不变的结构特征。

### 早停法 (Early Stopping)

最简单的正则化器：当验证集损失（Validation Loss）开始上升时停止训练。此时模型尚未过拟合。在实际操作中，你需记录每个 epoch（训练轮次）的验证集损失，保存最佳模型，并继续训练一个“耐心值”（Patience）窗口（通常为 5-20 个 epoch）。如果在耐心值窗口内验证集损失未得到改善，则停止训练并加载已保存的最佳模型。

### 何时应用何种策略

flowchart TD
    Gap{"Train-test<br/>accuracy gap?"} -->|"> 10%"| Heavy["Heavy regularization"]
    Gap -->|"5-10%"| Medium["Moderate regularization"]
    Gap -->|"< 5%"| Light["Light regularization"]

    Heavy --> D5["Dropout p=0.3-0.5"]
    Heavy --> WD2["Weight decay 0.01-0.1"]
    Heavy --> Aug["Aggressive data augmentation"]
    Heavy --> ES["Early stopping"]

    Medium --> D3["Dropout p=0.1-0.2"]
    Medium --> WD1["Weight decay 0.001-0.01"]
    Medium --> Norm["BatchNorm or LayerNorm"]

    Light --> D1["Dropout p=0.05-0.1"]
    Light --> WD0["Weight decay 1e-4"]


## 构建

### 步骤 1：随机失活（Dropout）（训练与评估模式）

import random
import math


class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
        self.mask = None

    def forward(self, x):
        if not self.training:
            return list(x)
        self.mask = []
        output = []
        for val in x:
            if random.random() < self.p:
                self.mask.append(0)
                output.append(0.0)
            else:
                self.mask.append(1)
                output.append(val / (1 - self.p))
        return output

    def backward(self, grad_output):
        grads = []
        for g, m in zip(grad_output, self.mask):
            if m == 0:
                grads.append(0.0)
            else:
                grads.append(g / (1 - self.p))
        return grads

### 步骤 2：L2 权重衰减（L2 Weight Decay）

def l2_regularization(weights, lambda_reg):
    penalty = 0.0
    for w in weights:
        penalty += w * w
    return lambda_reg * 0.5 * penalty

def l2_gradient(weights, lambda_reg):
    return [lambda_reg * w for w in weights]

### 步骤 3：批归一化（Batch Normalization）

class BatchNorm:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features
        self.training = True
        self.num_features = num_features

    def forward(self, batch):
        batch_size = len(batch)
        if self.training:
            mean = [0.0] * self.num_features
            for sample in batch:
                for j in range(self.num_features):
                    mean[j] += sample[j]
            mean = [m / batch_size for m in mean]

            var = [0.0] * self.num_features
            for sample in batch:
                for j in range(self.num_features):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v / batch_size for v in var]

            for j in range(self.num_features):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * var[j]
        else:
            mean = list(self.running_mean)
            var = list(self.running_var)

        self.x_hat = []
        output = []
        for sample in batch:
            normalized = []
            out_sample = []
            for j in range(self.num_features):
                x_h = (sample[j] - mean[j]) / math.sqrt(var[j] + self.eps)
                normalized.append(x_h)
                out_sample.append(self.gamma[j] * x_h + self.beta[j])
            self.x_hat.append(normalized)
            output.append(out_sample)
        return output

### 步骤 4：层归一化（Layer Normalization）

class LayerNorm:
    def __init__(self, num_features, eps=1e-5):
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        mean = sum(x) / len(x)
        var = sum((xi - mean) ** 2 for xi in x) / len(x)

        self.x_hat = []
        output = []
        for j in range(self.num_features):
            x_h = (x[j] - mean) / math.sqrt(var + self.eps)
            self.x_hat.append(x_h)
            output.append(self.gamma[j] * x_h + self.beta[j])
        return output

### 步骤 5：均方根归一化（RMSNorm）

class RMSNorm:
    def __init__(self, num_features, eps=1e-6):
        self.gamma = [1.0] * num_features
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        rms = math.sqrt(sum(xi * xi for xi in x) / len(x) + self.eps)
        output = []
        for j in range(self.num_features):
            output.append(self.gamma[j] * x[j] / rms)
        return output

### 步骤 6：带正则化与不带正则化的训练（Training With and Without Regularization）

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class RegularizedNetwork:
    def __init__(self, hidden_size=16, lr=0.05, dropout_p=0.0, weight_decay=0.0):
        random.seed(0)
        self.hidden_size = hidden_size
        self.lr = lr
        self.dropout_p = dropout_p
        self.weight_decay = weight_decay
        self.dropout = Dropout(p=dropout_p) if dropout_p > 0 else None

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x, training=True):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        if self.dropout and training:
            self.dropout.training = True
            self.h = self.dropout.forward(self.h)
        elif self.dropout:
            self.dropout.training = False
            self.h = self.dropout.forward(self.h)

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            self.w2[i] -= self.lr * (d_out * self.h[i] + self.weight_decay * self.w2[i])
            for j in range(2):
                self.w1[i][j] -= self.lr * (d_h * self.x[j] + self.weight_decay * self.w1[i][j])
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def evaluate(self, data):
        correct = 0
        total_loss = 0.0
        for x, y in data:
            pred = self.forward(x, training=False)
            eps = 1e-15
            p = max(eps, min(1 - eps, pred))
            total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
            if (pred >= 0.5) == (y >= 0.5):
                correct += 1
        return total_loss / len(data), correct / len(data) * 100

    def train_model(self, train_data, test_data, epochs=300):
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in train_data:
                pred = self.forward(x, training=True)
                self.backward(y)
                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            train_loss = total_loss / len(train_data)
            train_acc = correct / len(train_data) * 100
            test_loss, test_acc = self.evaluate(test_data)
            history.append((train_loss, train_acc, test_loss, test_acc))
            if epoch % 75 == 0 or epoch == epochs - 1:
                gap = train_acc - test_acc
                print(f"    Epoch {epoch:3d}: train_acc={train_acc:.1f}%, test_acc={test_acc:.1f}%, gap={gap:.1f}%")
        return history


## 使用方法

PyTorch 将所有归一化（Normalization）和正则化（Regularization）技术都封装为模块：

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10),
)

model.train()
out_train = model(torch.randn(32, 784))

model.eval()
out_test = model(torch.randn(1, 784))

切换 `model.train()` / `model.eval()` 模式至关重要。它会开启或关闭随机失活（Dropout），并指示批归一化（BatchNorm）使用当前批次的统计量还是累积的运行统计量。在推理（Inference）前忘记调用 `model.eval()` 是深度学习中最常见的错误之一。你的测试准确率会出现随机波动，因为此时随机失活仍处于激活状态，且批归一化仍在使用小批量（Mini-batch）统计量。

对于 Transformer 架构，其模式则有所不同：

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.ff(x))
        return x

这里使用的是层归一化（LayerNorm）而非批归一化。随机失活的概率参数 `p` 设为 0.1 而非 0.5。这些都是 Transformer 的默认配置。

## 交付成果

本课时将生成：
- `outputs/prompt-regularization-advisor.md` —— 一个用于诊断过拟合（Overfitting）并推荐合适正则化策略的提示词（Prompt）

## 练习

1. 为二维数据实现空间丢弃（Spatial Dropout）：与其丢弃单个神经元，不如丢弃整个特征通道。通过将连续的特征组视为通道并丢弃整个组来模拟该过程。在 `hidden_size=32` 的圆环数据集（circle dataset）上，将其训练-测试差距（train-test gap）与标准丢弃（Dropout）进行对比。

2. 结合第 05 课的标签平滑（Label Smoothing）与本课的丢弃（Dropout）进行实现。使用四种配置进行训练：均不使用、仅使用 Dropout、仅使用标签平滑、两者同时使用。分别测量每种配置下最终的训练-测试准确率差距。哪种组合产生的差距最小？

3. 在圆环数据集网络的隐藏层与激活函数之间添加一个批归一化（BatchNorm）层。分别在有无 BatchNorm 的情况下，使用 0.01、0.05 和 0.1 的学习率进行训练。BatchNorm 应能在基础网络（vanilla network）出现发散的较高学习率下实现稳定训练。

4. 实现早停法（Early Stopping）：记录每个训练轮次（epoch）的测试损失，保存最佳权重，若测试损失连续 20 个轮次未改善则停止训练。让正则化网络运行 1000 个轮次。报告取得最佳测试准确率的轮次编号，以及由此节省了多少个轮次的计算量。

5. 在 4 层网络（而非仅 2 层）上对比层归一化（LayerNorm）与 RMS 归一化（RMSNorm）。使用相同的权重初始化两者。训练 200 个轮次，并比较最终准确率、训练速度（每个轮次耗时）以及第一层的梯度幅值。验证 RMSNorm 在保持相同准确率的前提下是否具有更快的训练速度。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 过拟合 (Overfitting) | “模型死记硬背了数据” | 当模型的训练性能显著优于测试性能时，表明模型学习到了噪声而非有效信号 |
| 正则化 (Regularization) | “防止过拟合” | 任何用于约束模型复杂度以提升泛化能力的技术：如 Dropout、权重衰减、归一化、数据增强等 |
| 随机失活 (Dropout) | “随机删除神经元” | 在训练过程中以概率 p 将随机神经元置零，迫使网络学习冗余表示；等效于训练一个集成模型 (Ensemble) |
| 权重衰减 (Weight Decay) | “L2 惩罚” | 在每一步通过减去 lambda * w 将所有权重向零收缩；通过限制权重大小来惩罚模型复杂度 |
| 批归一化 (Batch Normalization) | “按批次归一化” | 在训练时使用批次统计量对批次维度上的层输出进行归一化，在推理时使用滑动平均值 |
| 层归一化 (Layer Normalization) | “按样本归一化” | 对每个样本内部的特征维度进行归一化；与批次大小无关，常用于批次大小可变的 Transformer 架构中 |
| RMSNorm (均方根归一化) | “不带均值的 LayerNorm” | 均方根归一化；去除了 LayerNorm 中的均值减法步骤，在保持同等精度的前提下实现约 10% 的加速 |
| 早停法 (Early Stopping) | “在过拟合前停止” | 当验证损失不再改善时终止训练；最简单的正则化手段之一，常与其他方法配合使用 |
| 数据增强 (Data Augmentation) | “用少量数据生成更多数据” | 对训练输入进行变换（如翻转、裁剪、添加噪声），以增加有效数据集规模并迫使模型学习不变性特征 |
| 泛化差距 (Generalization Gap) | “训练集与测试集的划分” | 训练性能与测试性能之间的差异；正则化的目标正是尽可能缩小这一差距 |

## 延伸阅读

- Srivastava 等人，《Dropout: A Simple Way to Prevent Neural Networks from Overfitting》（2014）—— Dropout 的原始论文，提出了集成学习视角的解释并包含大量实验
- Ioffe 与 Szegedy，《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》（2015）—— 引入了 BatchNorm 及其训练流程，是深度学习领域引用率最高的论文之一
- Zhang 与 Sennrich，《Root Mean Square Layer Normalization》（2019）—— 证明了 RMSNorm 在降低计算量的同时能达到与 LayerNorm 相当的精度；已被 LLaMA 和 Mistral 等模型采用
- Zhang 等人，《Understanding Deep Learning Requires Rethinking Generalization》（2017）—— 里程碑式论文，揭示了神经网络能够死记硬背随机标签，对传统的泛化理论提出了挑战