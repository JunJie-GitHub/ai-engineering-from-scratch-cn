# 损失函数（Loss Functions）

> 你的网络做出了预测，但真实标签（Ground Truth）却与之相悖。预测偏差有多大？这个数值就是损失（Loss）。选错损失函数，你的模型就会彻底优化错误的目标。

**类型：** 实战构建
**语言：** Python
**前置课程：** 第 03.04 课（激活函数（Activation Functions））
**时长：** 约 75 分钟

## 学习目标

- 从零开始实现均方误差（Mean Squared Error, MSE）、二元交叉熵（Binary Cross-Entropy）、分类交叉熵（Categorical Cross-Entropy）以及对比损失（Contrastive Loss, InfoNCE）及其梯度（Gradients）计算
- 通过演示“对所有输入均预测为 0.5”的失效模式，解释为何均方误差不适用于分类任务
- 将标签平滑（Label Smoothing）应用于交叉熵，并阐述其如何防止模型做出过度自信的预测
- 为回归、二分类、多分类以及嵌入学习（Embedding Learning）任务选择合适的损失函数

## 问题所在

在分类问题上最小化均方误差的模型会自信地将所有输入都预测为 0.5。它确实在最小化损失，但也彻底失去了实用价值。

损失函数是模型实际优化的唯一目标。不是准确率（Accuracy），不是 F1 分数（F1 Score），也不是你汇报给主管的任何其他指标。优化器（Optimizer）会计算损失函数的梯度并调整权重，以使该数值不断减小。如果损失函数未能准确反映你真正关心的目标，模型就会找到数学上代价最小的方式来“满足”它，而这种方式几乎永远不是你想要的。

来看一个具体例子。假设你有一个二分类任务，两类样本各占 50%。你使用均方误差作为损失函数。模型会对每个输入都输出 0.5。此时的平均均方误差为 0.25，这是在不进行任何实际学习的情况下所能达到的最小值。该模型毫无判别能力，但从技术上讲，它确实最小化了你的损失函数。如果切换为交叉熵（Cross-Entropy），同一个模型就会被强制将预测值推向 0 或 1，因为 `-log(0.5) = 0.693` 是一个极高的损失值，而 `-log(0.99) = 0.01` 则会奖励那些自信且正确的预测。损失函数的选择，决定了模型是真正在学习，还是在“刷指标”。

情况可能更糟。在自监督学习（Self-Supervised Learning）中，你甚至没有标签。对比损失完全定义了学习信号：什么算作相似，什么算作不同，以及模型应该以多大的力度将它们推开。如果对比损失设计不当，你的嵌入表示（Embeddings）就会坍缩至同一点——所有输入都映射为相同的向量。从技术上讲损失为零，但实际价值完全归零。

## 核心概念

### 均方误差 (Mean Squared Error, MSE)

回归任务的默认选择。计算预测值与目标值之间的平方差，并对所有样本求平均。

MSE = (1/n) * sum((y_pred - y_true)^2)

平方操作的意义在于：它对较大误差施加二次惩罚。误差为 2 时的代价是误差为 1 时的 4 倍；误差为 10 时的代价则是 100 倍。这使得 MSE 对异常值 (outliers) 非常敏感——单个严重偏离的预测值会主导整个损失值。

实际数值示例：假设你的模型用于预测房价，对大多数房屋的预测偏差为 1 万美元，但对某一栋豪宅的预测偏差高达 20 万美元。MSE 会极力去修正这栋豪宅的误差，但这可能会损害模型在其余 99 栋房屋上的预测性能。

MSE 关于预测值的梯度为：

dMSE/dy_pred = (2/n) * (y_pred - y_true)

梯度与误差呈线性关系。误差越大，梯度越大。这对回归任务是一个优点（大误差需要大幅修正），但对分类任务却是个缺点（分类任务希望以指数级而非线性方式惩罚那些高置信度的错误预测）。

### 交叉熵损失 (Cross-Entropy Loss)

分类任务的标准损失函数。其理论根基源于信息论——用于衡量预测概率分布与真实分布之间的差异（散度）。

**二元交叉熵 (Binary Cross-Entropy, BCE):**

BCE = -(y * log(p) + (1 - y) * log(1 - p))

其中 `y` 为真实标签（0 或 1），`p` 为预测概率。

`-log(p)` 的作用机制：当真实标签为 1 且预测值 `p = 0.99` 时，损失为 `-log(0.99) = 0.01`；而当预测值 `p = 0.01` 时，损失飙升至 `-log(0.01) = 4.6`。高达 460 倍的差异正是交叉熵生效的原因。它会严厉惩罚高置信度的错误预测，而对高置信度的正确预测几乎不予惩罚。

其梯度也印证了这一点：

dBCE/dp = -(y/p) + (1-y)/(1-p)

当 `y = 1` 且 `p` 接近 0 时，梯度为 `-1/p`，趋近于负无穷。模型会接收到极强的修正信号。当 `p` 接近 1 时，梯度则非常微小。预测已经正确，无需修正。

**分类交叉熵 (Categorical Cross-Entropy):**

适用于目标标签采用独热编码 (one-hot encoding) 的多分类任务。

CCE = -sum(y_i * log(p_i))

只有真实类别会对损失产生贡献（因为其他所有 `y_i` 均为 0）。假设有 10 个类别，正确类别的预测概率为 0.1（相当于随机猜测），则损失为 `-log(0.1) = 2.3`；若正确类别的概率为 0.9，损失则降至 `-log(0.9) = 0.105`。模型由此学会将概率质量集中在正确答案上。

### 为什么 MSE 不适用于分类任务

graph TD
    subgraph "MSE on Classification"
        P1["Predict 0.5 for class 1<br/>MSE = 0.25"]
        P2["Predict 0.9 for class 1<br/>MSE = 0.01"]
        P3["Predict 0.1 for class 1<br/>MSE = 0.81"]
    end
    subgraph "Cross-Entropy on Classification"
        C1["Predict 0.5 for class 1<br/>CE = 0.693"]
        C2["Predict 0.9 for class 1<br/>CE = 0.105"]
        C3["Predict 0.1 for class 1<br/>CE = 2.303"]
    end
    P3 -->|"MSE gradient<br/>flattens near<br/>saturation"| Slow["Slow correction"]
    C3 -->|"CE gradient<br/>explodes near<br/>wrong answer"| Fast["Fast correction"]

当预测值接近 0 或 1 时，MSE 的梯度会趋于平缓（受 Sigmoid 函数饱和区影响）。交叉熵的梯度恰好弥补了这一缺陷——`-log` 项抵消了 Sigmoid 的平缓区域，从而在模型最需要的地方提供强梯度信号。

### 标签平滑 (Label Smoothing)

标准的独热标签断言“这 100% 属于第 3 类，其他类别为 0%”。这是一种非常绝对的假设。标签平滑技术对此进行了软化处理：

smooth_label = (1 - alpha) * one_hot + alpha / num_classes

当 `alpha = 0.1` 且类别数为 10 时：目标向量不再是 `[0, 0, 1, 0, ...]`，而是变为 `[0.01, 0.01, 0.91, 0.01, ...]`。模型的目标值从 1.0 降为了 0.91。

其原理在于：若模型试图通过 Softmax 输出精确的 1.0，就必须将 logits 推向无穷大。这会导致模型过度自信，损害泛化能力，并使其在面对数据分布偏移 (distribution shift) 时变得脆弱。标签平滑将目标值上限限制在 0.9（当 `alpha=0.1` 时），从而将 logits 维持在合理范围内。GPT 及大多数现代模型均采用标签平滑或其等效技术。

### 对比损失 (Contrastive Loss)

无需标签，无需类别。仅依赖输入对，并回答一个问题：它们是相似还是不同？

**SimCLR 风格的对比损失 (NT-Xent / InfoNCE):**

选取一张图像，生成它的两个增强视图（如裁剪、旋转、色彩抖动）。这两个视图构成“正样本对”——它们的嵌入表示 (embeddings) 应当相似。批次中的其他所有图像则构成“负样本对”——它们的嵌入表示应当不同。

L = -log(exp(sim(z_i, z_j) / tau) / sum(exp(sim(z_i, z_k) / tau)))

其中 `sim()` 为余弦相似度，`z_i` 和 `z_j` 为正样本对，求和项遍历所有负样本，`tau`（温度系数 (temperature)）控制分布的尖锐程度。温度越低 = 负样本区分难度越大 = 分离力度越强。

实际数值示例：批次大小 (batch size) 为 256 意味着每个正样本对对应 255 个负样本。温度系数 `tau = 0.07`（SimCLR 默认值）。该损失函数在形式上类似于对相似度进行 Softmax 操作——它要求正样本对的相似度在所有 256 个选项中最高。

**三元组损失 (Triplet Loss):**

接收三个输入：锚点样本 (anchor)、正样本（同类）和负样本（异类）。

L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

间隔参数 `margin`（通常为 0.2-1.0）强制要求正负样本距离之间保持最小差距。如果负样本已经足够远，损失将为零——无梯度，不更新。这提升了训练效率，但需要精心设计三元组挖掘策略（即挑选距离锚点较近的困难负样本）。

### 焦点损失 (Focal Loss)

专为类别不平衡数据集设计。标准交叉熵对所有正确分类的样本一视同仁，而焦点损失会降低简单样本的权重：

FL = -alpha * (1 - p_t)^gamma * log(p_t)

其中 `p_t` 为真实类别的预测概率，`gamma` 控制聚焦程度。当 `gamma = 0` 时，退化为标准交叉熵。当 `gamma = 2`（默认值）时：

- 简单样本（`p_t = 0.9`）：权重 = `(0.1)^2 = 0.01`。实际上被忽略。
- 困难样本（`p_t = 0.1`）：权重 = `(0.9)^2 = 0.81`。保留完整的梯度信号。

焦点损失由 Lin 等人提出，最初用于目标检测任务。在该场景中，99% 的候选区域都是背景（简单负样本）。若不采用焦点损失，模型会淹没在大量简单背景样本中，永远学不会检测目标物体。引入该损失后，模型能将计算能力集中在那些关键且模棱两可的困难样本上。

### 损失函数决策树

flowchart TD
    Start["What is your task?"] --> Reg{"Regression?"}
    Start --> Cls{"Classification?"}
    Start --> Emb{"Learning embeddings?"}

    Reg -->|"Yes"| Outliers{"Outlier sensitive?"}
    Outliers -->|"Yes, penalize outliers"| MSE["Use MSE"]
    Outliers -->|"No, robust to outliers"| MAE["Use MAE / Huber"]

    Cls -->|"Binary"| BCE["Use Binary CE"]
    Cls -->|"Multi-class"| CCE["Use Categorical CE"]
    Cls -->|"Imbalanced"| FL["Use Focal Loss"]
    CCE -->|"Overconfident?"| LS["Add Label Smoothing"]

    Emb -->|"Paired data"| CL["Use Contrastive Loss"]
    Emb -->|"Triplets available"| TL["Use Triplet Loss"]
    Emb -->|"Large batch self-supervised"| NCE["Use InfoNCE"]

### 损失曲面 (Loss Landscape)

graph LR
    subgraph "Loss Surface Shape"
        MSE_S["MSE<br/>Smooth parabola<br/>Single minimum<br/>Easy to optimize"]
        CE_S["Cross-Entropy<br/>Steep near wrong answers<br/>Flat near correct answers<br/>Strong gradients where needed"]
        CL_S["Contrastive<br/>Many local minima<br/>Depends on batch composition<br/>Temperature controls sharpness"]
    end
    MSE_S -->|"Best for"| Reg2["Regression"]
    CE_S -->|"Best for"| Cls2["Classification"]
    CL_S -->|"Best for"| Emb2["Representation learning"]


## 构建

### 步骤 1：均方误差 (Mean Squared Error, MSE) 及其梯度

def mse(predictions, targets):
    n = len(predictions)
    total = 0.0
    for p, t in zip(predictions, targets):
        total += (p - t) ** 2
    return total / n

def mse_gradient(predictions, targets):
    n = len(predictions)
    grads = []
    for p, t in zip(predictions, targets):
        grads.append(2.0 * (p - t) / n)
    return grads

### 步骤 2：二元交叉熵 (Binary Cross-Entropy)

log(0) 的问题确实存在。如果模型对正样本的预测值恰好为 0，则 log(0) 会等于负无穷大。数值截断 (Clipping) 可以防止这种情况发生。

import math

def binary_cross_entropy(predictions, targets, eps=1e-15):
    n = len(predictions)
    total = 0.0
    for p, t in zip(predictions, targets):
        p_clipped = max(eps, min(1 - eps, p))
        total += -(t * math.log(p_clipped) + (1 - t) * math.log(1 - p_clipped))
    return total / n

def bce_gradient(predictions, targets, eps=1e-15):
    grads = []
    for p, t in zip(predictions, targets):
        p_clipped = max(eps, min(1 - eps, p))
        grads.append(-(t / p_clipped) + (1 - t) / (1 - p_clipped))
    return grads

### 步骤 3：结合 Softmax 的多分类交叉熵 (Categorical Cross-Entropy)

Softmax 函数将原始 logits 转换为概率分布。随后，我们针对独热编码 (One-Hot) 目标计算交叉熵。

def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]

def categorical_cross_entropy(logits, target_index, eps=1e-15):
    probs = softmax(logits)
    p = max(eps, probs[target_index])
    return -math.log(p)

def cce_gradient(logits, target_index):
    probs = softmax(logits)
    grads = list(probs)
    grads[target_index] -= 1.0
    return grads

Softmax 与交叉熵组合的梯度推导结果非常简洁：对于真实类别，梯度仅为（预测概率 - 1）；对于其他所有类别，梯度则为预测概率本身。这种优雅的简化并非巧合，这也正是 Softmax 与交叉熵通常搭配使用的原因。

### 步骤 4：标签平滑 (Label Smoothing)

def label_smoothed_cce(logits, target_index, num_classes, alpha=0.1, eps=1e-15):
    probs = softmax(logits)
    loss = 0.0
    for i in range(num_classes):
        if i == target_index:
            smooth_target = 1.0 - alpha + alpha / num_classes
        else:
            smooth_target = alpha / num_classes
        p = max(eps, probs[i])
        loss += -smooth_target * math.log(p)
    return loss

### 步骤 5：对比损失 (Contrastive Loss)（简化版 InfoNCE）

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)

def contrastive_loss(anchor, positive, negatives, temperature=0.07):
    sim_pos = cosine_similarity(anchor, positive) / temperature
    sim_negs = [cosine_similarity(anchor, neg) / temperature for neg in negatives]

    max_sim = max(sim_pos, max(sim_negs)) if sim_negs else sim_pos
    exp_pos = math.exp(sim_pos - max_sim)
    exp_negs = [math.exp(s - max_sim) for s in sim_negs]
    total_exp = exp_pos + sum(exp_negs)

    return -math.log(max(1e-15, exp_pos / total_exp))

### 步骤 6：分类任务中的 MSE 与交叉熵对比

使用这两种损失函数训练第 04 课中的相同网络（圆形数据集）。观察交叉熵如何更快地收敛。

import random

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


class LossComparisonNetwork:
    def __init__(self, loss_type="bce", hidden_size=8, lr=0.1):
        random.seed(0)
        self.loss_type = loss_type
        self.lr = lr
        self.hidden_size = hidden_size

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        if self.loss_type == "mse":
            d_loss = 2.0 * (self.out - target)
        else:
            eps = 1e-15
            p = max(eps, min(1 - eps, self.out))
            d_loss = -(target / p) + (1 - target) / (1 - p)

        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            self.w2[i] -= self.lr * d_out * self.h[i]
            for j in range(2):
                self.w1[i][j] -= self.lr * d_h * self.x[j]
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def compute_loss(self, pred, target):
        if self.loss_type == "mse":
            return (pred - target) ** 2
        else:
            eps = 1e-15
            p = max(eps, min(1 - eps, pred))
            return -(target * math.log(p) + (1 - target) * math.log(1 - p))

    def train(self, data, epochs=200):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                self.backward(y)
                total_loss += self.compute_loss(pred, y)
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append((avg_loss, accuracy))
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses


## 使用方法

PyTorch 内置了所有标准损失函数（Loss Function），并确保了数值稳定性（Numerical Stability）：

import torch
import torch.nn as nn
import torch.nn.functional as F

predictions = torch.tensor([0.9, 0.1, 0.7], requires_grad=True)
targets = torch.tensor([1.0, 0.0, 1.0])

mse_loss = F.mse_loss(predictions, targets)
bce_loss = F.binary_cross_entropy(predictions, targets)

logits = torch.randn(4, 10)
labels = torch.tensor([3, 7, 1, 9])
ce_loss = F.cross_entropy(logits, labels)
ce_smooth = F.cross_entropy(logits, labels, label_smoothing=0.1)

请使用 `F.cross_entropy`（而非 `F.nll_loss` 加上手动 Softmax）。它将 Log-Softmax 和负对数似然（Negative Log-Likelihood）合并为一个数值稳定的操作。单独应用 Softmax 再取对数的稳定性较差——在相减大指数值时会损失精度。

对于对比学习（Contrastive Learning），大多数团队会使用自定义实现或 `lightly`、`pytorch-metric-learning` 等库。其核心循环始终一致：计算成对相似度（Pairwise Similarities），在正样本和负样本上构建 Softmax，然后进行反向传播（Backpropagation）。

## 交付产物

本章节将生成以下文件：
- `outputs/prompt-loss-function-selector.md` -- 用于选择合适损失函数的可复用提示词（Prompt）
- `outputs/prompt-loss-debugger.md` -- 当损失曲线（Loss Curve）异常时用于诊断的提示词

## 练习

1. 实现 Huber 损失（Huber Loss，又称平滑 L1 损失），该损失函数在小误差时表现为均方误差（Mean Squared Error, MSE），在大误差时表现为平均绝对误差（Mean Absolute Error, MAE）。当 5% 的训练目标添加了随机噪声（即异常值）时，分别使用 MSE 和 Huber 损失训练一个预测 y = sin(x) 的回归网络（Regression Network）。比较最终的测试误差。

2. 在二分类训练循环中加入 Focal Loss（焦点损失）。创建一个类别不平衡的数据集（90% 为类别 0，10% 为类别 1）。在训练 200 个 Epoch 后，比较标准二元交叉熵（Binary Cross-Entropy, BCE）与 Focal Loss（gamma=2）在少数类召回率（Recall）上的表现。

3. 实现带有半困难负样本挖掘（Semi-hard Negative Mining）的三元组损失（Triplet Loss）。为 5 个类别生成二维嵌入（Embedding）数据。对于每个锚点（Anchor），找出比正样本距离更远但最接近的负样本（即半困难样本）。将其收敛速度与随机三元组选择进行对比。

4. 运行 MSE 与交叉熵的对比实验，但在训练过程中跟踪每一层的梯度幅值（Gradient Magnitudes）。绘制每个 Epoch 的平均梯度范数（Gradient Norm）曲线。验证在模型最不确定的早期 Epoch 中，交叉熵是否会产生更大的梯度。

5. 实现 KL 散度损失（KL Divergence Loss），并验证当真实分布为独热编码（One-hot）时，最小化 KL(true || predicted) 所产生的梯度与交叉熵相同。随后尝试使用软标签（Soft Targets，如知识蒸馏中的场景），此时“真实”分布来源于教师模型（Teacher Model）的 Softmax 输出。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 损失函数 (Loss function) | “模型错得有多离谱” | 一个可微函数，将预测值与目标值映射为一个标量，优化器通过最小化该标量来更新模型 |
| 均方误差 (MSE) | “平均平方误差” | 预测值与目标值差值的平方的均值；对较大误差施加二次方惩罚 |
| 交叉熵 (Cross-entropy) | “分类任务用的损失” | 使用 -log(p) 衡量预测概率分布与真实分布之间的差异（散度） |
| 二元交叉熵 (Binary cross-entropy) | “BCE” | 适用于二分类的交叉熵：-(y*log(p) + (1-y)*log(1-p)) |
| 标签平滑 (Label smoothing) | “软化目标标签” | 将硬性的 0/1 目标替换为软性值（如 0.1/0.9），以防止模型过度自信并提升泛化能力 |
| 对比损失 (Contrastive loss) | “拉近同类，推远异类” | 一种学习表征的损失函数，通过在嵌入空间 (embedding space) 中使相似样本对靠近、不相似样本对远离来实现 |
| InfoNCE | “CLIP/SimCLR 用的损失” | 基于相似度分数的归一化温度缩放交叉熵；将对比学习视为分类任务处理 |
| 焦点损失 (Focal loss) | “解决数据不平衡的方案” | 通过 (1-p_t)^gamma 对交叉熵进行加权，降低简单样本的权重，使模型专注于困难样本 |
| 三元组损失 (Triplet loss) | “锚点-正样本-负样本” | 在嵌入空间中，强制锚点与正样本的距离至少比与负样本的距离小一个边界值 (margin) |
| 温度系数 (Temperature) | “分布锐度调节旋钮” | 作用于逻辑值 (logits) 或相似度的标量除数，用于控制输出分布的集中程度；值越低，分布越尖锐 |

## 扩展阅读

- Lin 等人，《Focal Loss for Dense Object Detection》（2017）—— 提出 Focal Loss，用于解决目标检测（RetinaNet）中极端的类别不平衡问题
- Chen 等人，《A Simple Framework for Contrastive Learning of Visual Representations》（SimCLR，2020）—— 定义了基于 NT-Xent 损失的现代对比学习流程
- Szegedy 等人，《Rethinking the Inception Architecture》（2016）—— 将标签平滑作为一种正则化技术引入，现已成为大多数大型模型的标准配置
- Hinton 等人，《Distilling the Knowledge in a Neural Network》（2015）—— 利用软目标 (soft targets) 和 KL 散度 (KL divergence) 进行知识蒸馏，是模型压缩领域的奠基性工作