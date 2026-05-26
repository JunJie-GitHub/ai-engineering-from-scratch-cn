# 优化器 (Optimizers)

> 梯度下降 (Gradient Descent) 告诉你该朝哪个方向移动，但它并未说明该移动多远或多快。随机梯度下降 (SGD) 就像指南针，而 Adam (自适应矩估计) 则是带有实时路况的 GPS。

**类型：** 构建
**语言：** Python
**前置课程：** 第 03.05 课（损失函数 (Loss Functions)）
**时长：** 约 75 分钟

## 学习目标

- 使用 Python 从零实现随机梯度下降 (SGD)、带动量的随机梯度下降 (SGD with Momentum)、Adam 以及 AdamW 优化器
- 解释 Adam 的偏差校正 (Bias Correction) 如何补偿训练初期零初始化的矩估计 (Moment Estimates)
- 论证为何在相同任务上，AdamW 比结合 L2 正则化 (L2 Regularization) 的 Adam 能产生更好的泛化能力 (Generalization)
- 为 Transformer、卷积神经网络 (CNN)、生成对抗网络 (GAN) 以及微调 (Fine-tuning) 任务选择合适的优化器及默认超参数 (Hyperparameters)

## 问题背景

你已经计算出了梯度。你知道第 4,721 号权重应该减少 0.003 以降低损失。但这 0.003 是什么单位？经过了怎样的缩放？此外，在第 1 步和第 1,000 步时，你应该移动相同的幅度吗？

原始梯度下降 (Vanilla Gradient Descent) 在每一步对所有参数应用相同的学习率 (Learning Rate)：`w = w - lr * gradient`。这在实际训练神经网络时会引发三个棘手的问题。

首先是震荡 (Oscillation)。损失曲面 (Loss Landscape) 很少呈现平滑的碗状，它更像是一条狭长的山谷。梯度指向的是横跨山谷的方向（陡峭方向），而非沿着山谷的方向（平缓方向）。梯度下降会在狭窄的维度上来回反弹，而在有用的维度上进展甚微。你一定见过这种现象：损失快速下降后进入平台期，这并不是因为模型已经收敛，而是因为它陷入了震荡。

其次，对所有参数使用单一的学习率是错误的。某些权重需要大幅更新（它们处于早期的欠拟合 (Underfitting) 阶段），而另一些权重只需微调（它们已接近最优值）。适用于前者的学习率会破坏后者，反之亦然。

第三是鞍点 (Saddle Points)。在高维空间中，损失曲面存在大片梯度接近于零的平坦区域。基础 SGD 会以梯度的速度（实际上接近于零）在这些区域中缓慢爬行。模型看起来像是卡住了，但实际上并没有——它只是处于一片平坦区域，而另一侧仍有有效的下降路径。但 SGD 缺乏推动其穿越该区域的机制。

Adam 解决了上述所有问题。它为每个参数维护两个滑动平均值：梯度均值（动量 (Momentum)，用于处理震荡）和梯度平方均值（自适应学习率 (Adaptive Learning Rate)，用于处理不同量级）。结合前几步的偏差校正，它提供了一个只需使用默认超参数就能解决 80% 问题的通用优化器。本课将从零开始构建它，以便你准确理解它在剩余 20% 的场景中何时以及为何会失效。

## 核心概念

### 随机梯度下降 (Stochastic Gradient Descent, SGD)

最基础的优化器。在小批量数据 (mini-batch) 上计算梯度，并沿其反方向更新参数。

w = w - lr * gradient

“随机”一词意味着你使用数据的随机子集（即小批量数据）来估计梯度，而非使用完整数据集。这种噪声实际上是有用的——它有助于模型跳出尖锐的局部极小值 (local minima)。但噪声也会导致参数更新过程中的震荡。

学习率 (learning rate) 是唯一的调节旋钮。设置过高：损失函数会发散。设置过低：训练将遥遥无期。最优值取决于网络架构、数据分布、批量大小以及当前的训练阶段。对于现代网络中的标准 SGD，典型取值范围在 0.01 到 0.1 之间。但即使在单次训练过程中，理想的学习率也会动态变化。

### 动量法 (Momentum)

“小球滚下山坡”的比喻虽然被用滥了，但非常准确。动量法不仅依赖当前梯度进行步进，还会维护一个速度变量，用于累积历史梯度。

m_t = beta * m_{t-1} + gradient
w = w - lr * m_t

Beta（通常设为 0.9）控制保留多少历史信息。当 beta = 0.9 时，动量大致相当于过去 10 个梯度的平均值（1 / (1 - 0.9) = 10）。

动量法为何能解决震荡问题：方向一致的梯度会相互叠加，而方向相反的梯度则会相互抵消。在狭窄的谷底中，“横向”分量每一步都会改变符号并被削弱；而“纵向”分量则保持一致并被放大。最终结果是在有效方向上实现平滑加速。

实际数据对比：在病态的损失曲面 (loss landscape) 上，仅使用 SGD 可能需要 10,000 步才能收敛。而在相同问题上，加入动量（beta=0.9）的 SGD 通常只需 3,000 到 5,000 步。这种加速效果非常显著。

### RMSProp

首个真正行之有效的逐参数自适应学习率 (per-parameter adaptive learning rate) 方法。由 Hinton 在 Coursera 课程讲座中提出（从未正式发表）。

s_t = beta * s_{t-1} + (1 - beta) * gradient^2
w = w - lr * gradient / (sqrt(s_t) + epsilon)

`s_t` 用于跟踪梯度平方的滑动平均值。对于梯度持续较大的参数，更新时会除以一个较大的数（从而降低有效学习率）；对于梯度较小的参数，则除以较小的数（从而提高有效学习率）。

这解决了“所有参数共享单一学习率”的问题。如果一个权重已经经历了大幅更新，它很可能已接近最优值——此时应减缓其更新速度。如果一个权重更新幅度一直很小，它可能尚未充分训练——此时应加快其更新速度。

Epsilon（通常设为 1e-8）用于防止在参数未发生更新时出现除以零的错误。

### Adam：动量法 + RMSProp

Adam 结合了上述两种思想。它为每个参数维护两个指数移动平均值 (exponential moving averages)：

m_t = beta1 * m_{t-1} + (1 - beta1) * gradient        (first moment: mean)
v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2       (second moment: variance)

**偏差校正 (Bias correction)** 是大多数教程都会忽略的关键细节。在第 1 步时，`m_1 = (1 - beta1) * gradient`。若 `beta1 = 0.9`，结果仅为 `0.1 * gradient`——比实际值小了十倍。这是因为移动平均值尚未“预热”。偏差校正对此进行了补偿：

m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)

当 `beta1 = 0.9` 且处于第 1 步时：`m_hat = m_1 / (1 - 0.9) = m_1 / 0.1`，即等于实际梯度。到了第 100 步：`(1 - 0.9^100)` 约等于 1.0，校正项随之消失。偏差校正主要在前约 10 步起作用，50 步之后便不再重要。

参数更新公式：

w = w - lr * m_hat / (sqrt(v_hat) + epsilon)

Adam 默认参数：`lr = 0.001`, `beta1 = 0.9`, `beta2 = 0.999`, `epsilon = 1e-8`。这些默认值能解决 80% 的问题。当效果不佳时，优先调整 `lr`，其次是 `beta2`。几乎不需要修改 `beta1` 或 `epsilon`。

### AdamW：正确的权重衰减 (Weight Decay) 实现

L2 正则化 (L2 regularization) 会在损失函数中添加 `lambda * w^2` 项。在标准 SGD 中，这等价于权重衰减（即在每一步从权重中减去 `lambda * w`）。但在 Adam 中，这种等价关系不再成立。

Loshchilov 与 Hutter 的核心洞察：当你将 L2 项加入损失函数并由 Adam 处理梯度时，自适应学习率也会缩放正则化项。梯度方差较大的参数受到的正则化较弱，而方差较小的参数受到的正则化较强。这并非我们想要的效果——我们期望的是无论梯度统计特性如何，所有参数都能获得均匀的正则化约束。

AdamW 通过在 Adam 更新之后，直接对权重应用权重衰减来解决此问题：

w = w - lr * m_hat / (sqrt(v_hat) + epsilon) - lr * lambda * w

权重衰减项（`lr * lambda * w`）不会被 Adam 的自适应因子缩放。每个参数都会按相同比例进行收缩。

这看似是个微不足道的细节，实则不然。在几乎所有任务上，AdamW 收敛到的解都优于 Adam + L2 正则化。它是 PyTorch 中训练 Transformer、扩散模型 (diffusion models) 及大多数现代架构的默认优化器。BERT、GPT、LLaMA、Stable Diffusion 等模型均使用 AdamW 训练。

### 学习率：最重要的超参数

graph TD
    LR["Learning Rate"] --> TooHigh["Too high (lr > 0.01)"]
    LR --> JustRight["Just right"]
    LR --> TooLow["Too low (lr < 0.00001)"]

    TooHigh --> Diverge["Loss explodes<br/>NaN weights<br/>Training crashes"]
    JustRight --> Converge["Loss decreases steadily<br/>Reaches good minimum<br/>Generalizes well"]
    TooLow --> Stall["Loss decreases slowly<br/>Gets stuck in suboptimal minimum<br/>Wastes compute"]

    JustRight --> Schedule["Usually needs scheduling"]
    Schedule --> Warmup["Warmup: ramp from 0 to max<br/>First 1-10% of training"]
    Schedule --> Decay["Decay: reduce over time<br/>Cosine or linear"]

如果只调整一个超参数，那一定是学习率。学习率 10 倍的变动，其影响远超任何网络架构层面的决策。常见默认值如下：

- SGD：`lr = 0.01` 到 `0.1`
- Adam/AdamW：`lr = 1e-4` 到 `3e-4`
- 微调预训练模型：`lr = 1e-5` 到 `5e-5`
- 学习率预热 (warmup)：在前 1% 到 10% 的训练步数内线性递增

### 优化器对比

flowchart LR
    subgraph "Optimization Path"
        SGD_P["SGD<br/>Oscillates across valley<br/>Slow but finds flat minima"]
        Mom_P["SGD + Momentum<br/>Smoother path<br/>3x faster than SGD"]
        Adam_P["Adam<br/>Adapts per-parameter<br/>Fast convergence"]
        AdamW_P["AdamW<br/>Adam + proper decay<br/>Best generalization"]
    end
    SGD_P --> Mom_P --> Adam_P --> AdamW_P

### 各优化器的适用场景

flowchart TD
    Task["What are you training?"] --> Type{"Model type?"}

    Type -->|"Transformer / LLM"| AdamW["AdamW<br/>lr=1e-4, wd=0.01-0.1"]
    Type -->|"CNN / ResNet"| SGD_M["SGD + Momentum<br/>lr=0.1, momentum=0.9"]
    Type -->|"GAN"| Adam2["Adam<br/>lr=2e-4, beta1=0.5"]
    Type -->|"Fine-tuning"| AdamW2["AdamW<br/>lr=2e-5, wd=0.01"]
    Type -->|"Don't know yet"| Default["Start with AdamW<br/>lr=3e-4, wd=0.01"]


## 构建项目

### 步骤 1：标准随机梯度下降（Vanilla SGD）

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

### 步骤 2：带动量的随机梯度下降（SGD with Momentum）

class SGDMomentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None

    def step(self, params, grads):
        if self.velocities is None:
            self.velocities = [0.0] * len(params)
        for i in range(len(params)):
            self.velocities[i] = self.beta * self.velocities[i] + grads[i]
            params[i] -= self.lr * self.velocities[i]

### 步骤 3：Adam 优化器

import math

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)

### 步骤 4：AdamW 优化器

class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
            params[i] -= self.lr * self.weight_decay * params[i]

### 步骤 5：训练对比

使用全部四种优化器，在第 05 课的圆形数据集（circle dataset）上训练同一个两层网络，并对比它们的收敛性（convergence）。

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


class OptimizerTestNetwork:
    def __init__(self, optimizer, hidden_size=8):
        random.seed(0)
        self.hidden_size = hidden_size
        self.optimizer = optimizer

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def get_params(self):
        params = []
        for row in self.w1:
            params.extend(row)
        params.extend(self.b1)
        params.extend(self.w2)
        params.append(self.b2)
        return params

    def set_params(self, params):
        idx = 0
        for i in range(self.hidden_size):
            for j in range(2):
                self.w1[i][j] = params[idx]
                idx += 1
        for i in range(self.hidden_size):
            self.b1[i] = params[idx]
            idx += 1
        for i in range(self.hidden_size):
            self.w2[i] = params[idx]
            idx += 1
        self.b2 = params[idx]

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

    def compute_grads(self, target):
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        grads = [0.0] * (self.hidden_size * 2 + self.hidden_size + self.hidden_size + 1)
        idx = 0
        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            grads[idx] = d_h * self.x[0]
            grads[idx + 1] = d_h * self.x[1]
            idx += 2

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            grads[idx] = d_out * self.w2[i] * d_relu
            idx += 1

        for i in range(self.hidden_size):
            grads[idx] = d_out * self.h[i]
            idx += 1

        grads[idx] = d_out
        return grads

    def train(self, data, epochs=300):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                grads = self.compute_grads(y)
                params = self.get_params()
                self.optimizer.step(params, grads)
                self.set_params(params)

                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append((avg_loss, accuracy))
            if epoch % 75 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses


## 使用方法

PyTorch 优化器（Optimizer）负责处理参数组（Parameter Groups）、梯度裁剪（Gradient Clipping）以及学习率调度（Learning Rate Scheduling）：

import torch
import torch.optim as optim

model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(torch.randn(32, 784))
    loss = torch.nn.functional.cross_entropy(output, torch.randint(0, 10, (32,)))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

标准训练流程始终遵循以下顺序：`zero_grad` → `forward` → `loss` → `backward` → （可选 `clip`）→ `step` → （可选 `schedule`）。请牢记此顺序。顺序错误（例如在调用 `optimizer.step()` 之前调用 `scheduler.step()`）是引发隐蔽 Bug 的常见原因。

对于卷积神经网络（Convolutional Neural Network, CNN），许多从业者仍倾向于使用搭配阶梯（Step）或余弦（Cosine）调度策略的随机梯度下降（Stochastic Gradient Descent, SGD）结合动量（Momentum）（lr=0.1, momentum=0.9, weight_decay=1e-4）。SGD 倾向于寻找更平坦的极小值（Flatter Minima），这通常能带来更好的泛化（Generalization）能力。对于 Transformer 和大语言模型（Large Language Model, LLM），带有预热（Warmup）与余弦衰减（Cosine Decay）的 AdamW 是业界通用的默认配置。除非有经过严谨验证的理由，否则不建议盲目违背这一共识。

## 交付成果

本课时将产出：
- `outputs/prompt-optimizer-selector.md` —— 用于为任意网络架构选择合适优化器与学习率的决策提示词（Prompt）

## 练习

1. 实现 Nesterov 动量（Nesterov Momentum），即在“前瞻”位置（`w - lr * beta * v`）而非当前位置计算梯度。在 Circle 数据集上，将其收敛（Convergence）速度与标准动量进行对比。

2. 实现学习率预热（Learning Rate Warmup）调度策略：在前 10% 的训练步数中，学习率从 0 线性增长至 `max_lr`，随后按余弦曲线衰减至 0。分别使用“Adam + 预热”与“无预热的 Adam”进行训练。记录在 Circle 数据集上达到 90% 准确率所需的 Epoch 数量。

3. 在 Adam 训练过程中，追踪每个参数的有效学习率（Effective Learning Rate）。有效学习率的计算公式为 `lr * m_hat / (sqrt(v_hat) + eps)`。绘制训练至第 10、50 和 200 步时有效学习率的分布图。所有参数的更新速度是否一致？

4. 实现梯度裁剪（Gradient Clipping，按全局范数裁剪）。将最大梯度范数设为 1.0。在使用较高学习率（Adam 的 lr=0.01）的情况下，分别进行开启与关闭裁剪的训练。在 10 个不同的随机种子下，统计开启与关闭裁剪时训练发散（Divergence，损失值变为 NaN）的次数。

5. 在权重初始值较大的网络上对比 Adam 与 AdamW。将所有权重初始化为 [-5, 5] 范围内的随机值（远大于常规初始化）。设置 `weight_decay=0.1` 并训练 200 个 Epoch。绘制两种优化器在训练过程中权重的 L2 范数（L2 Norm）变化曲线。AdamW 应展现出更快的权重收缩速度。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 学习率 (Learning rate) | “步长” | 梯度更新时的标量乘数；训练过程中影响最大的单一超参数 |
| 随机梯度下降 (Stochastic Gradient Descent, SGD) | “基础梯度下降” | 随机梯度下降：通过减去学习率乘以梯度来更新权重，梯度基于小批量数据计算 |
| 动量 (Momentum) | “滚球类比” | 历史梯度的指数移动平均；抑制震荡并加速一致方向的更新 |
| RMSProp | “自适应学习率” | 将每个参数的梯度除以其近期梯度的运行均方根（RMS）；均衡各参数的学习率 |
| Adam | “默认优化器” | 结合动量（一阶矩）和 RMSProp（二阶矩），并在初始步骤进行偏差校正 |
| AdamW | “正确的 Adam 实现” | 采用解耦权重衰减的 Adam；直接将正则化应用于权重，而非通过梯度 |
| 偏差校正 (Bias correction) | “运行平均值的预热” | 除以 `(1 - beta^t)` 以补偿 Adam 矩估计初始化为零带来的偏差 |
| 权重衰减 (Weight decay) | “缩小权重” | 在每一步减去权重值的一定比例；一种惩罚大权重的正则化方法 |
| 学习率调度 (Learning rate schedule) | “随时间改变学习率” | 在训练期间调整学习率的函数；预热（warmup）+ 余弦衰减（cosine decay）是现代默认方案 |
| 梯度裁剪 (Gradient clipping) | “限制梯度范数” | 当梯度向量的范数超过阈值时对其进行缩放；防止梯度更新爆炸 |

## 延伸阅读

- Kingma & Ba，《Adam：一种随机优化方法》（2014）—— Adam 的原始论文，包含收敛性分析及偏差校正的推导过程
- Loshchilov & Hutter，《解耦权重衰减正则化》（2017）—— 证明了在 Adam 中 L2 正则化与权重衰减并不等价，并提出了 AdamW
- Smith，《用于训练神经网络的周期性学习率》（2017）—— 引入了学习率范围测试（LR range test）和周期性调度策略，免去了手动调优固定学习率的麻烦
- Ruder，《梯度下降优化算法综述》（2016）—— 关于各类优化器变体最全面的单篇综述，提供了清晰的对比与直观理解