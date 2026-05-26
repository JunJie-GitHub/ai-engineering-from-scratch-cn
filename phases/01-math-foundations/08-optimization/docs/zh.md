# 优化 (Optimization)

> 训练神经网络无非就是寻找山谷的最低点。

**类型：** 构建
**语言：** Python
**前置知识：** 第一阶段，第 04-05 课（导数 (Derivatives)、梯度 (Gradients)）
**时长：** 约 75 分钟

## 学习目标

- 从零实现标准梯度下降 (Vanilla Gradient Descent)、带动量的随机梯度下降 (SGD with Momentum) 以及 Adam
- 在罗森布罗克函数 (Rosenbrock Function) 上比较不同优化器 (Optimizer) 的收敛性，并解释 Adam 为何能自适应调整每个权重的学习率 (Learning Rate)
- 区分凸 (Convex) 与非凸 (Non-convex) 损失曲面 (Loss Landscapes)，并解释鞍点 (Saddle Points) 在高维空间中的作用
- 配置学习率调度策略 (Learning Rate Schedules)（如阶梯衰减 (Step Decay)、余弦退火 (Cosine Annealing)、预热 (Warmup)）以提升训练稳定性

## 问题描述

你有一个损失函数 (Loss Function)，它能告诉你模型的预测偏差有多大。你还有梯度，它们能指出使损失增大的方向。现在，你需要一种“下山”的策略。

最朴素的方法很简单：沿着梯度的反方向移动。步长由一个称为学习率的数值来缩放。不断重复。这就是梯度下降 (Gradient Descent)，它确实有效。但“有效”是有前提的。如果学习率过大，你会直接越过谷底，在两侧山壁间来回震荡；如果学习率过小，你将在成千上万步不必要的迭代中缓慢逼近答案。一旦遇到鞍点，即使尚未找到最小值，你的更新也会停滞不前。

深度学习中的每一个优化器都在回答同一个问题：如何更快、更可靠地抵达谷底？

## 核心概念

### 优化（Optimization）的含义

优化是指寻找使函数值最小化（或最大化）的输入值。在机器学习中，该函数即为损失（Loss），输入则是模型的权重（Weights）。训练过程本质上就是优化。

minimize L(w) where:
  L = loss function
  w = model weights (could be millions of parameters)

### 基础梯度下降（Vanilla Gradient Descent）

这是最基础的优化器（Optimizer）。计算损失相对于每个权重的梯度（Gradient）。沿梯度的反方向更新每个权重。步长由学习率（Learning Rate）进行缩放。

w = w - lr * gradient

这就是整个算法。仅一行代码。

graph TD
    A["* Starting point (high loss)"] --> B["Moving downhill along gradient"]
    B --> C["Approaching minimum"]
    C --> D["o Minimum (low loss)"]

### 学习率（Learning Rate）：最重要的超参数（Hyperparameter）

学习率控制着步长大小。它决定了模型收敛（Convergence）的方方面面。

graph LR
    subgraph TooLarge["Too Large (lr = 1.0)"]
        A1["Step 1"] -->|overshoot| A2["Step 2"]
        A2 -->|overshoot| A3["Step 3"]
        A3 -->|diverging| A4["..."]
    end
    subgraph TooSmall["Too Small (lr = 0.0001)"]
        B1["Step 1"] -->|tiny step| B2["Step 2"]
        B2 -->|tiny step| B3["Step 3"]
        B3 -->|10,000 steps later| B4["Minimum"]
    end
    subgraph JustRight["Just Right (lr = 0.01)"]
        C1["Start"] --> C2["..."] --> C3["Converged in ~100 steps"]
    end

合适的学习率没有固定公式，只能通过实验摸索。常见的初始值参考：Adam 优化器通常设为 0.001，带动量的随机梯度下降（Stochastic Gradient Descent, SGD）通常设为 0.01。

### SGD、批量梯度下降与小批量梯度下降的对比

基础梯度下降在每次更新前会计算整个数据集的梯度。这被称为批量梯度下降（Batch Gradient Descent）。它非常稳定，但速度较慢。

随机梯度下降（SGD）仅基于单个随机样本计算梯度并立即更新。它的更新过程带有噪声，但速度极快。

小批量梯度下降（Mini-batch Gradient Descent）则取两者之长。每次计算一个小批量样本（如 32、64、128 或 256 个）的梯度，然后进行更新。这也是目前业界实际最常用的方法。

| 变体 | 批量大小 | 梯度质量 | 单步速度 | 噪声 |
|---------|-----------|-----------------|---------------|-------|
| 批量梯度下降（Batch GD） | 整个数据集 | 精确 | 慢 | 无 |
| 随机梯度下降（SGD） | 1 个样本 | 噪声极大 | 快 | 高 |
| 小批量梯度下降（Mini-batch） | 32-256 | 良好估计 | 均衡 | 中等 |

SGD 和小批量梯度下降中的噪声并非缺陷。它有助于模型跳出浅层的局部极小值（Local Minima）和鞍点（Saddle Points）。

### 动量（Momentum）：滚下山坡的球

基础梯度下降仅关注当前梯度。如果梯度呈锯齿状变化（在狭窄的谷底很常见），优化进度就会很慢。动量机制通过将历史梯度累积为一个速度项（Velocity Term）来解决这一问题。

v = beta * v + gradient
w = w - lr * v

类比：一个滚下山坡的球。它不会在每个小凸起处停下并重新起步。它会在方向一致时不断加速，并抑制震荡。

graph TD
    subgraph Without["Without Momentum (zigzag, slow)"]
        W1["Start"] -->|left| W2[" "]
        W2 -->|right| W3[" "]
        W3 -->|left| W4[" "]
        W4 -->|right| W5[" "]
        W5 -->|left| W6[" "]
        W6 --> W7["Minimum"]
    end
    subgraph With["With Momentum (smooth, fast)"]
        M1["Start"] --> M2[" "] --> M3[" "] --> M4["Minimum"]
    end

`beta`（通常设为 0.9）控制历史信息的保留程度。`beta` 值越高，动量越大，路径越平滑，但对方向变化的响应也会越慢。

### Adam：自适应学习率（Adaptive Learning Rates）

不同的权重需要不同的学习率。对于很少出现大梯度的权重，当梯度终于变大时，应该迈出更大的步长；而对于持续出现巨大梯度的权重，则应该缩小步长。

Adam（Adaptive Moment Estimation，自适应矩估计）为每个权重跟踪两个指标：

1. 一阶矩（First Moment, m）：梯度的滑动平均值（类似于动量）
2. 二阶矩（Second Moment, v）：梯度平方的滑动平均值（反映梯度幅值）

m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient^2

m_hat = m / (1 - beta1^t)    bias correction
v_hat = v / (1 - beta2^t)    bias correction

w = w - lr * m_hat / (sqrt(v_hat) + epsilon)

除以 `sqrt(v_hat)` 是核心思想。梯度较大的权重会被一个较大的数除（有效步长变小），梯度较小的权重会被一个较小的数除（有效步长变大）。每个权重都获得了独立的自适应学习率。

默认超参数：`lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8`。这些默认值在大多数问题上表现良好。

### 学习率调度（Learning Rate Schedules）

固定学习率只是一种折中方案。在训练初期，你希望步长较大以快速推进；在训练后期，你希望步长较小以便在极小值附近进行微调。

常见的调度策略：

| 调度策略 | 公式 | 适用场景 |
|----------|---------|----------|
| 阶梯衰减（Step decay） | 每 N 个 epoch 执行 lr = lr * factor | 简单，便于手动控制 |
| 指数衰减（Exponential decay） | lr = lr_0 * decay^t | 平滑下降 |
| 余弦退火（Cosine annealing） | lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T)) | Transformer 模型、现代训练流程 |
| 预热+衰减（Warmup + decay） | 线性上升，随后衰减 | 大模型，防止训练初期不稳定 |

### 凸函数与非凸函数（Convex vs Non-convex）

凸函数（Convex）只有一个极小值。梯度下降总能找到它。例如二次函数 `f(x) = x^2` 就是凸函数。

神经网络的损失函数是非凸的（Non-convex）。它们包含许多局部极小值、鞍点和平坦区域。

graph LR
    subgraph Convex["Convex: One valley, one answer"]
        direction TB
        CV1["High loss"] --> CV2["Global minimum"]
    end
    subgraph NonConvex["Non-convex: Multiple valleys, saddle points"]
        direction TB
        NC1["Start"] --> NC2["Local minimum"]
        NC1 --> NC3["Saddle point"]
        NC1 --> NC4["Global minimum"]
    end

在实践中，高维神经网络中的局部极小值很少成为问题。大多数局部极小值的损失值与全局极小值（Global Minimum）非常接近。真正的障碍是鞍点（在某些方向平坦，在其他方向弯曲）。动量机制和小批量带来的噪声有助于模型跳出这些区域。

### 损失地形可视化（Loss Landscape Visualization）

损失是所有权重的函数。对于一个拥有 100 万个权重的模型，其损失地形存在于 1,000,001 维空间中。为了可视化它，我们通常在权重空间中随机选取两个方向，并绘制沿这两个方向的损失变化，从而生成一个二维曲面。

graph TD
    HL["High loss region"] --> SP["Saddle point"]
    HL --> LM["Local minimum"]
    SP --> LM
    SP --> GM["Global minimum"]
    LM -.->|"shallow barrier"| GM
    style HL fill:#ff6666,color:#000
    style SP fill:#ffcc66,color:#000
    style LM fill:#66ccff,color:#000
    style GM fill:#66ff66,color:#000

尖锐的极小值泛化能力（Generalization）较差，而平坦的极小值泛化能力较好。这也是带动量的 SGD 在最终测试准确率上往往优于 Adam 的原因之一：其引入的噪声能防止模型陷入尖锐的极小值。

## 构建

### 步骤 1：定义测试函数

Rosenbrock 函数 (Rosenbrock function) 是一种经典的优化基准测试 (optimization benchmark)。其最小值位于 (1, 1) 处，处于一个狭窄的弯曲山谷中，该位置相对容易定位，但沿其追踪却十分困难。

f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2

def rosenbrock(params):
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_gradient(params):
    x, y = params
    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    df_dy = 200 * (y - x ** 2)
    return [df_dx, df_dy]

### 步骤 2：基础梯度下降 (Vanilla Gradient Descent)

class GradientDescent:
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params, grads):
        return [p - self.lr * g for p, g in zip(params, grads)]

### 步骤 3：带动量的随机梯度下降 (SGD with Momentum)

class SGDMomentum:
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [0.0] * len(params)
        self.velocity = [
            self.momentum * v + g
            for v, g in zip(self.velocity, grads)
        ]
        return [p - self.lr * v for p, v in zip(params, self.velocity)]

### 步骤 4：Adam 优化器 (Adam)

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

        self.m = [
            self.beta1 * m + (1 - self.beta1) * g
            for m, g in zip(self.m, grads)
        ]
        self.v = [
            self.beta2 * v + (1 - self.beta2) * g ** 2
            for v, g in zip(self.v, grads)
        ]

        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v]

        return [
            p - self.lr * mh / (vh ** 0.5 + self.epsilon)
            for p, mh, vh in zip(params, m_hat, v_hat)
        ]

### 步骤 5：运行与对比

def optimize(optimizer, func, grad_func, start, steps=5000):
    params = list(start)
    history = [params[:]]
    for _ in range(steps):
        grads = grad_func(params)
        params = optimizer.step(params, grads)
        history.append(params[:])
    return history

start = [-1.0, 1.0]

gd_history = optimize(GradientDescent(lr=0.0005), rosenbrock, rosenbrock_gradient, start)
sgd_history = optimize(SGDMomentum(lr=0.0001, momentum=0.9), rosenbrock, rosenbrock_gradient, start)
adam_history = optimize(Adam(lr=0.01), rosenbrock, rosenbrock_gradient, start)

for name, history in [("GD", gd_history), ("SGD+M", sgd_history), ("Adam", adam_history)]:
    final = history[-1]
    loss = rosenbrock(final)
    print(f"{name:6s} -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")

预期输出：Adam 收敛 (converge) 速度最快。带动量的 SGD 遵循更平滑的路径。基础梯度下降在狭窄山谷中进展缓慢。

## 实践应用

在实际开发中，建议直接使用 PyTorch 或 JAX 提供的优化器（Optimizer）。它们内置了对参数组（Parameter Groups）、权重衰减（Weight Decay）、梯度裁剪（Gradient Clipping）以及 GPU 加速的支持。

import torch

model = torch.nn.Linear(784, 10)

sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam = torch.optim.Adam(model.parameters(), lr=0.001)
adamw = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=100)

经验法则：

- 优先使用 Adam（学习率 lr=0.001）。它在大多数情况下无需调参即可取得良好效果。
- 当追求极致的最终准确率且有时间进行调参时，可切换至带动量的随机梯度下降（SGD with Momentum，lr=0.01, momentum=0.9）。
- 训练 Transformer 模型时，请使用 AdamW（解耦权重衰减的 Adam）。
- 对于训练周期超过几个训练轮次（Epoch）的任务，务必使用学习率调度器（Learning Rate Schedule）。
- 若训练过程不稳定，请降低学习率；若训练速度过慢，则适当提高学习率。

## 交付内容

本节内容将生成一个用于选择合适优化器的提示词（Prompt）。详见 `outputs/prompt-optimizer-guide.md`。

此处构建的优化器类将在第三阶段从零开始训练神经网络时再次用到。

## 练习题

1. **学习率扫描（Learning Rate Sweep）。** 在 Rosenbrock 函数上分别使用学习率 [0.0001, 0.0005, 0.001, 0.005, 0.01] 运行基础梯度下降（Vanilla Gradient Descent）。绘制或打印每个学习率在 5000 步后的最终损失值（Loss）。找出仍能收敛的最大学习率。

2. **动量对比。** 在 Rosenbrock 函数上分别使用动量值 [0.0, 0.5, 0.9, 0.99] 运行带动量的 SGD。记录每一步的损失值。哪个动量值收敛最快？哪个会出现超调（Overshoot）？

3. **逃离鞍点（Saddle Point Escape）。** 定义函数 `f(x, y) = x^2 - y^2`（原点处为鞍点）。从 (0.01, 0.01) 开始初始化。对比基础梯度下降（Vanilla GD）、带动量的 SGD 以及 Adam 的表现。哪种优化器能成功逃离鞍点？

4. **实现学习率衰减。** 为 `GradientDescent` 类添加指数衰减调度：`lr = lr_0 * 0.999^step`。在 Rosenbrock 函数上对比使用衰减与不使用衰减时的收敛情况。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 梯度下降 (Gradient Descent) | “下山” | 通过减去按学习率缩放的梯度来更新权重。这是最基础的优化器。 |
| 学习率 (Learning Rate) | “步长” | 控制每次更新时权重移动距离的标量。设置过大会导致发散，过小则会浪费算力。 |
| 动量 (Momentum) | “保持滚动” | 将历史梯度累积为速度向量。能够抑制震荡，并在方向一致时加速移动。 |
| 随机梯度下降 (SGD) | “随机采样” | 在随机数据子集而非完整数据集上计算梯度。在实际应用中，几乎总是指小批量随机梯度下降 (Mini-batch SGD)。 |
| 小批量 (Mini-batch) | “一块数据” | 用于估计梯度的一小部分训练数据（通常为 32-256 个样本）。在计算速度与梯度准确性之间取得平衡。 |
| 自适应矩估计 (Adam) | “默认优化器” | 跟踪每个权重的梯度及其平方的滑动平均值，从而为每个权重分配独立的学习率。 |
| 偏差校正 (Bias Correction) | “修复冷启动” | Adam 的一阶矩和二阶矩初始值均为零。偏差校正通过除以 `(1 - beta^t)` 来补偿训练初期的偏差。 |
| 学习率调度 (Learning Rate Schedule) | “随时间调整学习率” | 在训练过程中动态调整学习率的函数。通常初期采用较大步长，后期采用较小步长。 |
| 凸函数 (Convex Function) | “单一山谷” | 任何局部最小值即为全局最小值的函数。梯度下降总能找到该最小值。但神经网络的损失函数并非凸函数。 |
| 鞍点 (Saddle Point) | “平坦但非最小值” | 梯度为零的点，但在某些方向上是最小值，在其他方向上却是最大值。在高维空间中十分常见。 |
| 损失地形 (Loss Landscape) | “地形图” | 损失函数在权重空间上的分布形态。通常通过沿两个随机方向切片来进行可视化。 |
| 收敛 (Convergence) | “到达目标” | 优化器已达到某一状态，继续迭代无法再显著降低损失值。 |

## 延伸阅读

- [Sebastian Ruder：梯度下降优化算法综述](https://ruder.io/optimizing-gradient-descent/) - 全面综述了所有主流优化器
- [动量为何真正有效 (Distill)](https://distill.pub/2017/momentum/) - 动量动态的交互式可视化
- [Adam：一种随机优化方法 (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980) - Adam 的原始论文，篇幅短小且易于阅读
- [神经网络损失地形可视化 (Li et al., 2018)](https://arxiv.org/abs/1712.09913) - 揭示了尖锐最小值与平坦最小值差异的论文