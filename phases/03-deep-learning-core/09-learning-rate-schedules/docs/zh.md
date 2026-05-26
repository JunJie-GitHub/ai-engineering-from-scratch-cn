# 学习率调度与预热

> 学习率（Learning Rate）是唯一最重要的超参数（Hyperparameter）。不是网络架构（Architecture），不是数据集规模（Dataset Size），也不是激活函数（Activation Function）。就是学习率。如果你只调整一个参数，那就调整它。

**类型：** 构建
**语言：** Python
**前置课程：** 第 03.06 课（优化器 (Optimizers)），第 03.08 课（权重初始化 (Weight Initialization)）
**时长：** 约 90 分钟

## 学习目标

- 从零开始实现恒定（Constant）、阶梯衰减（Step Decay）、余弦退火（Cosine Annealing）、预热+余弦（Warmup + Cosine）以及 1cycle 学习率调度策略
- 演示学习率选择的三种失败模式：发散（Divergence，过高）、停滞（Stalling，过低）和振荡（Oscillation，无衰减）
- 解释为什么基于 Adam 的优化器需要预热（Warmup），以及它如何稳定训练初期
- 在同一任务上比较所有五种调度的收敛速度（Convergence Speed），并根据给定的训练预算选择合适的策略

## 问题所在

将学习率设为 0.1，训练会发散——损失值在 3 个步骤内飙升至无穷大。设为 0.0001，训练如同爬行——经过 100 个轮次（Epochs）后，模型几乎仍停留在随机初始状态。设为 0.01，训练在前 50 个轮次表现正常，随后损失值会在一个永远无法达到的最小值附近振荡，因为步长太大了。

最优的学习率并非固定不变，它会随着训练过程动态调整。在训练初期，你需要较大的步长来快速探索参数空间；而在训练后期，你需要极小的步长来收敛至尖锐的最小值。一个 90% 准确率的模型与一个 95% 准确率的模型之间的差距，往往仅仅在于学习率调度策略的不同。

过去三年发布的每一个主流模型都采用了学习率调度策略。Llama 3 使用了峰值 `lr=3e-4`，配合 2000 步的预热和余弦衰减至 `3e-5`。GPT-3 使用了 `lr=6e-4`，并在 3.75 亿个 token 上进行预热。这些并非随意设定的数值，而是耗资数百万美元进行大规模超参数搜索（Hyperparameter Sweeps）后得出的结果。

你必须深入理解调度策略，因为默认设置通常无法直接解决你的特定问题。当你微调（Fine-tune）预训练模型时，合适的调度策略与从零开始训练截然不同。当你增大批次大小（Batch Size）时，预热周期也需要相应调整。当训练在第 10,000 步崩溃时，你需要能够判断这究竟是调度策略的问题，还是其他原因导致的。

## 核心概念

### 恒定学习率 (Constant Learning Rate)

最简单的方法。选定一个数值，在每一步训练中都使用它。

lr(t) = lr_0

很少能达到最优效果。它在训练后期往往过高（导致在最小值附近震荡），而在训练初期又往往过低（在微小步长上浪费算力）。适用于小型模型和调试阶段。对于训练时间超过一小时的任何任务来说，这都是一个糟糕的选择。

### 阶梯衰减 (Step Decay)

源自 ResNet 时代的经典方法。在固定的训练轮数 (Epoch) 处，按固定比例（通常为 10 倍）降低学习率。

lr(t) = lr_0 * gamma^(floor(epoch / step_size))

其中 `gamma = 0.1` 且 `step_size = 30` 表示：每 30 个 Epoch 学习率下降 10 倍。ResNet-50 曾采用此策略——初始学习率 `lr=0.1`，在第 30、60 和 90 个 Epoch 时分别下降 10 倍。

问题在于：最佳衰减点高度依赖于数据集和模型架构。一旦更换任务，就需要重新调整衰减时机。此外，衰减过程是突变的——学习率突然变化时，损失值 (Loss) 可能会出现尖峰。

### 余弦退火 (Cosine Annealing)

遵循余弦曲线，从最大学习率平滑衰减至最小值：

lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))

其中 `t` 为当前步数，`T` 为总步数。

当 `t=0` 时，余弦项为 1，因此 `lr = lr_max`。当 `t=T` 时，余弦项为 -1，因此 `lr = lr_min`。衰减初期较为平缓，中期加速，末期再次趋于平缓。

这是目前大多数现代训练任务的默认选择。除了 `lr_max` 和 `lr_min` 之外，无需调整其他超参数 (Hyperparameter)。余弦形状符合经验观察：大部分学习过程发生在训练中期——在该关键阶段，你希望保持合理的步长。

### 预热 (Warmup)：为何要从较小的学习率开始

Adam 等自适应优化器 (Adaptive Optimizer) 会维护梯度 (Gradient) 均值和方差 (Variance) 的滑动估计值。在第 0 步时，这些估计值初始化为零。最初的几次梯度更新基于极不准确的统计量。如果在此期间学习率过大，模型将迈出巨大且方向错误的步长。

预热机制解决了这一问题。从一个极小的学习率开始（通常为 `lr_max / warmup_steps` 甚至为零），并在前 N 步内线性递增至 `lr_max`。当达到完整学习率时，Adam 的统计量已经趋于稳定。

lr(t) = lr_max * (t / warmup_steps)     for t < warmup_steps

典型的预热时长：占总训练步数的 1%~5%。Llama 3 训练了约 1.8 万亿个 Token，预热了 2000 步。GPT-3 则在 3.75 亿个 Token 期间进行预热。

### 线性预热 + 余弦衰减 (Linear Warmup + Cosine Decay)

现代训练的默认方案。先线性递增，再进行余弦衰减：

if t < warmup_steps:
    lr(t) = lr_max * (t / warmup_steps)
else:
    progress = (t - warmup_steps) / (total_steps - warmup_steps)
    lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))

Llama、GPT、PaLM 以及大多数现代 Transformer 架构均采用此策略。预热防止了训练初期的不稳定性，而余弦衰减则帮助模型收敛至一个较优的极小值。

### 1cycle 策略 (1cycle Policy)

Leslie Smith 于 2018 年提出：在训练的前半段，将学习率从低值递增至高值，然后在后半段再递减回来。这看似违反直觉——为什么要在训练中途*提高*学习率？

理论依据：较高的学习率通过向优化轨迹添加噪声，起到了正则化 (Regularization) 的作用。在递增阶段，模型能够探索更广阔的损失曲面 (Loss Landscape)，从而找到更优的吸引域 (Basin)。随后的递减阶段则在该最佳吸引域内进行精细优化。

Phase 1 (0 to T/2):    lr ramps from lr_max/25 to lr_max
Phase 2 (T/2 to T):    lr ramps from lr_max to lr_max/10000

在固定的算力预算下，1cycle 策略通常比余弦退火训练得更快。代价是：你必须提前知道总训练步数。

### 调度曲线形状 (Schedule Shapes)

graph LR
    subgraph "Constant"
        C1["lr"] --- C2["lr"] --- C3["lr"]
    end

    subgraph "Step Decay"
        S1["0.1"] --- S2["0.1"] --- S3["0.01"] --- S4["0.001"]
    end

    subgraph "Cosine Annealing"
        CS1["lr_max"] --> CS2["gradual"] --> CS3["steep"] --> CS4["lr_min"]
    end

    subgraph "Warmup + Cosine"
        WC1["0"] --> WC2["lr_max"] --> WC3["cosine"] --> WC4["lr_min"]
    end

### 决策流程图 (Decision Flowchart)

flowchart TD
    Start["Choosing a LR schedule"] --> Know{"Know total<br/>training steps?"}

    Know -->|"Yes"| Budget{"Compute budget?"}
    Know -->|"No"| Constant["Use constant LR<br/>with manual decay"]

    Budget -->|"Large (days/weeks)"| WarmCos["Warmup + Cosine Decay<br/>(Llama/GPT default)"]
    Budget -->|"Small (hours)"| OneCycle["1cycle Policy<br/>(fastest convergence)"]
    Budget -->|"Moderate"| Cosine["Cosine Annealing<br/>(safe default)"]

    WarmCos --> Warmup["Warmup = 1-5% of steps"]
    OneCycle --> FindLR["Find lr_max with LR range test"]
    Cosine --> MinLR["Set lr_min = lr_max / 10"]

### 已发表模型的实际参数 (Real Numbers from Published Models)

graph TD
    subgraph "Published LR Configs"
        L3["Llama 3 (405B)<br/>Peak: 3e-4<br/>Warmup: 2000 steps<br/>Schedule: Cosine to 3e-5"]
        G3["GPT-3 (175B)<br/>Peak: 6e-4<br/>Warmup: 375M tokens<br/>Schedule: Cosine to 0"]
        R50["ResNet-50<br/>Peak: 0.1<br/>Warmup: none<br/>Schedule: Step decay x0.1 at 30,60,90"]
        B["BERT (340M)<br/>Peak: 1e-4<br/>Warmup: 10K steps<br/>Schedule: Linear decay"]
    end


## 构建项目

### 步骤 1：调度函数 (Schedule Functions)

每个函数接收当前训练步数 (step)，并返回该步对应的学习率 (learning rate)。

import math


def constant_schedule(step, lr=0.01, **kwargs):
    return lr


def step_decay_schedule(step, lr=0.1, step_size=100, gamma=0.1, **kwargs):
    return lr * (gamma ** (step // step_size))


def cosine_schedule(step, lr=0.01, total_steps=1000, lr_min=1e-5, **kwargs):
    if step >= total_steps:
        return lr_min
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * step / total_steps))


def warmup_cosine_schedule(step, lr=0.01, total_steps=1000, warmup_steps=100, lr_min=1e-5, **kwargs):
    if total_steps <= warmup_steps:
        return lr * (step / max(warmup_steps, 1))
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))


def one_cycle_schedule(step, lr=0.01, total_steps=1000, **kwargs):
    mid = max(total_steps // 2, 1)
    if step < mid:
        return (lr / 25) + (lr - lr / 25) * step / mid
    else:
        progress = (step - mid) / max(total_steps - mid, 1)
        return lr * (1 - progress) + (lr / 10000) * progress

### 步骤 2：可视化所有调度策略

打印基于文本的图表，展示每种调度策略在训练过程中的演变趋势。

def visualize_schedule(name, schedule_fn, total_steps=500, **kwargs):
    steps = list(range(0, total_steps, total_steps // 20))
    if total_steps - 1 not in steps:
        steps.append(total_steps - 1)

    lrs = [schedule_fn(s, total_steps=total_steps, **kwargs) for s in steps]
    max_lr = max(lrs) if max(lrs) > 0 else 1.0

    print(f"\n{name}:")
    for s, lr_val in zip(steps, lrs):
        bar_len = int(lr_val / max_lr * 40)
        bar = "#" * bar_len
        print(f"  Step {s:4d}: lr={lr_val:.6f} {bar}")

### 步骤 3：训练网络 (Training Network)

在圆形数据集 (circle dataset) 上使用一个简单的两层网络，架构与之前的课程相同，但此次我们将改变学习率调度策略。

import random


def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def relu(x):
    return max(0.0, x)


def relu_deriv(x):
    return 1.0 if x > 0 else 0.0


def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


def train_with_schedule(schedule_fn, schedule_name, data, epochs=300, base_lr=0.05, **kwargs):
    random.seed(0)
    hidden_size = 8
    total_steps = epochs * len(data)

    std = math.sqrt(2.0 / 2)
    w1 = [[random.gauss(0, std) for _ in range(2)] for _ in range(hidden_size)]
    b1 = [0.0] * hidden_size
    w2 = [random.gauss(0, std) for _ in range(hidden_size)]
    b2 = 0.0

    step = 0
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for x, target in data:
            lr = schedule_fn(step, lr=base_lr, total_steps=total_steps, **kwargs)

            z1 = []
            h = []
            for i in range(hidden_size):
                z = w1[i][0] * x[0] + w1[i][1] * x[1] + b1[i]
                z1.append(z)
                h.append(relu(z))

            z2 = sum(w2[i] * h[i] for i in range(hidden_size)) + b2
            out = sigmoid(z2)

            error = out - target
            d_out = error * out * (1 - out)

            for i in range(hidden_size):
                d_h = d_out * w2[i] * relu_deriv(z1[i])
                w2[i] -= lr * d_out * h[i]
                for j in range(2):
                    w1[i][j] -= lr * d_h * x[j]
                b1[i] -= lr * d_h
            b2 -= lr * d_out

            total_loss += (out - target) ** 2
            if (out >= 0.5) == (target >= 0.5):
                correct += 1
            step += 1

        avg_loss = total_loss / len(data)
        accuracy = correct / len(data) * 100
        epoch_losses.append(avg_loss)

    return epoch_losses

### 步骤 4：对比所有调度策略

使用每种调度策略训练相同的网络，并对比最终损失 (loss) 和收敛行为 (convergence behavior)。

def compare_schedules(data):
    configs = [
        ("Constant", constant_schedule, {}),
        ("Step Decay", step_decay_schedule, {"step_size": 15000, "gamma": 0.1}),
        ("Cosine", cosine_schedule, {"lr_min": 1e-5}),
        ("Warmup+Cosine", warmup_cosine_schedule, {"warmup_steps": 3000, "lr_min": 1e-5}),
        ("1cycle", one_cycle_schedule, {}),
    ]

    print(f"\n{'Schedule':<20} {'Start Loss':>12} {'Mid Loss':>12} {'End Loss':>12} {'Best Loss':>12}")
    print("-" * 70)

    for name, schedule_fn, extra_kwargs in configs:
        losses = train_with_schedule(schedule_fn, name, data, epochs=300, base_lr=0.05, **extra_kwargs)
        mid_idx = len(losses) // 2
        best = min(losses)
        print(f"{name:<20} {losses[0]:>12.6f} {losses[mid_idx]:>12.6f} {losses[-1]:>12.6f} {best:>12.6f}")

### 步骤 5：学习率过高与过低的影响

演示三种典型情况：过高（导致发散 divergence）、过低（导致收敛极慢 crawling）以及设置适中。

def lr_sensitivity(data):
    learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]

    print("\nLR Sensitivity (constant schedule, 100 epochs):")
    print(f"  {'LR':>10} {'Start Loss':>12} {'End Loss':>12} {'Status':>15}")
    print("  " + "-" * 52)

    for lr in learning_rates:
        losses = train_with_schedule(constant_schedule, f"lr={lr}", data, epochs=100, base_lr=lr)
        start = losses[0]
        end = losses[-1]

        if end > start or math.isnan(end) or end > 1.0:
            status = "DIVERGED"
        elif end > start * 0.9:
            status = "BARELY MOVED"
        elif end < 0.15:
            status = "CONVERGED"
        else:
            status = "LEARNING"

        end_str = f"{end:.6f}" if not math.isnan(end) else "NaN"
        print(f"  {lr:>10.4f} {start:>12.6f} {end_str:>12} {status:>15}")


## 使用方法

PyTorch 在 `torch.optim.lr_scheduler` 模块中提供了调度器（Scheduler）：

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = optim.Adam(model.parameters(), lr=3e-4)

scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)

for step in range(1000):
    loss = train_step(model, optimizer)
    scheduler.step()

对于预热（Warmup）+ 余弦退火（Cosine Annealing）策略，可以使用 lambda 调度器，或 HuggingFace 提供的 `get_cosine_schedule_with_warmup` 函数：

from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=100000,
)

该 HuggingFace 函数是大多数 Llama 和 GPT 微调（Fine-tuning）脚本所采用的方案。如果不确定如何选择，建议使用预热 + 余弦退火策略，并将预热步数设置为总步数的 3% 到 5%。该策略几乎适用于所有场景。

## 交付成果

本章节将生成：
- `outputs/prompt-lr-schedule-advisor.md` -- 一个提示词（Prompt），用于根据你的训练配置推荐合适的学习率调度策略与超参数（Hyperparameters）

## 练习

1. 实现指数衰减（Exponential Decay）：lr(t) = lr_0 * gamma^t，其中 gamma = 0.999。在 circle 数据集上将其与余弦退火进行对比。

2. 实现学习率范围测试（Learning Rate Range Test，由 Leslie Smith 提出）：在几百个训练步数内，将学习率从 1e-7 指数级增加至 1。绘制损失（Loss）随学习率变化的曲线。最佳的最大学习率位于损失开始上升之前的临界点。

3. 使用预热 + 余弦退火策略进行训练，但调整预热长度：分别设置为总步数的 0%、1%、5%、10% 和 20%。寻找训练最稳定的最佳平衡点。

4. 实现带热重启的余弦退火（Cosine Annealing with Warm Restarts，SGDR）：每隔 T 步将学习率重置为 lr_max 并再次衰减。在更长的训练周期中，将其与标准余弦退火进行对比。

5. 构建一个“调度器手术刀（Schedule Surgeon）”工具：监控训练损失，当损失趋于稳定时自动从预热切换至余弦退火；若损失长时间陷入平台期（Plateau），则自动降低学习率。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 学习率 (Learning Rate) | “模型学习的速度” | 用于乘以梯度以确定参数更新步长的标量 |
| 调度策略 (Schedule) | “随时间改变学习率” | 将训练步数映射为学习率的函数，旨在优化模型收敛过程 |
| 预热 (Warmup) | “从较小的学习率开始” | 在前 N 个训练步中将学习率从接近零线性提升至目标值，以稳定优化器统计量 |
| 余弦退火 (Cosine Annealing) | “平滑的学习率衰减” | 在训练过程中，使学习率沿余弦曲线从 `lr_max` 衰减至 `lr_min` |
| 阶梯衰减 (Step Decay) | “在里程碑处降低学习率” | 在固定的 epoch 间隔处，将学习率乘以一个系数（通常为 0.1） |
| 1Cycle 策略 (1Cycle Policy) | “先升后降” | Leslie Smith 提出的一种方法，在单个周期内先提升后降低学习率，以加速收敛 |
| 学习率范围测试 (LR Range Test) | “寻找最佳学习率” | 在短暂训练过程中逐步提高学习率，以找到损失函数开始发散时的临界值 |
| 带热重启的余弦调度 (Cosine with Warm Restarts) | “重置并重复” | 周期性地将学习率重置为 `lr_max` 并再次衰减（即 SGDR 算法） |
| 最小学习率 (Eta min) | “学习率的下限” | 调度策略衰减所达到的最低学习率值 |
| 峰值学习率 (Peak Learning Rate) | “最大学习率” | 训练过程中达到的最高学习率，通常出现在预热阶段之后 |

## 扩展阅读

- Loshchilov & Hutter，《SGDR: Stochastic Gradient Descent with Warm Restarts》（2017）—— 引入了余弦退火与热重启机制
- Smith，《Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates》（2018）—— 提出 1Cycle 策略的论文
- Touvron 等人，《Llama 2: Open Foundation and Fine-Tuned Chat Models》（2023）—— 记录了大规模训练中使用的预热与余弦调度方案
- Goyal 等人，《Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour》（2017）—— 提出了大批次训练中的线性缩放规则与预热方法