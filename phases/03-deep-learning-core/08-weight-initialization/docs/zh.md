# 权重初始化 (Weight Initialization) 与训练稳定性 (Training Stability)

> 初始化错误，训练根本无法开始。初始化正确，50 层网络也能像 3 层一样顺畅训练。

**类型：** 实践
**语言：** Python
**前置课程：** 第 03.04 课（激活函数 (Activation Functions)），第 03.07 课（正则化 (Regularization)）
**时长：** 约 90 分钟

## 学习目标

- 实现零初始化 (Zero Initialization)、随机初始化 (Random Initialization)、Xavier/Glorot 初始化以及 Kaiming/He 初始化策略，并测量它们对信号穿过 50 层网络时激活值幅度 (Activation Magnitudes) 的影响
- 推导 Xavier 初始化为何使用 Var(w) = 2/(fan_in + fan_out)，而 Kaiming 初始化使用 Var(w) = 2/fan_in
- 演示零初始化带来的对称性问题 (Symmetry Problem)，并解释为何仅靠随机缩放比例仍不足够
- 为不同的激活函数匹配正确的初始化策略：Sigmoid/Tanh 使用 Xavier，ReLU/GELU 使用 Kaiming

## 问题所在

将所有权重初始化为零。网络什么也学不到。每个神经元计算相同的函数，接收相同的梯度，并进行完全相同的更新。经过 10,000 个轮次 (Epochs) 后，你那包含 512 个神经元的隐藏层依然是 512 个完全相同的神经元副本。你为 512 个参数付了钱，却只得到了 1 个。

将权重初始化得过大。激活值会在网络中呈爆炸式增长。到第 10 层时，数值会飙升至 1e15。到第 20 层时，它们会溢出至无穷大。梯度在反向传播时也会遵循完全相同的轨迹。

从标准正态分布 (Standard Normal Distribution) 中随机初始化权重。对于 3 层网络或许有效。但在 50 层网络中，信号会坍缩至零或爆炸至无穷大，这完全取决于随机缩放比例是略微偏小还是略微偏大。“有效”与“失效”之间的界限极其微妙。

权重初始化是深度学习中最被低估的决策之一。网络架构能发顶会论文，优化器能收获大量博客文章，而初始化往往只配得到一个脚注。但如果这一步走错，其他一切都毫无意义——你的网络在训练开始前就已经“死亡”了。

## 核心概念

### 对称性问题 (Symmetry Problem)

层中的每个神经元 (neuron) 都具有相同的结构：将输入与权重相乘，加上偏置 (bias)，然后应用激活函数 (activation function)。如果所有权重都初始化为相同的值（全零是极端情况），那么每个神经元都会计算出相同的输出。在反向传播 (backpropagation) 过程中，每个神经元接收到的梯度 (gradient) 也是相同的。在参数更新步骤中，每个神经元的权重变化量完全一致。

网络将陷入停滞。尽管网络拥有成百上千个参数，但它们的变化步调完全一致。这种现象被称为对称性 (symmetry)，而随机初始化 (random initialization) 正是打破这种对称性的直接方法。通过让每个神经元在权重空间 (weight space) 中从不同的起点开始，它们就能学习到不同的特征。

但仅仅“随机”是不够的。随机性的*尺度* (scale) 决定了网络能否成功训练。

### 方差在层间的传播 (Variance Propagation Through Layers)

考虑一个具有 `fan_in` 个输入的单层：

z = w1*x1 + w2*x2 + ... + w_n*x_n

如果每个权重 $w_i$ 从方差 (variance) 为 `Var(w)` 的分布中采样，且每个输入 $x_i$ 的方差为 `Var(x)`，则输出的方差为：

Var(z) = fan_in * Var(w) * Var(x)

如果 `Var(w) = 1` 且 `fan_in = 512`，输出方差将是输入方差的 512 倍。经过 10 层后：512^10 = 1.2e27。你的信号已经爆炸 (exploded)。

如果 `Var(w) = 0.001`，每层的输出方差会缩小为原来的 0.001 * 512 = 0.512。经过 10 层后：0.512^10 = 0.00013。你的信号已经消失 (vanished)。

我们的目标是：选择合适的 `Var(w)`，使得 `Var(z) = Var(x)`。这样，信号的幅度 (magnitude) 就能在层间保持恒定。

### Xavier/Glorot 初始化 (Xavier/Glorot Initialization)

Glorot 和 Bengio（2010）针对 Sigmoid 和 Tanh 激活函数推导出了该问题的解法。为了在前向传播 (forward pass) 和反向传播 (backward pass) 中均保持方差恒定：

Var(w) = 2 / (fan_in + fan_out)

在实际应用中，权重通常从以下分布中采样：

w ~ Uniform(-limit, limit)  where limit = sqrt(6 / (fan_in + fan_out))

或者：

w ~ Normal(0, sqrt(2 / (fan_in + fan_out)))

这种方法之所以有效，是因为 Sigmoid 和 Tanh 在零点附近近似线性，而经过合理初始化的激活值恰好落在这个区域。因此，方差能够在数十层网络中保持稳定。

### Kaiming/He 初始化 (Kaiming/He Initialization)

ReLU 激活函数会“杀死”一半的输出（所有负值变为零）。由于平均而言一半的输入会被置零，有效的 `fan_in` 实际上减半了。Xavier 初始化并未考虑这一点——它低估了所需的方差。

He 等人（2015）对此公式进行了调整：

Var(w) = 2 / fan_in

权重从以下分布中采样：

w ~ Normal(0, sqrt(2 / fan_in))

公式中的系数 2 正是为了补偿 ReLU 将一半激活值置零的影响。如果没有这个系数，信号每经过一层就会缩小约 0.5 倍。经过 50 层后：0.5^50 = 8.8e-16。Kaiming 初始化有效避免了这一问题。

### Transformer 初始化 (Transformer Initialization)

GPT-2 引入了一种不同的模式。残差连接 (residual connections) 将每个子层 (sub-layer) 的输出与其输入相加：

x = x + sublayer(x)

每次相加操作都会增加方差。在包含 N 个残差层的网络中，方差会随 N 成比例增长。GPT-2 将残差层的权重缩放为 1/sqrt(2N)（其中 N 为层数）。这确保了累积信号的幅度保持稳定。

Llama 3（4050 亿参数，126 层）采用了类似的方案。如果不进行这种缩放，残差流 (residual stream) 在穿过 126 层注意力 (attention) 和前馈网络块 (feedforward blocks) 后将会无限增长。

flowchart TD
    subgraph "Zero Init"
        Z1["Layer 1<br/>All weights = 0"] --> Z2["Layer 2<br/>All neurons identical"]
        Z2 --> Z3["Layer 3<br/>Still identical"]
        Z3 --> ZR["Result: 1 effective neuron<br/>regardless of width"]
    end

    subgraph "Xavier Init"
        X1["Layer 1<br/>Var = 2/(fan_in+fan_out)"] --> X2["Layer 2<br/>Signal stable"]
        X2 --> X3["Layer 50<br/>Signal stable"]
        X3 --> XR["Result: Trains with<br/>sigmoid/tanh"]
    end

    subgraph "Kaiming Init"
        K1["Layer 1<br/>Var = 2/fan_in"] --> K2["Layer 2<br/>Signal stable"]
        K2 --> K3["Layer 50<br/>Signal stable"]
        K3 --> KR["Result: Trains with<br/>ReLU/GELU"]
    end

### 50 层网络中的激活幅度 (Activation Magnitude Through 50 Layers)

graph LR
    subgraph "Mean Activation Magnitude"
        direction LR
        L1["Layer 1"] --> L10["Layer 10"] --> L25["Layer 25"] --> L50["Layer 50"]
    end

    subgraph "Results"
        R1["Random N(0,1): EXPLODES by layer 5"]
        R2["Random N(0,0.01): Vanishes by layer 10"]
        R3["Xavier + Sigmoid: ~1.0 at layer 50"]
        R4["Kaiming + ReLU: ~1.0 at layer 50"]
    end

### 选择合适的初始化方法 (Choosing the Right Init)

flowchart TD
    Start["What activation?"] --> Act{"Activation type?"}

    Act -->|"Sigmoid / Tanh"| Xavier["Xavier/Glorot<br/>Var = 2/(fan_in + fan_out)"]
    Act -->|"ReLU / Leaky ReLU"| Kaiming["Kaiming/He<br/>Var = 2/fan_in"]
    Act -->|"GELU / Swish"| Kaiming2["Kaiming/He<br/>(same as ReLU)"]
    Act -->|"Transformer residual"| GPT["Scale by 1/sqrt(2N)<br/>N = num layers"]

    Xavier --> Check["Verify: activation magnitudes<br/>stay between 0.5 and 2.0<br/>through all layers"]
    Kaiming --> Check
    Kaiming2 --> Check
    GPT --> Check


## 构建

### 步骤 1：初始化策略

初始化权重矩阵（Weight Matrix）的四种方法。每种方法都会返回一个列表的列表（即二维矩阵），包含 `fan_in` 列和 `fan_out` 行。

import math
import random


def zero_init(fan_in, fan_out):
    return [[0.0 for _ in range(fan_in)] for _ in range(fan_out)]


def random_init(fan_in, fan_out, scale=1.0):
    return [[random.gauss(0, scale) for _ in range(fan_in)] for _ in range(fan_out)]


def xavier_init(fan_in, fan_out):
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]


def kaiming_init(fan_in, fan_out):
    std = math.sqrt(2.0 / fan_in)
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]

### 步骤 2：激活函数

我们需要 Sigmoid、Tanh 和 ReLU 来测试每种初始化策略与其预期激活函数的匹配效果。

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def tanh_act(x):
    return math.tanh(x)


def relu(x):
    return max(0.0, x)

### 步骤 3：通过 50 层网络的前向传播

将随机数据输入深层网络，并测量每一层的平均激活幅度（Activation Magnitude）。

def forward_deep(init_fn, activation_fn, n_layers=50, width=64, n_samples=100):
    random.seed(42)
    layer_magnitudes = []

    inputs = [[random.gauss(0, 1) for _ in range(width)] for _ in range(n_samples)]

    for layer_idx in range(n_layers):
        weights = init_fn(width, width)
        biases = [0.0] * width

        new_inputs = []
        for sample in inputs:
            output = []
            for neuron_idx in range(width):
                z = sum(weights[neuron_idx][j] * sample[j] for j in range(width)) + biases[neuron_idx]
                output.append(activation_fn(z))
            new_inputs.append(output)
        inputs = new_inputs

        magnitudes = []
        for sample in inputs:
            magnitudes.append(sum(abs(v) for v in sample) / width)
        mean_mag = sum(magnitudes) / len(magnitudes)
        layer_magnitudes.append(mean_mag)

    return layer_magnitudes

### 步骤 4：实验设计

运行所有组合：零初始化（Zero Initialization）、随机初始化 N(0,1)、随机初始化 N(0,0.01)、Xavier 初始化（Xavier Initialization）配合 Sigmoid、Xavier 初始化配合 Tanh、Kaiming 初始化（Kaiming Initialization）配合 ReLU。打印关键层级的幅度值。

def run_experiment():
    configs = [
        ("Zero init + Sigmoid", lambda fi, fo: zero_init(fi, fo), sigmoid),
        ("Random N(0,1) + ReLU", lambda fi, fo: random_init(fi, fo, 1.0), relu),
        ("Random N(0,0.01) + ReLU", lambda fi, fo: random_init(fi, fo, 0.01), relu),
        ("Xavier + Sigmoid", xavier_init, sigmoid),
        ("Xavier + Tanh", xavier_init, tanh_act),
        ("Kaiming + ReLU", kaiming_init, relu),
    ]

    print(f"{'Strategy':<30} {'L1':>10} {'L5':>10} {'L10':>10} {'L25':>10} {'L50':>10}")
    print("-" * 80)

    for name, init_fn, act_fn in configs:
        mags = forward_deep(init_fn, act_fn)
        row = f"{name:<30}"
        for idx in [0, 4, 9, 24, 49]:
            val = mags[idx]
            if val > 1e6:
                row += f" {'EXPLODED':>10}"
            elif val < 1e-6:
                row += f" {'VANISHED':>10}"
            else:
                row += f" {val:>10.4f}"
        print(row)

### 步骤 5：对称性演示

演示零初始化如何导致所有神经元（Neurons）产生完全相同的输出。

def symmetry_demo():
    random.seed(42)
    weights = zero_init(2, 4)
    biases = [0.0] * 4

    inputs = [0.5, -0.3]
    outputs = []
    for neuron_idx in range(4):
        z = sum(weights[neuron_idx][j] * inputs[j] for j in range(2)) + biases[neuron_idx]
        outputs.append(sigmoid(z))

    print("\nSymmetry Demo (4 neurons, zero init):")
    for i, out in enumerate(outputs):
        print(f"  Neuron {i}: output = {out:.6f}")
    all_same = all(abs(outputs[i] - outputs[0]) < 1e-10 for i in range(len(outputs)))
    print(f"  All identical: {all_same}")
    print(f"  Effective parameters: 1 (not {len(weights) * len(weights[0])})")

### 步骤 6：逐层幅度报告

打印可视化条形图，展示数据流经 50 层网络时的激活幅度变化。

def magnitude_report(name, magnitudes):
    print(f"\n{name}:")
    for i, mag in enumerate(magnitudes):
        if i % 5 == 0 or i == len(magnitudes) - 1:
            if mag > 1e6:
                bar = "X" * 50 + " EXPLODED"
            elif mag < 1e-6:
                bar = "." + " VANISHED"
            else:
                bar_len = min(50, max(1, int(mag * 10)))
                bar = "#" * bar_len
            print(f"  Layer {i+1:3d}: {bar} ({mag:.6f})")


## 使用方法

PyTorch 将这些方法作为内置函数提供：

import torch
import torch.nn as nn

layer = nn.Linear(512, 256)

nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)

nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

nn.init.zeros_(layer.bias)

当你调用 `nn.Linear(512, 256)` 时，PyTorch 默认采用 Kaiming 均匀初始化（Kaiming uniform initialization）。这就是为什么大多数简单网络能够“开箱即用”——PyTorch 已经替你做出了正确的选择。然而，当你构建自定义架构或网络深度超过 20 层时，你需要理解其底层机制，并可能需要手动覆盖默认配置。

对于 Transformer 模型，HuggingFace 的模型通常在其 `_init_weights` 方法中处理权重初始化（weight initialization）。GPT-2 的实现会将残差投影（residual projections）按 `1/sqrt(N)` 进行缩放。如果你从零开始构建 Transformer，则需要自行实现这一逻辑。

## 交付成果

本课时将产出：
- `outputs/prompt-init-strategy.md` -- 一个用于诊断权重初始化问题并推荐合适策略的提示词（prompt）

## 练习

1. 添加 LeCun 初始化（LeCun initialization）（方差 Var = 1/fan_in，专为 SELU 激活函数（SELU activation）设计）。使用 LeCun 初始化 + tanh 运行 50 层实验，并与 Xavier 初始化（Xavier initialization） + tanh 进行对比。

2. 实现 GPT-2 的残差缩放（residual scaling）：在将输出添加到残差流（residual stream）之前，将每一层的输出乘以 `1/sqrt(2*N)`。分别在有缩放和无缩放的情况下运行 50 层网络，测量残差幅值（residual magnitude）的增长速度。

3. 创建一个“初始化健康检查”（init health check）函数，该函数接收网络的层维度（layer dimensions）和激活函数类型，随后推荐正确的初始化方案，并在当前初始化可能导致问题时发出警告。

4. 分别使用 `fan_in = 16` 和 `fan_in = 1024` 运行实验。Xavier 和 Kaiming 初始化会根据 `fan_in` 自适应调整，但随机初始化（random initialization）不会。请展示随着网络层变大，“有效”与“失效”之间的差距是如何扩大的。

5. 实现正交初始化（orthogonal initialization）（生成随机矩阵，计算其奇异值分解 SVD，并使用正交矩阵 U）。在 50 层 ReLU 网络中，将其与 Kaiming 初始化进行对比。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 权重初始化 (Weight initialization) | “随机设置初始权重” | 选择初始权重值的策略，决定了网络是否具备可训练性 |
| 对称性破缺 (Symmetry breaking) | “让神经元各不相同” | 通过随机初始化确保神经元学习不同的特征，而非执行相同的计算函数 |
| 扇入 (Fan-in) | “神经元的输入数量” | 传入连接的数量，决定了输入方差在加权和中的累积程度 |
| 扇出 (Fan-out) | “神经元的输出数量” | 传出连接的数量，在反向传播过程中对维持梯度方差至关重要 |
| Xavier/Glorot 初始化 (Xavier/Glorot init) | “Sigmoid 初始化” | Var(w) = 2/(fan_in + fan_out)，专为在 Sigmoid 和 Tanh 激活函数中保持方差而设计 |
| Kaiming/He 初始化 (Kaiming/He init) | “ReLU 初始化” | Var(w) = 2/fan_in，考虑了 ReLU 激活函数会将一半的激活值置零的特性 |
| 方差传播 (Variance propagation) | “信号在层间如何放大或缩小” | 基于权重尺度，逐层分析激活方差如何变化的数学推导 |
| 残差缩放 (Residual scaling) | “GPT-2 的初始化技巧” | 将残差连接权重按 1/sqrt(2N) 进行缩放，以防止方差在 N 个 Transformer 层中不断累积 |
| 死亡网络 (Dead network) | “什么都训练不起来” | 因初始化不当导致所有梯度为零或所有激活值进入饱和状态的网络 |
| 激活值爆炸 (Exploding activations) | “数值趋向无穷大” | 当权重方差过大时，导致激活值幅度在层间呈指数级增长的现象 |

## 延伸阅读

- Glorot & Bengio，《Understanding the difficulty of training deep feedforward neural networks》（2010）—— 提出 Xavier 初始化 (Xavier initialization) 的原始论文，包含详细的方差分析
- He 等人，《Delving Deep into Rectifiers》（2015）—— 针对 ReLU 网络引入了 Kaiming 初始化 (Kaiming initialization)
- Radford 等人，《Language Models are Unsupervised Multitask Learners》（2019）—— GPT-2 论文，提出了残差缩放初始化 (residual scaling initialization)
- Mishkin & Matas，《All You Need is a Good Init》（2016）—— 提出层序单位方差初始化 (layer-sequential unit-variance initialization)，作为解析公式的一种经验替代方案