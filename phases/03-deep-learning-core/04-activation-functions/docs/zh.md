# 激活函数 (Activation Functions)

> 如果没有非线性（nonlinearity），你的百层网络不过是一个花哨的矩阵乘法。激活函数就是让神经网络能够以曲线方式进行“思考”的闸门。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 03.03 课（反向传播 (Backpropagation)）
**Time:** 约 75 分钟

## 学习目标

- 从零开始实现 sigmoid、tanh、ReLU、Leaky ReLU、GELU、Swish 和 softmax 及其导数
- 通过测量不同激活函数在 10 层以上网络中的激活值幅度，诊断梯度消失问题（vanishing gradient problem）
- 检测 ReLU 网络中的死亡神经元（dead neurons），并解释 GELU 为何能避免这种失效模式
- 为给定架构（Transformer、CNN、RNN、输出层）选择合适的激活函数

## 问题所在

堆叠两个线性变换：y = W2(W1x + b1) + b2。展开后得到：y = W2W1x + W2b1 + b2。这实际上就是 y = Ax + c —— 一个单一的线性变换。无论你堆叠多少个线性层，最终结果都会坍缩为一次矩阵乘法。你的百层网络与单层网络具有相同的表示能力（representational power）。

这并非理论上的奇思妙想。它意味着深层线性网络根本无法学习异或（XOR）逻辑，无法对螺旋数据集进行分类，也无法识别人脸。如果没有激活函数，网络的深度就只是一种假象。

激活函数打破了线性关系。它们通过非线性函数对每一层的输出进行变换，赋予网络弯曲决策边界（decision boundaries）、逼近任意函数以及真正进行学习的能力。但如果选错了激活函数，你的梯度可能会消失至零（例如深层网络中的 sigmoid）、爆炸至无穷大（未经谨慎初始化的无界激活函数），或者导致神经元永久死亡（带有较大负偏置（bias）的 ReLU）。激活函数的选择直接决定了你的网络是否能够进行有效学习。

## 核心概念

### 为什么需要非线性

矩阵乘法具有可组合性。将向量先乘以矩阵 A 再乘以矩阵 B，等同于直接乘以矩阵 AB。这意味着堆叠十个线性层在数学上等价于一个包含一个大矩阵的线性层。所有的参数、所有的深度——都白费了。你需要某种东西来打破这种线性链式结构。这正是激活函数（Activation Function）的作用。

证明如下。一个线性层计算 `f(x) = Wx + b`。堆叠两层：

Layer 1: h = W1 * x + b1
Layer 2: y = W2 * h + b2

代入计算：

y = W2 * (W1 * x + b1) + b2
y = (W2 * W1) * x + (W2 * b1 + b2)
y = A * x + c

结果仍然只是一个线性层。如果在层之间插入一个非线性激活函数 `g()`：

h = g(W1 * x + b1)
y = W2 * h + b2

此时代入过程被打破。`W2 * g(W1 * x + b1) + b2` 无法再简化为单一的线性变换。网络因此能够表示非线性函数。每增加一个带有激活函数的层，都会提升网络的表征能力（Representational Capacity）。

### Sigmoid

神经网络最早使用的激活函数。

sigmoid(x) = 1 / (1 + e^(-x))

输出范围：(0, 1)。函数平滑且可微，能将任意实数映射为类似概率的值。

其导数为：

sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

该导数的最大值为 0.25，出现在 x = 0 处。在反向传播（Backpropagation）过程中，梯度会逐层相乘。经过十层 Sigmoid 意味着梯度最多会被乘以 0.25 十次：

0.25^10 = 0.000000953674

不到原始信号的百万分之一。这就是梯度消失问题（Vanishing Gradient Problem）。浅层网络的梯度变得极小，导致权重几乎无法更新。网络看似在学习（深层的损失在下降），但前几层实际上已经“冻结”。深层的 Sigmoid 网络根本无法训练。

另一个问题是：Sigmoid 的输出始终为正（0 到 1），这意味着权重的梯度符号始终相同。这会导致梯度下降（Gradient Descent）过程中出现“之”字形震荡。

### Tanh

Sigmoid 的零中心化（Zero-centered）版本。

tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

输出范围：(-1, 1)。零中心化特性消除了上述的“之”字形震荡问题。

其导数为：

tanh'(x) = 1 - tanh(x)^2

在 x = 0 处导数最大值为 1.0，是 Sigmoid 的四倍。但梯度消失问题依然存在。当输入为较大的正数或负数时，导数仍会趋近于零。经过十层后梯度依然会被严重压缩，只是程度稍轻。

### ReLU：突破性进展

线性整流单元（Rectified Linear Unit, ReLU）。该函数由 Nair 和 Hinton 于 2010 年在深度学习领域推广（其本身可追溯至 Fukushima 1969 年的工作），彻底改变了领域格局。

relu(x) = max(0, x)

输出范围：[0, 正无穷)。其导数极其简单：

relu'(x) = 1  if x > 0
            0  if x <= 0

对于正输入不存在梯度消失问题。梯度恰好为 1，直接无损传递。这正是深层网络变得可训练的原因——ReLU 在各层之间保持了梯度的幅度。

但它存在一种失效模式：神经元死亡问题（Dead Neuron Problem）。如果某个神经元的加权输入始终为负（由于较大的负偏置或不幸的权重初始化），其输出将始终为零，梯度也始终为零，导致权重永远无法更新。该神经元将永久“死亡”。在实践中，ReLU 网络在训练期间可能有 10% 到 40% 的神经元会死亡。

### Leaky ReLU

解决神经元死亡问题最简单的方案。

leaky_relu(x) = x        if x > 0
                alpha * x if x <= 0

其中 `alpha` 是一个较小的常数，通常取 0.01。负半轴具有一个微小的斜率而非零，因此即使神经元处于“死亡”状态，仍能接收到梯度信号并有机会恢复。

### GELU：现代默认选择

高斯误差线性单元（Gaussian Error Linear Unit, GELU）。由 Hendrycks 和 Gimpel 于 2016 年提出。它是 BERT、GPT 以及大多数现代 Transformer 架构的默认激活函数。

gelu(x) = x * Phi(x)

其中 `Phi(x)` 是标准正态分布的累积分布函数（Cumulative Distribution Function, CDF）。实际计算中使用的近似公式为：

gelu(x) ~= 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

GELU 处处平滑，允许较小的负值输出（不同于 ReLU 的硬性截断为零），并且具有概率解释：它根据输入在高斯分布下为正的概率来对其进行加权。这种平滑的门控机制（Gating Mechanism）在 Transformer 架构中优于 ReLU，因为它提供了更优的梯度流，并彻底避免了神经元死亡问题。

### Swish / SiLU

一种自门控（Self-gated）激活函数，由 Ramachandran 等人于 2017 年通过自动化搜索发现。

swish(x) = x * sigmoid(x)

Swish 的数学形式即为 `x * sigmoid(x)`。Google 通过在激活函数空间中进行自动化搜索发现了它——这相当于用神经网络来设计神经网络的一部分。

与 GELU 类似，它也是平滑、非单调的，并允许较小的负值。两者的区别很微妙：Swish 使用 Sigmoid 进行门控，而 GELU 使用高斯 CDF。在实际应用中，两者的性能几乎相同。Swish 常用于 EfficientNet 及部分视觉模型，而 GELU 则在语言模型中占据主导地位。

### Softmax：输出层激活函数

不用于隐藏层。Softmax 将原始得分向量（Logits）转换为概率分布（Probability Distribution）。

softmax(x_i) = e^(x_i) / sum(e^(x_j) for all j)

每个输出值都在 0 到 1 之间，且所有输出值之和为 1。这使其成为多分类（Multi-class Classification）任务标准的最终激活函数。最大的 Logit 会获得最高的概率，但与 `argmax` 不同，Softmax 是可微的，并且保留了关于相对置信度（Relative Confidence）的信息。

### 函数形状对比

graph LR
    subgraph "Activation Functions"
        S["Sigmoid<br/>Range: (0,1)<br/>Saturates both ends"]
        T["Tanh<br/>Range: (-1,1)<br/>Zero-centered"]
        R["ReLU<br/>Range: [0,inf)<br/>Dead neurons"]
        G["GELU<br/>Range: ~(-0.17,inf)<br/>Smooth gating"]
    end
    S -->|"Vanishing gradient"| Problem["Deep networks<br/>don't train"]
    T -->|"Less severe but<br/>still vanishes"| Problem
    R -->|"Gradient = 1<br/>for x > 0"| Solution["Deep networks<br/>train fast"]
    G -->|"Smooth gradient<br/>everywhere"| Solution

### 梯度流对比

graph TD
    Input["Input Signal"] --> L1["Layer 1"]
    L1 --> L5["Layer 5"]
    L5 --> L10["Layer 10"]
    L10 --> Output["Output"]

    subgraph "Gradient at Layer 1"
        SigGrad["Sigmoid: ~0.000001"]
        TanhGrad["Tanh: ~0.001"]
        ReluGrad["ReLU: ~1.0"]
        GeluGrad["GELU: ~0.8"]
    end

### 何时使用何种激活函数

flowchart TD
    Start["What are you building?"] --> Hidden{"Hidden layers<br/>or output?"}

    Hidden -->|"Hidden layers"| Arch{"Architecture?"}
    Hidden -->|"Output layer"| Task{"Task type?"}

    Arch -->|"Transformer / NLP"| GELU["Use GELU"]
    Arch -->|"CNN / Vision"| ReLU["Use ReLU or Swish"]
    Arch -->|"RNN / LSTM"| Tanh["Use Tanh"]
    Arch -->|"Simple MLP"| ReLU2["Use ReLU"]

    Task -->|"Binary classification"| Sigmoid["Use Sigmoid"]
    Task -->|"Multi-class classification"| Softmax["Use Softmax"]
    Task -->|"Regression"| Linear["Use Linear (no activation)"]


## 构建

### 步骤 1：实现所有激活函数（activation function）及其导数（derivative）

每个函数接收一个浮点数（float）并返回一个浮点数。每个导数函数接收相同的输入并返回梯度（gradient）。

import math

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_act(x):
    return math.tanh(x)

def tanh_derivative(x):
    t = math.tanh(x)
    return 1 - t * t

def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def leaky_relu_derivative(x, alpha=0.01):
    return 1.0 if x > 0 else alpha

def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

def gelu_derivative(x):
    phi = 0.5 * (1 + math.erf(x / math.sqrt(2)))
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return phi + x * pdf

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

def softmax(xs):
    max_x = max(xs)
    exps = [math.exp(x - max_x) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]

### 步骤 2：可视化梯度消失区域

在 -5 到 5 范围内均匀选取 100 个点计算梯度。打印文本直方图（histogram），展示各激活函数的梯度接近于零的区间。

def gradient_scan(name, derivative_fn, start=-5, end=5, n=100):
    step = (end - start) / n
    near_zero = 0
    healthy = 0
    for i in range(n):
        x = start + i * step
        g = derivative_fn(x)
        if abs(g) < 0.01:
            near_zero += 1
        else:
            healthy += 1
    pct_dead = near_zero / n * 100
    print(f"{name:15s}: {healthy:3d} healthy, {near_zero:3d} near-zero ({pct_dead:.0f}% dead zone)")

gradient_scan("Sigmoid", sigmoid_derivative)
gradient_scan("Tanh", tanh_derivative)
gradient_scan("ReLU", relu_derivative)
gradient_scan("Leaky ReLU", leaky_relu_derivative)
gradient_scan("GELU", gelu_derivative)
gradient_scan("Swish", swish_derivative)

### 步骤 3：梯度消失（vanishing gradient）实验

分别使用 Sigmoid 与 ReLU，将信号前向传播（forward pass）通过 N 个层（layer）。测量激活值幅度（magnitude）的变化情况。

import random

def vanishing_gradient_experiment(activation_fn, name, n_layers=10, n_inputs=5):
    random.seed(42)
    values = [random.gauss(0, 1) for _ in range(n_inputs)]

    print(f"\n{name} through {n_layers} layers:")
    for layer in range(n_layers):
        weights = [random.gauss(0, 1) for _ in range(n_inputs)]
        z = sum(w * v for w, v in zip(weights, values))
        activated = activation_fn(z)
        magnitude = abs(activated)
        bar = "#" * int(magnitude * 20)
        print(f"  Layer {layer+1:2d}: magnitude = {magnitude:.6f} {bar}")
        values = [activated] * n_inputs

vanishing_gradient_experiment(sigmoid, "Sigmoid")
vanishing_gradient_experiment(relu, "ReLU")
vanishing_gradient_experiment(gelu, "GELU")

### 步骤 4：死亡神经元（dead neuron）检测器

构建一个 ReLU 网络，传入随机输入，统计从未被激活（fire）的神经元数量。

def dead_neuron_detector(n_inputs=5, hidden_size=20, n_samples=1000):
    random.seed(0)
    weights = [[random.gauss(0, 1) for _ in range(n_inputs)] for _ in range(hidden_size)]
    biases = [random.gauss(0, 1) for _ in range(hidden_size)]

    fire_counts = [0] * hidden_size

    for _ in range(n_samples):
        inputs = [random.gauss(0, 1) for _ in range(n_inputs)]
        for neuron_idx in range(hidden_size):
            z = sum(w * x for w, x in zip(weights[neuron_idx], inputs)) + biases[neuron_idx]
            if relu(z) > 0:
                fire_counts[neuron_idx] += 1

    dead = sum(1 for c in fire_counts if c == 0)
    rarely_fire = sum(1 for c in fire_counts if 0 < c < n_samples * 0.05)
    healthy = hidden_size - dead - rarely_fire

    print(f"\nDead Neuron Report ({hidden_size} neurons, {n_samples} samples):")
    print(f"  Dead (never fired):     {dead}")
    print(f"  Barely alive (<5%):     {rarely_fire}")
    print(f"  Healthy:                {healthy}")
    print(f"  Dead neuron rate:       {dead/hidden_size*100:.1f}%")

    for i, c in enumerate(fire_counts):
        status = "DEAD" if c == 0 else "WEAK" if c < n_samples * 0.05 else "OK"
        bar = "#" * (c * 40 // n_samples)
        print(f"  Neuron {i:2d}: {c:4d}/{n_samples} fires [{status:4s}] {bar}")

dead_neuron_detector()

### 步骤 5：训练对比 -- Sigmoid vs ReLU vs GELU

在圆形数据集（dataset）（圆内点标记为类别 1，圆外点标记为类别 0）上，使用三种不同的激活函数训练结构相同的双层网络（two-layer network）。对比它们的收敛速度（convergence speed）。

def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class ActivationNetwork:
    def __init__(self, activation_fn, activation_deriv, hidden_size=8, lr=0.1):
        random.seed(0)
        self.act = activation_fn
        self.act_d = activation_deriv
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
            self.h.append(self.act(z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        error = self.out - target
        d_out = error * self.out * (1 - self.out)

        for i in range(self.hidden_size):
            d_h = d_out * self.w2[i] * self.act_d(self.z1[i])
            self.w2[i] -= self.lr * d_out * self.h[i]
            for j in range(2):
                self.w1[i][j] -= self.lr * d_h * self.x[j]
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def train(self, data, epochs=200):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                self.backward(y)
                total_loss += (pred - y) ** 2
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append(avg_loss)
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses


data = make_circle_data()

configs = [
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("ReLU", relu, relu_derivative),
    ("GELU", gelu, gelu_derivative),
]

results = {}
for name, act_fn, act_d_fn in configs:
    print(f"\n=== Training with {name} ===")
    net = ActivationNetwork(act_fn, act_d_fn, hidden_size=8, lr=0.1)
    losses = net.train(data, epochs=200)
    results[name] = losses

print("\n=== Final Loss Comparison ===")
for name, losses in results.items():
    print(f"  {name:10s}: start={losses[0]:.4f} -> end={losses[-1]:.4f} (improvement: {(1 - losses[-1]/losses[0])*100:.1f}%)")


## 实际应用

PyTorch 提供了所有这些激活函数的函数式 (functional) 和模块式 (module) 两种形式：

import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(4, 10)

relu_out = F.relu(x)
gelu_out = F.gelu(x)
sigmoid_out = torch.sigmoid(x)
swish_out = F.silu(x)

logits = torch.randn(4, 5)
probs = F.softmax(logits, dim=1)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.GELU(),
    nn.Linear(64, 32),
    nn.GELU(),
    nn.Linear(32, 5),
)

Transformer 隐藏层：GELU。卷积神经网络 (CNN) 隐藏层：ReLU。分类任务输出层：softmax。回归任务输出层：无（线性 (linear)）。概率输出层：sigmoid。掌握这些即可。建议从这些默认配置起步，仅在具备充分依据时再进行调整。

循环神经网络 (RNN) 和长短期记忆网络 (LSTM) 使用 tanh 处理隐藏状态 (hidden state)，使用 sigmoid 处理门控 (gates)。但如果你如今从零开始构建模型，大概率已不再使用 RNN。若你的 ReLU 网络中出现神经元死亡 (dead neuron) 现象，请切换至 GELU。除非有特定理由，否则不建议使用 Leaky ReLU——GELU 不仅能解决神经元死亡问题，还能提供更优的梯度流 (gradient flow)。

## 交付物

本课程的产出包括：
- `outputs/prompt-activation-selector.md` -- 一个可复用的提示词 (prompt)，可帮助你为任意网络架构 (architecture) 选择合适的激活函数 (activation function)

## 练习

1. 实现参数化 ReLU (Parametric ReLU, PReLU)，其中负斜率参数 alpha 为可学习参数 (learnable parameter)。在圆形数据集 (circle dataset) 上进行训练，并与固定参数的 Leaky ReLU 进行对比。

2. 将梯度消失 (vanishing gradient) 实验的网络层数从 10 层改为 50 层。绘制 sigmoid、tanh、ReLU 和 GELU 在每一层的梯度幅值 (magnitude)。观察每种激活函数的信号在到达哪一层时实际上已衰减至零？

3. 实现指数线性单元 (Exponential Linear Unit, ELU)：当 x > 0 时，elu(x) = x；当 x <= 0 时，elu(x) = alpha * (e^x - 1)。在相同网络架构下，将其神经元死亡率 (dead neuron rate) 与 ReLU 进行对比。

4. 构建一个在训练期间运行的“梯度健康监控器 (gradient health monitor)”：在每个训练轮次 (epoch)，计算每一层的平均梯度幅值。当任意层的梯度低于 0.001 或超过 100 时，打印警告信息。

5. 修改训练对比实验，使用第 01 课中的 XOR 数据集 (XOR dataset) 替代圆形数据集。在 XOR 任务上，哪种激活函数收敛 (converges) 最快？为何其结果与圆形数据集不同？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 激活函数 (Activation function) | “非线性部分” | 应用于每个神经元输出的函数，用于打破线性关系，使网络能够学习非线性映射 |
| 梯度消失 (Vanishing gradient) | “深层网络中梯度会消失” | 当激活函数的导数小于 1 时，梯度在逐层传递过程中呈指数级衰减，导致浅层网络无法训练 |
| 梯度爆炸 (Exploding gradient) | “梯度会爆炸” | 当有效乘数大于 1 时，梯度在逐层传递过程中呈指数级增长，导致训练过程不稳定 |
| 死亡神经元 (Dead neuron) | “停止学习的神经元” | 输入始终为负的 ReLU 神经元，其输出和梯度均为零 |
| Sigmoid | “将值压缩到 0-1 之间” | 逻辑函数 1/(1+e^-x)，历史上具有重要意义，但在深层网络中会导致梯度消失 |
| ReLU | “将负值截断为零” | max(0, x) —— 通过保持梯度幅值，使深度学习变得切实可行的激活函数 |
| GELU | “Transformer (Transformer) 的激活函数” | 高斯误差线性单元 (Gaussian Error Linear Unit)，一种平滑的激活函数，根据输入为正的概率对其进行加权 |
| Swish/SiLU | “自门控 ReLU” | x * sigmoid(x)，通过自动化搜索发现，应用于 EfficientNet |
| Softmax | “将得分转换为概率” | 将一组未归一化得分 (logits) 向量归一化为概率分布，其中所有值均在 (0,1) 范围内且总和为 1 |
| Leaky ReLU | “不会死亡的 ReLU” | max(alpha*x, x)，其中 alpha 为较小值（如 0.01），通过允许微小的负梯度来防止神经元死亡 |
| 饱和区 (Saturation) | “Sigmoid 的平坦部分” | 激活函数导数趋近于零的区域，会阻断梯度流动 |
| Logit | “Softmax 之前的原始得分” | 应用 Softmax 或 Sigmoid 之前，网络最后一层的未归一化输出 |

## 延伸阅读

- Nair & Hinton，《Rectified Linear Units Improve Restricted Boltzmann Machines》（2010）—— 引入 ReLU 并使得深层网络训练成为可能的论文
- Hendrycks & Gimpel，《Gaussian Error Linear Units (GELUs)》（2016）—— 引入了后来成为 Transformer 默认配置的激活函数
- Ramachandran 等人，《Searching for Activation Functions》（2017）—— 利用自动化搜索发现 Swish，证明了激活函数的设计可以实现自动化
- Glorot & Bengio，《Understanding the difficulty of training deep feedforward neural networks》（2010）—— 诊断了梯度消失/爆炸问题并提出 Xavier 初始化 (Xavier initialization) 方法的论文
- Goodfellow, Bengio, Courville，《Deep Learning》第 6.3 章 (https://www.deeplearningbook.org/) —— 对隐藏单元和激活函数进行了严谨的论述