# 从零实现反向传播

> 反向传播（Backpropagation）是让模型学习成为可能的算法。没有它，神经网络不过是昂贵的随机数生成器。

**类型：** 实战
**语言：** Python
**前置课程：** 第 03.02 课（多层网络）
**预计耗时：** 约 120 分钟

## 学习目标

- 实现一个基于 Value 的自动微分（autograd）引擎，该引擎能够构建计算图（computational graph）并通过拓扑排序（topological sort）计算梯度
- 利用链式法则（chain rule）推导加法、乘法和 Sigmoid 函数的反向计算（backward pass）过程
- 仅使用你从零实现的反向传播引擎，在 XOR 问题和圆形分类任务上训练多层网络
- 识别深层 Sigmoid 网络中的梯度消失问题（vanishing gradient problem），并解释梯度为何会呈指数级衰减

## 问题背景

你的网络包含一个隐藏层，具有 768 个输入和 3072 个输出，这意味着共有 2,359,296 个权重。当网络做出错误预测时，究竟是哪些权重导致了误差？如果逐个测试每个权重，需要进行 230 万次前向传播（forward pass）。而反向传播只需一次反向计算就能计算出全部 230 万个梯度。这不仅仅是一种优化手段，更是“可训练”与“不可能”之间的本质区别。

朴素的做法是：取出一个权重，对其进行微小扰动，再次运行前向传播，观察损失（loss）是上升还是下降。这样就能得到该权重的梯度。接着，对网络中的每一个权重重复此操作。再乘以数千个训练步数和数百万个数据点，你将需要地质年代般漫长的时间才能训练出任何有用的模型。

反向传播正是为了解决这一难题而生。只需一次前向传播和一次反向计算，即可计算出所有梯度。其核心技巧在于将微积分中的链式法则系统地应用于计算图。正是这一算法让深度学习走向实用。没有它，我们至今仍会受困于简单的玩具级问题。

## 核心概念

### 链式法则（Chain Rule）在网络中的应用

你在第一阶段第五课中已经接触过链式法则。快速回顾一下：如果 y = f(g(x))，那么 dy/dx = f'(g(x)) * g'(x)。你需要沿着函数链将导数相乘。

在神经网络（Neural Network）中，“链”指的是从输入到损失（Loss）的一系列操作序列。每一层都会应用权重（Weights）、添加偏置（Biases），并通过激活函数（Activation Function）。损失函数会将最终输出与目标值进行比较。反向传播（Backpropagation）则沿着这条链逆向追溯，计算每个操作对误差的贡献程度。

### 计算图（Computational Graphs）

每次前向传播（Forward Pass）都会构建一个图。图中的每个节点代表一个操作（如乘法、加法、Sigmoid 函数）。每条边负责向前传递数值，并向后传递梯度（Gradient）。

graph LR
    x["x"] --> mul["*"]
    w["w"] --> mul
    mul -- "z1 = w*x" --> add["+"]
    b["b"] --> add
    add -- "z2 = z1 + b" --> sig["sigmoid"]
    sig -- "a = sigmoid(z2)" --> loss["Loss"]
    y["target"] --> loss

前向传播：数值从左向右流动。x 和 w 相乘得到 z1 = w*x。加上 b 得到 z2。经过 Sigmoid 函数得到激活值 a。最后使用损失函数将 a 与目标值 y 进行比较。

反向传播：梯度从右向左流动。从 dL/da（损失随激活值的变化率）开始。乘以 da/dz2（Sigmoid 函数的导数），得到 dL/dz2。将其拆分为 dL/db（由于 z2 = z1 + b，因此等于 dL/dz2）和 dL/dz1。接着计算 dL/dw = dL/dz1 * x 以及 dL/dx = dL/dz1 * w。

在反向传播过程中，图中的每个节点只负责一项任务：接收来自上游的梯度，乘以自身的局部导数（Local Derivative），然后将其向下游传递。

### 前向传播与反向传播

graph TB
    subgraph Forward["Forward Pass"]
        direction LR
        f1["Input x"] --> f2["z = Wx + b"]
        f2 --> f3["a = sigmoid(z)"]
        f3 --> f4["Loss = (a - y)^2"]
    end
    subgraph Backward["Backward Pass"]
        direction RL
        b4["dL/dL = 1"] --> b3["dL/da = 2(a-y)"]
        b3 --> b2["dL/dz = dL/da * a(1-a)"]
        b2 --> b1["dL/dW = dL/dz * x\ndL/db = dL/dz"]
    end
    Forward --> Backward

前向传播会保存所有中间值：z、a 以及每一层的输入。反向传播需要这些已保存的值来计算梯度。这正是反向传播核心的“内存-计算权衡”（Memory-Computation Tradeoff）。你通过消耗内存（存储激活值）来换取计算速度（只需一次遍历，而非数百万次）。

### 网络中的梯度流动

对于一个三层网络，梯度会逐层链式传递：

graph RL
    L["Loss"] -- "dL/da3" --> L3["Layer 3\na3 = sigmoid(z3)"]
    L3 -- "dL/dz3 = dL/da3 * sigmoid'(z3)" --> L2["Layer 2\na2 = sigmoid(z2)"]
    L2 -- "dL/dz2 = dL/da2 * sigmoid'(z2)" --> L1["Layer 1\na1 = sigmoid(z1)"]
    L1 -- "dL/dz1 = dL/da1 * sigmoid'(z1)" --> I["Input"]

在每一层，梯度都会乘以 Sigmoid 函数的导数。该导数为 a * (1 - a)，其最大值仅为 0.25（当 a = 0.5 时）。经过三层网络后，梯度最多被乘以 0.25^3 = 0.0156。若经过十层网络：0.25^10 = 0.000001。

### 梯度消失（Vanishing Gradients）

这就是梯度消失问题。Sigmoid 函数将其输出压缩在 0 到 1 之间，其导数始终小于 0.25。当堆叠足够多的 Sigmoid 层时，梯度会衰减至近乎为零。由于网络浅层接收到的梯度接近于零，它们几乎无法进行有效学习。

sigmoid(z):     Output range [0, 1]
sigmoid'(z):    Max value 0.25 (at z = 0)

After 5 layers:   gradient * 0.25^5 = 0.001x original
After 10 layers:  gradient * 0.25^10 = 0.000001x original

这就是深层 Sigmoid 网络几乎无法训练的原因。解决方案——ReLU（Rectified Linear Unit）及其变体——将在第四课中详细讲解。目前你只需明白：反向传播机制本身是完美无误的，问题出在它所要穿越的激活函数特性上。

### 推导两层网络的梯度

以下是针对包含输入 x、使用 Sigmoid 的隐藏层、使用 Sigmoid 的输出层以及均方误差（Mean Squared Error, MSE）损失的网络的具体数学推导。

前向传播：
z1 = W1 * x + b1
a1 = sigmoid(z1)
z2 = W2 * a1 + b2
a2 = sigmoid(z2)
L = (a2 - y)^2

反向传播（逐步应用链式法则）：
dL/da2 = 2(a2 - y)
da2/dz2 = a2 * (1 - a2)
dL/dz2 = dL/da2 * da2/dz2 = 2(a2 - y) * a2 * (1 - a2)

dL/dW2 = dL/dz2 * a1
dL/db2 = dL/dz2

dL/da1 = dL/dz2 * W2
da1/dz1 = a1 * (1 - a1)
dL/dz1 = dL/da1 * da1/dz1

dL/dW1 = dL/dz1 * x
dL/db1 = dL/dz1

每一个梯度都是从损失函数开始，沿路径回溯的局部导数的乘积。反向传播的本质正是如此。

## 构建

### 步骤 1：Value 节点

计算过程中的每个数值都会转换为一个 `Value`。它负责存储数据、梯度 (gradient) 以及自身的创建方式（以便后续能够反向计算梯度）。

class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._children = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

此时梯度尚未初始化（为 0.0）。反向传播函数也尚未定义（空操作）。`_children` 属性用于记录生成当前节点的上游 `Value`，以便后续对计算图进行拓扑排序 (topological sort)。

### 步骤 2：带反向传播函数的运算操作

每个运算操作都会创建一个新的 `Value`，并定义梯度如何通过该节点反向流动。

def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += out.grad
        other.grad += out.grad

    out._backward = _backward
    return out

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out

对于加法运算：d(a+b)/da = 1，d(a+b)/db = 1。因此，两个输入节点会直接接收输出节点的梯度。

对于乘法运算：d(a*b)/da = b，d(a*b)/db = a。每个输入节点接收的梯度为另一个输入节点的值乘以输出节点的梯度。

`+=` 累加操作至关重要。一个 `Value` 可能会被多个运算复用，其最终梯度是所有路径传递回来的梯度之和。

### 步骤 3：Sigmoid 与损失函数

import math

def sigmoid(self):
    x = self.data
    x = max(-500, min(500, x))
    s = 1.0 / (1.0 + math.exp(-x))
    out = Value(s, (self,), 'sigmoid')

    def _backward():
        self.grad += (s * (1 - s)) * out.grad

    out._backward = _backward
    return out

Sigmoid 函数的导数为：sigmoid(x) * (1 - sigmoid(x))。我们在前向传播 (forward pass) 中已经计算出了 sigmoid(x) = s，直接复用即可，无需额外计算。

def mse_loss(predicted, target):
    diff = predicted + Value(-target)
    return diff * diff

单个输出的均方误差 (Mean Squared Error, MSE) 为：(predicted - target)^2。我们将减法操作表示为加上一个取负值的 `Value`。

### 步骤 4：反向传播 (Backward Pass)

拓扑排序确保我们按正确的顺序处理节点——在通过某个节点传播梯度之前，该节点接收到的梯度已经完全累加完毕。

def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)

    build_topo(self)
    self.grad = 1.0
    for v in reversed(topo):
        v._backward()

从损失节点开始（梯度初始化为 1.0，因为 dL/dL = 1）。沿着排序后的计算图反向遍历。每个节点的 `_backward` 方法会将梯度推送到其子节点（上游节点）。

### 步骤 5：Layer 与 Network

import random

class Neuron:
    def __init__(self, n_inputs):
        scale = (2.0 / n_inputs) ** 0.5
        self.weights = [Value(random.uniform(-scale, scale)) for _ in range(n_inputs)]
        self.bias = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        return act.sigmoid()

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params


class Network:
    def __init__(self, sizes):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Layer(sizes[i], sizes[i + 1]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            if not isinstance(x, list):
                x = [x]
        return x[0] if len(x) == 1 else x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

`Neuron` 接收输入，计算加权和加上偏置 (bias)，然后应用 Sigmoid 激活函数。权重初始化采用 `sqrt(2/n_inputs)` 进行缩放，以防止在较深的网络中 Sigmoid 函数进入饱和区 (saturation)。`Layer` 是 `Neuron` 的列表，而 `Network` 则是 `Layer` 的列表。`parameters()` 方法会收集所有可学习的 `Value`，以便后续进行参数更新。

### 步骤 6：在 XOR 数据集上训练

random.seed(42)
net = Network([2, 4, 1])

xor_data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

learning_rate = 1.0

for epoch in range(1000):
    total_loss = Value(0.0)
    for inputs, target in xor_data:
        x = [Value(i) for i in inputs]
        pred = net(x)
        loss = mse_loss(pred, target)
        total_loss = total_loss + loss

    net.zero_grad()
    total_loss.backward()

    for p in net.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {total_loss.data:.6f}")

print("\nXOR Results:")
for inputs, target in xor_data:
    x = [Value(i) for i in inputs]
    pred = net(x)
    print(f"  {inputs} -> {pred.data:.4f} (expected {target})")

观察损失值逐渐下降的过程。从最初的随机预测到最终输出正确的 XOR 结果，完全由反向传播算法计算梯度并引导权重向正确方向调整所驱动。

### 步骤 7：圆形分类任务

在第 02 课中，你曾手动调整权重来完成圆形分类。现在，让网络自行学习这些权重。

random.seed(7)

def generate_circle_data(n=100):
    data = []
    for _ in range(n):
        x1 = random.uniform(-1.5, 1.5)
        x2 = random.uniform(-1.5, 1.5)
        label = 1.0 if x1 * x1 + x2 * x2 < 1.0 else 0.0
        data.append(([x1, x2], label))
    return data

circle_data = generate_circle_data(80)

circle_net = Network([2, 8, 1])
learning_rate = 0.5

for epoch in range(2000):
    random.shuffle(circle_data)
    total_loss_val = 0.0
    for inputs, target in circle_data:
        x = [Value(i) for i in inputs]
        pred = circle_net(x)
        loss = mse_loss(pred, target)
        circle_net.zero_grad()
        loss.backward()
        for p in circle_net.parameters():
            p.data -= learning_rate * p.grad
        total_loss_val += loss.data

    if epoch % 200 == 0:
        correct = 0
        for inputs, target in circle_data:
            x = [Value(i) for i in inputs]
            pred = circle_net(x)
            predicted_class = 1.0 if pred.data > 0.5 else 0.0
            if predicted_class == target:
                correct += 1
        accuracy = correct / len(circle_data) * 100
        print(f"Epoch {epoch:4d} | Loss: {total_loss_val:.4f} | Accuracy: {accuracy:.1f}%")

此处我们使用在线随机梯度下降 (Online Stochastic Gradient Descent, SGD)——在每个样本处理后立即更新权重，而不是累积整个批次 (batch) 的梯度。这能更快地打破对称性，并避免在整个损失曲面 (loss landscape) 上导致 Sigmoid 饱和。每个训练周期 (epoch) 打乱数据顺序，可防止网络死记硬背样本顺序。

无需手动调参。网络能够自行发现圆形的决策边界 (decision boundary)。这正是反向传播的强大之处：你只需定义网络架构、损失函数和数据，算法便会自动求解出最优权重。

## 上手使用

PyTorch 仅需几行代码即可完成上述所有操作。核心思想完全一致——自动微分（autograd）在前向传播（forward pass）过程中构建计算图（computational graph），并沿反向路径追踪以计算梯度（gradients）。

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid(),
)
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
criterion = nn.MSELoss()

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

for epoch in range(1000):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("PyTorch XOR Results:")
with torch.no_grad():
    for i in range(4):
        pred = model(X[i])
        print(f"  {X[i].tolist()} -> {pred.item():.4f} (expected {y[i].item()})")

`loss.backward()` 对应你代码中的 `total_loss.backward()`。`optimizer.step()` 对应你手动实现的 `p.data -= lr * p.grad`。`optimizer.zero_grad()` 对应你的 `net.zero_grad()`。算法相同，只是采用了工业级强度的实现。PyTorch 负责处理 GPU 加速、混合精度（mixed precision）、梯度检查点（gradient checkpointing）以及数百种层类型。但反向传播（backward pass）本质上仍是将相同的链式法则（chain rule）应用于相同的计算图。

训练（training）过程依次执行前向传播、反向传播，随后更新权重（weights）。推理（inference）则仅执行前向传播。不计算梯度，也不更新参数。这一区别至关重要，因为生产环境（production）中实际运行的正是推理过程。当你调用 Claude 或 GPT 等 API 时，你实际上是在进行推理——你的提示词（prompt）在网络中向前流动，最终从另一端输出词元（tokens）。模型权重不会发生任何改变。理解反向传播（backpropagation）之所以重要，正是因为它塑造了该网络中的每一个权重。

## 交付成果

本课程的产出物包括：
- `outputs/prompt-gradient-debugger.md` —— 一个可复用的提示词，用于诊断任意神经网络中的梯度问题（如梯度消失 vanishing、梯度爆炸 exploding、非数值 NaN）

## 练习

1. 为 `Value` 类添加 `__sub__` 方法（实现 `a - b = a + (-1 * b)`）。接着实现 `__neg__` 方法。通过手动计算简单表达式（如 `(a - b)^2`）的结果进行对比，验证梯度（gradient）是否正确。

2. 为 `Value` 添加 `relu` 方法（输出 `max(0, x)`，当 `x > 0` 时导数为 1，否则为 0）。在隐藏层（hidden layer）中用 `relu` 替换 `sigmoid`，并再次在 XOR 任务上进行训练。对比收敛速度。你应该会观察到训练速度更快——这为第 04 课的内容做了铺垫。

3. 在 `Value` 上实现用于整数次幂的 `__pow__` 方法。使用它来替换 `mse_loss`，改用标准的 `(predicted - target) ** 2` 表达式。验证其梯度是否与原始实现一致。

4. 在训练循环（training loop）中加入梯度裁剪（gradient clipping）：调用 `backward()` 后，将所有梯度限制在 `[-1, 1]` 范围内。训练一个更深的网络（4 层及以上且使用 `sigmoid`），并对比使用与不使用裁剪时的损失曲线（loss curve）。这是你应对梯度爆炸（exploding gradient）的第一道防线。

5. 构建一个可视化：在 XOR 任务上完成训练后，打印网络中每个参数的梯度。找出梯度最小的层。这将演示你在“概念”部分读到的梯度消失（vanishing gradient）问题。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 反向传播（Backpropagation） | “网络在学习” | 一种算法，通过在计算图（computational graph）中反向应用链式法则（chain rule），为每个权重计算 dL/dw |
| 计算图（Computational graph） | “网络结构” | 一种有向无环图，其中节点表示运算操作，边负责传递数值（前向）和梯度（反向） |
| 链式法则（Chain rule） | “导数相乘” | 若 y = f(g(x))，则 dy/dx = f'(g(x)) * g'(x) —— 反向传播的数学基础 |
| 梯度（Gradient） | “最陡峭上升的方向” | 损失函数相对于某个参数的偏导数 —— 指示如何调整该参数以降低损失 |
| 梯度消失（Vanishing gradient） | “深层网络学不到东西” | 梯度在穿过具有饱和激活函数（如 sigmoid）的层时呈指数级衰减 |
| 前向传播（Forward pass） | “运行网络” | 通过依次应用每一层的操作并存储中间值，从输入计算出输出 |
| 反向传播（Backward pass） | “计算梯度” | 逆向遍历计算图，利用链式法则在每个节点处累积梯度 |
| 学习率（Learning rate） | “学习速度有多快” | 一个标量，用于控制更新权重时的步长：w_new = w_old - lr * gradient |
| 拓扑排序（Topological sort） | “正确的顺序” | 图中节点的一种排列方式，确保每个节点出现在其依赖的所有节点之后 —— 保证梯度在传播前已完全累积 |
| 自动微分（Autograd） | “自动求导” | 一种在前向计算过程中构建计算图并自动计算梯度的系统 —— PyTorch 引擎的核心机制 |

## 延伸阅读

- Rumelhart, Hinton & Williams, "Learning representations by back-propagating errors" (1986) —— 该论文使反向传播（backpropagation）成为主流，并使得多层网络训练（multi-layer network training）成为可能。
- 3Blue1Brown, "Neural Networks" 系列 (https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) —— 对反向传播及网络中梯度流（gradient flow）最出色的可视化讲解。