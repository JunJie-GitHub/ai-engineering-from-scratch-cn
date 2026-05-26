# 多层网络与前向传播

> 单个神经元只能画一条直线。将它们堆叠起来，你就能绘制任何图形。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 01（数学基础），课程 03.01（感知机）
**预计时长：** 约 90 分钟

## 学习目标

- 从零开始构建多层网络（Multi-Layer Network），使用 Layer 和 Network 类实现完整的前向传播（Forward Pass）
- 追踪网络每一层的矩阵维度，并识别形状不匹配（Shape Mismatch）问题
- 解释堆叠非线性激活函数（Nonlinear Activation Functions）如何使网络能够学习曲线决策边界（Curved Decision Boundaries）
- 使用 2-2-1 架构和手动调整的 Sigmoid 权重（Sigmoid Weights）解决异或（XOR）问题

## 核心问题

单个神经元本质上只是一个“画线器”。仅此而已。它只能在数据中划出一条直线。而人工智能中的每一个实际问题——无论是图像识别、语言理解，还是下围棋——都需要处理曲线。将神经元堆叠成层，正是获得曲线拟合能力的关键。

1969 年，明斯基（Minsky）和帕珀特（Papert）证明了这一局限性是致命的：单层网络根本无法学习异或（XOR）问题。不是“难以学习”，而是从数学上就不可能。XOR 的真值表将 [0,1] 和 [1,0] 归为一类，将 [0,0] 和 [1,1] 归为另一类。没有任何一条直线能将它们分开。

这一结论直接导致神经网络领域的研究资金中断了十余年。事后看来，解决方案其实显而易见：放弃单层结构，将神经元堆叠成多层。让第一层将输入空间切割并转化为新的特征，再让第二层将这些特征组合起来，做出任何单一直线都无法实现的决策。

这种堆叠结构就是多层网络。它是当今所有投入生产环境的深度学习模型的基础。而前向传播——即数据从输入端流经隐藏层最终到达输出端的过程——则是你在实现其他任何功能之前，必须首先构建的核心模块。

## 核心概念

### 层：输入层、隐藏层与输出层

多层网络包含三种类型的层：

**输入层（Input Layer）**——严格来说并不算一层。它仅用于存放原始数据。两个特征对应两个输入节点。此处不进行任何计算。

**隐藏层（Hidden Layer）**——实际进行计算的地方。每个神经元接收上一层的所有输出，应用权重（weight）和偏置（bias），然后将结果传入激活函数（activation function）。之所以称为“隐藏”，是因为你在训练数据中永远无法直接看到这些值。

**输出层（Output Layer）**——给出最终结果。对于二分类任务，使用一个带有 Sigmoid 函数的神经元；对于多分类任务，每个类别对应一个神经元。

graph LR
    subgraph Input["Input Layer"]
        x1["x1"]
        x2["x2"]
    end
    subgraph Hidden["Hidden Layer (3 neurons)"]
        h1["h1"]
        h2["h2"]
        h3["h3"]
    end
    subgraph Output["Output Layer"]
        y["y"]
    end
    x1 --> h1
    x1 --> h2
    x1 --> h3
    x2 --> h1
    x2 --> h2
    x2 --> h3
    h1 --> y
    h2 --> y
    h3 --> y

这是一个 2-3-1 网络。包含两个输入、三个隐藏神经元和一个输出。每条连接都带有一个权重。每个神经元（输入层除外）都带有一个偏置。

每一层都会生成一个数值向量，称为隐藏状态（hidden state）。对于文本，隐藏状态会增加维度——将单个词编码为 768 个数值以捕捉语义信息。对于图像，它们会降低维度——将数百万像素压缩为易于处理的表示形式。隐藏状态正是模型“学习”发生的地方。

### 神经元与激活函数

每个神经元执行三个步骤：

1. 将每个输入乘以其对应的权重
2. 对所有乘积求和并加上偏置
3. 将求和结果传入激活函数

目前，我们使用的激活函数是 Sigmoid：

sigmoid(z) = 1 / (1 + e^(-z))

Sigmoid 函数会将任意数值压缩到 (0, 1) 区间内。较大的正输入会趋近于 1，较大的负输入会趋近于 0，零则映射为 0.5。正是这种平滑曲线使得学习成为可能——与感知机（perceptron）的硬阶跃函数不同，Sigmoid 在任意点都存在梯度（gradient）。

### 前向传播：数据流动过程

前向传播（forward pass）将输入数据逐层推入网络，直至到达输出层。前向传播过程中不会发生任何学习。它纯粹是计算过程：相乘、相加、激活、循环往复。

graph TD
    X["Input: [x1, x2]"] --> WH["Multiply by Weight Matrix W1 (2x3)"]
    WH --> BH["Add Bias Vector b1 (3,)"]
    BH --> AH["Apply sigmoid to each element"]
    AH --> H["Hidden Output: [h1, h2, h3]"]
    H --> WO["Multiply by Weight Matrix W2 (3x1)"]
    WO --> BO["Add Bias Vector b2 (1,)"]
    BO --> AO["Apply sigmoid"]
    AO --> Y["Output: y"]

在每一层中，依次执行以下三个操作：

z = W * input + b       (linear transformation)
a = sigmoid(z)           (activation)

前一层的输出即为下一层的输入。这就是完整的前向传播过程。

### 矩阵维度

追踪维度是深度学习（deep learning）中最关键的调试技能。以下是该 2-3-1 网络的维度变化：

| 步骤 | 操作 | 维度 | 结果形状 |
|------|-----------|------------|-------------|
| 输入 | x | -- | (2,) |
| 隐藏层线性变换 | W1 * x + b1 | W1: (3, 2), b1: (3,) | (3,) |
| 隐藏层激活 | sigmoid(z1) | -- | (3,) |
| 输出层线性变换 | W2 * h + b2 | W2: (1, 3), b2: (1,) | (1,) |
| 输出层激活 | sigmoid(z2) | -- | (1,) |

规则如下：第 k 层的权重矩阵 W 的形状为 `(neurons_in_layer_k, neurons_in_layer_k_minus_1)`。行数对应当前层的神经元数量，列数对应上一层的神经元数量。如果形状无法对齐，说明代码存在错误。

### 万能近似定理

1989 年，George Cybenko 证明了一个惊人的结论：只要隐藏层包含足够多的神经元，单隐藏层神经网络就能以任意精度逼近任何连续函数。

这并不意味着单隐藏层总是最优选择。它仅说明该架构在理论上具备这种能力。在实践中，更深的网络（层数更多、每层神经元更少）相比浅而宽的网络，能够以更少的总参数量学习到相同的函数。这正是深度学习有效的原因。

直观理解：隐藏层中的每个神经元学习一个“凸起”或特征。只要将足够多的凸起放置在正确的位置，就能逼近任何平滑曲线。神经元越多，凸起越多，逼近效果越好。

graph LR
    subgraph FewNeurons["4 Hidden Neurons"]
        A["Rough approximation"]
    end
    subgraph MoreNeurons["16 Hidden Neurons"]
        B["Close approximation"]
    end
    subgraph ManyNeurons["64 Hidden Neurons"]
        C["Near-perfect fit"]
    end
    FewNeurons --> MoreNeurons --> ManyNeurons

### 可组合性

神经网络具有可组合性。你可以将它们堆叠、串联或并行运行。Whisper 模型使用编码器（encoder）网络处理音频，并使用独立的解码器（decoder）网络生成文本。现代大语言模型（LLM）仅包含解码器。BERT 仅包含编码器。T5 则是编码器-解码器架构。架构的选择决定了模型的能力边界。

## 构建

纯 Python 实现。不使用 NumPy。所有矩阵运算均从零手写。

### 步骤 1：Sigmoid 激活函数 (Sigmoid Activation)

import math

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))

将输入值限制 (Clamp) 在 [-500, 500] 区间内可防止数值溢出。`math.exp(500)` 的值很大但仍是有限的，而 `math.exp(1000)` 则会返回无穷大。

### 步骤 2：Layer 类 (Layer Class)

深度学习 (Deep Learning) 中最核心的运算是矩阵乘法 (Matrix Multiplication)。无论是每一层、每一个注意力头 (Attention Head)，还是每一次前向传播 (Forward Pass)，底层无一例外都是矩阵乘法。一个线性层 (Linear Layer) 接收输入向量，将其与权重矩阵 (Weight Matrix) 相乘，再加上偏置向量 (Bias Vector)：y = Wx + b。仅这一个公式，就占据了神经网络中 90% 的计算量。

一个 Layer 包含一个权重矩阵和一个偏置向量。它的 `forward` 方法接收输入向量，并返回经过激活后的输出。

class Layer:
    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        if weights is not None:
            self.weights = weights
        else:
            import random
            self.weights = [
                [random.uniform(-1, 1) for _ in range(n_inputs)]
                for _ in range(n_neurons)
            ]
        if biases is not None:
            self.biases = biases
        else:
            self.biases = [0.0] * n_neurons

    def forward(self, inputs):
        self.last_input = inputs
        self.last_output = []
        for neuron_idx in range(len(self.weights)):
            z = sum(
                w * x for w, x in zip(self.weights[neuron_idx], inputs)
            )
            z += self.biases[neuron_idx]
            self.last_output.append(sigmoid(z))
        return self.last_output

权重矩阵的形状为 `(n_neurons, n_inputs)`。矩阵的每一行代表一个神经元对所有输入的权重。`forward` 方法会遍历所有神经元，计算加权和并加上偏置，应用 Sigmoid 函数，最后收集结果。

### 步骤 3：Network 类 (Network Class)

一个 Network 本质上是一个层的列表。前向传播将它们串联起来：第 k 层的输出将作为第 k+1 层的输入。

class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

这就是完整的前向传播过程。仅需四行逻辑代码。数据输入后，流经每一层，最终从另一端输出。

### 步骤 4：使用手动调优权重解决 XOR 问题

在第一课中，我们通过组合 OR、NAND 和 AND 感知机 (Perceptron) 解决了 XOR 问题。现在，我们将使用自定义的 `Layer` 和 `Network` 类实现相同的功能。采用 2-2-1 架构：两个输入节点、两个隐藏层神经元、一个输出节点。

hidden = Layer(
    n_inputs=2,
    n_neurons=2,
    weights=[[20.0, 20.0], [-20.0, -20.0]],
    biases=[-10.0, 30.0],
)

output = Layer(
    n_inputs=2,
    n_neurons=1,
    weights=[[20.0, 20.0]],
    biases=[-30.0],
)

xor_net = Network([hidden, output])

xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

for inputs, expected in xor_data:
    result = xor_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    print(f"  {inputs} -> {result[0]:.6f} (rounded: {predicted}, expected: {expected})")

较大的权重值（20 和 -20）会使 Sigmoid 函数的行为近似于阶跃函数 (Step Function)。第一个隐藏神经元近似实现 OR 逻辑，第二个近似实现 NAND 逻辑。输出神经元将它们组合成 AND 逻辑，从而等效于 XOR。

### 步骤 5：圆形区域分类

一个更具挑战性的问题：将二维平面上的点分类为位于原点为中心、半径为 0.5 的圆内或圆外。这需要一条曲线决策边界 (Decision Boundary)，而单个感知机是无法实现的。

import random
import math

random.seed(42)

data = []
for _ in range(200):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    label = 1 if (x * x + y * y) < 0.25 else 0
    data.append(([x, y], label))

circle_net = Network([
    Layer(n_inputs=2, n_neurons=8),
    Layer(n_inputs=8, n_neurons=1),
])

在随机权重下，网络的分类效果会很差。但前向传播依然可以正常运行。这正是关键所在——前向传播仅仅是计算过程。而学习正确的权重则依赖于反向传播 (Backpropagation)，这将在第三课中讲解。

correct = 0
for inputs, expected in data:
    result = circle_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    if predicted == expected:
        correct += 1

print(f"Accuracy with random weights: {correct}/{len(data)} ({100*correct/len(data):.1f}%)")

随机权重会导致准确率极低，通常甚至不如直接猜测多数类。经过训练（第三课）后，同样的架构配合 8 个隐藏神经元将能够绘制出一条曲线边界，从而将圆内与圆外的点清晰分开。

## 使用它

PyTorch 仅需四行代码即可完成上述所有操作：

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Sigmoid(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
output = model(x)
print(output)

`nn.Linear(2, 8)` 对应你之前实现的 `Layer` 类：包含形状为 (8, 2) 的权重矩阵（weight matrix）和形状为 (8,) 的偏置向量（bias vector）。`nn.Sigmoid()` 对应逐元素（element-wise）应用的 Sigmoid 函数（Sigmoid function）。`nn.Sequential` 则对应你的 `Network` 类：按顺序将各层（layer）串联起来。

两者的区别在于速度与规模。PyTorch 可在 GPU（图形处理器）上运行，能够处理包含数百万样本的批次（batch），并自动计算用于反向传播（backpropagation）的梯度（gradient）。但其前向传播（forward pass）的逻辑与你刚刚从零构建的代码完全一致。

## 交付使用

本课时将生成一个可复用的提示词（prompt），用于设计网络架构（network architecture）：

- `outputs/prompt-network-architect.md`

当你需要针对特定问题决定网络层数、每层神经元（neuron）数量以及使用何种激活函数（activation function）时，即可使用该提示词。

## 练习

1. 构建一个 2-4-2-1 网络（包含两个隐藏层（hidden layer）），并使用随机权重在 XOR 数据上运行前向传播。打印中间隐藏层的输出，观察特征表示（representation）在每一层是如何变换的。

2. 将圆形分类器中隐藏层的大小从 8 改为 2，再改为 32。每次均使用随机权重运行前向传播。隐藏层神经元数量的改变是否会影响输出范围或分布？为什么？

3. 在 `Network` 类中实现一个 `count_parameters` 方法，用于返回可训练权重和偏置的总数。在一个 784-256-128-10 网络（经典的 MNIST 架构）上测试该方法。它包含多少个参数（parameter）？

4. 为一个 3-4-4-2 网络构建前向传播。向其输入 RGB 颜色值（已归一化至 0-1 范围），并观察两个输出结果。这是一个用于二分类简单颜色分类器的架构。

5. 将 Sigmoid 函数替换为“泄漏阶跃”（leaky step）函数：若 z < 0 则返回 0.01 * z，否则返回 1.0。使用第 4 步中手动调优的相同权重，在 XOR 数据上运行前向传播。它还能正常工作吗？为什么平滑的 Sigmoid 函数比硬截断（hard cutoff）更受青睐？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 前向传播 (Forward pass) | “运行模型” | 将输入数据逐层传递——乘以权重、加上偏置、进行激活——最终生成输出结果 |
| 隐藏层 (Hidden layer) | “中间部分” | 位于输入层与输出层之间、其数值在原始数据中无法直接观测到的任意网络层 |
| 多层网络 (Multi-layer network) | “深度神经网络” | 神经元层按顺序堆叠而成，每一层的输出作为下一层的输入 |
| 激活函数 (Activation function) | “非线性部分” | 在线性变换之后应用的函数，用于为决策边界引入非线性曲线 |
| Sigmoid 函数 (Sigmoid) | “S型曲线” | sigma(z) = 1/(1+e^(-z))，将任意实数压缩至 (0,1) 区间，处处平滑且可导 |
| 权重矩阵 (Weight matrix) | “参数” | 形状为 (当前层神经元数, 上一层神经元数) 的矩阵 W，包含可学习的连接强度 |
| 偏置向量 (Bias vector) | “偏移量” | 在矩阵乘法之后相加的向量，使得即使所有输入为零时神经元也能被激活 |
| 通用近似定理 (Universal approximation) | “神经网络能学习任何东西” | 只要隐藏层包含足够多的神经元，单个隐藏层即可近似任意连续函数——但“足够多”可能意味着数十亿个 |
| 线性变换 (Linear transformation) | “矩阵乘法步骤” | z = W * x + b，激活前的计算步骤，负责将输入映射到新的特征空间 |
| 决策边界 (Decision boundary) | “分类器切换的地方” | 输入空间中的一个曲面，网络输出在此处跨越分类阈值 |

## 进一步阅读

- Michael Nielsen，《Neural Networks and Deep Learning》第 1-2 章 (http://neuralnetworksanddeeplearning.com/) —— 对前向传播和网络结构最清晰的免费讲解，并配有交互式可视化演示
- Cybenko，《Approximation by Superpositions of a Sigmoidal Function》（1989）—— 通用近似定理的原始论文，行文出乎意料地通俗易懂
- 3Blue1Brown，《But what is a neural network?》（https://www.youtube.com/watch?v=aircAruvnKk）—— 20 分钟的可视化导览，深入讲解网络层、权重与前向传播，帮助构建正确的思维模型
- Goodfellow, Bengio, Courville，《Deep Learning》第 6 章 (https://www.deeplearningbook.org/) —— 多层网络的标准参考教材，可免费在线阅读