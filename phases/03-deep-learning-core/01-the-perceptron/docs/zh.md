# 感知机 (Perceptron)

> 感知机是神经网络的“原子”。将其拆解开来，你会发现权重（weights）、偏置（bias）以及一个决策机制。

**类型：** 构建
**语言：** Python
**前置要求：** 第一阶段（线性代数直觉）
**预计耗时：** 约 60 分钟

## 学习目标

- 使用 Python 从零实现感知机，涵盖权重更新规则（weight update rule）与阶跃激活函数（step activation function）
- 解释为何单个感知机仅能解决线性可分问题（linearly separable problems），并演示其在异或（XOR）问题上的失败案例
- 通过组合或门（OR）、与非门（NAND）和与门（AND）构建多层感知机（multi-layer perceptron），以解决 XOR 问题
- 利用 Sigmoid 激活函数（sigmoid activation）与反向传播（backpropagation）训练一个两层网络，使其自动学习 XOR 逻辑

## 问题背景

你已经了解了向量（vectors）和点积（dot products）。你也知道矩阵能够将输入转换为输出。但机器究竟是如何*学习*该使用哪种变换的呢？

感知机正是为了解答这个问题而生。它是最基础的学习机器：接收输入，乘以权重，加上偏置，然后做出二分类决策。接着进行调整。仅此而已。迄今为止构建的所有神经网络，本质上都是将这一思想层层堆叠而成。

理解感知机，就意味着理解代码中“学习”的真正含义：不断调整数值，直到输出结果与实际情况相符。

## 核心概念

### 一个神经元，一个决策

感知机（Perceptron）接收 n 个输入，将每个输入乘以对应的权重（Weight），求和后加上偏置（Bias），最后将结果传入激活函数（Activation Function）。

graph LR
    x1["x1"] -- "w1" --> sum["Σ(wi*xi) + b"]
    x2["x2"] -- "w2" --> sum
    x3["x3"] -- "w3" --> sum
    bias["bias"] --> sum
    sum --> step["step(z)"]
    step --> out["output (0 or 1)"]

阶跃函数（Step Function）的处理方式非常直接：如果加权和加上偏置的结果大于等于 0，则输出 1；否则输出 0。

step(z) = 1  if z >= 0
           0  if z < 0

这是一个线性分类器（Linear Classifier）。权重和偏置共同定义了一条直线（或在更高维度中为超平面（Hyperplane）），将输入空间划分为两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制出一条直线：

  x2
  ┤
  │  Class 1        /
  │    (0)          /
  │                /
  │               / w1·x1 + w2·x2 + b = 0
  │              /
  │             /     Class 2
  │            /        (1)
  ┼───────────/──────────── x1

直线一侧的所有输入输出为 0，另一侧的所有输入输出为 1。训练过程会不断调整这条直线的位置，直到它能正确地将不同类别的样本分开。

### 学习规则

感知机学习规则非常简单：

For each training example (x, y_true):
    y_pred = predict(x)
    error = y_true - y_pred

    For each weight:
        w_i = w_i + learning_rate * error * x_i
    bias = bias + learning_rate * error

如果预测正确，误差为 0，参数保持不变。如果预测为 0 但实际应为 1，权重会增加；如果预测为 1 但实际应为 0，权重会减小。学习率（Learning Rate）控制着每次调整的幅度。

### 异或（XOR）问题

感知机的局限性在此显现。观察以下逻辑门：

AND gate:           OR gate:            XOR gate:
x1  x2  out         x1  x2  out         x1  x2  out
0   0   0           0   0   0           0   0   0
0   1   0           0   1   1           0   1   1
1   0   0           1   0   1           1   0   1
1   1   1           1   1   1           1   1   0

与门（AND）和或门（OR）是线性可分（Linearly Separable）的：你可以画一条直线将 0 和 1 分开。但异或门（XOR）不是。没有任何一条直线能够将 [0,1] 和 [1,0] 与 [0,0] 和 [1,1] 分开。

AND (separable):        XOR (not separable):

  x2                      x2
  1 ┤  0     1            1 ┤  1     0
    │     /                 │
  0 ┤  0 / 0              0 ┤  0     1
    ┼──/──────── x1         ┼──────────── x1
       line works!          no single line works!

这是一个根本性的限制。单个感知机只能解决线性可分问题。明斯基（Minsky）和帕珀特（Papert）在 1969 年证明了这一点，这导致神经网络研究在随后的十年里几乎陷入停滞。

解决方法：将感知机堆叠成层。多层感知机（Multi-Layer Perceptron）可以通过将两个线性决策组合成一个非线性决策来解决异或问题。

## 动手实现

### 步骤 1：感知机 (Perceptron) 类

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >= 0 else 0

    def train(self, training_data, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return
        print(f"Did not converge after {epochs} epochs")

### 步骤 2：在逻辑门 (Logic Gates) 上进行训练

and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

or_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

not_data = [
    ([0], 1),
    ([1], 0),
]

print("=== AND Gate ===")
p_and = Perceptron(2)
p_and.train(and_data)
for inputs, _ in and_data:
    print(f"  {inputs} -> {p_and.predict(inputs)}")

print("\n=== OR Gate ===")
p_or = Perceptron(2)
p_or.train(or_data)
for inputs, _ in or_data:
    print(f"  {inputs} -> {p_or.predict(inputs)}")

print("\n=== NOT Gate ===")
p_not = Perceptron(1)
p_not.train(not_data)
for inputs, _ in not_data:
    print(f"  {inputs} -> {p_not.predict(inputs)}")

### 步骤 3：观察异或 (XOR) 的失败

xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

print("\n=== XOR Gate (single perceptron) ===")
p_xor = Perceptron(2)
p_xor.train(xor_data, epochs=1000)
for inputs, expected in xor_data:
    result = p_xor.predict(inputs)
    status = "OK" if result == expected else "WRONG"
    print(f"  {inputs} -> {result} (expected {expected}) {status}")

它永远无法收敛 (Converge)。这有力地证明了单个感知机无法学习异或逻辑。

### 步骤 4：使用双层网络解决异或问题

核心思路：异或逻辑等价于 `(x1 OR x2) AND NOT (x1 AND x2)`。我们将三个感知机组合起来：

graph LR
    x1["x1"] --> OR["OR neuron"]
    x1 --> NAND["NAND neuron"]
    x2["x2"] --> OR
    x2 --> NAND
    OR --> AND["AND neuron"]
    NAND --> AND
    AND --> out["output"]

def xor_network(x1, x2):
    or_neuron = Perceptron(2)
    or_neuron.weights = [1.0, 1.0]
    or_neuron.bias = -0.5

    nand_neuron = Perceptron(2)
    nand_neuron.weights = [-1.0, -1.0]
    nand_neuron.bias = 1.5

    and_neuron = Perceptron(2)
    and_neuron.weights = [1.0, 1.0]
    and_neuron.bias = -1.5

    hidden1 = or_neuron.predict([x1, x2])
    hidden2 = nand_neuron.predict([x1, x2])
    output = and_neuron.predict([hidden1, hidden2])
    return output


print("\n=== XOR Gate (multi-layer network) ===")
for inputs, expected in xor_data:
    result = xor_network(inputs[0], inputs[1])
    print(f"  {inputs} -> {result} (expected {expected})")

所有四种情况均正确。将感知机堆叠成层，能够构建出单个感知机无法实现的决策边界 (Decision Boundaries)。

### 步骤 5：训练双层网络

步骤 4 中的权重是手动设定的。这种方法对异或问题有效，但对于事先不知道正确权重的实际问题则行不通。解决方案是：用 Sigmoid 函数 (Sigmoid Function) 替换阶跃函数 (Step Function)，并通过反向传播 (Backpropagation) 自动学习权重。

class TwoLayerNetwork:
    def __init__(self, learning_rate=0.5):
        import random
        random.seed(0)
        self.w_hidden = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
        self.b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.w_output = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_output = random.uniform(-1, 1)
        self.lr = learning_rate

    def sigmoid(self, x):
        import math
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_outputs = []
        for i in range(2):
            z = sum(w * x for w, x in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(self.sigmoid(z))
        z_out = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = self.sigmoid(z_out)
        return self.output

    def train(self, training_data, epochs=10000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                output = self.forward(inputs)
                error = target - output
                total_error += error ** 2

                d_output = error * output * (1 - output)

                saved_w_output = self.w_output[:]
                hidden_deltas = []
                for i in range(2):
                    h = self.hidden_outputs[i]
                    hd = d_output * saved_w_output[i] * h * (1 - h)
                    hidden_deltas.append(hd)

                for i in range(2):
                    self.w_output[i] += self.lr * d_output * self.hidden_outputs[i]
                self.b_output += self.lr * d_output

                for i in range(2):
                    for j in range(len(inputs)):
                        self.w_hidden[i][j] += self.lr * hidden_deltas[i] * inputs[j]
                    self.b_hidden[i] += self.lr * hidden_deltas[i]

net = TwoLayerNetwork(learning_rate=2.0)
net.train(xor_data, epochs=10000)
for inputs, expected in xor_data:
    result = net.forward(inputs)
    predicted = 1 if result >= 0.5 else 0
    print(f"  {inputs} -> {result:.4f} (rounded: {predicted}, expected {expected})")

与步骤 4 相比，这里有两个关键区别。首先，Sigmoid 函数取代了阶跃函数——它是平滑的，因此存在梯度 (Gradients)。其次，`train` 方法将误差从输出层反向传播至隐藏层 (Hidden Layer)，并根据每个权重对误差的贡献程度按比例进行调整。这就是仅用 20 行代码实现的反向传播算法。

这为第 03 课奠定了基础。`d_output` 和 `hidden_deltas` 背后的数学原理，正是应用于网络图结构的链式法则 (Chain Rule)。我们将在下一课中进行严谨的推导。

## 上手使用

你刚刚从零构建的所有功能，只需一行导入语句即可实现：

from sklearn.linear_model import Perceptron as SkPerceptron
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 0, 0, 1])

clf = SkPerceptron(max_iter=100, tol=1e-3)
clf.fit(X, y)
print([clf.predict([x])[0] for x in X])

仅需五行代码。你之前编写的 30 行 `Perceptron` 类也能实现相同的功能。`sklearn` 版本额外增加了收敛性检查（convergence checks）、多种损失函数（loss functions）以及稀疏输入支持，但其核心循环完全一致：计算加权和（weighted sum）、应用阶跃函数（step function）、在出现误差时更新权重。

真正的差距在大规模应用中才会显现。在生产环境的神经网络（neural networks）中，主要变化如下：

- 阶跃函数被替换为 Sigmoid、ReLU 或其他平滑激活函数（activation functions）
- 权重（weights）通过反向传播（backpropagation，见第 03 课）自动学习
- 网络层数加深：从 3 层、10 层到 100 层以上
- 核心原理保持不变：每一层都基于前一层的输出构建新的特征（features）

单个感知机（perceptron）只能划分出直线边界。将它们堆叠起来，你就能拟合任意复杂的形状。

## 课程产出

本课程的产出包括：
- `outputs/skill-perceptron.md` - 一份技能文档，涵盖何时需要单层架构（single-layer architectures）与多层架构（multi-layer architectures）

## 练习

1. 在 NAND 门（与非门，通用逻辑门——任何逻辑电路均可由 NAND 门构建）上训练一个感知机。验证其权重和偏置能否构成有效的决策边界（decision boundary）。
2. 修改 `Perceptron` 类，以记录每个训练轮次（epoch）的决策边界（w1*x1 + w2*x2 + b = 0）。打印在 AND 门（与门）训练过程中该直线的移动轨迹。
3. 构建一个具有 3 个输入的感知机，仅当 3 个输入中至少有 2 个为 1 时才输出 1（多数表决函数）。该问题是否线性可分（linearly separable）？为什么？

## 核心术语

| 术语 | 通俗理解 | 实际含义 |
|------|----------|----------|
| 感知机（Perceptron） | “一个假的神经元” | 一种线性分类器：将输入与权重进行点积运算，加上偏置后，通过阶跃函数输出 |
| 权重（Weight） | “输入有多重要” | 一个乘数，用于缩放每个输入对最终决策的贡献度 |
| 偏置（Bias） | “阈值” | 一个常数，用于平移决策边界，使得感知机即使在输入全为零时也能被激活 |
| 激活函数（Activation function） | “把数值压缩的东西” | 应用于加权和之后的函数——感知机使用阶跃函数，现代网络使用 Sigmoid 或 ReLU |
| 线性可分（Linearly separable） | “你能在它们之间画一条线” | 指存在一个超平面（hyperplane）能够完美将不同类别分开的数据集 |
| XOR 问题（XOR problem） | “感知机搞不定的东西” | 证明了单层网络无法学习非线性可分函数 |
| 决策边界（Decision boundary） | “分类器切换判断的地方” | 将输入空间划分为两个类别的超平面 w*x + b = 0 |
| 多层感知机（Multi-layer perceptron） | “真正的神经网络” | 将感知机按层堆叠，每一层的输出作为下一层的输入 |

## 延伸阅读

- Frank Rosenblatt，《感知机（Perceptron）：大脑中信息存储与组织的概率模型》（1958 年）——开启一切的奠基性论文
- Minsky & Papert，《感知机（Perceptrons）》（1969 年）——该书证明了单层网络（single-layer networks）无法解决异或（XOR）问题，并导致感知机研究停滞了十年
- Michael Nielsen，《神经网络（Neural Networks）与深度学习（Deep Learning）》第 1 章 (http://neuralnetworksanddeeplearning.com/) ——免费在线资源，对感知机如何组合构建为网络提供了最直观的图解说明