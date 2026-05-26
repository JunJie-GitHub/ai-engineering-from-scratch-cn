# 构建你自己的迷你框架

> 你已经构建了神经元（neuron）、层（layer）、网络（network）、反向传播（backpropagation）、激活函数（activation function）、损失函数（loss function）、优化器（optimizer）、正则化（regularization）、初始化（initialization）以及学习率调度（learning rate schedule）。它们目前都是独立的组件。现在，将它们串联起来，组装成一个完整的框架。不是 PyTorch，不是 TensorFlow，而是完全属于你自己的。

**类型：** 构建
**语言：** Python
**前置要求：** 第三阶段全部内容（第 01-09 课）
**预计耗时：** 约 120 分钟

## 学习目标

- 构建一个完整的深度学习框架（约 500 行代码），包含模块（Module）、线性层（Linear）、ReLU、Sigmoid、Dropout、批归一化（BatchNorm）、序列容器（Sequential）、损失函数、优化器以及数据加载器（DataLoader）
- 解释模块（Module）抽象机制（前向传播 forward、反向传播 backward、参数 parameters），以及为何需要切换训练/评估模式（train/eval mode）
- 将所有组件整合为一个可运行的训练循环（training loop），用于在圆形分类任务上训练一个 4 层网络
- 将你框架中的每个组件与 PyTorch 中的对应实现进行映射（如 nn.Module、nn.Sequential、optim.Adam、DataLoader）

## 问题背景

经过前十节课的学习，你积累了大量构建模块，但它们分散在不同的文件中。这里有一个 Value 类，那里有一个训练循环，权重初始化在另一个文件，学习率调度又在别处。想要训练一个网络，你不得不从五节不同的课程中复制粘贴代码，然后手动将它们拼接在一起。

这正是框架所要解决的问题。PyTorch 提供了 nn.Module、nn.Sequential、optim.Adam、DataLoader 以及一套将它们串联起来的训练循环模式。TensorFlow 则提供了 keras.Layer、keras.Sequential 和 keras.optimizers.Adam。这些并非魔法，而是一种组织模式，让你能够定义、训练和评估网络，而无需每次都重新搭建底层管线。

接下来，你将用大约 500 行 Python 代码实现同样的功能。不依赖 numpy，也没有任何外部依赖。这个框架将能够定义任意前馈网络（feedforward network），使用 SGD 或 Adam 进行训练，支持数据批处理（data batching）、Dropout 和批归一化（batch normalization）、任意激活函数，以及学习率调度。

完成本项目后，你将彻底明白在 PyTorch 中编写 model = nn.Sequential(...) 时底层究竟发生了什么。你会理解为什么需要 model.train() 和 model.eval()。你会明白为什么 optimizer.zero_grad() 需要作为一个独立的调用来执行。你将透彻理解这一切，因为所有底层逻辑都是你亲手构建的。

## 核心概念

### 模块抽象（Module Abstraction）

PyTorch 中的每一层都继承自 `nn.Module`。一个模块（Module）承担三项核心职责：

1. **forward()** —— 根据输入计算输出
2. **parameters()** —— 返回所有可训练权重（Trainable Weights）
3. **backward()** —— 计算梯度（Gradients）（在 PyTorch 中由自动微分（Autograd）处理，在我们的实现中为显式调用）

线性层（Linear Layer）是一个模块，ReLU 激活函数（ReLU Activation）是一个模块，Dropout 层（Dropout Layer）是一个模块，批归一化层（Batch Normalization Layer）也是一个模块。它们都遵循相同的接口。

### 顺序容器（Sequential Container）

`nn.Sequential` 用于将多个模块串联起来。前向传播（Forward Pass）：数据依次流经模块 1、模块 2、模块 3。反向传播（Backward Pass）：沿链条反向进行。该容器本身也是一个模块——它同样具备 `forward()`、`parameters()` 和 `backward()` 方法。这体现了组合模式（Composite Pattern）：一系列模块的组合本身依然是一个模块。

### 训练模式与评估模式（Training vs Evaluation Mode）

在训练期间，Dropout 会随机将部分神经元置零，而在评估期间则让所有信号直接通过。批归一化（Batch Normalization）在训练时使用当前批次的统计量，在评估时则使用滑动平均值（Running Averages）。`train()` 和 `eval()` 方法用于切换这两种行为。每个模块都包含一个 `training` 标志位。

### 优化器（Optimizer）

优化器利用参数的梯度来更新参数。随机梯度下降（Stochastic Gradient Descent, SGD）：`param -= lr * grad`。Adam 优化器：维护动量（Momentum）和方差估计（Variance Estimates），随后进行更新。优化器并不感知网络架构——它仅能看到一个扁平化的参数列表及其对应的梯度。

### 数据加载器（DataLoader）

批次处理（Batching）主要出于两个原因。首先，在处理大规模问题时，无法将整个数据集一次性加载到内存中。其次，小批量梯度下降（Mini-batch Gradient Descent）引入的噪声有助于模型跳出局部极小值（Local Minima）。`DataLoader` 负责将数据划分为多个批次，并可选择在每个训练轮次（Epoch）之间进行数据打乱（Shuffle）。

### 框架架构（Framework Architecture）

graph TD
    subgraph "Modules"
        Linear["Linear<br/>W*x + b"]
        ReLU["ReLU<br/>max(0, x)"]
        Sigmoid["Sigmoid<br/>1/(1+e^-x)"]
        Dropout["Dropout<br/>random zero mask"]
        BatchNorm["BatchNorm<br/>normalize activations"]
    end

    subgraph "Containers"
        Sequential["Sequential<br/>chains modules"]
    end

    subgraph "Loss Functions"
        MSE["MSELoss<br/>(pred - target)^2"]
        BCE["BCELoss<br/>binary cross-entropy"]
    end

    subgraph "Optimizers"
        SGD["SGD<br/>param -= lr * grad"]
        Adam["Adam<br/>adaptive moments"]
    end

    subgraph "Data"
        DataLoader["DataLoader<br/>batching + shuffle"]
    end

    Sequential --> |"contains"| Linear
    Sequential --> |"contains"| ReLU
    Sequential --> |"forward/backward"| MSE
    SGD --> |"updates"| Sequential
    DataLoader --> |"feeds"| Sequential

### 训练循环（Training Loop）

sequenceDiagram
    participant DL as DataLoader
    participant M as Model
    participant L as Loss
    participant O as Optimizer

    loop Each Epoch
        DL->>M: batch of inputs
        M->>M: forward pass (layer by layer)
        M->>L: predictions
        L->>L: compute loss
        L->>M: backward pass (gradients)
        M->>O: parameters + gradients
        O->>M: updated parameters
        O->>O: zero gradients
    end

### 模块层级结构（Module Hierarchy）

classDiagram
    class Module {
        +forward(x)
        +backward(grad)
        +parameters()
        +train()
        +eval()
    }

    class Linear {
        -weights
        -biases
        +forward(x)
        +backward(grad)
    }

    class ReLU {
        +forward(x)
        +backward(grad)
    }

    class Sequential {
        -modules[]
        +forward(x)
        +backward(grad)
        +parameters()
    }

    Module <|-- Linear
    Module <|-- ReLU
    Module <|-- Sequential
    Sequential *-- Module


## 构建

### 步骤 1：模块基类 (Module Base Class)

每个网络层（Layer）都必须实现的抽象接口。

class Module:
    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

### 步骤 2：线性层 (Linear Layer)

神经网络的基础构建模块。负责存储权重（Weights）和偏置（Biases），在前向传播（Forward Pass）中计算 Wx + b，并在反向传播（Backward Pass）中计算权重和输入的梯度。

import math
import random


class Linear(Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        std = math.sqrt(2.0 / fan_in)
        self.weights = [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
        self.biases = [0.0] * fan_out
        self.weight_grads = [[0.0] * fan_in for _ in range(fan_out)]
        self.bias_grads = [0.0] * fan_out
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.input = None

    def forward(self, x):
        self.input = x
        output = []
        for i in range(self.fan_out):
            val = self.biases[i]
            for j in range(self.fan_in):
                val += self.weights[i][j] * x[j]
            output.append(val)
        return output

    def backward(self, grad):
        input_grad = [0.0] * self.fan_in
        for i in range(self.fan_out):
            self.bias_grads[i] += grad[i]
            for j in range(self.fan_in):
                self.weight_grads[i][j] += grad[i] * self.input[j]
                input_grad[j] += grad[i] * self.weights[i][j]
        return input_grad

    def parameters(self):
        params = []
        for i in range(self.fan_out):
            for j in range(self.fan_in):
                params.append((self.weights, i, j, self.weight_grads))
            params.append((self.biases, i, None, self.bias_grads))
        return params

### 步骤 3：激活模块 (Activation Modules)

将 ReLU、Sigmoid 和 Tanh 实现为模块。每个模块都会缓存反向传播所需的中间值。

class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = [1.0 if v > 0 else 0.0 for v in x]
        return [max(0.0, v) for v in x]

    def backward(self, grad):
        return [g * m for g, m in zip(grad, self.mask)]


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = []
        for v in x:
            v = max(-500, min(500, v))
            self.output.append(1.0 / (1.0 + math.exp(-v)))
        return self.output

    def backward(self, grad):
        return [g * o * (1 - o) for g, o in zip(grad, self.output)]


class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = [math.tanh(v) for v in x]
        return self.output

    def backward(self, grad):
        return [g * (1 - o * o) for g, o in zip(grad, self.output)]

### 步骤 4：Dropout 模块

在训练期间随机将部分元素置零。将剩余元素按 1/(1-p) 进行缩放，以保持期望值不变。在评估模式（Eval Mode）下不执行任何操作。

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training:
            return x
        self.mask = [0.0 if random.random() < self.p else 1.0 / (1 - self.p) for _ in x]
        return [v * m for v, m in zip(x, self.mask)]

    def backward(self, grad):
        if self.mask is None:
            return grad
        return [g * m for g, m in zip(grad, self.mask)]

### 步骤 5：BatchNorm 模块

对批次（Batch）中每个特征的激活值进行归一化，使其均值为零、方差为一。同时维护滑动统计量（Running Statistics）以供评估模式使用。

class BatchNorm(Module):
    def __init__(self, size, momentum=0.1, eps=1e-5):
        super().__init__()
        self.size = size
        self.gamma = [1.0] * size
        self.beta = [0.0] * size
        self.gamma_grads = [0.0] * size
        self.beta_grads = [0.0] * size
        self.running_mean = [0.0] * size
        self.running_var = [1.0] * size
        self.momentum = momentum
        self.eps = eps
        self.x_norm = None
        self.std_inv = None
        self.batch_input = None

    def forward_batch(self, batch):
        batch_size = len(batch)
        output_batch = []

        if self.training:
            mean = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    mean[j] += sample[j]
            mean = [m / batch_size for m in mean]

            var = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v / batch_size for v in var]

            self.std_inv = [1.0 / math.sqrt(v + self.eps) for v in var]

            self.x_norm = []
            self.batch_input = batch
            for sample in batch:
                normed = [(sample[j] - mean[j]) * self.std_inv[j] for j in range(self.size)]
                self.x_norm.append(normed)
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

            for j in range(self.size):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * var[j]
        else:
            std_inv = [1.0 / math.sqrt(v + self.eps) for v in self.running_var]
            for sample in batch:
                normed = [(sample[j] - self.running_mean[j]) * std_inv[j] for j in range(self.size)]
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

        return output_batch

    def forward(self, x):
        result = self.forward_batch([x])
        return result[0]

    def backward(self, grad):
        if self.x_norm is None:
            return grad
        for j in range(self.size):
            self.gamma_grads[j] += self.x_norm[0][j] * grad[j]
            self.beta_grads[j] += grad[j]
        return [grad[j] * self.gamma[j] * self.std_inv[j] for j in range(self.size)]

    def parameters(self):
        params = []
        for j in range(self.size):
            params.append((self.gamma, j, None, self.gamma_grads))
            params.append((self.beta, j, None, self.beta_grads))
        return params

### 步骤 6：Sequential 容器

将多个模块串联起来。前向传播从左至右执行，反向传播从右至左执行。

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, grad):
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params

    def train(self):
        self.training = True
        for module in self.modules:
            module.train()

    def eval(self):
        self.training = False
        for module in self.modules:
            module.eval()

### 步骤 7：损失函数 (Loss Functions)

包含均方误差（MSE）和二元交叉熵（Binary Cross-Entropy）。每个函数都会返回损失值，并提供一个 `backward()` 方法用于返回梯度。

class MSELoss:
    def __call__(self, predicted, target):
        self.predicted = predicted
        self.target = target
        n = len(predicted)
        self.loss = sum((p - t) ** 2 for p, t in zip(predicted, target)) / n
        return self.loss

    def backward(self):
        n = len(self.predicted)
        return [2 * (p - t) / n for p, t in zip(self.predicted, self.target)]


class BCELoss:
    def __call__(self, predicted, target):
        self.predicted = predicted
        self.target = target
        eps = 1e-7
        n = len(predicted)
        self.loss = 0
        for p, t in zip(predicted, target):
            p = max(eps, min(1 - eps, p))
            self.loss += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        self.loss /= n
        return self.loss

    def backward(self):
        eps = 1e-7
        n = len(self.predicted)
        grads = []
        for p, t in zip(self.predicted, self.target):
            p = max(eps, min(1 - eps, p))
            grads.append((-t / p + (1 - t) / (1 - p)) / n)
        return grads

### 步骤 8：SGD 与 Adam 优化器 (Optimizers)

两者均接收参数列表，并利用梯度更新权重。

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.params = parameters
        self.lr = lr

    def step(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                container[i][j] -= self.lr * grad_container[i][j]
            else:
                container[i] -= self.lr * grad_container[i]

    def zero_grad(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)

    def step(self):
        self.t += 1
        for idx, (container, i, j, grad_container) in enumerate(self.params):
            if j is not None:
                g = grad_container[i][j]
            else:
                g = grad_container[i]

            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * g
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * g * g

            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

            update = self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

            if j is not None:
                container[i][j] -= update
            else:
                container[i] -= update

    def zero_grad(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0

### 步骤 9：DataLoader

将数据划分为批次（Batches），并可选择在每个训练轮次（Epoch）开始时打乱数据顺序。

class DataLoader:
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = [self.data[i] for i in batch_indices]
            inputs = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            yield inputs, targets

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

### 步骤 10：在圆形分类任务上训练 4 层网络

将所有组件整合在一起。定义模型、选择损失函数与优化器，并运行训练循环。

def make_circle_data(n=500, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], [label]))
    return data


def train():
    random.seed(42)

    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 16),
        ReLU(),
        Linear(16, 8),
        ReLU(),
        Linear(8, 1),
        Sigmoid(),
    )

    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    data = make_circle_data(500)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    loader = DataLoader(train_data, batch_size=16, shuffle=True)

    model.train()

    for epoch in range(100):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_inputs, batch_targets in loader:
            batch_loss = 0
            for x, t in zip(batch_inputs, batch_targets):
                pred = model.forward(x)
                loss = criterion(pred, t)
                batch_loss += loss

                optimizer.zero_grad()
                grad = criterion.backward()
                model.backward(grad)
                optimizer.step()

                predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
                if predicted_class == t[0]:
                    total_correct += 1
                total_samples += 1

            total_loss += batch_loss

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100

        if epoch % 10 == 0 or epoch == 99:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Train Accuracy: {accuracy:.1f}%")

    model.eval()
    correct = 0
    for x, t in test_data:
        pred = model.forward(x)
        predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
        if predicted_class == t[0]:
            correct += 1
    test_accuracy = correct / len(test_data) * 100
    print(f"\nTest Accuracy: {test_accuracy:.1f}% ({correct}/{len(test_data)})")

    return model, test_accuracy


## 上手使用

以下是你刚刚构建内容的 PyTorch 等效实现：

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_predictions = model(test_inputs)

结构完全相同。`Sequential`、`Linear`、`ReLU`、`Sigmoid`、`BCELoss`、`Adam`、`zero_grad`、`backward`、`step`、`train`、`eval`。每个概念都是一一对应的。区别在于 PyTorch 会自动处理自动微分（autograd）（无需在每个模块中手动实现 `backward()`），支持 GPU 运行，且经过了多年的深度优化。但核心骨架是完全一致的。

现在当你看到 PyTorch 代码时，你能清楚地知道每一行具体在做什么。这种透彻的理解正是本教程的核心目的。

## 交付成果

本课时将产出：
- `outputs/prompt-framework-architect.md` -- 用于利用框架抽象（framework abstraction）设计神经网络架构（neural network architecture）的提示词（prompt）

## 练习

1. 为多分类（multi-class classification）任务添加一个 `SoftmaxCrossEntropyLoss` 类。对预测值执行 Softmax 操作，计算交叉熵损失（cross-entropy loss），并处理合并的反向传播（backward pass）。在一个三分类螺旋数据集（spiral dataset）上对其进行测试。

2. 在优化器（optimizer）中实现学习率调度（learning rate scheduling）：添加 `set_lr()` 方法，并接入第 09 课中的余弦调度（cosine schedule）。使用预热（warmup）结合余弦策略训练圆形分类器，并与恒定学习率（constant learning rate）进行对比。

3. 为 `Sequential` 添加 `save()` 和 `load()` 方法，用于将所有权重（weights）序列化至 JSON 文件并重新加载。验证加载后的模型是否能输出与原始模型一致的预测结果。

4. 在 Adam 优化器中实现权重衰减（weight decay，即 L2 正则化（L2 regularization））。添加 `weight_decay` 参数，使权重在每一步更新时都向零收缩。对比 `decay=0` 与 `decay=0.01` 两种设置下的训练效果。

5. 将逐样本训练循环替换为标准的小批量梯度累积（mini-batch gradient accumulation）：累积一个批次（batch）内所有样本的梯度，随后除以批次大小（batch size）并执行一次优化器步进（optimizer step）。评估该改动是否会改变模型的收敛速度（convergence speed）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 模块 (Module) | “一个层” | 框架中的基础抽象——任何包含 `forward()`、`backward()` 和 `parameters()` 的对象 |
| 序列容器 (Sequential) | “按顺序堆叠层” | 一种将模块串联起来的容器，在前向传播时按顺序应用它们，在反向传播时按逆序应用 |
| 前向传播 (Forward pass) | “运行网络” | 按顺序将输入传入每个模块以计算输出 |
| 反向传播 (Backward pass) | “计算梯度” | 将损失梯度按逆序通过每个模块传播，以计算参数梯度 |
| 参数 (Parameters) | “可训练的权重” | 网络中所有可由优化器更新的值——包括权重和偏置 |
| 优化器 (Optimizer) | “更新权重的组件” | 一种利用梯度更新参数的算法，实现了 SGD、Adam 或其他更新规则 |
| 数据加载器 (DataLoader) | “提供数据的组件” | 一种迭代器，负责将数据集划分为批次，并可选择在每个训练轮次之间进行打乱 |
| 训练模式 (Training mode) | "model.train()" | 一个标志位，用于启用随机行为（如 Dropout）以及使用批次统计量的批归一化 (Batch Normalization) |
| 评估模式 (Evaluation mode) | "model.eval()" | 一个标志位，用于禁用 Dropout 并在批归一化中使用运行统计量 |
| 梯度清零 (Zero grad) | “清除梯度” | 在计算下一个批次的梯度之前，将所有参数梯度重置为零 |

## 延伸阅读

- Paszke 等人，《PyTorch: An Imperative Style, High-Performance Deep Learning Library》（2019）—— 阐述 PyTorch 设计决策的论文
- Chollet，《Deep Learning with Python, Second Edition》（2021）—— 第 3 章使用相同的模块/层抽象讲解了 Keras 的内部机制
- Johnson，《Tiny-DNN》（https://github.com/tiny-dnn/tiny-dnn）—— 一个仅包含头文件的 C++ 深度学习框架，有助于理解框架内部机制