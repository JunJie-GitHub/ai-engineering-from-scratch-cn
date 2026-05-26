# PyTorch 简介

> 你已经用活塞和曲轴亲手打造了引擎。现在，来学习一下大家真正在驾驶的那一款吧。

**类型：** 构建
**语言：** Python
**前置课程：** 第 03.10 课（构建你自己的微型框架）
**时长：** 约 75 分钟

## 学习目标

- 使用 PyTorch 的 nn.Module、nn.Sequential 和 autograd 构建并训练神经网络
- 使用 PyTorch 张量（Tensor）、GPU 加速以及标准训练循环（zero_grad、forward、loss、backward、step）
- 将你从零构建的微型框架组件转换为对应的 PyTorch 实现
- 对纯 Python 框架与 PyTorch 在相同任务上的训练速度进行性能分析（Profiling）与对比

## 问题所在

你已经拥有一个可运行的微型框架。其中包含线性层（Linear Layer）、ReLU、Dropout、批归一化（Batch Normalization）、Adam 优化器、数据加载器（DataLoader）以及训练循环。它完全使用纯 Python 实现，能够在圆形分类问题上训练一个 4 层网络。

但在解决相同问题时，它的速度比 PyTorch 慢了 500 倍。

你的微型框架通过嵌套的 Python 循环逐样本进行处理。而 PyTorch 会将相同的操作分发至经过优化的 C++/CUDA 内核，并在 GPU 上运行。在单张 NVIDIA A100 显卡上，PyTorch 仅需约 6 小时即可在 ImageNet（128 万张图像）数据集上训练 ResNet-50（2560 万参数）。如果换作你的框架，完成相同任务大约需要 3000 小时——前提是它不会先耗尽内存。

速度并非唯一的差距。你的框架不支持 GPU。没有自动微分（Automatic Differentiation）——你为每个模块手动编写了 backward()。不支持模型序列化（Serialization）。不支持分布式训练（Distributed Training）。不支持混合精度（Mixed Precision）。如果不使用 print 语句，你甚至无法调试梯度流（Gradient Flow）。

PyTorch 完美填补了上述所有空白。更重要的是，它保留了你已经建立的完全相同的思维模型（Mental Model）：Module、forward()、parameters()、backward()、optimizer.step()。这些概念可以一一对应迁移。语法也几乎完全一致。唯一的区别在于，PyTorch 在你从零设计的相同接口背后，封装了长达十年的系统工程（Systems Engineering）积累。

## 核心概念

### PyTorch 为何胜出

2015 年，TensorFlow 要求在运行任何代码前必须先定义静态计算图（static computation graph）。你需要先构建图、编译它，然后再将数据输入其中。调试意味着盯着计算图的可视化界面。修改网络架构则意味着从头重新构建整个图。

PyTorch 于 2017 年发布，秉持着截然不同的理念：即时执行（eager execution）。你编写 Python 代码，它立即运行。`y = model(x)` 会立刻计算出 y 的值，而不是“向图中添加一个稍后计算 y 的节点”。这意味着标准的 Python 调试工具可以直接使用。`print()` 能用，`pdb` 能用，前向传播（forward pass）中的 `if/else` 也能正常工作。

到 2020 年，市场已经给出了答案。PyTorch 在机器学习（machine learning）研究论文中的占比从 2017 年的 7% 飙升至 2022 年的 75% 以上。Meta、Google DeepMind、OpenAI、Anthropic 和 Hugging Face 均将 PyTorch 作为其核心框架。作为回应，TensorFlow 2.x 也引入了即时执行模式——这无异于默认 PyTorch 的设计是正确的。

经验教训：开发者体验具有复利效应。一个运行速度慢 10% 但调试速度快 50% 的框架，永远会胜出。

### 张量（Tensors）

张量（tensor）是一个多维数组，具有三个关键属性：形状（shape）、数据类型（dtype）和设备（device）。

import torch

x = torch.zeros(3, 4)           # shape: (3, 4), dtype: float32, device: cpu
x = torch.randn(2, 3, 224, 224) # batch of 2 RGB images, 224x224
x = torch.tensor([1, 2, 3])     # from a Python list

**形状（Shape）** 表示维度。标量的形状为 `()`，向量为 `(n,)`，矩阵为 `(m, n)`，一批图像的形状为 `(batch, channels, height, width)`。

**数据类型（Dtype）** 控制精度和内存占用。

| dtype | 位数 | 范围 | 使用场景 |
|-------|------|-------|----------|
| float32 | 32 | 约 7 位十进制数 | 默认训练 |
| float16 | 16 | 约 3.3 位十进制数 | 混合精度训练 |
| bfloat16 | 16 | 与 float32 范围相同，精度较低 | 大语言模型（LLM）训练 |
| int8 | 8 | -128 到 127 | 量化推理 |

**设备（Device）** 决定计算发生的位置。

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(3, 4, device=device)
x = x.to("cuda")
x = x.cpu()

每个操作都要求所有张量位于同一设备上。这是初学者最常遇到的 PyTorch 错误：`RuntimeError: Expected all tensors to be on the same device`。解决方法是在计算前将所有数据移动到同一设备上。

**重塑形状（Reshaping）** 的时间复杂度为 O(1)（常数时间）——它只修改元数据，不改变实际数据。

x = torch.randn(2, 3, 4)
x.view(2, 12)      # reshape to (2, 12) -- must be contiguous
x.reshape(6, 4)    # reshape to (6, 4) -- works always
x.permute(2, 0, 1) # reorder dimensions
x.unsqueeze(0)     # add dimension: (1, 2, 3, 4)
x.squeeze()        # remove size-1 dimensions

### 自动微分（Autograd）

你的迷你框架要求为每个模块手动实现 `backward()`。PyTorch 则不需要。它会将张量上的每一次操作记录到一个有向无环图（directed acyclic graph，即计算图）中，然后反向遍历该图以自动计算梯度。

graph LR
    x["x (leaf)"] --> mul["*"]
    w["w (leaf, requires_grad)"] --> mul
    mul --> add["+"]
    b["b (leaf, requires_grad)"] --> add
    add --> loss["loss"]
    loss --> |".backward()"| add
    add --> |"grad"| b
    add --> |"grad"| mul
    mul --> |"grad"| w

与你的框架的关键区别在于：PyTorch 使用基于磁带（tape-based）的自动微分。在前向传播期间，每个操作都会被追加到一条“磁带”上。调用 `.backward()` 时，系统会反向重放这条磁带。

x = torch.randn(3, requires_grad=True)
y = x ** 2 + 3 * x
z = y.sum()
z.backward()
print(x.grad)  # dz/dx = 2x + 3

自动微分的三条规则：

1. 只有 `requires_grad=True` 的叶子张量（leaf tensors）才会累积梯度
2. 梯度默认会累积——每次反向传播前需调用 `optimizer.zero_grad()`
3. `torch.no_grad()` 会禁用梯度追踪（用于模型评估阶段）

### nn.Module

`nn.Module` 是 PyTorch 中所有神经网络组件的基类。你在第 10 课中已经构建过类似的抽象。PyTorch 的版本在此基础上增加了自动参数注册、递归模块发现、设备管理以及状态字典（state dict）序列化功能。

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

当你在 `__init__` 中将 `nn.Module` 或 `nn.Parameter` 赋值为属性时，PyTorch 会自动注册它们。`model.parameters()` 会递归收集所有已注册的参数。这就是为什么你再也不需要像在迷你框架中那样手动收集权重了。

核心构建模块：

| 模块 | 功能 | 参数量 |
|--------|-------------|------------|
| nn.Linear(in, out) | 线性变换 Wx + b | in*out + out |
| nn.Conv2d(in_ch, out_ch, k) | 二维卷积 | in_ch*out_ch*k*k + out_ch |
| nn.BatchNorm1d(features) | 归一化激活值 | 2 * features |
| nn.Dropout(p) | 随机置零 | 0 |
| nn.ReLU() | max(0, x) | 0 |
| nn.GELU() | 高斯误差线性单元 | 0 |
| nn.Embedding(vocab, dim) | 查找表 | vocab * dim |
| nn.LayerNorm(dim) | 逐样本归一化 | 2 * dim |

### 损失函数与优化器

PyTorch 内置了你之前所构建的所有组件的生产级版本。

**损失函数**（来自 `torch.nn`）：

| 损失函数 | 任务 | 输入要求 |
|------|------|-------|
| nn.MSELoss() | 回归 | 任意形状 |
| nn.CrossEntropyLoss() | 多分类 | 原始 logits（非 softmax 输出） |
| nn.BCEWithLogitsLoss() | 二分类 | 原始 logits（非 sigmoid 输出） |
| nn.L1Loss() | 回归（鲁棒性强） | 任意形状 |
| nn.CTCLoss() | 序列对齐 | 对数概率 |

注意：`CrossEntropyLoss` 内部结合了 `LogSoftmax` 和 `NLLLoss`。请传入原始 logits，而不是 softmax 的输出。这是一个常见错误，会导致梯度计算错误且不会报错。

**优化器**（来自 `torch.optim`）：

| 优化器 | 适用场景 | 典型学习率 |
|-----------|-------------|-----------|
| SGD(params, lr, momentum) | 卷积神经网络（CNN）、调优成熟的流程 | 0.01--0.1 |
| Adam(params, lr) | 默认起点 | 1e-3 |
| AdamW(params, lr, weight_decay) | Transformer、微调 | 1e-4--1e-3 |
| LBFGS(params) | 小规模、二阶优化 | 1.0 |

### 训练循环

每个 PyTorch 训练循环都遵循相同的五步模式。你在第 10 课中已经了解过这一点。

sequenceDiagram
    participant D as DataLoader
    participant M as Model
    participant L as Loss fn
    participant O as Optimizer

    loop Each Epoch
        D->>M: batch = next(dataloader)
        M->>L: predictions = model(batch)
        L->>L: loss = criterion(predictions, targets)
        L->>M: loss.backward()
        O->>M: optimizer.step()
        O->>O: optimizer.zero_grad()
    end

标准范式：

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

批次循环内仅需五行代码。正是这五行代码训练出了 GPT-4、Stable Diffusion 和 LLaMA。网络架构会变，数据会变，但这五行代码永远不会变。

### Dataset 与 DataLoader

PyTorch 的 `Dataset` 是一个抽象类，包含两个方法：`__len__` 和 `__getitem__`。`DataLoader` 则为其封装了批处理（batching）、数据打乱（shuffling）以及多进程数据加载功能。

from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

`num_workers=4` 会生成 4 个子进程，在 GPU 训练当前批次时并行加载数据。对于受磁盘 I/O 限制的任务（如大尺寸图像、音频），仅凭这一项设置就能使训练速度翻倍。

### GPU 训练

将模型迁移至 GPU：

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

这会递归地将所有参数和缓冲区移动到 GPU。随后在训练过程中移动每个批次的数据：

inputs, targets = inputs.to(device), targets.to(device)

**混合精度（mixed precision）** 在现代 GPU（如 A100、H100、RTX 4090）上可将内存占用减半并使吞吐量翻倍。其原理是在前向/反向传播中使用 float16，同时保持主权重为 float32：

from torch.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, targets in loader:
    with autocast(device_type="cuda"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

### 对比：迷你框架 vs PyTorch vs JAX

| 特性 | 迷你框架（第10课） | PyTorch | JAX |
|---------|---------------------|---------|-----|
| 自动微分 | 手动实现 backward() | 基于磁带的自动微分 | 函数式变换 |
| 执行模式 | 即时执行（Python 循环） | 即时执行（C++ 内核） | 追踪 + JIT 编译 |
| GPU 支持 | 无 | 支持（CUDA, ROCm, MPS） | 支持（CUDA, TPU） |
| 速度（MNIST MLP） | ~300秒/轮 | ~0.5秒/轮 | ~0.3秒/轮 |
| 模块系统 | 自定义 Module 类 | nn.Module | 无状态函数（Flax/Equinox） |
| 调试 | print() | print(), pdb, breakpoint() | 较难（JIT 追踪会破坏 print） |
| 生态系统 | 无 | Hugging Face, Lightning, timm | Flax, Optax, Orbax |
| 学习曲线 | 你自己构建的 | 中等 | 陡峭（函数式范式） |
| 生产环境应用 | 玩具问题 | Meta, OpenAI, Anthropic, HF | Google DeepMind, Midjourney |

## 构建

仅使用 PyTorch 底层原语 (Primitives) 在 MNIST 数据集上训练的一个 3 层多层感知机 (Multilayer Perceptron)。不使用任何高级封装，也不依赖 `torchvision.datasets`。我们将自行下载并解析原始数据。

### 步骤 1：从原始文件加载 MNIST 数据

MNIST 数据集以 4 个 gzip 压缩文件的形式提供：训练图像（60,000 x 28 x 28）、训练标签、测试图像（10,000 x 28 x 28）以及测试标签。我们将下载这些文件并解析其二进制格式。

import torch
import torch.nn as nn
import struct
import gzip
import urllib.request
import os

def download_mnist(path="./mnist_data"):
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    os.makedirs(path, exist_ok=True)
    for f in files:
        filepath = os.path.join(path, f)
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(base_url + f, filepath)

def load_images(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        images = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        images = images.reshape(num, rows * cols).float() / 255.0
    return images

def load_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
        labels = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    return labels

### 步骤 2：定义模型

一个 3 层多层感知机 (Multilayer Perceptron)：784 -> 256 -> 128 -> 10。使用 ReLU 激活函数 (ReLU Activation)。引入随机失活 (Dropout) 进行正则化。为保持简洁，不使用批归一化 (Batch Normalization)。

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

输出层生成 10 个原始未归一化预测值 (Logits)（每个数字类别对应一个）。此处不显式使用 Softmax，因为 `CrossEntropyLoss` 会在内部自动处理该操作。

参数量计算：784*256 + 256 + 256*128 + 128 + 128*10 + 10 = 235,146。以现代标准来看，这个规模非常小。作为对比，GPT-2 small 拥有 1.24 亿参数。该模型仅需几秒即可完成训练。

### 步骤 3：训练循环

采用经典的“前向传播-计算损失-反向传播-参数更新”模式。

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

注意在评估阶段使用了 `torch.no_grad()`。这会禁用自动求导机制 (Autograd)，从而降低内存占用并加速推理过程。如果不加此上下文管理器，PyTorch 会构建一个你根本用不到的计算图 (Computational Graph)。

### 步骤 4：整合所有组件

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_mnist()
    train_images = load_images("./mnist_data/train-images-idx3-ubyte.gz")
    train_labels = load_labels("./mnist_data/train-labels-idx1-ubyte.gz")
    test_images = load_images("./mnist_data/t10k-images-idx3-ubyte.gz")
    test_labels = load_labels("./mnist_data/t10k-labels-idx1-ubyte.gz")

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )

    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Parameters: {num_params:,}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print()

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        print(
            f"Epoch {epoch+1:2d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    torch.save(model.state_dict(), "mnist_mlp.pt")
    print(f"\nModel saved to mnist_mlp.pt")
    print(f"Final test accuracy: {test_acc:.4f}")

训练 10 个训练轮次 (Epochs) 后的预期输出：测试准确率约为 97.8%。CPU 训练时间：约 30 秒。GPU 训练时间：约 5 秒。若使用相同架构的自研微型框架：约 45 分钟。

## 上手使用

### 快速对比：迷你框架（Mini Framework） vs PyTorch

| 迷你框架（第 10 课） | PyTorch |
|---------------------------|---------|
| `model = Sequential(Linear(784, 256), ReLU(), ...)` | `model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), ...)` |
| `pred = model.forward(x)` | `pred = model(x)` |
| `optimizer.zero_grad()` | `optimizer.zero_grad()` |
| `grad = criterion.backward()` 然后 `model.backward(grad)` | `loss.backward()` |
| `optimizer.step()` | `optimizer.step()` |
| 不支持 GPU | `model.to("cuda")` |
| 需为每个模块手动执行反向传播 | 自动微分（Autograd）自动处理一切 |

两者的接口几乎完全一致。真正的区别在于底层实现。

### 保存与加载模型

torch.save(model.state_dict(), "model.pt")

model = MNISTModel()
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()

务必保存 `state_dict()`（参数字典），而非整个模型对象。保存模型对象会依赖 `pickle` 序列化，在重构代码时极易引发兼容性问题。而状态字典（State Dict）具有良好的可移植性。

### 学习率调度（Learning Rate Scheduling）

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10
)
for epoch in range(10):
    train_one_epoch(model, train_loader, criterion, optimizer, device)
    scheduler.step()

PyTorch 内置了 15 种以上的调度器（Scheduler）：StepLR、ExponentialLR、CosineAnnealingLR、OneCycleLR、ReduceLROnPlateau 等。它们均可无缝接入相同的优化器接口。

## 交付产出

本课将生成两份产出物：

- `outputs/prompt-pytorch-debugger.md` —— 用于诊断常见 PyTorch 训练故障的提示词（Prompt）
- `outputs/skill-pytorch-patterns.md` —— PyTorch 训练模式（Training Patterns）的技能参考指南

## 练习

1. **添加批归一化（Batch Normalization）。** 在每个线性层之后（激活函数之前）插入 `nn.BatchNorm1d`。对比仅使用 Dropout 的版本，评估测试准确率与训练速度。使用批归一化后，模型应在更少的训练轮次（Epoch）内达到 98% 以上的准确率。

2. **实现学习率查找器（Learning Rate Finder）。** 使用指数递增的学习率（从 1e-7 到 1.0）训练一个轮次（Epoch）。绘制损失值（Loss）与学习率（LR）的关系曲线。最佳学习率通常位于损失值开始上升之前的临界点。利用该方法为 MNIST 模型挑选更优的学习率。

3. **迁移至 GPU 并启用混合精度（Mixed Precision）。** 在训练循环中加入 `torch.amp.autocast` 和 `GradScaler`。分别测量在 GPU 上开启与关闭混合精度时的吞吐量（样本数/秒）。在 A100 显卡上，预计可获得约 2 倍的加速效果。

4. **构建自定义数据集（Dataset）。** 下载 Fashion-MNIST 数据集（格式与 MNIST 相同，但类别为服装物品）。实现一个包含 `__getitem__` 和 `__len__` 方法的 `FashionMNISTDataset(Dataset)` 类。使用相同的多层感知机（MLP）进行训练并对比准确率。Fashion-MNIST 难度更高——预期准确率约为 88%，而 MNIST 约为 98%。

5. **将 Adam 替换为 SGD + 动量（Momentum）。** 使用 `SGD(params, lr=0.01, momentum=0.9)` 进行训练。对比两者的收敛曲线。随后加入 `CosineAnnealingLR` 调度器，观察 SGD 是否能在第 10 个轮次（Epoch）前追平 Adam 的性能。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 张量（Tensor） | “多维数组” | 一种带有类型信息且感知计算设备的数组，其内置的每一项操作都原生支持自动微分（Automatic Differentiation） |
| 自动微分引擎（Autograd） | “自动反向传播” | 一种基于计算磁带（tape-based）的系统，在前向传播期间记录所有操作，随后按逆序重放以精确计算梯度 |
| 模块基类（`nn.Module`） | “一个网络层” | 所有可微计算模块的基类——负责注册参数、支持嵌套结构，并自动管理训练与评估模式 |
| 状态字典（`state_dict`） | “模型权重” | 一个将参数名称映射到张量的有序字典（`OrderedDict`）——代表已训练模型的可移植、可序列化状态 |
| 反向传播方法（`.backward()`） | “计算梯度” | 反向遍历计算图，为每个 `requires_grad=True` 的叶子张量计算并累加梯度 |
| 设备迁移方法（`.to(device)`） | “移动到 GPU” | 递归地将模型的所有参数和缓冲区转移至指定计算设备（CPU、CUDA、MPS） |
| 数据加载器（`DataLoader`） | “数据管道” | 一个迭代器，负责从数据集（`Dataset`）中批量读取、打乱数据，并支持可选的并行数据加载 |
| 混合精度训练（Mixed precision） | “使用 float16” | 在前向和反向传播中使用 `float16` 以提升训练速度，同时保留 `float32` 主权重以确保数值稳定性 |
| 动态图执行（Eager execution） | “立即执行” | 操作在调用时即刻执行，而非推迟至后续的编译步骤——这是 PyTorch 区别于 TensorFlow 1.x 的核心设计哲学 |
| 梯度清零（`zero_grad`） | “重置梯度” | 在下一次反向传播前将所有参数梯度清零，因为 PyTorch 默认采用梯度累加机制 |

## 延伸阅读

- Paszke 等人，《PyTorch: An Imperative Style, High-Performance Deep Learning Library》（2019）——阐述 PyTorch 设计权衡的原始论文
- PyTorch 官方教程：“Learning PyTorch with Examples”（https://pytorch.org/tutorials/beginner/pytorch_with_examples.html）——从张量到 `nn.Module` 的官方学习路径
- PyTorch 性能调优指南（https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html）——涵盖混合精度、`DataLoader` 工作进程、锁页内存（pinned memory）及其他生产环境优化技巧
- Horace He，《Making Deep Learning Go Brrrr》（https://horace.io/brrr_intro.html）——解析 GPU 训练为何高效，并提供针对 PyTorch 的专属优化策略