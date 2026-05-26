# 卷积神经网络（CNNs）—— 从 LeNet 到 ResNet

> 过去三十年里，每一个主流的卷积神经网络（CNN）都遵循着相同的“卷积–非线性–下采样”（conv–nonlinearity–downsample）配方，只是在此基础上附加了一个新想法。请按顺序学习这些核心思想。

**Type:** 学习 + 实践
**Languages:** Python
**Prerequisites:** 第3阶段第11课（PyTorch）、第4阶段第01课（图像基础）、第4阶段第02课（从零实现卷积）
**Time:** 约75分钟

## 学习目标

- 梳理 LeNet-5 -> AlexNet -> VGG -> Inception -> ResNet 的架构演进脉络，并阐明每个系列所贡献的单一核心创新点
- 使用 PyTorch 实现 LeNet-5、VGG 风格模块（block）以及 ResNet 的 BasicBlock，每个实现均控制在 40 行代码以内
- 解释残差连接（residual connections）为何能将一个原本无法训练的 1000 层网络转化为业界领先（state-of-the-art）的模型
- 阅读现代骨干网络（backbone）（如 ResNet-18、ResNet-50）的代码，并在查看源码前预测其输出形状（output shape）、感受野（receptive field）和参数量（parameter count）

## 问题背景

2011 年，表现最佳的 ImageNet 分类器在 Top-5 准确率（top-5 accuracy）上仅达到约 74%。2012 年，AlexNet 将其提升至 85%。到了 2015 年，ResNet 更是达到了 96%。这期间没有引入新数据，也没有新一代 GPU 的加持。性能的提升完全源于架构层面的创新。作为一名一线计算机视觉工程师，必须清楚每个想法源自哪篇论文，因为你在 2026 年部署到生产环境的每一个骨干网络（backbone），都是对这些经典模块的重新组合。此外，这些思想还在不断跨界迁移：分组卷积（grouped convolutions）从 CNN 延伸到了 Transformer，残差连接（residual connections）从 ResNet 普及到了现有的每一个大语言模型（large language model, LLM），而批归一化（batch normalisation）则活跃在扩散模型（diffusion models）中。

按顺序研究这些网络还能让你避免一个常见误区：在 LeNet 规模的网络就能解决问题时，却盲目追求市面上最大的模型。MNIST 数据集根本不需要 ResNet。了解每个模型家族的扩展曲线（scaling curve），能帮你准确判断在何种场景下该选择何种规模的模型。

## 核心概念

### 改变计算机视觉（Computer Vision）的四大理念

timeline
    title Four ideas, four families
    1998 : LeNet-5 : Conv + pool + FC for digits, trained on CPU, 60k params
    2012 : AlexNet : Deeper + ReLU + dropout + two GPUs, won ImageNet by 10 points
    2014 : VGG / Inception : 3x3 stacks (VGG), parallel filter sizes (Inception)
    2015 : ResNet : Identity skip connections unlock 100+ layer training

在经典视觉领域，没有任何其他进展能比这四次飞跃更为重要。

### LeNet-5（1998）

由 Yann LeCun 提出的数字识别器。包含 60,000 个参数。采用两个卷积-池化块、两个全连接层（fully connected layers）以及 tanh 激活函数。它确立了所有卷积神经网络（Convolutional Neural Network, CNN）所继承的基础模板：

input (1, 32, 32)
  conv 5x5 -> (6, 28, 28)
  avg pool 2x2 -> (6, 14, 14)
  conv 5x5 -> (16, 10, 10)
  avg pool 2x2 -> (16, 5, 5)
  flatten -> 400
  dense -> 120
  dense -> 84
  dense -> 10

现代语境下被称为 CNN 的所有架构——即交替的卷积层与下采样（downsampling）连接至一个小型分类器头（classifier head）——本质上都是增加了更多层数、更大通道数以及更优激活函数的 LeNet。

### AlexNet（2012）

三项变革共同攻克了 ImageNet 数据集：

1. **ReLU** 替代 tanh。梯度消失（gradient vanishing）问题得以缓解，训练速度提升了六倍。
2. 在全连接头中引入 **Dropout**。正则化（regularization）从此成为网络的一个标准层，而非一种临时技巧。
3. **深度与宽度**。包含五个卷积层、三个密集层，参数量达 6000 万，模型被拆分至两块 GPU 上进行训练。

该论文的图 2 依然将 GPU 拆分展示为两条并行数据流。这种并行设计仅是应对硬件限制的权宜之计，而非架构层面的深刻洞见——但上述三大理念至今仍存在于你使用的每一个模型中。

### VGG（2014）

VGG 提出了一个问题：如果仅使用 3x3 卷积核并不断加深网络，会发生什么？

stack:   conv 3x3 -> conv 3x3 -> pool 2x2
repeat:  16 or 19 conv layers

两个连续的 3x3 卷积层与单个 5x5 卷积层具有相同的感受野（receptive field），但参数量更少（2*9*C^2 = 18C^2 对比 25*C^2），且中间多了一个 ReLU 激活函数。VGG 将这一观察转化为完整的网络架构。其极简的设计——仅重复使用单一类型的模块——使其成为后续所有架构的基准参考。

代价：1.38 亿参数，训练缓慢，推理成本高昂。

### Inception（2014，同年）

针对“我该使用多大的卷积核尺寸（kernel size）？”这一问题，Google 给出的答案是：全部使用，并行处理。

flowchart LR
    IN["Input feature map"] --> A["1x1 conv"]
    IN --> B["3x3 conv"]
    IN --> C["5x5 conv"]
    IN --> D["3x3 max pool"]
    A --> CAT["Concatenate<br/>along channel axis"]
    B --> CAT
    C --> CAT
    D --> CAT
    CAT --> OUT["Next block"]

    style IN fill:#dbeafe,stroke:#2563eb
    style CAT fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a

每个分支各司其职：1x1 用于通道混合（channel mixing），3x3 用于提取局部纹理，5x5 用于捕捉更大范围的模式，池化层用于提取平移不变特征（shift-invariant features）。随后的拼接操作（concat）允许下一层自主选择最有用的分支特征。Inception v1 在每个分支内部使用 1x1 卷积作为瓶颈层（bottleneck），以将参数量控制在合理范围内。

### 退化问题（Degradation Problem）

到了 2015 年，VGG-19 表现良好，但 VGG-32 却失效了。理论上加深网络应能提升性能，但当层数超过约 20 层后，训练损失和测试损失反而双双恶化。这并非过拟合（overfitting），而是优化器（optimizer）无法找到有效权重，因为梯度在逐层传递过程中发生了连乘衰减。

Plain deep network:
  y = f_L( f_{L-1}( ... f_1(x) ... ) )

Gradient wrt early layer:
  dL/dW_1 = dL/dy * df_L/df_{L-1} * ... * df_2/df_1 * df_1/dW_1

Each multiplicative term has magnitude roughly (weight magnitude) * (activation gain).
Stack 100 of them with gains < 1 and the gradient is effectively zero.

VGG 在 19 层时能够正常工作，得益于同期发表的批归一化（Batch Normalization）技术有效维持了激活值的尺度。但即便如此，批归一化也无法挽救超过 30 层左右的极深网络。

### ResNet（2015）

何恺明、张祥雨、任少卿、孙剑提出了一项改变一切的改进：

standard block:   y = F(x)
residual block:   y = F(x) + x

其中的 `+ x` 意味着该层始终可以通过将 `F(x)` 趋近于零来选择“什么都不做”。如今，一个 1000 层的 ResNet 最差情况也不过等同于一个单层网络，因为每个额外的残差块（residual block）都提供了一个简单的“逃生通道”。有了这一保证，优化器便愿意让每个残差块都发挥*一点*作用——而将这种微小的作用叠加 100 次，便足以达到当时的最先进水平（state-of-the-art）。

flowchart LR
    X["Input x"] --> F["F(x)<br/>conv + BN + ReLU<br/>conv + BN"]
    X -.->|identity skip| PLUS(["+"])
    F --> PLUS
    PLUS --> RELU["ReLU"]
    RELU --> OUT["y"]

    style X fill:#dbeafe,stroke:#2563eb
    style PLUS fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a

该模块的两种变体在各类架构中随处可见：

- **BasicBlock**（ResNet-18、ResNet-34）：包含两个 3x3 卷积层，跳跃连接（skip connection）跨越这两个卷积层。
- **Bottleneck**（ResNet-50、-101、-152）：采用 1x1 降维、3x3 中间卷积、1x1 升维的结构，跳跃连接跨越这三个模块。在通道数较高时计算成本更低。

当跳跃连接需要跨越下采样操作（步长 stride=2）时，恒等映射路径会被替换为步长为 2 的 1x1 卷积，以匹配特征图形状。

### 残差连接为何在计算机视觉之外同样重要

这一理念的核心并非仅仅为了图像分类。它的真正意义在于，将深度网络从“祈祷梯度能够顺利传递”的玄学，转变为可靠且可扩展的工程化工具。在下一阶段你将读到的每一个 Transformer 架构中，每个模块都采用了完全相同的跳跃连接。没有 ResNet，就不会有 GPT。

## 构建

### 步骤 1：LeNet-5

一个精简且忠实还原的 LeNet 模型。采用 Tanh 激活函数 (Tanh activation) 和平均池化 (average pooling)。唯一向现代实践妥协的地方是，我们在下游使用了 `nn.CrossEntropyLoss`，而非原始的高斯连接 (Gaussian connections)。

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

net = LeNet5()
x = torch.randn(1, 1, 32, 32)
print(f"output: {net(x).shape}")
print(f"params: {sum(p.numel() for p in net.parameters()):,}")

预期输出：`output: torch.Size([1, 10])`，`params: 61,706`。这就是开启现代计算机视觉 (computer vision) 时代的完整数字分类器。

### 步骤 2：VGG 模块 (VGG block)

一个可复用的模块：包含两个 3x3 卷积层 (convolutional layers)、ReLU 激活函数 (ReLU activation)、批归一化 (batch normalization) 和最大池化 (max pooling)。

class VGGBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.pool(x)

class MiniVGG(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stack = nn.Sequential(
            VGGBlock(3, 32),
            VGGBlock(32, 64),
            VGGBlock(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.stack(x))

net = MiniVGG()
x = torch.randn(1, 3, 32, 32)
print(f"output: {net(x).shape}")
print(f"params: {sum(p.numel() for p in net.parameters()):,}")

在 CIFAR 尺寸输入上堆叠三个 VGG 模块，配合一个自适应池化层 (adaptive pooling layer) 和一个线性层 (linear layer)。参数量约为 29 万。对于 CIFAR-10 数据集来说绰绰有余。

### 步骤 3：ResNet 基础模块 (ResNet BasicBlock)

这是 ResNet-18 和 ResNet-34 的核心构建单元。

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)

卷积层设置 `bias=False` 是批归一化的常规做法——因为 BN 的 beta 参数已经处理了偏置 (bias)，同时保留卷积偏置纯属浪费。`shortcut`（捷径连接）仅在步幅 (stride) 或通道数 (channel count) 发生变化时才需要真实的卷积操作；否则，它只是一个无操作的恒等映射 (identity mapping)。

### 步骤 4：微型 ResNet

堆叠四组 BasicBlock，即可得到一个适用于 CIFAR 尺寸输入的可用 ResNet 模型。

class TinyResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_group(32, 32, num_blocks=2, stride=1)
        self.layer2 = self._make_group(32, 64, num_blocks=2, stride=2)
        self.layer3 = self._make_group(64, 128, num_blocks=2, stride=2)
        self.layer4 = self._make_group(128, 256, num_blocks=2, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def _make_group(self, in_c, out_c, num_blocks, stride):
        blocks = [BasicBlock(in_c, out_c, stride=stride)]
        for _ in range(num_blocks - 1):
            blocks.append(BasicBlock(out_c, out_c, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)

net = TinyResNet()
x = torch.randn(1, 3, 32, 32)
print(f"output: {net(x).shape}")
print(f"params: {sum(p.numel() for p in net.parameters()):,}")

共四组，每组包含两个模块。在第 2、3、4 组的起始处使用步幅为 2 的卷积。每次下采样 (downsampling) 时通道数翻倍。参数量约为 280 万。这正是能够平滑扩展至 ResNet-152 的标准架构配方。

### 步骤 5：比较参数与特征效率 (parameter-to-feature efficiency)

将相同的输入依次通过这三个网络，并对比它们的参数量。

def summary(name, net, x):
    y = net(x)
    params = sum(p.numel() for p in net.parameters())
    print(f"{name:12s}  input {tuple(x.shape)} -> output {tuple(y.shape)}  params {params:>10,}")

x = torch.randn(1, 3, 32, 32)
summary("LeNet5",     LeNet5(),       torch.randn(1, 1, 32, 32))
summary("MiniVGG",    MiniVGG(),      x)
summary("TinyResNet", TinyResNet(),   x)

三个模型，代表三个时代，参数量相差三个数量级。在 CIFAR-10 数据集上的准确率方面，经过几个训练周期 (epochs) 后大致为：LeNet 60%，MiniVGG 89%，TinyResNet 93%。

## 上手使用

`torchvision.models` 提供了上述所有模型的预训练版本（Pretrained Versions）。不同模型族的调用签名（Call Signature）完全一致，这正是主干网络抽象（Backbone Abstraction）设计的核心目的。

from torchvision.models import resnet18, ResNet18_Weights, vgg16, VGG16_Weights

r18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
r18.eval()

print(f"ResNet-18 params: {sum(p.numel() for p in r18.parameters()):,}")
print(r18.layer1[0])
print()

v16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
v16.eval()
print(f"VGG-16   params: {sum(p.numel() for p in v16.parameters()):,}")

ResNet-18 拥有 1170 万参数，而 VGG-16 高达 1.38 亿。两者在 ImageNet 上的 Top-1 准确率（Top-1 Accuracy）却非常接近（69.8% 对比 71.6%）。残差连接（Residual Connections）为你带来了 12 倍的参数效率优势。这就是为什么从 2016 年到 2021 年视觉 Transformer（Vision Transformer, ViT）问世之前，ResNet 变体一直占据主导地位——并且在计算资源受限的实际部署场景中，它们至今仍是主流选择。

对于迁移学习（Transfer Learning），标准流程始终如一：加载预训练权重、冻结主干网络、替换分类头（Classifier Head）。

for p in r18.parameters():
    p.requires_grad = False
r18.fc = nn.Linear(r18.fc.in_features, 10)

仅需三行代码。你现在就拥有了一个用于 10 分类 CIFAR 数据集的分类器，它直接继承了 ImageNet 预训练所学习到的特征表示（Feature Representations）。

## 交付产出

本节内容将生成：

- `outputs/prompt-backbone-selector.md` — 一个提示词（Prompt），可根据任务类型、数据集规模和计算预算，自动选择合适的卷积神经网络（Convolutional Neural Network, CNN）家族（LeNet/VGG/ResNet/MobileNet/ConvNeXt）。
- `outputs/skill-residual-block-reviewer.md` — 一项技能脚本，用于读取 PyTorch 模块并标记跳跃连接（Skip Connection）错误（如步长变化时缺失捷径连接、捷径连接的激活顺序错误、批归一化（Batch Normalization, BN）层相对于加法操作的位置不当）。

## 练习

1. **(简单)** 手动逐层计算 `TinyResNet` 的参数量，并与 `sum(p.numel() for p in net.parameters())` 的结果进行对比。参数预算主要消耗在何处——卷积层（Convolutions）、批归一化层，还是分类头？
2. **(中等)** 实现瓶颈块（Bottleneck Block，结构为 1x1 -> 3x3 -> 1x1 并包含跳跃连接），并使用它构建一个适用于 CIFAR 数据集的 ResNet-50 风格网络。将其参数量与 `TinyResNet` 进行对比。
3. **(困难)** 移除 `BasicBlock` 中的跳跃连接，分别在 CIFAR-10 数据集上训练一个 34 层的“普通”网络和一个 34 层的 ResNet，各训练 10 个轮次（Epochs）。绘制两者的训练损失（Training Loss）随轮次变化的曲线。复现 He 等人论文中图 1 的结果：即深层普通网络收敛到的损失值反而高于其较浅的对应网络。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 主干网络 (Backbone) | “模型主体” | 产生特征图 (Feature map) 并馈送至任务头 (Task head) 的卷积块堆叠结构 |
| 残差连接 (Residual connection) | “跳跃连接 (Skip connection)” | `y = F(x) + x`；通过将 F 设为零，使优化器 (Optimizer) 能够学习恒等映射 (Identity mapping)，从而让任意深度的网络均可训练 |
| 基础块 (BasicBlock) | “带跳跃连接的两个 3x3 卷积” | ResNet-18/34 的构建单元：conv-BN-ReLU-conv-BN-add-ReLU |
| 瓶颈块 (Bottleneck) | “1x1 降维，3x3 卷积，1x1 升维” | ResNet-50/101/152 的构建单元；在高通道数时计算成本较低，因为 3x3 卷积在缩减后的通道宽度上运行 |
| 退化问题 (Degradation problem) | “网络越深效果越差” | 当普通卷积层超过约 20 层时，训练误差和测试误差均会上升；该问题通过残差连接解决，而非依赖更多数据 |
| 茎部网络 (Stem) | “第一层” | 将 3 通道输入转换为基础特征宽度的初始卷积层；在 ImageNet 上通常为 7x7 步长 2，在 CIFAR 上通常为 3x3 步长 1 |
| 头部网络 (Head) | “分类器” | 位于主干网络最后一个模块之后的层：自适应池化 (Adaptive pooling)、展平 (Flatten)、全连接层 (Linear layer(s)) |
| 迁移学习 (Transfer learning) | “预训练权重 (Pretrained weights)” | 加载在 ImageNet 上训练好的主干网络，并仅针对你的任务微调 (Fine-tuning) 头部网络 |

## 进一步阅读

- [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385) — ResNet 论文；其中的每一张图都值得仔细研读
- [Very Deep Convolutional Networks (Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556) — VGG 论文；至今仍是解释“为何使用 3x3 卷积”的最佳参考文献
- [ImageNet Classification with Deep CNNs (Krizhevsky et al., 2012)](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) — AlexNet 论文；终结了手工特征时代的开山之作
- [Going Deeper with Convolutions (Szegedy et al., 2014)](https://arxiv.org/abs/1409.4842) — Inception v1 论文；其并行滤波器思想至今仍可见于视觉 Transformer (Vision Transformer) 中