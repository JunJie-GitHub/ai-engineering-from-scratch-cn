# 视频理解——时序建模

> 视频是一系列图像加上连接它们的物理规律。每个视频模型要么将时间视为额外的轴（3D 卷积（3D Conv）），要么视为需要注意力机制（Attention）处理的序列（Transformer），要么视为只需提取一次并进行池化（Pooling）的特征（2D+池化（2D+Pool））。

**类型：** 学习与实践
**语言：** Python
**前置知识：** 第 4 阶段第 03 课（卷积神经网络（CNNs））、第 4 阶段第 04 课（图像分类（Image Classification））
**时长：** 约 45 分钟

## 学习目标

- 区分三种主流的视频建模方法（2D+池化（2D+Pool）、3D 卷积（3D Conv）、时空 Transformer（Spatio-Temporal Transformer）），并预测它们在计算成本与准确率之间的权衡
- 在 PyTorch 中实现帧采样（Frame Sampling）、时序池化（Temporal Pooling）以及一个 2D+池化基线分类器（Baseline Classifier）
- 解释为何 I3D 的“膨胀”3D 卷积核（Inflated 3D Kernels）能够很好地迁移 ImageNet 权重，以及因式分解 (2+1)D 卷积（Factorised (2+1)D Conv）有何不同
- 熟悉标准的动作识别（Action Recognition）数据集与评估指标：Kinetics-400/600、UCF101、Something-Something V2；以及片段（Clip）级和视频（Video）级的 Top-1 准确率（Top-1 Accuracy）

## 问题背景

一段 30 秒、30 fps 的视频包含 900 张图像。直观来看，视频分类相当于将图像分类运行 900 次，然后进行某种形式的聚合。当动作在几乎每一帧中都清晰可见时（如体育、烹饪、健身视频），这种方法行之有效；但当动作本身由运动轨迹定义时，该方法就会严重失效：例如“将某物从左推到右”，在每一帧中看起来都只是两个静止的物体。

每个视频架构的核心问题在于：时序结构（Temporal Structure）在何时以及如何被建模？这一问题的答案决定了其他所有方面——计算成本、预训练策略（Pretraining Strategy）、能否复用 ImageNet 权重，以及模型训练所使用的数据集。

本课程的篇幅特意比静态图像课程更短。核心的图像处理机制已经就位，视频理解主要围绕“时序”展开：采样、建模与聚合。

## 核心概念

### 三大架构体系

flowchart LR
    V["Video clip<br/>(T frames)"] --> A1["2D + pool<br/>run 2D CNN per frame,<br/>average over time"]
    V --> A2["3D conv<br/>convolve over<br/>T x H x W"]
    V --> A3["Spatio-temporal<br/>transformer<br/>attention over<br/>(t, h, w) tokens"]

    A1 --> C["Logits"]
    A2 --> C
    A3 --> C

    style A1 fill:#dbeafe,stroke:#2563eb
    style A2 fill:#fef3c7,stroke:#d97706
    style A3 fill:#dcfce7,stroke:#16a34a

### 2D + 池化（2D + pool）

选取一个二维卷积神经网络（2D CNN，如 ResNet、EfficientNet 或 ViT）。对每一帧采样图像独立运行该网络。对逐帧提取的嵌入向量（embeddings）进行平均池化（或最大池化、注意力池化）。将池化后的向量输入分类器。

优点：
- ImageNet 预训练权重可直接迁移（transfer learning）。
- 实现最为简单。
- 计算成本低：仅需 T 帧 × 单张图像推理成本。

缺点：
- 无法建模运动信息。动作仅被视为外观特征的聚合。
- 时间池化具有顺序不变性；“开门”和“关门”在模型看来完全相同。

适用场景：外观特征主导的任务、在小型视频数据集上进行迁移学习、构建初始基线模型。

### 3D 卷积（3D convolutions）

将二维（H, W）卷积核替换为三维（T, H, W）卷积核。网络同时在空间和时间维度上进行卷积运算。早期代表架构：C3D、I3D、SlowFast。

I3D 技巧：加载预训练的二维 ImageNet 模型，通过沿新增的时间轴复制权重来“膨胀”（inflate）每个二维卷积核。一个 3x3 的二维卷积核会变为 3x3x3 的三维卷积核。这使得三维模型能够继承强大的预训练权重，而无需从零开始训练。

优点：
- 直接对运动信息进行建模。
- I3D 膨胀技术可直接实现迁移学习。

缺点：
- 浮点运算次数（FLOPs）约为对应二维模型的 T/8 倍（以时间核大小为 3 且堆叠 3 次为例）。
- 时间卷积核通常较小；建模长程运动需要金字塔结构或双流（dual-stream）架构。

适用场景：以运动信号为主的动作识别任务（如 Something-Something V2，或包含大量运动类别的 Kinetics 数据集）。

### 时空 Transformer（Spatio-temporal transformers）

将视频切分为时空图块（space-time patches）网格，并在所有图块上应用注意力机制（attention）。代表模型：TimeSformer、ViViT、Video Swin、VideoMAE。

关键的注意力模式：
- **联合注意力（Joint）** — 对 (t, h, w) 维度进行统一的大规模注意力计算。计算复杂度与 `T*H*W` 呈二次方关系，开销较大。
- **分离注意力（Divided）** — 每个模块包含两次注意力计算：一次沿时间维度，一次沿空间维度。计算复杂度近似线性扩展。
- **分解注意力（Factorised）** — 在不同模块间交替进行时间注意力与空间注意力计算。

优点：
- 在所有主流基准测试中均达到最先进（SOTA）精度。
- 可通过图块膨胀技术从图像 Transformer（如 ViT）迁移权重。
- 借助稀疏注意力（sparse attention）机制支持长上下文视频处理。

缺点：
- 计算资源消耗大。
- 需谨慎选择注意力模式，否则运行时间会急剧膨胀。

适用场景：大规模数据集、高保真视频理解、视频+文本多模态任务。

### 帧采样（Frame sampling）

一段 10 秒、30 fps 的视频片段包含 300 帧；将全部 300 帧输入任何模型都是资源浪费。标准策略如下：

- **均匀采样（Uniform sampling）** — 在视频片段中均匀抽取 T 帧。2D+池化架构的默认选择。
- **密集采样（Dense sampling）** — 随机截取连续的 T 帧窗口。常用于 3D 卷积，因为运动建模依赖相邻帧。
- **多片段采样（Multi-clip）** — 从同一视频中采样多个 T 帧窗口，分别进行分类，在测试阶段对预测结果取平均。

T 通常取 8、16、32 或 64。T 值越大，时间信号越丰富，但计算开销也相应增加。

### 评估方法（Evaluation）

两个评估层级：
- **片段级准确率（Clip-level accuracy）** — 模型输入单个 T 帧片段，输出 Top-k 预测结果。
- **视频级准确率（Video-level accuracy）** — 对同一视频的多个片段级预测结果取平均；该指标通常更高且更稳定。

务必同时报告这两项指标。若某模型片段级得分为 78%、视频级为 82%，说明其高度依赖测试时平均（test-time averaging）；若得分为 80% / 81%，则说明其单片段鲁棒性更强。

### 常见数据集

- **Kinetics-400 / 600 / 700** — 通用动作识别数据集。包含约 40 万个视频片段；提供 YouTube 链接（其中许多现已失效）。
- **Something-Something V2** — 以运动定义动作的数据集（如“将 X 从左向右移动”）。无法通过 2D+池化架构解决。
- **UCF-101**、**HMDB-51** — 较早期、规模较小，但论文中仍常被引用。
- **AVA** — 时空动作定位（action localisation）数据集；难度高于单纯的动作分类。

## 构建

### 步骤 1：帧采样器 (Frame Sampler)

均匀采样器 (Uniform Sampler) 和密集采样器 (Dense Sampler)，适用于帧列表或视频张量 (Video Tensor)。

import numpy as np

def sample_uniform(num_frames_total, T):
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    step = num_frames_total / T
    return [int(i * step) for i in range(T)]


def sample_dense(num_frames_total, T, rng=None):
    rng = rng or np.random.default_rng()
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    start = int(rng.integers(0, num_frames_total - T + 1))
    return list(range(start, start + T))

两者均返回 `T` 个索引，用于对视频张量进行切片。

### 步骤 2：2D+池化基线模型 (2D+Pool Baseline)

对每一帧运行 2D ResNet-18，对特征进行平均池化 (Average Pooling)，然后进行分类。

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FramePool(nn.Module):
    def __init__(self, num_classes=400, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*(list(backbone.children())[:-1]))  # global avg pool kept
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (N, T, 3, H, W)
        N, T = x.shape[:2]
        x = x.view(N * T, *x.shape[2:])
        feats = self.features(x).view(N, T, -1)
        pooled = feats.mean(dim=1)
        return self.head(pooled)

model = FramePool(num_classes=10)
x = torch.randn(2, 8, 3, 224, 224)
print(f"output: {model(x).shape}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

该模型包含一千一百万个参数，基于 ImageNet 预训练，逐帧运行、池化并分类。在外观主导型任务上，该基线模型的性能通常与标准的 3D 模型相差仅 5-10 个百分点，有时甚至更优，因为它复用了更强大的 ImageNet 主干网络 (Backbone)。

### 步骤 3：I3D 风格的膨胀 3D 卷积 (Inflated 3D Conv)

通过沿新增的时间轴重复权重，将单个 2D 卷积 (2D Conv) 转换为 3D 卷积 (3D Conv)。

def inflate_2d_to_3d(conv2d, time_kernel=3):
    out_c, in_c, kh, kw = conv2d.weight.shape
    weight_3d = conv2d.weight.data.unsqueeze(2)  # (out, in, 1, kh, kw)
    weight_3d = weight_3d.repeat(1, 1, time_kernel, 1, 1) / time_kernel
    conv3d = nn.Conv3d(in_c, out_c, kernel_size=(time_kernel, kh, kw),
                        padding=(time_kernel // 2, conv2d.padding[0], conv2d.padding[1]),
                        stride=(1, conv2d.stride[0], conv2d.stride[1]),
                        bias=False)
    conv3d.weight.data = weight_3d
    return conv3d

conv2d = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
conv3d = inflate_2d_to_3d(conv2d, time_kernel=3)
print(f"2D weight shape:  {tuple(conv2d.weight.shape)}")
print(f"3D weight shape:  {tuple(conv3d.weight.shape)}")
x = torch.randn(1, 3, 8, 56, 56)
print(f"3D output shape:  {tuple(conv3d(x).shape)}")

除以 `time_kernel` 可保持激活值 (Activation) 的幅度大致恒定——这对于在首次前向传播时不破坏批归一化 (Batch Normalization) 的统计量至关重要。

### 步骤 4：分解式 (2+1)D 卷积 (Factorised (2+1)D Conv)

将 3D 卷积拆分为 2D（空间）卷积和 1D（时间）卷积。在保持相同感受野 (Receptive Field) 的同时减少参数量，并在部分基准测试中取得更高的准确率。

class Conv2Plus1D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        mid_c = (in_c * out_c * kernel_size * kernel_size * kernel_size) \
                // (in_c * kernel_size * kernel_size + out_c * kernel_size)
        self.spatial = nn.Conv3d(in_c, mid_c, kernel_size=(1, kernel_size, kernel_size),
                                 padding=(0, kernel_size // 2, kernel_size // 2), bias=False)
        self.bn = nn.BatchNorm3d(mid_c)
        self.act = nn.ReLU(inplace=True)
        self.temporal = nn.Conv3d(mid_c, out_c, kernel_size=(kernel_size, 1, 1),
                                  padding=(kernel_size // 2, 0, 0), bias=False)

    def forward(self, x):
        return self.temporal(self.act(self.bn(self.spatial(x))))

c = Conv2Plus1D(3, 64)
x = torch.randn(1, 3, 8, 56, 56)
print(f"(2+1)D output: {tuple(c(x).shape)}")

完整的 R(2+1)D 网络等同于将 ResNet-18 中的每个 3x3 卷积替换为 `Conv2Plus1D`。

## 使用方式

以下两个库涵盖了生产环境中的视频处理工作：

- `torchvision.models.video` — 提供 R(2+1)D、MViT、Swin3D 模型，并附带在 Kinetics 数据集上预训练的权重。其 API 与图像模型保持一致。
- `pytorchvideo`（Meta 出品）— 包含模型库（Model Zoo）、针对 Kinetics / SSv2 / AVA 数据集的数据加载器（Data Loaders）以及标准数据变换（Transforms）。

对于视觉-语言视频模型（Vision-Language Video Models，如视频描述生成 Video Captioning、视频问答 Video QA），请使用 `transformers` 库（支持 `VideoMAE`、`VideoLLaMA`、`InternVideo` 等）。

## 交付成果

本课程的产出物包括：

- `outputs/prompt-video-architecture-picker.md` — 一个提示词（Prompt），可根据外观与运动特征的侧重、数据集规模及计算预算，自动选择 2D+池化（2D+pool）/ I3D / (2+1)D / Transformer 架构。
- `outputs/skill-frame-sampler-auditor.md` — 一项技能（Skill），用于检查视频流水线（Pipeline）中的帧采样器（Sampler），并标记常见缺陷：索引差一错误（Off-by-one Index）、当 `num_frames < T` 时采样不均匀、缺乏保持宽高比的裁剪（Aspect-preserving Crop）等。

## 练习

1. **（简单）** 计算 T=8 时 FramePool 与 I3D 风格 3D ResNet（T=8）的近似浮点运算次数（FLOPs）。论证为何 2D+池化（2D+pool）的计算成本低 3-5 倍。
2. **（中等）** 生成一个合成视频数据集：包含沿随机方向运动的随机球体，并按运动方向标注（“从左到右”、“从右到左”、“对角向上”）。在该数据集上训练 FramePool。展示其准确率接近随机猜测水平，从而证明仅凭外观特征不足以完成运动识别任务。
3. **（困难）** 构建一个 R(2+1)D-18 模型：将 ResNet-18 中的每个 `Conv2d` 替换为 `Conv2Plus1D`。使用在 ImageNet 上预训练的 ResNet-18 权重，对第一个卷积层的权重进行膨胀初始化（Inflation）。在练习 2 的运动数据集上进行训练，并使其性能超越 FramePool。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 2D + 池化 (2D + pool) | “逐帧分类器” | 对每个采样帧运行 2D 卷积神经网络（CNN），在时间维度上对特征进行平均池化（Average-pool），最后进行分类 |
| 3D 卷积 (3D convolution) | “时空卷积核” | 在 (T, H, W) 维度上进行卷积操作的卷积核；能够原生地对运动特征进行建模 |
| 权重膨胀 (Inflation) | “将 2D 权重提升至 3D” | 通过将 2D 卷积权重沿新增的时间轴重复来初始化 3D 卷积权重，随后除以时间维度卷积核大小（kernel_T）以保持激活值尺度不变 |
| (2+1)D 卷积 | “分解卷积” | 将 3D 卷积分解为 2D 空间卷积 + 1D 时间卷积；参数量更少，且两者之间引入了额外的非线性激活 |
| 分离注意力 (Divided attention) | “先时间后空间” | Transformer 模块中每层包含两个注意力机制：一个针对同一帧内的 Token，另一个针对同一空间位置跨帧的 Token |
| 视频片段 (Clip) | “T 帧窗口” | 采样得到的 T 帧连续子序列；视频模型处理的基本单元 |
| 片段准确率 vs 视频准确率 (Clip vs video accuracy) | “两种评估设置” | 片段准确率 = 每个视频仅采样一次进行评估；视频准确率 = 对多个采样片段的预测结果取平均 |
| Kinetics 数据集 | “视频领域的 ImageNet” | 包含 400-700 个动作类别、30 万+ YouTube 视频片段，是标准的视频预训练语料库 |

## 扩展阅读

- [I3D: Quo Vadis, Action Recognition (Carreira & Zisserman, 2017)](https://arxiv.org/abs/1705.07750) — 引入了膨胀（inflation）技术与 Kinetics 数据集
- [R(2+1)D: A Closer Look at Spatiotemporal Convolutions (Tran et al., 2018)](https://arxiv.org/abs/1711.11248) — 采用分解卷积（factorised conv），至今仍是强有力的基线（baseline）
- [TimeSformer: Is Space-Time Attention All You Need? (Bertasius et al., 2021)](https://arxiv.org/abs/2102.05095) — 首个表现优异的视频 Transformer（Transformer）模型
- [VideoMAE (Tong et al., 2022)](https://arxiv.org/abs/2203.12602) — 面向视频的掩码自编码器（Masked Autoencoder）预训练；当前主流的预训练方案（pretraining recipe）