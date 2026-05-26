# 语义分割（Semantic Segmentation）— U-Net

> 分割本质上是对每个像素进行分类。U-Net 通过将下采样编码器（downsampling encoder）与上采样解码器（upsampling decoder）相结合，并在两者之间建立跳跃连接（skip connections），成功实现了这一架构。

**类型：** 构建
**语言：** Python
**前置知识：** 第4阶段第03课（卷积神经网络 CNNs）、第4阶段第04课（图像分类 Image Classification）
**时长：** 约75分钟

## 学习目标

- 区分语义分割（semantic segmentation）、实例分割（instance segmentation）和全景分割（panoptic segmentation），并为特定问题选择合适的任务
- 在 PyTorch 中从零构建 U-Net，包含编码器模块（encoder blocks）、瓶颈层（bottleneck）、使用转置卷积（transposed convolutions）的解码器以及跳跃连接
- 实现逐像素交叉熵（pixel-wise cross-entropy）、Dice 损失（Dice loss）以及两者的组合损失，该组合损失目前是医疗和工业分割领域的默认标准
- 读取各类别的交并比（IoU）和 Dice 指标，并诊断低分是源于小目标召回率（recall）、边界精度（boundary accuracy）不足，还是类别不平衡（class imbalance）

## 问题背景

分类任务为每张图像输出一个标签，检测任务为每张图像输出少量边界框，而分割任务则为每个像素输出一个标签。对于尺寸为 `H x W` 的输入，其输出是一个形状为 `H x W`（语义分割）或 `H x W x N_instances`（实例分割）的张量。这意味着每张图像需要进行数百万次预测，而非仅仅一次。

正是这种结构使得分割技术成为几乎所有密集预测（dense-prediction）视觉产品的核心：医学影像（肿瘤掩膜）、自动驾驶（道路、车道线、障碍物）、卫星遥感（建筑轮廓、作物边界）、文档解析（版面区域）以及机器人技术（可抓取区域）。这些任务都无法仅通过给物体画框来解决，它们需要精确的轮廓（silhouette）。

该架构面临的问题表述简单，但解决起来却不易：网络需要同时捕捉图像的全局上下文（global context，即场景类型）与局部像素细节（local pixel detail，即精确区分道路与人行道像素）。传统的卷积神经网络（CNN）通过空间压缩来获取上下文，但这会丢失细节。而 U-Net 的设计则成功兼顾了两者。

## 核心概念

### 语义分割 vs 实例分割 vs 全景分割

flowchart LR
    IN["Input image"] --> SEM["Semantic<br/>(pixel → class)"]
    IN --> INS["Instance<br/>(pixel → object id,<br/>only foreground classes)"]
    IN --> PAN["Panoptic<br/>(every pixel → class + id)"]

    style SEM fill:#dbeafe,stroke:#2563eb
    style INS fill:#fef3c7,stroke:#d97706
    style PAN fill:#dcfce7,stroke:#16a34a

- **语义分割 (Semantic Segmentation)** 表示“这个像素是道路，那个像素是汽车”。相邻的两辆车会合并成一个单一的色块。
- **实例分割 (Instance Segmentation)** 表示“这个像素是3号车，那个像素是5号车”。它会忽略背景中的“_stuff_”（即天空、道路、草地等无固定形状的类别）。
- **全景分割 (Panoptic Segmentation)** 将两者统一：每个像素都获得一个类别标签，每个实例都分配一个唯一ID，同时分割“_stuff_”（背景）和“_things_”（前景物体）。

本节课程主要讲解语义分割。下一节课程（Mask R-CNN）将讲解实例分割。

### U-Net 的网络结构

flowchart LR
    subgraph ENC["Encoder (contracting)"]
        E1["64<br/>H x W"] --> E2["128<br/>H/2 x W/2"]
        E2 --> E3["256<br/>H/4 x W/4"]
        E3 --> E4["512<br/>H/8 x W/8"]
    end
    subgraph BOT["Bottleneck"]
        B1["1024<br/>H/16 x W/16"]
    end
    subgraph DEC["Decoder (expanding)"]
        D4["512<br/>H/8 x W/8"] --> D3["256<br/>H/4 x W/4"]
        D3 --> D2["128<br/>H/2 x W/2"]
        D2 --> D1["64<br/>H x W"]
    end
    E4 --> B1 --> D4
    E1 -. skip .-> D1
    E2 -. skip .-> D2
    E3 -. skip .-> D3
    E4 -. skip .-> D4
    D1 --> OUT["1x1 conv<br/>classes"]

    style ENC fill:#dbeafe,stroke:#2563eb
    style BOT fill:#fef3c7,stroke:#d97706
    style DEC fill:#dcfce7,stroke:#16a34a

编码器 (Encoder) 会将空间分辨率减半四次，同时将通道数翻倍。解码器 (Decoder) 则执行相反操作：将空间分辨率翻倍四次，同时将通道数减半。跳跃连接 (Skip Connections) 会在每个分辨率层级上，将编码器对应的特征图与解码器的特征图进行拼接。最后的 1x1 卷积层会在完整分辨率下将通道数从 `64` 映射到 `num_classes`。

为什么跳跃连接是必要的：当解码器尝试输出像素级预测时，它只能看到尺寸很小的特征图。如果没有跳跃连接，它将无法精确定位边缘，因为这些细节信息在编码器中已被压缩。跳跃连接将编码器在下采样过程中计算出的高分辨率特征图直接传递给解码器。

### 转置卷积与双线性上采样

解码器需要扩大空间维度。通常有两种选择：

- **转置卷积 (Transposed Convolution)** (`nn.ConvTranspose2d`) —— 可学习的上采样方式。早期 U-Net 的默认选择。如果步长 (stride) 和卷积核大小 (kernel size) 不能整除，可能会产生棋盘格伪影 (checkerboard artifacts)。
- **双线性上采样 (Bilinear Upsample) + 3x3 卷积** —— 先进行平滑上采样，再接一个卷积层。伪影更少，参数量更小，现已成为现代架构的默认选择。

这两种方法在实际项目中都很常见。对于初次构建 U-Net 而言，双线性上采样是更稳妥的选择。

### 像素网格上的交叉熵损失

对于包含 C 个类别的语义分割任务，模型输出形状为 `(N, C, H, W)`。目标标签形状为 `(N, H, W)`，包含整数类别 ID。交叉熵损失 (Cross-Entropy Loss) 的计算方式与图像分类完全相同，只是将其应用于每一个空间位置：

Loss = mean over (n, h, w) of -log( softmax(logits[n, :, h, w])[target[n, h, w]] )

PyTorch 中的 `F.cross_entropy` 原生支持这种张量形状，无需进行重塑 (reshape)。

### Dice 损失及其必要性

交叉熵损失会平等对待每一个像素。当某一类别在画面中占据绝对主导时（例如医学影像中：99% 为背景，1% 为肿瘤），这种做法就会失效。网络只需将所有像素都预测为背景，就能获得 99% 的准确率，但模型实际上毫无用处。

Dice 损失 (Dice Loss) 通过直接优化预测掩码 (Mask) 与真实掩码之间的重叠区域来解决这一问题：

Dice(p, y) = 2 * sum(p * y) / (sum(p) + sum(y) + epsilon)
Dice_loss = 1 - Dice

其中 `p` 是某一类别的 sigmoid/softmax 概率图，`y` 是二值化的真实掩码。仅当重叠完全匹配时，损失才为零。由于它是基于比率计算的，因此类别不平衡问题对其没有影响。

在实际应用中，通常使用**组合损失 (Combined Loss)**：

L = L_cross_entropy + lambda * L_dice       (lambda ~ 1)

交叉熵损失在训练初期能提供稳定的梯度；而 Dice 损失则在训练后期专注于让模型真正拟合掩码的形状。这种组合是医学影像领域的默认标准，在任何类别不平衡的数据集上都极难被超越。

### 评估指标

- **像素准确率 (Pixel Accuracy)** —— 预测正确的像素百分比。计算成本低。但在不平衡数据上会失效，原因与分类任务中的准确率指标相同。
- **各类别 IoU (交并比, Intersection over Union)** —— 计算每个类别掩码的交并比；对所有类别取平均值即为 mIoU (Mean IoU)。
- **Dice 系数 (像素级 F1 分数)** —— 与 IoU 类似；`Dice = 2 * IoU / (1 + IoU)`。医学影像领域更偏好 Dice，而自动驾驶社区更偏好 IoU；两者呈单调相关关系。
- **边界 F1 (Boundary F1)** —— 衡量预测边界与真实边界的接近程度，即使微小的偏移也会受到惩罚。对于半导体检测等高精度任务至关重要。

报告评估结果时应列出每个类别的 IoU，而不仅仅是 mIoU。当九个类别的 IoU 达到 85% 时，mIoU 会掩盖掉某个仅 15% 的类别表现。

### 输入分辨率的权衡

U-Net 的编码器会将分辨率减半四次，因此输入尺寸必须能被 16 整除。医学图像通常为 512x512 或 1024x1024。自动驾驶的裁剪图像常为 2048x1024。U-Net 的显存开销与 `H * W * C_max` 成正比，在 1024x1024 分辨率且瓶颈层通道数为 1024 的情况下，仅前向传播就会消耗数 GB 的显存 (VRAM)。

两种标准的应对方案：
1. 图像分块 (Tiling) —— 将输入切分为 256x256 的重叠图块进行处理，最后再拼接。
2. 使用空洞卷积 (Dilated Convolutions) 替换瓶颈层 —— 在保持较高空间分辨率的同时扩大感受野（如 DeepLab 系列架构）。

对于初始模型而言，使用 256x256 的输入配合基础通道数为 64 的 U-Net，可以在 8 GB 显存下轻松完成训练。

## 构建

### 步骤 1：编码器模块 (Encoder block)

包含两个 3x3 卷积层 (convolutional layer)，均配有批归一化 (batch normalization) 和 ReLU 激活函数。第一个卷积层改变通道数；第二个卷积层保持通道数不变。

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

该模块在整个网络中被重复使用。`bias=False` 是因为批归一化层 (BN) 的 beta 参数已经处理了偏置。

### 步骤 2：下采样与上采样模块 (Down and up blocks)

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_c, out_c),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

仅检查空间维度的形状 (`shape[-2:]`) 能够处理尺寸不能被 16 整除的输入；在拼接 (concatenation) 之前，使用安全的 `F.interpolate` 对张量进行对齐。如果比较完整形状，通道数差异也会触发该检查，而这应当引发明确的报错，而非静默地进行插值 (interpolation)。

### 步骤 3：U-Net 架构

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.d4 = Down(base * 8, base * 16)
        self.u1 = Up(base * 16 + base * 8, base * 8)
        self.u2 = Up(base * 8 + base * 4, base * 4)
        self.u3 = Up(base * 4 + base * 2, base * 2)
        self.u4 = Up(base * 2 + base, base)
        self.outc = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)

net = UNet(in_channels=3, num_classes=2, base=32)
x = torch.randn(1, 3, 256, 256)
print(f"output: {net(x).shape}")
print(f"params: {sum(p.numel() for p in net.parameters()):,}")

输出形状为 `(1, 2, 256, 256)` —— 空间尺寸与输入相同，通道数为 `num_classes`。当 `base=32` 时，参数量约为 770 万 (7.7M)。

### 步骤 4：损失函数 (Losses)

def dice_loss(logits, targets, num_classes, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * targets_one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1 - dice.mean()


def combined_loss(logits, targets, num_classes, lam=1.0):
    ce = F.cross_entropy(logits, targets)
    dc = dice_loss(logits, targets, num_classes)
    return ce + lam * dc, {"ce": ce.item(), "dice": dc.item()}

Dice 系数 (Dice coefficient) 按类别分别计算后取平均值（宏观 Dice / macro Dice）。参数 `eps` 用于防止当批次中缺失某些类别时出现除以零的错误。

### 步骤 5：IoU 评估指标 (IoU metric)

@torch.no_grad()
def iou_per_class(logits, targets, num_classes):
    preds = logits.argmax(dim=1)
    ious = torch.zeros(num_classes)
    for c in range(num_classes):
        pred_c = (preds == c)
        true_c = (targets == c)
        inter = (pred_c & true_c).sum().float()
        union = (pred_c | true_c).sum().float()
        ious[c] = (inter / union) if union > 0 else torch.tensor(float("nan"))
    return ious

返回一个长度为 C 的向量。`nan` 用于标记当前批次中不存在的类别 —— 在计算平均交并比 (mean IoU, mIoU) 时，不应将这些类别纳入平均计算。

### 步骤 6：用于端到端验证的合成数据集 (Synthetic dataset)

在彩色背景上生成几何图形，迫使网络学习识别形状特征，而非单纯记忆像素颜色。

import numpy as np
from torch.utils.data import Dataset, DataLoader

def synthetic_segmentation(num_samples=200, size=64, seed=0):
    rng = np.random.default_rng(seed)
    images = np.zeros((num_samples, size, size, 3), dtype=np.float32)
    masks = np.zeros((num_samples, size, size), dtype=np.int64)
    for i in range(num_samples):
        bg = rng.uniform(0, 1, (3,))
        images[i] = bg
        masks[i] = 0
        num_shapes = rng.integers(1, 4)
        for _ in range(num_shapes):
            cls = int(rng.integers(1, 3))
            color = rng.uniform(0, 1, (3,))
            cx, cy = rng.integers(10, size - 10, size=2)
            r = int(rng.integers(4, 12))
            yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
            if cls == 1:
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
            else:
                mask = (np.abs(xx - cx) < r) & (np.abs(yy - cy) < r)
            images[i][mask] = color
            masks[i][mask] = cls
        images[i] += rng.normal(0, 0.02, images[i].shape)
        images[i] = np.clip(images[i], 0, 1)
    return images, masks


class SegDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i]).permute(2, 0, 1).float()
        mask = torch.from_numpy(self.masks[i]).long()
        return img, mask

共包含三个类别：背景 (0)、圆形 (1) 和方形 (2)。网络必须学会根据形状进行区分。

### 步骤 7：训练循环 (Training loop)

def train_one_epoch(model, loader, optimizer, device, num_classes):
    model.train()
    loss_sum, total = 0.0, 0
    iou_sum = torch.zeros(num_classes)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss, _ = combined_loss(logits, y, num_classes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        iou_sum += iou_per_class(logits, y, num_classes).nan_to_num(0)
    return loss_sum / total, iou_sum / len(loader)

在合成数据集上运行 10 到 30 个训练周期 (epochs)，观察形状类别的 mIoU 如何攀升至 0.9 以上。请注意，`nan_to_num(0)` 会将批次中缺失的类别视为 0；若要获得准确的各类别 IoU，应在评估阶段根据类别是否存在进行掩码处理，并使用 `torch.nanmean` 跨批次计算平均值，而非在此处直接求平均。

## 使用指南

在生产环境中，`segmentation_models_pytorch`（简称 `smp`）将标准的分割架构（segmentation architecture）与任意 `torchvision` 或 `timm` 骨干网络（backbone）进行封装。仅需三行代码：

import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3,
)

在实际工作中，还需了解以下几点：
- **DeepLabV3+** 使用空洞卷积（dilated convolutions）替代基于最大池化的下采样，使瓶颈层（bottleneck）保持分辨率；在卫星图像和自动驾驶数据上能生成更清晰的边界。
- **SegFormer** 将卷积编码器替换为分层 Transformer（hierarchical Transformer）；在多项基准测试中达到当前最优水平（State-of-the-Art, SOTA）。
- **Mask2Former** / **OneFormer** 在单一架构中统一了语义分割（semantic segmentation）、实例分割（instance segmentation）和全景分割（panoptic segmentation）。

这三者均可在 `smp` 或 `transformers` 库中作为即插即用（drop-in replacement）的替代方案，且兼容相同的数据加载器（data loader）。

## 交付成果

本课时将生成以下文件：

- `outputs/prompt-segmentation-task-picker.md` —— 一个提示词（prompt），用于根据给定任务在语义分割、实例分割和全景分割之间进行选择，并推荐相应的架构。
- `outputs/skill-segmentation-mask-inspector.md` —— 一项技能（skill），用于报告类别分布、预测掩码（mask）统计信息，以及预测不足或边界模糊的类别。

## 练习

1. **(简单)** 为二分类分割任务（前景与背景）实现 `bce_dice_loss`。在合成的双类别数据集上进行验证：当前景像素仅占 5% 时，组合损失函数的收敛速度应快于单独使用二元交叉熵损失（Binary Cross-Entropy, BCE）。
2. **(中等)** 将 `nn.Upsample + conv` 上采样模块替换为 `nn.ConvTranspose2d` 上采样模块。在合成数据集上分别训练两者并比较平均交并比（mean Intersection over Union, mIoU）。观察转置卷积（transposed convolution）版本中棋盘格伪影（checkerboard artifacts）出现的位置。
3. **(困难)** 选取一个真实的分割数据集（如 Oxford-IIIT Pets、Cityscapes 迷你划分或某个医学子集），训练 U-Net 模型，使其交并比（Intersection over Union, IoU）与 `smp.Unet` 参考模型的差距控制在 2 个点以内。报告每个类别的 IoU，并指出哪些类别在损失函数中加入 Dice 损失（Dice loss）后受益最大。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 语义分割 (Semantic Segmentation) | “为每个像素打标签” | 将每个像素分类至 C 个类别中；同一类别的不同实例会合并 |
| 实例分割 (Instance Segmentation) | “为每个物体打标签” | 区分同一类别的不同实例；仅针对前景目标 |
| 全景分割 (Panoptic Segmentation) | “语义 + 实例” | 每个像素分配一个类别；每个实体实例 (Thing Instance) 还会分配一个唯一 ID |
| 跳跃连接 (Skip Connection) | “U-Net 桥接” | 将编码器 (Encoder) 特征拼接至分辨率匹配的解码器 (Decoder) 特征中；保留高频细节 |
| 转置卷积 (Transposed Convolution) | “反卷积” | 可学习的上采样操作；可能产生棋盘格伪影 (Checkerboard Artifacts) |
| Dice 损失 (Dice Loss) | “重叠损失” | 1 - 2\|A ∩ B\| / (\|A\| + \|B\|)；直接优化掩码重叠度，对类别不平衡具有鲁棒性 |
| 平均交并比 (mIoU) | “平均交并比” | 各类别交并比 (IoU) 的平均值；分割任务社区标准评估指标 |
| 边界 F1 分数 (Boundary F1) | “边界精度” | 仅基于边界像素计算的 F1 分数；对精度要求极高的任务至关重要 |

## 扩展阅读

- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597) — 原始论文；被广泛引用的架构图位于第 2 页
- [Fully Convolutional Networks (Long et al., 2015)](https://arxiv.org/abs/1411.4038) — 首次将分割任务转化为端到端卷积问题的论文
- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) — 工业级分割任务的参考实现；涵盖所有标准架构与标准损失函数
- [Lessons learned from training SOTA segmentation (kaggle.com competitions)](https://www.kaggle.com/code/iafoss/carvana-unet-pytorch) — 详细解析了测试时增强 (Test-Time Augmentation, TTA)、伪标签 (Pseudo-labeling) 和类别权重 (Class Weights) 在真实数据中的重要性