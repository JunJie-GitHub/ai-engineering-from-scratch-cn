# 图像基础 (Image Fundamentals) — 像素 (Pixels)、通道 (Channels)、色彩空间 (Color Spaces)

> 图像本质上是光采样数据构成的张量 (Tensor)。你未来使用的每一个视觉模型 (Vision Model) 都源于这一基本事实。

**类型：** 构建 (Build)
**语言：** Python
**前置课程：** 第一阶段第12课（张量运算 (Tensor Operations)）、第三阶段第11课（PyTorch 入门 (Intro to PyTorch)）
**时长：** 约45分钟

## 学习目标

- 解释连续场景 (continuous scene) 如何被离散化 (discretized) 为像素 (pixels)，以及为何采样 (sampling) 与量化 (quantization) 的决策会设定每个下游模型 (downstream model) 的性能上限
- 将图像作为 NumPy 数组 (NumPy arrays) 进行读取、切片与检查，并能在 HWC 与 CHW 布局 (HWC and CHW layouts) 之间熟练切换
- 在 RGB、灰度 (grayscale)、HSV 与 YCbCr 之间进行转换，并阐明每种色彩空间 (color space) 存在的理由
- 严格按照 torchvision 的要求应用像素级预处理 (pixel-level preprocessing)（包含归一化 (normalize)、标准化 (standardize)、调整尺寸 (resize) 与通道优先 (channel-first)）

## 问题

你阅读的每篇论文、下载的每个预训练权重 (pretrained weight)、调用的每个视觉 API (vision API)，都假设输入具有特定的编码 (encoding)。当模型需要 `float32` 时，如果你传入 `uint8` 图像，它依然会运行——但会悄无声息地输出垃圾结果。将 BGR 输入到基于 RGB 训练的网络中，准确率会骤降十个点。当模型期望通道在前 (channels-first) 的输入时，如果你给它通道在后 (channels-last) 的数据，第一个卷积层 (convolutional layer) 就会把高度误当作特征通道。这些情况都不会抛出错误。它们只会毁掉你的评估指标，让你花上一周时间去排查一个实际上出在文件加载方式上的 Bug。

一旦明白卷积 (convolution) 在什么数据上滑动，它其实并不复杂。难点在于，“一张图像”对相机、JPEG 解码器、PIL、OpenCV、torchvision 以及 CUDA 内核 (CUDA kernel) 而言，含义各不相同。每个技术栈都有其自身的轴顺序、字节范围和通道约定。如果视觉工程师搞不清这些差异，交付的流水线 (pipeline) 必然存在缺陷。

本节课将夯实基础，以便本阶段后续内容能在此基础上展开。学完本课后，你将明白什么是像素 (pixel)，为什么每个像素对应三个数值而非一个，“使用 ImageNet 统计值 (ImageNet stats) 进行归一化 (normalization)” 究竟做了什么，以及如何在后续课程默认的两三种数据布局 (layout) 之间进行转换。

## 概念

### 完整预处理流程概览

每个生产环境的视觉系统（Vision System）都遵循相同的可逆变换序列。只要有一步出错，模型看到的输入就会与训练时不同。

flowchart LR
    A["Image file<br/>(JPEG/PNG)"] --> B["Decode<br/>uint8 HWC"]
    B --> C["Convert<br/>colorspace<br/>(RGB/BGR/YCbCr)"]
    C --> D["Resize<br/>shorter side"]
    D --> E["Center crop<br/>model size"]
    E --> F["Divide by 255<br/>float32 [0,1]"]
    F --> G["Subtract mean<br/>Divide by std"]
    G --> H["Transpose<br/>HWC → CHW"]
    H --> I["Batch<br/>CHW → NCHW"]
    I --> J["Model"]

    style A fill:#fef3c7,stroke:#d97706
    style J fill:#ddd6fe,stroke:#7c3aed
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#bfdbfe,stroke:#2563eb

红色和蓝色方框是 80% 静默失败（Silent Failures）的根源：遗漏标准化（Standardization）和布局（Layout）错误。

### 像素是采样点，而非小方块

相机传感器统计落在微型探测器网格上的光子。每个探测器在极短时间内积分光线，并输出与击中光子数成正比的电压。随后，传感器将该电压离散化为整数。一个探测器对应一个像素。

Continuous scene                 Sensor grid                     Digital image
(infinite detail)                (H x W detectors)               (H x W integers)

    ~~~~~                        +--+--+--+--+--+                 210 198 180 155 120
   ~   ~   ~                     |  |  |  |  |  |                 205 195 178 152 118
  ~ light ~      ---->           +--+--+--+--+--+     ---->       200 190 175 150 115
   ~~~~~                         |  |  |  |  |  |                 195 185 170 148 112
                                 +--+--+--+--+--+                 188 180 165 145 108

这一步涉及两个关键选择，它们决定了后续所有处理的上限：

- **空间采样（Spatial Sampling）** 决定场景中每度视角包含多少个探测器。数量过少会导致边缘出现锯齿（混叠/Aliasing）；数量过多则会使存储和计算成本激增。
- **强度量化（Intensity Quantization）** 决定电压被划分成多少个离散区间。8 位提供 256 个级别，是显示设备的标准。10、12、16 位能提供更平滑的渐变，对医学成像、高动态范围（HDR）和原始传感器流水线至关重要。

像素不是带有面积的彩色方块，而是单次测量值。当你调整大小或旋转图像时，实际上是在对该测量网格进行重采样（Resampling）。

### 为什么是三个通道

单个探测器统计整个可见光谱的光子——这生成的是灰度图像。为了获取色彩，传感器在网格上覆盖红、绿、蓝滤光片的马赛克阵列。经过去马赛克（Demosaicing）处理后，每个空间位置都会得到三个整数：分别来自附近红、绿、蓝滤光探测器的响应值。这三个整数构成了像素的 RGB 三元组。

One pixel in memory:

    (R, G, B) = (210, 140, 30)   <- reddish-orange

An H x W RGB image:

    shape (H, W, 3)     stored as   H rows of W pixels of 3 values
                                    each in [0, 255] for uint8

“三”并非魔法。深度相机（Depth Camera）会增加 Z 通道。卫星图像会添加红外和紫外波段。医学扫描通常只有一个通道（X 光、CT）或多个通道（高光谱/Hyperspectral）。通道数位于张量的最后一个轴，卷积层（Conv Layers）会学习在该轴上进行特征混合。

### 两种布局约定：HWC 与 CHW

同一个张量，两种排列顺序。每个库都会选择其中一种。

HWC (height, width, channels)           CHW (channels, height, width)

   W ->                                    H ->
  +-----+-----+-----+                     +-----+-----+
H |R G B|R G B|R G B|                   C |R R R R R R|
| +-----+-----+-----+                   | +-----+-----+
v |R G B|R G B|R G B|                   v |G G G G G G|
  +-----+-----+-----+                     +-----+-----+
                                          |B B B B B B|
                                          +-----+-----+

   PIL, OpenCV, matplotlib,              PyTorch, most deep learning
   almost every image file on disk       frameworks, cuDNN kernels

采用 CHW 是因为卷积核（Convolution Kernels）在高度（H）和宽度（W）上滑动。将通道轴放在首位，意味着每个卷积核在每个通道上看到的都是一个连续的二维平面，这有利于高效的向量化计算。磁盘格式保留 HWC 是因为它与传感器输出扫描线的方式相匹配。

这行代码你会敲上千遍：

img_chw = img_hwc.transpose(2, 0, 1)      # NumPy
img_chw = img_hwc.permute(2, 0, 1)        # PyTorch tensor

内存布局可视化：

flowchart TB
    subgraph HWC["HWC — pixels stored interleaved (PIL, OpenCV, JPEG)"]
        H1["row 0: R G B | R G B | R G B ..."]
        H2["row 1: R G B | R G B | R G B ..."]
        H3["row 2: R G B | R G B | R G B ..."]
    end
    subgraph CHW["CHW — channels stored as stacked planes (PyTorch, cuDNN)"]
        C1["plane R: entire H x W of red values"]
        C2["plane G: entire H x W of green values"]
        C3["plane B: entire H x W of blue values"]
    end
    HWC -->|"transpose(2, 0, 1)"| CHW
    CHW -->|"transpose(1, 2, 0)"| HWC

### 字节范围与数据类型（dtype）

主流约定有三种：

| Convention | dtype | Range | Where you see it |
|------------|-------|-------|------------------|
| 原始（Raw） | `uint8` | [0, 255] | 磁盘文件、PIL、OpenCV 输出 |
| 归一化（Normalized） | `float32` | [0.0, 1.0] | 执行 `img.astype('float32') / 255` 之后 |
| 标准化（Standardized） | `float32` | 约 [-2, +2] | 减去均值并除以标准差之后 |

卷积网络（Convolutional Networks）是在标准化输入上训练的。ImageNet 统计值 `mean=[0.485, 0.456, 0.406]` 和 `std=[0.229, 0.224, 0.225]` 是基于 [0, 1] 归一化像素计算的，完整 ImageNet 训练集三个通道的算术平均值和标准差。将原始 `uint8` 数据输入到期望标准化浮点数的模型中，是应用视觉领域最常见的静默失败原因。

### 色彩空间及其存在意义

RGB 是采集格式，但对模型而言并不总是最有用的表示形式。

 RGB               HSV                       YCbCr / YUV

 R red             H hue (angle 0-360)       Y luminance (brightness)
 G green           S saturation (0-1)        Cb chroma blue-yellow
 B blue            V value/brightness (0-1)  Cr chroma red-green

 Linear to         Separates color from      Separates brightness from
 sensor output     brightness. Useful for    color. JPEG and most video
                   color thresholding, UI    codecs compress the chroma
                   sliders, simple filters   channels harder because the
                                             human eye is less sensitive
                                             to chroma detail than to Y.

对于大多数现代卷积神经网络（CNN），你直接输入 RGB。你会在以下场景遇到其他色彩空间：

- **HSV** —— 传统计算机视觉（CV）代码、基于颜色的分割、白平衡。
- **YCbCr** —— 解析 JPEG 内部结构、视频流水线、仅对 Y 通道操作的超分辨率模型。
- **灰度（Grayscale）** —— 光学字符识别（OCR）、文档模型，以及任何颜色属于干扰变量而非有效信号的场景。

从 RGB 转换到灰度是加权求和而非简单平均，因为人眼对绿色的敏感度高于红色或蓝色：

Y = 0.299 R + 0.587 G + 0.114 B       (ITU-R BT.601, the classic weights)

### 宽高比、缩放与插值

每个模型都有固定的输入尺寸（大多数 ImageNet 分类器为 224x224，现代检测器为 384x384 或 512x512）。你的图像很少能直接匹配。三种关键的缩放策略：

- **缩放短边后中心裁剪（Resize shorter side, then center crop）** —— ImageNet 标准做法。保持宽高比，舍弃边缘像素条带。
- **缩放并填充（Resize and pad）** —— 保持宽高比和所有像素，添加黑边。目标检测和 OCR 的标准做法。
- **直接缩放到目标尺寸** —— 拉伸图像。计算成本低，会扭曲几何形状，但对许多分类任务足够有效。

插值方法（Interpolation Method）决定了当新网格与旧网格不对齐时，如何计算中间像素：

Nearest neighbour     fastest, blocky, only choice for masks/labels
Bilinear              fast, smooth, default for most image resizing
Bicubic               slower, sharper on upscaling
Lanczos               slowest, best quality, used for final display

经验法则：训练使用双线性插值（Bilinear），用于查看的资产使用双三次插值（Bicubic）或 Lanczos 插值，包含整数类别 ID 的内容使用最近邻插值（Nearest Neighbour）。

## 构建

### 步骤 1：加载图像并检查其形状 (shape)

使用 Pillow 加载任意 JPEG 或 PNG 图像，将其转换为 NumPy 数组，并打印结果。为了提供一个可离线运行且具有确定性的示例，我们将合成一张图像。

import numpy as np
from PIL import Image

def synthetic_rgb(h=128, w=192, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    r = (np.sin(xx * 6) * 0.5 + 0.5) * 255
    g = yy * 255
    b = (1 - yy) * xx * 255
    rgb = np.stack([r, g, b], axis=-1) + rng.normal(0, 6, (h, w, 3))
    return np.clip(rgb, 0, 255).astype(np.uint8)

arr = synthetic_rgb()
# Or load from disk:
# arr = np.asarray(Image.open("your_image.jpg").convert("RGB"))

print(f"type:   {type(arr).__name__}")
print(f"dtype:  {arr.dtype}")
print(f"shape:  {arr.shape}     # (H, W, C)")
print(f"min:    {arr.min()}")
print(f"max:    {arr.max()}")
print(f"pixel at (0, 0): {arr[0, 0]}")

预期输出：`shape: (H, W, 3)`，`dtype: uint8`，范围 `[0, 255]`。无论这些字节来自相机、JPEG 解码器还是合成生成器，这都是标准的磁盘存储表示形式。

### 步骤 2：分离通道并重排布局

分别提取 R、G、B 通道，然后将其从 HWC 格式转换为 PyTorch 所需的 CHW 格式。

R = arr[:, :, 0]
G = arr[:, :, 1]
B = arr[:, :, 2]
print(f"R shape: {R.shape}, mean: {R.mean():.1f}")
print(f"G shape: {G.shape}, mean: {G.mean():.1f}")
print(f"B shape: {B.shape}, mean: {B.mean():.1f}")

arr_chw = arr.transpose(2, 0, 1)
print(f"\nHWC shape: {arr.shape}")
print(f"CHW shape: {arr_chw.shape}")

每个通道对应一个灰度平面。CHW 仅仅是重新排列了坐标轴；当内存布局允许时，严格来说并不需要复制数据。

### 步骤 3：灰度与 HSV 转换

首先通过加权求和计算灰度图，然后手动实现 RGB 到 HSV 的转换。

def rgb_to_grayscale(rgb):
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (rgb.astype(np.float32) @ weights).astype(np.uint8)

def rgb_to_hsv(rgb):
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    cmax = np.max(rgb_f, axis=-1)
    cmin = np.min(rgb_f, axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask = delta > 0
    rmax = mask & (cmax == r)
    gmax = mask & (cmax == g)
    bmax = mask & (cmax == b)
    h[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6
    h[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2
    h[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4
    h = h * 60.0

    s = np.where(cmax > 0, delta / cmax, 0)
    v = cmax
    return np.stack([h, s, v], axis=-1)

gray = rgb_to_grayscale(arr)
hsv = rgb_to_hsv(arr)
print(f"gray shape: {gray.shape}, range: [{gray.min()}, {gray.max()}]")
print(f"hsv   shape: {hsv.shape}")
print(f"hue range: [{hsv[..., 0].min():.1f}, {hsv[..., 0].max():.1f}] degrees")
print(f"sat range: [{hsv[..., 1].min():.2f}, {hsv[..., 1].max():.2f}]")
print(f"val range: [{hsv[..., 2].min():.2f}, {hsv[..., 2].max():.2f}]")

色调 (Hue) 以角度为单位输出，饱和度 (Saturation) 和明度 (Value) 的范围为 `[0, 1]`。这与 OpenCV 的 `hsv_full` 约定一致。

### 步骤 4：归一化 (normalization)、标准化 (standardization) 及其逆操作

将原始字节数据转换为预训练 ImageNet 模型所期望的精确张量 (tensor)，然后再转换回来。

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_imagenet(rgb_uint8):
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)
    return x

def deprocess_imagenet(chw_float32):
    x = chw_float32.transpose(1, 2, 0)
    x = x * std + mean
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x

x = preprocess_imagenet(arr)
print(f"preprocessed shape: {x.shape}     # (C, H, W)")
print(f"preprocessed dtype: {x.dtype}")
print(f"preprocessed mean per channel:  {x.mean(axis=(1, 2)).round(3)}")
print(f"preprocessed std  per channel:  {x.std(axis=(1, 2)).round(3)}")

roundtrip = deprocess_imagenet(x)
max_diff = np.abs(roundtrip.astype(int) - arr.astype(int)).max()
print(f"roundtrip max pixel diff: {max_diff}    # should be 0 or 1")

每个通道的均值应接近于零，标准差应接近于一。这对预处理/后处理函数正是 torchvision 中 `transforms.Normalize` 调用在底层所执行的操作。

### 步骤 5：使用三种插值 (interpolation) 方法调整图像大小

在放大图像时比较最近邻 (nearest)、双线性 (bilinear) 和双三次 (bicubic) 插值，以便清晰地观察差异。

target = (arr.shape[0] * 3, arr.shape[1] * 3)

nearest = np.asarray(Image.fromarray(arr).resize(target[::-1], Image.NEAREST))
bilinear = np.asarray(Image.fromarray(arr).resize(target[::-1], Image.BILINEAR))
bicubic = np.asarray(Image.fromarray(arr).resize(target[::-1], Image.BICUBIC))

def local_roughness(x):
    gy = np.diff(x.astype(float), axis=0)
    gx = np.diff(x.astype(float), axis=1)
    return float(np.abs(gy).mean() + np.abs(gx).mean())

for name, out in [("nearest", nearest), ("bilinear", bilinear), ("bicubic", bicubic)]:
    print(f"{name:>8}  shape={out.shape}  roughness={local_roughness(out):6.2f}")

最近邻插值的粗糙度得分最高，因为它保留了硬边缘。双线性插值最为平滑。双三次插值介于两者之间，在避免阶梯状伪影的同时保留了视觉上的清晰度。

## 使用方法

`torchvision.transforms` 将上述所有内容整合到一个可组合流水线 (composable pipeline) 中。下面的代码精确复现了 `preprocess_imagenet` 的功能，并额外增加了调整大小 (resize) 与裁剪 (crop) 操作。

import torch
from torchvision import transforms
from PIL import Image

img = Image.fromarray(synthetic_rgb(256, 256))

pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

x = pipeline(img)
print(f"tensor type:  {type(x).__name__}")
print(f"tensor dtype: {x.dtype}")
print(f"tensor shape: {tuple(x.shape)}      # (C, H, W)")
print(f"per-channel mean: {x.mean(dim=(1, 2)).tolist()}")
print(f"per-channel std:  {x.std(dim=(1, 2)).tolist()}")

batch = x.unsqueeze(0)
print(f"\nbatched shape: {tuple(batch.shape)}   # (N, C, H, W) — ready for a model")

共包含四个步骤，且必须严格按照此顺序执行：`Resize(256)` 将图像的较短边缩放至 256 像素；`CenterCrop(224)` 从图像中心截取 224x224 的区域；`ToTensor()` 将像素值除以 255，并将通道布局从 HWC (Height-Width-Channel) 转换为 CHW (Channel-Height-Width)；`Normalize` 则减去 ImageNet 均值 (ImageNet mean) 并除以标准差。若颠倒该顺序，将静默地改变传入模型的数据。

## 发布

本课程将产出以下内容：

- `outputs/prompt-vision-preprocessing-audit.md` — 一个提示词（Prompt），可将任意模型卡片（Model Card）或数据集卡片（Dataset Card）转化为一份检查清单，列出团队必须严格遵循的精确预处理不变量（Preprocessing Invariants）。
- `outputs/skill-image-tensor-inspector.md` — 一项技能（Skill），针对任意图像形状的张量（Tensor）或数组（Array），报告其数据类型（dtype）、布局（Layout）、取值范围（Range），并判断其数据呈现为原始（Raw）、归一化（Normalized）还是标准化（Standardized）状态。

## 练习

1. **(Easy)** 使用 OpenCV (`cv2.imread`) 和 Pillow 加载一张 JPEG 图像。打印两者的形状 (shape) 以及位于 `(0, 0)` 的像素 (pixel) 值。解释通道顺序 (channel order) 的差异，然后编写一行转换代码，使 OpenCV 数组 (array) 与 Pillow 的完全一致。
2. **(Medium)** 编写 `standardize(img, mean, std)` 及其逆函数 (inverse)，使它们在任何 uint8 图像上都能通过 `roundtrip_max_diff <= 1` 的测试。你的函数必须能在单次调用中同时处理 HWC 格式的单张图像和 NCHW 格式的批次 (batch) 数据。
3. **(Hard)** 获取一个 3 通道的 ImageNet 标准化 (ImageNet-standardized) 张量 (tensor)，并将其输入一个 1x1 卷积 (1x1 convolution) 层，该层学习将 RGB 加权混合 (weighted mixture) 为单个灰度通道 (grayscale channel)。将权重 (weights) 初始化为 `[0.299, 0.587, 0.114]` 并冻结 (freeze) 它们，验证输出结果与你手动编写的 `rgb_to_grayscale` 之间的差异在浮点误差 (floating-point error) 范围内。还有哪些经典的色彩空间变换 (color-space transforms) 可以表示为 1x1 卷积 (convolutions)？

## 关键术语

| 术语 (Term) | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 像素 (Pixel) | “一个彩色方块” | 网格特定位置处的光强度采样值——彩色图像对应三个数值，灰度图像对应一个数值 |
| 通道 (Channel) | “颜色” | 堆叠构成图像张量 (image tensor) 的并行空间网格之一；在 HWC 中为最后一个轴，在 CHW 中为第一个轴 |
| HWC / CHW (HWC / CHW) | “形状” | 图像张量的轴排列顺序；磁盘文件与 PIL 使用 HWC，PyTorch 与 cuDNN 使用 CHW |
| 归一化 (Normalize) | “缩放图像” | 除以 255 使像素值落在 [0, 1] 区间内——这是必要步骤，但并非充分条件 |
| 标准化 (Standardize) | “零中心化” | 逐通道减去均值 (mean) 并除以标准差 (std)，使输入数据的分布与模型训练时的分布保持一致 |
| 灰度转换 (Grayscale conversion) | “对通道求平均” | 采用 0.299/0.587/0.114 作为系数的加权求和，以匹配人类对亮度的感知特性 |
| 插值 (Interpolation) | “缩放时如何选取像素” | 当新网格与旧网格未对齐时，决定输出像素值的计算规则——标签处理常用 nearest，模型训练常用 bilinear，图像显示常用 bicubic |
| 宽高比 (Aspect ratio) | “宽度除以高度” | 用于区分“缩放并填充 (resize and pad)”与“缩放并拉伸 (resize and stretch)”的比例关系 |

## 延伸阅读

- [Charles Poynton — A Guided Tour of Color Space](https://poynton.ca/PDFs/Guided_tour.pdf) — 对为何存在众多色彩空间（Color Space）以及各自适用场景最清晰的技术解析
- [PyTorch Vision Transforms Docs](https://pytorch.org/vision/stable/transforms.html) — 你在生产环境中实际组合使用的完整变换（Transforms）流水线
- [How JPEG Works (Colt McAnlis)](https://www.youtube.com/watch?v=F1kYBnY6mwg) — 对色度子采样（Chroma Subsampling）、离散余弦变换（DCT）以及 JPEG 为何编码 YCbCr 而非 RGB 的直观视觉讲解
- [ImageNet Preprocessing Conventions (torchvision models)](https://pytorch.org/vision/stable/models.html) — `mean=[0.485, 0.456, 0.406]` 的标准依据，以及为何模型库（Model Zoo）中的每个模型都要求使用该值