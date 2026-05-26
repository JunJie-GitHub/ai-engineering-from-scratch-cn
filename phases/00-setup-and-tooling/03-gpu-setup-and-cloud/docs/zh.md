# GPU（图形处理器）配置与云平台

> 使用 CPU（中央处理器）进行训练足以满足学习需求。实际项目训练则需要 GPU（图形处理器）。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 0，第 01 课
**时长：** 约 45 分钟

## 学习目标

- 使用 `nvidia-smi` 和 PyTorch 的 CUDA API 验证本地图形处理器 (GPU) 的可用性
- 配置搭载 T4 GPU 的 Google Colab 以进行免费的云端实验
- 对中央处理器 (CPU) 与 GPU 上的矩阵乘法进行基准测试，并测量加速比
- 使用半精度浮点数 (fp16) 经验法则估算显存 (VRAM) 可容纳的最大模型

## 问题

第 1-3 阶段的大多数课程在中央处理器 (CPU) 上即可流畅运行。但一旦开始训练卷积神经网络 (CNNs)、Transformer (Transformers) 或大语言模型 (LLMs)（第 4 阶段及以后），你就需要图形处理器 (GPU) 加速。在 CPU 上需要 8 小时的训练任务，在 GPU 上仅需 10 分钟。

你有三种选择：本地 GPU、云端 GPU 或 Google Colab（免费）。

## 概念

您的选项：

1. 本地 NVIDIA GPU (图形处理器)
   成本：$0（您已拥有）
   设置：安装 CUDA + cuDNN
   最适合：日常使用、大型数据集

2. Google Colab (免费层级)
   成本：$0
   设置：无需配置
   最适合：快速实验、本地无 GPU (图形处理器)

3. 云端 GPU (图形处理器) (Lambda, RunPod, Vast.ai)
   成本：$0.20-2.00/小时
   设置：SSH (安全外壳协议) + 安装
   最适合：正式训练、大型模型


## 构建

### 选项 1：本地 NVIDIA 图形处理器 (GPU)

检查您的设备是否已配备：

nvidia-smi

安装支持 统一计算设备架构 (CUDA) 的 PyTorch：

import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

### 选项 2：Google Colab

1. 访问 [colab.research.google.com](https://colab.research.google.com)
2. 依次选择 Runtime（运行时） > Change runtime type（更改运行时类型） > T4 GPU
3. 运行 `!nvidia-smi` 进行验证

可将本课程提供的 Notebook（交互式笔记本）直接上传至 Colab。

### 选项 3：云端 GPU

若使用 Lambda Labs、RunPod 或 Vast.ai 等平台：

ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"

### 没有 GPU？没问题。

大多数课程章节均可在 中央处理器 (CPU) 上运行。需要 GPU 的章节会明确标注，并提供 Colab 链接。

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")


## 实战：GPU 与 CPU 基准测试对比

这是AI工程（AI Engineering）课程中的一节课。

--- BEGIN TEXT ---
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
--- END TEXT ---

## 练习

1. 运行上述基准测试 (benchmark)，并对比 CPU 与 GPU 的耗时
2. 如果没有 GPU，请在 Google Colab 上运行并进行对比
3. 检查你的 GPU 显存 (GPU memory) 容量，并估算可容纳的最大模型规模（经验法则 (rule of thumb)：fp16 精度下每个参数 (parameter) 约占 2 字节）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 统一计算设备架构 (CUDA) | "GPU 编程" | NVIDIA 的并行计算平台，允许开发者直接在 GPU 上运行代码 |
| 显存 (VRAM) | "GPU 内存" | GPU 上的专用视频内存，独立于系统内存。它直接限制了可加载模型的大小。 |
| 半精度浮点格式 (fp16) | "半精度" | 16 位浮点数格式，内存占用仅为 fp32 的一半，且精度损失微乎其微 |
| 张量核心 (Tensor Core) | "快速矩阵硬件" | 专为矩阵乘法优化的 GPU 核心，运算速度比普通核心快 4 到 8 倍 |