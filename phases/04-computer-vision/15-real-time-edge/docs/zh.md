# 实时视觉——边缘部署

> 边缘推理（Edge Inference）是一门在仅有 2 GB 内存的设备上，让准确率达到 90% 的模型以 30 fps 运行的技术。每一个百分点的准确率提升，都需要以毫秒级的延迟（Latency）作为交换。

**类型：** 学习 + 实践
**编程语言：** Python
**前置知识：** 第 4 阶段第 04 课（图像分类），第 10 阶段第 11 课（量化（Quantization））
**预计耗时：** 约 75 分钟

## 学习目标

- 测量任意 PyTorch 模型的推理延迟（Inference Latency）、峰值内存占用和吞吐量（Throughput），并解读浮点运算次数（FLOPs）/ 参数量（Params）/ 延迟之间的权衡关系
- 使用 PyTorch 的训练后量化（Post-Training Quantization）将视觉模型量化至 INT8，并验证准确率损失小于 1%
- 将模型导出为 ONNX 格式，并使用 ONNX Runtime 或 TensorRT 进行编译；列举三种最常见的导出失败情况及其修复方法
- 说明在边缘设备资源受限的情况下，何时应选择 MobileNetV3、EfficientNet-Lite、ConvNeXt-Tiny 或 MobileViT

## 问题背景

训练阶段的视觉模型堪称“浮点数巨兽”。它拥有 1 亿（100M）参数，每次前向传播（Forward Pass）需要 10 GFLOPs 的计算量，并占用 2 GB 显存（VRAM）。这些资源根本无法塞进手机、车载信息娱乐系统、工业相机或无人机中。要将视觉系统真正落地交付，就意味着必须在缩小 100 倍的资源预算内，实现相同的预测能力。

主要依靠三个关键调节因素来完成这项工作：模型选择（采用相同训练策略但更小的架构）、量化（使用 INT8 替代 FP32）以及推理运行时（Inference Runtime，如 ONNX Runtime、TensorRT、Core ML、TFLite）。正确配置这三者，决定了你的项目是只能在工作站上跑的演示程序，还是能部署在 30 美元摄像头模块上的正式产品。

本课程首先建立度量体系（“无法测量，就无法优化”），随后逐一讲解这三个关键因素。目标并非掌握所有边缘运行时环境，而是让你了解有哪些可用的优化手段，以及如何验证每一项优化是否达到了预期效果。

## 核心概念

### 三大资源预算

flowchart LR
    M["Model"] --> LAT["Latency<br/>ms per image"]
    M --> MEM["Memory<br/>peak MB"]
    M --> PWR["Power<br/>mJ per inference"]

    LAT --> SHIP["Ship / no-ship<br/>decision"]
    MEM --> SHIP
    PWR --> SHIP

    style LAT fill:#fecaca,stroke:#dc2626
    style MEM fill:#fef3c7,stroke:#d97706
    style PWR fill:#dbeafe,stroke:#2563eb

- **延迟（Latency）**：p50、p95、p99。仅取 p50 平均值会掩盖对实时系统至关重要的长尾延迟表现。
- **峰值内存（Peak Memory）**：设备运行期间遇到的最大内存占用，而非稳态平均值。这一点至关重要，因为内存溢出（Out of Memory, OOM）在嵌入式目标设备上会导致致命崩溃。
- **功耗/能耗（Power / Energy）**：电池供电设备每次推理消耗的毫焦耳（mJ）。通常用 CPU/GPU 利用率乘以时间来近似估算。

边缘部署的决策依据通常是一张包含（模型、延迟、内存、准确率）的表格。表中的每一项数据都必须在目标设备上实测得出，而非在工作站上估算。

### 性能测量准则

每个边缘性能分析（edge profile）都应遵循以下三条准则：

1. **预热（Warm up）**：在正式测量前，使用 5-10 次虚拟前向传播（dummy forward passes）对模型进行预热。冷缓存和即时编译（Just-In-Time, JIT compilation）会导致首次测量数据缺乏代表性。
2. **同步（Synchronise）**：在计时代码块前后使用 `torch.cuda.synchronize()` 同步 GPU 任务。若不进行同步，你测量的将是内核调度时间，而非实际的内核执行时间。
3. **固定输入尺寸（Fix input sizes）**：将输入尺寸设置为生产环境的实际分辨率。224x224 分辨率下的延迟绝不等同于 512x512 下的延迟。

### 将 FLOPs 作为代理指标

FLOPs（每次推理的浮点运算次数，Floating-Point Operations per inference）是一种低成本且与设备无关的延迟代理指标。它适用于模型架构对比，但若作为绝对的实际运行时间（Wall-clock Time）参考则会产生误导。在实际应用中，FLOPs 高出 10% 的模型反而可能快 2 倍，这通常是因为它使用了更契合硬件的算子（例如深度可分离卷积编译效率高，而大型 7x7 卷积则不然）。

准则：使用 FLOPs 进行架构搜索，使用设备端实测延迟进行部署决策。

### 一段话讲清量化

将 FP32 权重和激活值替换为 INT8。在支持 INT8 内核的硬件上（包括所有现代移动 SoC 和配备 Tensor Core 的 NVIDIA GPU），模型体积可缩小至 1/4，内存带宽需求降至 1/4，计算量减少 2-4 倍。在视觉任务中，采用训练后静态量化（Post-Training Static Quantisation, PTQ）通常仅会导致 0.1% 到 1% 的准确率下降。

类型：

- **动态量化（Dynamic Quantisation）**：将权重量化为 INT8，激活值仍以浮点格式计算。实现简单，但加速效果有限。
- **静态量化（训练后，Static / Post-Training）**：量化权重，并在小型校准集上标定激活值范围。速度远快于动态量化。
- **量化感知训练（Quantisation-Aware Training, QAT）**：在训练过程中模拟量化操作，使模型能够适应量化带来的误差。准确率最高，但需要标注数据。

在视觉任务中，PTQ 仅需付出 5% 的精力即可获得 95% 的收益。仅当 PTQ 导致的准确率损失无法接受时，才考虑使用 QAT。

### 剪枝与知识蒸馏

- **剪枝（Pruning）**：移除不重要的权重（基于幅值）或通道（结构化剪枝）。在参数量过大的模型上效果显著，但对本身已高度紧凑的架构作用有限。
- **知识蒸馏（Distillation）**：训练一个小型学生模型来拟合大型教师模型的逻辑输出值（logits）。通常能挽回因模型压缩而损失的大部分准确率，是生产级边缘模型的标准做法。

### 推理运行时框架

- **PyTorch 动态图模式（Eager Mode）**：速度较慢，不适用于部署。仅限开发调试使用。
- **TorchScript**：已过时。已被 `torch.compile` 和 ONNX 导出取代。
- **ONNX Runtime**：中立的推理运行时。CPU、CUDA、CoreML、TensorRT、OpenVINO 均提供 ONNX 执行提供程序（Execution Providers）。建议从此入手。
- **TensorRT**：NVIDIA 的推理编译器。在 NVIDIA GPU（工作站与 Jetson 平台）上延迟表现最佳。可与 ONNX Runtime 集成或独立使用。
- **Core ML**：Apple 为 iOS/macOS 提供的运行时框架。需导出为 `.mlmodel` 或 `.mlpackage` 格式。
- **TFLite**：Google 为 Android/ARM 平台提供的运行时框架。需导出为 `.tflite` 格式。
- **OpenVINO**：Intel 为 CPU/VPU 提供的运行时框架。需导出为 `.xml` + `.bin` 格式。

实际工作流：导出 PyTorch 模型 -> 转换为 ONNX -> 根据目标平台选择对应的运行时。ONNX 已成为该领域的通用标准。

### 边缘架构选型指南

| 预算约束 | 模型 | 选型理由 |
|--------|-------|-----|
| < 300 万参数 | MobileNetV3-Small | 兼容性极佳，适合作为基准模型 |
| 300 万 - 1000 万 | EfficientNet-Lite-B0 | 在 TFLite 上具备最佳的参效比（准确率/参数） |
| 1000 万 - 2000 万 | ConvNeXt-Tiny | 参效比最优，对 CPU 友好 |
| 2000 万 - 3000 万 | MobileViT-S 或 EfficientViT | 具备 ImageNet 级别准确率的 Transformer 架构 |
| 3000 万 - 8000 万 | Swin-V2-Tiny | 适用于支持窗口注意力机制（Window Attention）的技术栈 |

除非有特殊原因，否则建议将上述所有模型均量化为 INT8。

## 构建

### 步骤 1：正确测量延迟 (Latency)

import time
import torch

def measure_latency(model, input_shape, device="cpu", warmup=10, iters=50):
    model = model.to(device).eval()
    x = torch.randn(input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "p99_ms": times[int(len(times) * 0.99)],
        "mean_ms": sum(times) / len(times),
    }

进行预热 (Warm-up)、执行同步操作 (Synchronize)，并使用 `time.perf_counter()`。报告百分位数 (Percentiles)，而非仅仅提供平均值 (Mean)。

### 步骤 2：参数量 (Parameter Count) 与浮点运算次数 (FLOPs) 统计

def parameter_count(model):
    return sum(p.numel() for p in model.parameters())

def flops_estimate(model, input_shape):
    """
    Rough FLOP count for a conv/linear-only model. For production use `fvcore` or `ptflops`.
    """
    total = 0
    def conv_hook(m, inp, out):
        nonlocal total
        c_out, c_in, kh, kw = m.weight.shape
        h, w = out.shape[-2:]
        total += 2 * c_in * c_out * kh * kw * h * w
    def linear_hook(m, inp, out):
        nonlocal total
        total += 2 * m.in_features * m.out_features
    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    model.eval()
    with torch.no_grad():
        model(torch.randn(input_shape))
    for h in hooks:
        h.remove()
    return total

在实际项目中，请使用 `fvcore.nn.FlopCountAnalysis` 或 `ptflops`；它们能够正确处理所有类型的模块 (Module)。

### 步骤 3：训练后静态量化 (Post-Training Static Quantization)

def quantise_ptq(model, calibration_loader, backend="x86"):
    import torch.ao.quantization as tq
    model = model.eval().cpu()
    model.qconfig = tq.get_default_qconfig(backend)
    tq.prepare(model, inplace=True)
    with torch.no_grad():
        for x, _ in calibration_loader:
            model(x)
    tq.convert(model, inplace=True)
    return model

流程包含配置、准备（插入观察者 (Observers)）、使用真实数据校准 (Calibrate) 以及转换（融合 (Fuse) 与量化 (Quantise)）。该过程要求模型预先进行层融合（将 `Conv -> BN -> ReLU` 转换为 `ConvBnReLU`），可通过 `torch.ao.quantization.fuse_modules` 实现。

### 步骤 4：导出为 ONNX 格式

def export_onnx(model, sample_input, path="model.onnx"):
    model = model.eval()
    torch.onnx.export(
        model,
        sample_input,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    return path

在 2026 年，`opset_version=17` 是稳妥的默认选择。`dynamic_axes` 参数允许你以任意批次大小 (Batch Size) 运行该 ONNX 模型。

### 步骤 5：基准测试 (Benchmark) 与方案对比

import torch.nn as nn
from torchvision.models import mobilenet_v3_small

def compare_regimes():
    model = mobilenet_v3_small(weights=None, num_classes=10)
    params = parameter_count(model)
    flops = flops_estimate(model, (1, 3, 224, 224))
    lat_fp32 = measure_latency(model, (1, 3, 224, 224), device="cpu")
    print(f"FP32 MobileNetV3-Small: {params:,} params  {flops/1e9:.2f} GFLOPs  "
          f"p50={lat_fp32['p50_ms']:.2f}ms  p95={lat_fp32['p95_ms']:.2f}ms")

对 `resnet50`、`efficientnet_v2_s` 和 `convnext_tiny` 运行相同的函数，即可生成部署决策所需的对比表格。

## 实际应用

生产环境技术栈通常会收敛为以下三种路径之一：

- **Web / 无服务器（Serverless）**：PyTorch -> ONNX -> ONNX Runtime（CPU 或 CUDA 提供程序）。部署最简单，对大多数场景已足够。
- **NVIDIA 边缘设备（NVIDIA Edge）**：PyTorch -> ONNX -> TensorRT。延迟表现最佳，但工程投入最大。
- **移动端（Mobile）**：PyTorch -> ONNX -> Core ML（iOS）或 TFLite（Android）。导出前需进行量化（Quantisation）。

在性能测量方面，`torch-tb-profiler`、`nvprof` / `nsys` 以及 macOS 上的 Instruments 可提供逐层（Layer-by-layer）的性能细分数据。`benchmark_app`（OpenVINO）和 `trtexec`（TensorRT）则提供独立的命令行（CLI）基准测试数值。

## 交付内容

本章节将产出以下内容：

- `outputs/prompt-edge-deployment-planner.md` —— 一个提示词（Prompt），可根据目标设备和延迟服务等级协议（SLA）自动选择主干网络（Backbone）、量化策略及推理运行时（Runtime）。
- `outputs/skill-latency-profiler.md` —— 一项技能脚本，用于生成完整的延迟基准测试脚本，包含预热（Warmup）、同步（Synchronisation）、百分位数（Percentiles）计算及内存跟踪功能。

## 练习

1. **（简单）** 在 CPU 上测量 `resnet18`、`mobilenet_v3_small`、`efficientnet_v2_s` 和 `convnext_tiny` 在 224x224 分辨率下的 p50 延迟。输出结果表格，并指出哪种架构具有最佳的每毫秒准确率（Accuracy-per-ms）。
2. **（中等）** 对 `mobilenet_v3_small` 应用训练后静态量化（Post-Training Static Quantisation）。在 CIFAR-10 或类似数据集的预留子集（Held-out Subset）上，报告 FP32 与 INT8 的延迟对比及准确率损失。
3. **（困难）** 将 `convnext_tiny` 导出为 ONNX 格式，使用 `CPUExecutionProvider` 在 `onnxruntime` 中运行，并与 PyTorch 即时执行（Eager）模式基线进行延迟对比。找出 ONNX Runtime 首次实现加速的网络层，并解释其原因。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 延迟（Latency） | “有多快” | 从输入到输出的耗时；关注 p50/p95/p99 百分位数，而非平均值 |
| 浮点运算次数（FLOPs） | “模型大小” | 单次前向传播的浮点运算量；粗略代表计算成本 |
| INT8 量化（INT8 Quantisation） | “8 位” | 将 FP32 权重和激活值替换为 8 位整数；体积缩小约 4 倍，速度提升 2-4 倍 |
| 训练后量化（PTQ） | “训练后量化” | 无需重新训练即可对已训练模型进行量化；实现简单，通常已足够 |
| 量化感知训练（QAT） | “量化感知训练” | 在训练过程中模拟量化操作；准确率最佳，但需要标注数据 |
| 开放神经网络交换（ONNX） | “中立格式” | 所有主流推理运行时均支持的模型交换格式 |
| TensorRT | “NVIDIA 编译器” | 将 ONNX 编译为针对 NVIDIA GPU 优化的推理引擎 |
| 知识蒸馏（Distillation） | “教师 -> 学生” | 训练小型模型以模仿大型模型的逻辑输出（Logits）；可恢复大部分损失的准确率 |

## 延伸阅读

- [EfficientNet (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946) — 针对高效架构的复合缩放（compound scaling）方法
- [MobileNetV3 (Howard et al., 2019)](https://arxiv.org/abs/1905.02244) — 集成 h-swish 与 squeeze-excite 的移动端优先（mobile-first）架构
- [A Practical Guide to TensorRT Optimization (NVIDIA)](https://developer.nvidia.com/blog/accelerating-model-inference-with-tensorrt-tips-and-best-practices-for-pytorch-users/) — 如何在实际部署中获取论文中的吞吐量（throughput）数据
- [ONNX Runtime docs](https://onnxruntime.ai/docs/) — 量化（quantisation）、图优化（graph optimisation）与执行提供程序（provider）选择