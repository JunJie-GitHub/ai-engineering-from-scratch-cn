---
name: skill-latency-profiler
description: 编写包含预热、同步、百分位数和内存追踪的完整延迟基准测试脚本
version: 1.0.0
phase: 4
lesson: 15
tags: [边缘计算, 部署, 性能剖析, 基准测试]
---

# 延迟分析器 (Latency Profiler)

为任意 PyTorch 模型生成严谨的延迟基准测试。产出下游团队真正可信赖的报告。

## 适用场景

- 在选定部署模型前，对比多个候选骨干网络 (backbone)。
- 量化 (quantisation) 或剪枝 (pruning) 操作的前后对比。
- 更换推理运行时 (runtime) 后（如 Eager 模式 vs ONNX vs TensorRT）。
- 生成部署就绪性报告。

## 输入参数

- `model`：PyTorch `nn.Module`。
- `input_shape`：元组，例如 `(1, 3, 224, 224)`。
- `device`：`cpu` | `cuda` | `mps`。
- `warmup`：默认值为 10。
- `iters`：默认值为 100。

## 检查项

### 1. 预热 (Warmup)
在不计时的情况下运行模型 `warmup` 次。用于捕获首次前向传播的 JIT 编译开销及冷缓存效应。

### 2. 同步 (Synchronisation)
对于 `cuda` 设备，在每次计时的前向传播前后调用 `torch.cuda.synchronize()`。
对于 `mps` 设备，调用 `torch.mps.synchronize()`。

### 3. 计时器 (Timer)
使用 `time.perf_counter()` 进行墙上时钟 (wall-clock) 测量，并将结果转换为毫秒。

### 4. 百分位数 (Percentiles)
对完整的耗时列表进行排序。报告 `p50, p90, p95, p99, mean, std`。

### 5. 内存 (Memory)
对于 `cuda` 设备，在运行结束后调用 `torch.cuda.max_memory_allocated()` 并减去基线值。
对于 `cpu` 设备，在运行前后使用 `tracemalloc` 或 `psutil.Process().memory_info().rss` 进行测量。

### 6. 批量大小扫描 (Batch-size sweep)
可选地针对 `batch_size in [1, 4, 16, 32]` 重复基准测试，以揭示吞吐量 (throughput) 与延迟之间的权衡关系。

## 输出模板

import time
import torch
import psutil, os

def profile(model, input_shape, device="cpu", warmup=10, iters=100):
    proc = psutil.Process(os.getpid())
    baseline_rss = proc.memory_info().rss / 1e6

    model = model.to(device).eval()
    x = torch.randn(input_shape, device=device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(iters):
            sync()
            t0 = time.perf_counter()
            model(x)
            sync()
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    mean = sum(times) / len(times)
    std  = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5

    def pct(p):
        idx = max(0, min(len(times) - 1, int(len(times) * p) - 1))
        return times[idx]

    report = {
        "p50_ms":  pct(0.50),
        "p90_ms":  pct(0.90),
        "p95_ms":  pct(0.95),
        "p99_ms":  pct(0.99),
        "mean_ms": mean,
        "std_ms":  std,
        "rss_mb":  proc.memory_info().rss / 1e6 - baseline_rss,
    }
    if device == "cuda":
        report["peak_cuda_mb"] = torch.cuda.max_memory_allocated() / 1e6

    return report

## 核心规范

- 始终执行预热；切勿信任首次前向传播的计时结果。
- 关注百分位数而非平均值——单个异常值可能使平均值翻倍，但对 p50 的影响微乎其微。
- 使用与生产环境相同的 `input_shape`；224x224 分辨率下的延迟绝不等于 384x384 下的延迟。
- 对于 CUDA 设备，绝不可省略 `torch.cuda.synchronize()`；缺少它，所得数据将毫无意义。
- 记录数据时需同时附带 PyTorch 版本、CUDA 版本及设备名称，否则数据将失去可比性。