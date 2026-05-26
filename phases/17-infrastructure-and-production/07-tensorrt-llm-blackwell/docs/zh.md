# 在 Blackwell 架构上使用 FP8 与 NVFP4 的 TensorRT-LLM

> TensorRT-LLM 仅支持 NVIDIA 硬件，但在 Blackwell 架构上表现卓越。在配备 Dynamo 编排的 GB200 NVL72 系统上，SemiAnalysis InferenceX 于 2026 年第一至第二季度测得，120B 模型的推理成本为每百万 Token 0.012 美元，而 H100 + vLLM 方案的成本为 0.09 美元/百万 Token，两者存在 7 倍的经济性差距。该架构融合了三种浮点精度方案：FP8 对 KV 缓存 (KV Cache) 和注意力核函数 (Attention Kernels) 依然至关重要，因为它具备所需的动态范围；NVFP4（4 位微缩放浮点数）负责处理权重和激活值；多 Token 预测 (Multi-Token Prediction, MTP) 与分离式预填充/解码 (Disaggregated Prefill/Decode) 在此基础上再带来 2-3 倍的性能提升。Day-0 模型支持可直接加载 FP4 权重，无需训练后转换。对于 2026 年的工程团队而言，关键在于：TRT-LLM 是 NVIDIA 的闭源技术栈，采用它意味着以牺牲可移植性来换取吞吐量。在正式投入前，请务必针对您的模型组合与硬件配置进行详细的成本收益测算。

**Type:** 学习
**Languages:** Python（标准库，简易 FP8/NVFP4 内存与成本计算器）
**Prerequisites:** 第 17 阶段 · 04（vLLM 服务内部原理），第 10 阶段 · 13（量化）
**Time:** 约 75 分钟

## 学习目标

- 解释为何即使权重采用 NVFP4，FP8 对 KV 缓存 (KV Cache) 和注意力机制 (Attention) 依然至关重要。
- 计算前沿模型在 BF16、FP8 和 NVFP4 精度下的高带宽内存占用 (HBM Footprint)，并分析节省的内存来源。
- 列举 TRT-LLM 所利用的 Blackwell 专属特性（Day-0 FP4 支持、MTP、分离式服务、全互联通信原语 (All-to-All Primitives)）。
- 评估在何种情况下，TRT-LLM 的 NVIDIA 生态绑定所带来的 7 倍成本优势（相较于 Hopper 架构上的 vLLM）是值得的。

## 问题背景

2026 年推理经济性 (Inference Economics) 的核心指标是“每美元能处理多少 Token”。该指标的答案取决于四个层层叠加的技术选型：硬件代际（Hopper H100/H200 对比 Blackwell B200/GB200）、计算精度（BF16 → FP8 → NVFP4）、服务引擎（vLLM 对比 SGLang 对比 TRT-LLM）以及编排架构（单体架构对比分离式架构对比 Dynamo）。

在 Hopper 架构搭配 vLLM 的方案中，一个 120B 混合专家模型 (Mixture of Experts, MoE) 的运行成本约为每百万 Token 0.09 美元。而在 Blackwell 架构搭配 TRT-LLM + Dynamo 的方案中，同一模型的成本降至约 0.012 美元——便宜了 7 倍。这部分差距部分源于硬件（Blackwell 的单 GPU 大语言模型吞吐量是 Hopper 的 11-15 倍）。另一部分则源于软件栈的优化：FP4 权重、MTP 草稿生成、分离式预填充/解码，以及用于 MoE 专家通信的 NVLink 5 全互联通信原语 (All-to-All Primitives)。

在 NVIDIA 技术栈之外，你无法复现这一性能表现。这正是其中的权衡取舍——以牺牲可移植性换取极致的经济性。理解各项技术栈选型分别贡献了多少性能差距，正是本课的核心目的。

## 核心概念

### 为什么 FP8 仍然是 KV Cache（键值缓存）的底线

2026 年的一个常见误区：认为 NVFP4 适用于所有场景。事实并非如此。KV Cache 需要 FP8（8-bit floating point，8 位浮点数），因为它存储的注意力键（attention keys）和值（attention values）跨越了极宽的动态范围（dynamic range）。将 KV 量化为 FP4 会导致灾难性的精度损失——分布的尾部会截断，注意力分数（attention scores）也会崩溃。FP8 的指数位（exponent bits）为 KV Cache 提供了所需的数值范围。

NVFP4（2025-2026）适用于权重（weights）和激活值（activations）。微缩放（microscaling）：每个权重块都拥有独立的缩放因子（scale factor），使得小块数据能够在不损失逐张量缩放（per-tensor scale）精度的前提下适应不同的动态范围。对于激活值而言，FP4 表现良好，因为单层内的激活值范围相对较小。

典型的 Blackwell 配置：

- 权重：NVFP4（4-bit microscaling，4 位微缩放）。
- 激活值：NVFP4。
- KV Cache：FP8。
- 注意力累加器（attention accumulator）：FP32（用于保证 Softmax 稳定性）。

### TRT-LLM 使用的 Blackwell 专属原语

- **Day-0 FP4 权重**：模型提供商直接交付 FP4 权重；TRT-LLM 加载时无需进行训练后转换（post-training conversion）。FP4 无需经过 AWQ / GPTQ 步骤。
- **多令牌预测（Multi-token prediction, MTP）**：与 EAGLE（第 17 阶段 · 05）理念相同，但已集成到 TRT-LLM 构建流程中。
- **分离式服务（Disaggregated serving）**：预填充（prefill）和解码（decode）部署在独立的 GPU 资源池中，KV Cache 通过 NVLink 或 InfiniBand 传输。理念与 Dynamo（第 17 阶段 · 20）相同。
- **全对全通信原语（All-to-all communication primitives）**：与 Hopper 架构相比，NVLink 5 将混合专家模型（Mixture of Experts, MoE）专家间的通信延迟降低了 3 倍。TRT-LLM 的 MoE 内核（kernels）已针对此进行了优化。
- **NVFP4 + MXFP8 微缩放**：在 Blackwell Tensor Core（张量核心）上通过硬件加速处理缩放因子。

### 你需要记住的关键数据

- 通过 TRT-LLM 运行 GPT-OSS-120B 模型时，HGX B200 的成本为每百万令牌（M tokens）0.02 美元。
- 通过 Dynamo（编排 TRT-LLM）运行时，GB200 NVL72 的成本为每百万令牌 0.012 美元。
- 在可比工作负载下，H100 + vLLM 的成本约为每百万令牌 0.09 美元。
- TRT-LLM 在 2026 年三个月的更新中实现了 2.8 倍的吞吐量（throughput）提升。
- 单 GPU 的大语言模型（LLM）吞吐量，Blackwell 相比 Hopper 提升 11-15 倍。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 在所有提交的任务中均占据主导地位。

### FP4 在质量上的实际代价

NVFP4 是一种激进的量化方案。在重度推理工作负载（如思维链（chain-of-thought）、数学计算、长上下文代码生成）中，FP4 权重会导致可见的质量下降。逐块校准（per-block calibration）可以缓解但无法完全消除这一问题。发布推理模型的团队通常采用 FP8 权重 + FP4 激活值作为折中方案，或者坚持在 H200 上全程使用 FP8。

准则：在正式采用 NVFP4 权重之前，务必在评估集（eval set）上验证任务质量。

### 为什么这是一个绑定 NVIDIA 的决策

TRT-LLM 基于 C++ + CUDA + 闭源内核构建。模型必须针对特定的 GPU SKU 进行编译。不支持 AMD、Intel 或 ARM。如果你的基础设施（infrastructure）策略是多供应商混合架构，那么 TRT-LLM 不适合作为 TRT-LLM 服务层的首选——你仍然可以在混合硬件上使用 vLLM 提供服务。如果你采用纯 NVIDIA 架构，那么 7 倍的性能差距足以抵消供应商绑定的成本。

### 2026 年实战指南

对于年度推理账单超过 1 亿美元的企业，继续使用 Hopper + vLLM 意味着将错失 7-10 倍的成本优化空间。应将成本主导型工作负载迁移至 Blackwell + TRT-LLM + Dynamo 架构。将实验层保留在 H100 + vLLM 上以保障模型迭代速度。在生产环境部署前，务必对每个经过 NVFP4 转换的模型进行质量验证。

### 分离式架构的额外收益

TRT-LLM 的分离式服务（prefill 和 decode 资源池独立）已在第 17 阶段 · 20 中深入探讨。在 Blackwell 架构上，性能乘数效应会叠加：FP4 权重 × MTP 加速 × 分离式部署 × 缓存感知路由（cache-aware routing）。文中提到的 7 倍提升正是基于这一完整技术栈得出的。

## Use It

`code/main.py` computes HBM footprint, decode throughput (memory-bound regime), and $/M-tokens for a model across three stacks: H100 + BF16 + vLLM, H100 + FP8 + vLLM, B200 + NVFP4/FP8 + TRT-LLM. Run it to see the compounding effect and the share of the gap each change contributes.

## Ship It

This lesson produces `outputs/skill-trtllm-blackwell-advisor.md`. Given a workload, model size, and annual token volume, it decides whether the Blackwell + TRT-LLM stack is worth the NVIDIA-lock.

## Exercises

1. Run `code/main.py`. On a 120B MoE with 30% active parameters, compute the memory-bandwidth-limited decode throughput on H100 BF16, H100 FP8, and B200 NVFP4/FP8. Where does the biggest jump come from?
2. A customer spends $2M/year on H100 + vLLM. What is the break-even number of Blackwell GPUs they need to buy to amortize a migration to TRT-LLM in 12 months, given the 7x economic gap?
3. You see accuracy drop 3 points on MATH after NVFP4 weight conversion. Name two recovery paths: one quality-first (keep FP8 weights), one cost-first (calibrate with in-domain data).
4. Read the MLPerf v6.0 inference results. Which task has the smallest Blackwell-over-Hopper gap, and why?
5. Compute the HBM needed for a 405B model at NVFP4 weights + FP8 KV cache at 128k context. Does it fit on a single GB200 NVL72 node?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| FP8 | "eight-bit float" | 8-bit floating point; used for KV cache and attention due to dynamic range |
| NVFP4 | "four-bit micro" | NVIDIA's 4-bit microscaling FP format; weights and activations on Blackwell |
| MXFP8 | "MX eight" | Microscaling FP8 variant; hardware-accelerated on Blackwell Tensor Cores |
| Day-0 FP4 | "ship FP4 weights" | Model providers release weights already in FP4; no post-train conversion step |
| MTP | "multi-token prediction" | TRT-LLM's integrated speculative-decoding draft (Phase 17 · 05) |
| Disaggregated serving | "split prefill/decode" | Prefill and decode on separate GPU pools; KV transferred over NVLink/IB |
| All-to-all | "MoE expert comm" | Communication pattern routing tokens to expert GPUs; NVLink 5 cuts 3x |
| InferenceX | "SemiAnalysis inference bench" | The 2026 industry-accepted cost-per-token benchmark |

## Further Reading

- [NVIDIA — Blackwell Ultra MLPerf 推理 (MLPerf Inference) v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月的 MLPerf 测试结果。
- [NVIDIA — Blackwell 上的混合专家 (MoE) 推理](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 全互联 (all-to-all) 与 MoE 内核。
- [TensorRT-LLM 概述](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方引擎文档。
- [NVIDIA — 推出 Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — 构建于 TRT-LLM 之上的解耦编排 (disaggregated orchestration)。
- [MLPerf 推理](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 性能数据的基准测试 (benchmark) 套件。