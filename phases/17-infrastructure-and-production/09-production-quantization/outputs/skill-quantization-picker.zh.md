---
name: quantization-picker
description: 根据硬件、推理引擎、工作负载和质量容忍度，选择 2026 年的量化格式，并生成校准与验证计划。
version: 1.0.0
phase: 17
lesson: 09
tags: [quantization, awq, gptq, gguf, fp8, nvfp4, calibration]
---

根据给定的硬件（CPU / H100 / H200 / B200 / GB200 及数量）、推理引擎（Inference Engine，如 llama.cpp / vLLM / TRT-LLM / SGLang）、模型（参数量 + 任务类型——常规对话 / 逻辑推理 / 代码生成 / 多 LoRA 适配）以及质量容忍度（Quality Tolerance，可接受 HumanEval / MATH / MMLU 基准测试下降 N 分），选择一种量化格式（Quantization Format）并制定验证计划（Validation Plan）。

输出内容：

1. **格式推荐**。从以下选项中选择其一：GGUF Q4_K_M、GGUF Q5_K_M、GPTQ-Int4 + Marlin、AWQ-Int4 + Marlin、FP8、NVFP4 + FP8 KV，或组合堆叠方案。需依据决策树进行论证：CPU → GGUF；逻辑推理 → FP8；vLLM 上的多 LoRA → GPTQ；常规 GPU 对话 → AWQ；Blackwell 架构已验证 → NVFP4。
2. **显存预算**。报告权重（Weights）+ KV 缓存（KV Cache，按指定并发数 × 上下文长度计算）+ 激活值（Activations）的占用情况。确认其是否适配目标 GPU，或明确指出是否需要多 GPU 部署。
3. **校准计划**。数据集来源（AWQ/GPTQ 需匹配垂直领域数据；通用 C4/WikiText 仅作为备选）。样本数量（领域数据建议 500-2000 条）。验证集（从校准池中预留 10% 作为独立验证集）。
4. **验证计划**。评估集（Eval Set）需与任务匹配：代码生成使用 HumanEval，逻辑推理使用 MATH/MMLU，对话使用 MT-Bench。对比 BF16 基线模型与量化模型。若性能下降幅度 ≤ 质量容忍度，即可发布上线。
5. **KV 缓存决策**。需与权重量化分开考虑。逻辑推理任务推荐 FP8 KV；若注意力机制精度处于临界状态，则使用 BF16 KV；INT8 KV 仅在完成验证后方可采用。
6. **回滚路径**。在磁盘上保留 BF16/FP8 权重文件；设置切换标志，以便在生产环境质量下降时快速回退。

硬性拒绝条件：
- 在未使用评估集验证的情况下，为重度推理工作负载推荐 NVFP4 权重。
- 为垂直领域模型使用通用网页数据进行校准。必须始终使用领域内数据。
- 在高带宽内存（HBM）预算中遗漏 KV 缓存。必须逐项列出。
- 在未指明底层计算内核（Kernels）的情况下宣称吞吐量（Throughput）数据（Marlin-AWQ 与普通 AWQ 的性能差距可达 10 倍）。

拒绝规则：
- 若工作负载本身对质量极为敏感（如开放式创意生成、边缘案例推理），则拒绝激进的 INT4 量化。应保留 FP8 或 BF16。
- 若推理引擎为 llama.cpp，则拒绝除 GGUF 以外的任何格式。格式与引擎的匹配是基本前提。
- 若用户无法运行 1000 样本的评估测试，则予以拒绝。生产环境中严禁盲目量化。

输出要求：生成一页纸的量化选型报告，列出所选格式、HBM 预算、校准计划、验证计划、KV 缓存决策及回滚路径。最后附加一段“下一步测量指标”，根据核心风险指明需关注的一项指标：评估集性能差值（eval-set delta）、峰值并发下的 KV 缓存压力，或实际批次大小（Batch Size）下的吞吐量。