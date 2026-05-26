---
name: 量化技能
description: 根据硬件、质量与延迟约束，为部署大语言模型（LLM）选择合适的量化策略
version: 1.0.0
phase: 10
lesson: 11
tags: [量化, 推理, 部署, 优化, fp8, int4, int8, gptq, awq, gguf]
---

# 量化决策框架

在部署语言模型时，请使用此框架选择合适的数值格式（Number Format）、量化方法（Quantization Method）以及质量验证策略。

## 输入要求

请提供：
- **模型**（名称、参数量、原始精度）
- **目标硬件**（GPU 型号/显存、CPU、Apple Silicon、边缘设备）
- **延迟目标**（每秒生成 Token 数、首 Token 延迟时间）
- **质量底线**（最大可接受困惑度（Perplexity）增幅、基准测试（Benchmark）差异）
- **服务模式**（批处理大小、最大上下文长度、并发用户数）

## 快速选择指南

| 您的场景 | 格式 | 方法 | 预期质量损失 |
|---------------|--------|--------|----------------------|
| H100 GPU，追求最大吞吐量 | FP8 E4M3 | H100 原生类型转换 | < 0.1% |
| A100/A10，需要 2 倍吞吐量 | INT8 | LLM.int8() 或 SmoothQuant | < 0.5% |
| 单张 24GB GPU，70B 模型 | INT4 | AWQ 或 GPTQ | 1-3% |
| MacBook / Apple Silicon | INT4 GGUF | 通过 llama.cpp 使用 Q4_K_M | 1-2% |
| 移动端 / 边缘设备 | INT4 或 INT3 | QAT（量化感知训练） + 设备专用方案 | 2-5% |
| 极致压缩，允许一定损失 | INT2 | QuIP# 或 AQLM | 5-15% |
| 训练（混合精度） | BF16 + FP32 累加 | 框架原生支持 | 0% |

## 按组件选择精度

并非所有张量（Tensor）都应采用相同的处理方式。

| 组件 | 安全最低精度 | 推荐精度 | 避免使用 |
|-----------|-------------|-------------|-------|
| FFN（前馈神经网络）权重 | INT4 | INT4 (AWQ/GPTQ) | 未经 QAT 的 INT2 |
| 注意力权重 | INT4 | INT8 或 FP8 | INT2 |
| 嵌入层（Embedding Layer） | INT8 | FP16（保持原始） | INT4 |
| 输出头（Output Head） | INT8 | FP16（保持原始） | INT4 |
| KV 缓存（KV Cache） | FP8 | FP8 或 INT8 | 长上下文下的 INT4 |
| 注意力 Logits | FP16 | FP16 或 BF16 | INT8 |
| 激活值（推理阶段） | INT8 | FP8 或 INT8 | INT4 |

## 方法对比

### GPTQ
- **适用场景：** GPU 推理，且需要兼容 Hugging Face 的模型格式
- **校准数据：** 128 个样本，每个样本 2048 个 Token
- **耗时：** 在 A100 上处理 70B 模型约需 30-60 分钟
- **工具链：** `auto-gptq`, `exllama`, `exllamav2`
- **优势：** 经过充分验证，Hugging Face 上拥有庞大的模型库
- **劣势：** 量化应用速度慢于 AWQ，在部分模型上的质量略逊于 AWQ

### AWQ
- **适用场景：** GPU 推理，追求最佳的单位比特质量（Quality-per-bit）
- **校准数据：** 128 个样本
- **耗时：** 在 A100 上处理 70B 模型约需 15-30 分钟
- **工具链：** `autoawq`, `vLLM`（原生支持）
- **优势：** INT4 质量最佳，应用速度快，与 vLLM 深度集成
- **劣势：** 模型库规模小于 GPTQ

### GGUF
- **适用场景：** CPU 推理、Apple Silicon 平台、llama.cpp 生态
- **变体：** Q2_K, Q3_K_S/M/L, Q4_K_S/M, Q5_K_S/M, Q6_K, Q8_0, F16
- **推荐默认值：** Q4_K_M（质量与体积的最佳平衡）
- **工具链：** `llama.cpp`, `ollama`, `LM Studio`
- **优势：** 文件自包含，支持混合精度，生态庞大
- **劣势：** 对 GPU 优化不足（专为 CPU/Metal 设计）

### SmoothQuant
- **适用场景：** GPU 上的 INT8 推理，需要同时对权重和激活值进行量化
- **核心思想：** 通过逐通道缩放（Per-channel Scaling），将量化难度从激活值转移至权重
- **工具链：** `smoothquant`, `TensorRT-LLM`
- **优势：** 实现 W8A8（权重与激活值均为 INT8），带来 2 倍加速
- **劣势：** 仅支持 INT8，无法扩展至 INT4

## 质量验证流程

量化完成后，在部署前请执行以下验证：

1. **困惑度（Perplexity）测试。** 在 WikiText-2 或您的领域语料库上进行计算。差异值（Delta）< 0.5 为优秀，0.5-1.0 为良好，> 2.0 则存在问题。

2. **基准测试（Benchmark）扫描。** 运行 MMLU（通用知识）、GSM8K（数学）、HumanEval（代码）。数学与代码任务对精度损失最为敏感。

3. **输出对比（Output Comparison）。** 分别从原始模型和量化模型生成 100 条回复。使用大语言模型作为裁判（LLM-as-judge）计算胜率。目标：量化模型在超过 90% 的提示词（prompts）中胜出或打平。

4. **延迟测量（Latency Measurement）。** 分别在批次大小（batch size）为 1 和你的目标批次大小下，测量每秒生成的 token 数（tokens/second）。验证推理加速带来的收益是否足以弥补模型质量的损失。

5. **长上下文测试（Long-context Test）。** 若需服务长上下文（> 4K tokens），请在你的最大上下文长度下进行测试。KV 缓存（KV cache）的量化误差会随序列长度累积放大。

## 显存预算计算器（Memory Budget Calculator）

Weight memory (GB) = parameters (B) * bits / 8 / 1.073741824
KV cache per token (MB) = 2 * num_layers * d_model * bits / 8 / 1048576
KV cache for context (GB) = kv_per_token * max_context_length / 1024
Activation memory (GB) ~ 1-4 GB (relatively constant, depends on batch size)
Total = weight_memory + kv_cache + activation_memory + overhead (10-20%)

以 Llama 3 70B 模型在 INT4 量化、32K 上下文长度为例：
- 权重（Weights）：70B * 4 / 8 / 1.07 = 32.6 GB
- KV 缓存（FP16）：2 * 80 * 8192 * 16 / 8 / 1e9 * 32768 ≈ 40 GB
- KV 缓存（FP8）：≈ 20 GB
- 使用 FP8 KV 缓存的总显存：≈ 55 GB（可装入单张 80GB A100 显卡）

## 常见错误（Common Mistakes）

| 错误做法 | 失败原因 | 修复方案 |
|---------|-------------|-----|
| 将嵌入层（Embedding Layer）量化为 INT4 | 首层误差会在整个模型中被逐级放大 | 将嵌入层保持在 FP16 或 INT8 |
| 对 INT4 使用逐张量（Per-tensor）缩放因子 | 单个异常行会破坏所有行的精度 | 改用逐通道（Per-channel）或逐组（Per-group）缩放因子 |
| 未对 GPTQ/AWQ 进行校准 | 缺乏代表性数据会导致缩放因子计算错误 | 使用 128 个来自你业务领域的样本进行校准 |
| 所有层使用相同的位宽 | 模型首层和末层对量化更为敏感 | 采用混合精度：为首层和末层分配更高位宽 |
| 在极长上下文下量化 KV 缓存 | 误差会随序列长度呈二次方累积 | KV 缓存使用 FP8 而非 INT4 |
| 跳过质量验证 | 部分模型量化效果较差（尤其在临界条件下） | 务必运行困惑度（Perplexity）测试与下游任务评估 |

## 部署方案（Deployment Recipes）

### 方案 1：在 GPU 服务器上使用 vLLM 与 AWQ
pip install vllm autoawq
vllm serve model-awq --quantization awq --dtype half --max-model-len 8192

### 方案 2：在 MacBook 上使用 llama.cpp 与 GGUF
./llama-server -m model.Q4_K_M.gguf -c 4096 -ngl 99

### 方案 3：在 H100 上使用 TensorRT-LLM 与 FP8
trtllm-build --model_dir model --output_dir engine --dtype float16 --use_fp8
