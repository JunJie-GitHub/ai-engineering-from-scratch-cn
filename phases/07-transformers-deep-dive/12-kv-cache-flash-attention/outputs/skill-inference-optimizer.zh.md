---
name: inference-optimizer
description: 为新的推理部署 (inference deployment) 选择注意力机制实现 (attention implementation)、KV 缓存 (KV cache) 策略、量化 (quantization) 方案及投机解码 (speculative decoding)。
version: 1.0.0
phase: 7
lesson: 12
tags: [transformers, 推理 (inference), flash-attention, kv-cache]
---

给定一个推理部署配置（模型名称及参数、目标硬件、并发数、最大上下文长度、延迟服务等级目标 (latency SLO)、吞吐量目标 (throughput target)），输出以下内容：

1. 服务栈 (serving stack)。vLLM（默认生产环境）、SGLang（单 token 延迟最低）、TensorRT-LLM（NVIDIA 硬件最优）、llama.cpp（边缘设备/CPU）、MLX（Apple 芯片）。附一句选择理由。
2. 注意力机制实现。Flash Attention 2（Ampere/Ada 架构默认）、Flash Attention 3（Hopper 架构）、Flash Attention 4（Blackwell 架构，仅支持前向传播 (forward-only)）。需指定备选方案。
3. KV 缓存。数据类型 (dtype)（默认 fp16，若支持则用 fp8）、分页式 (paged) 与连续式 (contiguous) 对比、前缀缓存 (prefix caching) 开关状态、并行采样 (parallel sampling) 时的共享 KV。
4. 量化。fp16 / bf16（默认）、int8（仅权重 (weight-only)）、针对权重的 AWQ / GPTQ / GGUF。仅在经过基准测试验证后才启用激活值量化 (activation quantization)。
5. 额外加速方案。投机解码（EAGLE 2 / Medusa / 草稿模型）、连续批处理 (continuous batching)（始终开启）、分块预填充 (chunked prefill)（适用于长提示词负载）、若存在重复提示词则启用前缀缓存。

拒绝为训练任务部署 Flash Attention 4 —— 其发布初期仅支持前向传播。在未针对目标任务进行质量影响基准测试前，拒绝推荐 fp8 KV 缓存。标记任何未采用分组查询注意力 (Grouped Query Attention, GQA) 的 70B+ 参数模型，指出其在 32K+ 上下文长度下 KV 缓存将难以管理。对于任何包含重复系统提示词的智能体/工具调用 (agent/tool-calling) 部署，必须要求开启前缀缓存。