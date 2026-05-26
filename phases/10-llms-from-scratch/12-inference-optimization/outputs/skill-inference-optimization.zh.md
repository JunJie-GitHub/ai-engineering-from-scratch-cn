---
name: 推理优化技能
description: 诊断并优化大语言模型推理服务的吞吐量、延迟和成本
version: 1.0.0
phase: 10
lesson: 12
tags: [推理, kv缓存, 批处理, 投机解码, vllm, 优化]
---

# 大语言模型推理优化模式 (LLM Inference Optimization Pattern)

包含两个阶段：预填充 (prefill，计算受限且并行) 和解码 (decode，内存受限且串行)。每项优化均针对其中一个或两个阶段。

Request -> Prefill (process prompt) -> Decode (generate tokens) -> Response
              |                            |
         Compute-bound               Memory-bound
         Optimize: fusion,           Optimize: batching,
         prefix caching              quantization, speculation

## 决策框架

### 步骤 1：识别瓶颈

测量工作负载的 ops:byte（操作数:字节数）比率：

| ops:byte | 受限类型 | 优化方向 |
|----------|-------|-----------------|
| < 50 | 内存受限 (Memory-bound) | 量化 KV 缓存，增加批处理大小 |
| 50-200 | 过渡阶段 | 两者皆重要，从批处理开始 |
| > 200 | 计算受限 (Compute-bound) | 内核融合，张量并行，FP8 |

### 步骤 2：选择推理引擎

- **默认推荐**：vLLM（模型支持最广，采用 PagedAttention，提供 OpenAI 兼容 API）
- **多轮对话 / 结构化输出**：SGLang（基于 RadixAttention 的前缀缓存，约束解码）
- **极致 NVIDIA 吞吐量**：TensorRT-LLM（内核融合，H100 上的 FP8 支持）

### 步骤 3：按顺序应用优化策略

1. **KV 缓存 (KV cache)** -- 始终开启，无副作用
2. **连续批处理 (Continuous batching)** -- 始终开启，无副作用（vLLM/SGLang 默认启用）
3. **前缀缓存 (Prefix caching)** -- 若存在共享的系统提示词则启用（大多数聊天机器人适用）
4. **量化 (Quantization)** -- KV 缓存采用 INT8/FP8 可将内存占用降低 2-4 倍，且质量损失极小
5. **投机解码 (Speculative decoding)** -- 当延迟比吞吐量更重要时添加
6. **张量并行 (Tensor parallelism)** -- 当模型无法装入单张 GPU 时，跨 GPU 拆分

## KV 缓存内存计算公式

per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_param
total = per_token * sequence_length * num_concurrent_users

常见模型快速参考（BF16 精度）：

| 模型 | 单 Token 占用 | 100 用户 @ 4K 上下文 |
|-------|-----------|----------------|
| Llama 3 8B | 32 KB | 12.5 GB |
| Llama 3 70B | 320 KB | 125 GB |
| Llama 3 405B | 504 KB | 197 GB |

## 投机解码检查清单

- 草稿模型 (Draft model) 应比目标模型小 5-10 倍（例如为 70B 模型使用 8B 草稿模型）
- 接受率 (Acceptance rate) 需 > 70% 才能实现显著加速
- 最适合可预测文本（代码、结构化输出、自然语言）
- 最不适合高创造性/重采样任务（降低温度参数有助于改善）
- 对于大多数工作负载，性能排序为：EAGLE > 草稿-目标模型 > n-gram

## 常见误区

- 以 batch=1 运行解码阶段（内存受限，GPU 计算单元 95% 处于空闲状态）
- 分配连续的 KV 缓存块（应使用 PagedAttention，可实现接近零的内存碎片）
- 当 80% 的请求共享相同系统提示词时，忽略前缀缓存
- 为模型权重过度分配 GPU 显存，导致 KV 缓存无可用空间
- 仅测量吞吐量而忽略延迟（首字延迟 TTFT 高达 10 秒时，高吞吐量毫无意义）
- 在高温度参数下使用投机解码（接受率会降至 50% 以下）

## 监控检查清单

- 首字延迟 (Time to first token, TTFT)：预填充阶段延迟，交互式使用目标 < 500ms
- 词间延迟 (Inter-token latency, ITL)：解码速度，流式输出目标 < 50ms
- 吞吐量 (Throughput)：所有并发用户的总 token/秒 数
- KV 缓存利用率：已分配缓存的实际使用百分比
- 批处理利用率：每次迭代中批处理槽位的填充百分比
- 队列深度：等待批处理槽位的请求数量