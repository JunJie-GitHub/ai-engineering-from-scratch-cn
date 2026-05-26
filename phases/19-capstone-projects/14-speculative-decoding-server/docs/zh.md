# 综合项目 (Capstone) 14 — 投机解码 (Speculative Decoding) 推理服务器

> vLLM 0.7 集成的 EAGLE-3 在真实流量下可实现 2.5-3 倍的吞吐量 (Throughput) 提升。P-EAGLE（AWS 2026）进一步推动了并行推测 (Parallel Speculation) 技术。SGLang 的 SpecForge 实现了大规模草稿头 (Draft Heads) 训练。Red Hat 的 Speculators 中心发布了针对主流开源模型的对齐草稿模型。TensorRT-LLM 使投机解码在 NVIDIA 平台上成为原生支持的一等特性。2026 年的生产级服务栈通常采用 vLLM 或 SGLang，搭配 EAGLE 系列草稿模型、FP8 或 INT4 量化 (Quantization)，以及基于队列等待时间 (Queue-Wait) 的 HPA (Horizontal Pod Autoscaler)。本综合项目旨在以 2.5 倍以上的基线吞吐量部署两个开源模型，并提供完整的尾部延迟 (Tail Latency) 报告。

**Type:** 综合项目
**Languages:** Python（服务部署）、C++ / CUDA（内核检查）、YAML（配置文件）
**Prerequisites:** 阶段 3（深度学习）、阶段 7（Transformer）、阶段 10（从零构建大语言模型）、阶段 17（基础设施）
**Phases exercised:** P3 · P7 · P10 · P17
**Time:** 30 小时

## 问题背景

到 2026 年，投机解码已成为一项通用技术。EAGLE-3 草稿头基于目标模型的隐藏状态进行训练，可提前预测 N 个词元 (Token)；目标模型仅需单次前向传播即可完成验证。60-80% 的接受率 (Acceptance Rate) 可转化为 2-3 倍的端到端吞吐量。vLLM 0.7 已原生集成该功能。SGLang 结合 SpecForge 提供了完整的训练流水线。Red Hat 的 Speculators 发布了针对 Llama 3.3 70B、Qwen3-Coder-30B MoE 和 GPT-OSS-120B 的对齐草稿模型。

真正的技术难点在于服务运维，而非模型本身。接受率会随流量分布（ShareGPT 对话数据、代码数据或垂直领域数据）而发生漂移。在发生拒绝时的尾部延迟甚至比不使用投机解码时更差——因此必须报告多个批次大小 (Batch Size) 下的 p99 延迟，而不仅仅是稳态下的词元每秒吞吐量。与 Anthropic / OpenAI API 相比，每百万词元的成本是衡量方案可行性的关键指标。

## 核心概念

投机解码包含两个层级。**草稿模型** (Draft Model，如 EAGLE-3 头、N-gram 或较小的目标对齐模型) 在每一步提出 k 个候选词元。**目标模型** (Target Model) 在单次前向传播中验证所有 k 个词元；任何被接受的前缀将替换原有的贪婪解码路径。接受率取决于草稿模型与目标模型的对齐程度以及输入数据的分布。

在大多数流量场景下，EAGLE-3 的表现优于 N-gram 草稿模型。P-EAGLE 通过并行推测技术构建更深的草稿树。其权衡在于：由于验证阶段计算量更大，发生拒绝时的 p99 延迟会更高。服务配置必须按批次大小分桶报告延迟，以暴露这一问题。

部署环境为 Kubernetes。vLLM 0.7 在每个 GPU 或张量并行 (Tensor Parallel) 分片上运行一个副本。HPA 基于队列等待时间而非 CPU 使用率进行自动扩缩容。FP8（Marlin）和 INT4（AWQ）量化技术可将 GPU 显存占用控制在 H100 / H200 的规格范围内。端到端报告需包含吞吐量、接受率、批次大小为 1/8/32 时的 p50/p99 延迟，以及每百万词元的成本（$/1M tokens）。

## 架构设计

request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric

## 技术栈 (Stack)

- 服务框架 (Serving)：vLLM 0.7 或 SGLang 0.4
- 推测解码方法 (Speculative methods)：EAGLE-3 草稿头 (heads)、P-EAGLE 并行推测 (parallel)、n-gram 回退 (ngram fallback)
- 草稿模型训练 (Draft training)：SpecForge (SGLang) 或 Red Hat Speculators
- 目标模型 (Target models)：Llama 3.3 70B、Qwen3-Coder-30B MoE (混合专家模型)、GPT-OSS-120B
- 量化 (Quantization)：FP8 (Marlin)、INT4 AWQ
- 部署 (Deployment)：Kubernetes + NVIDIA 设备插件；基于队列等待指标的 HPA (水平 Pod 自动扩缩容)
- 评估 (Eval)：ShareGPT、MT-Bench-v2、GSM8K、HumanEval，用于测量跨领域接受率 (acceptance rate)
- 参考基准 (Reference)：TensorRT-LLM 推测解码，作为厂商基线

## 构建指南 (Build It)

1. **目标模型准备。** 选择 Llama 3.3 70B。通过 Marlin 将其量化为 FP8。在 1xH100（或 2x 张量并行 (tensor-parallel)）环境下使用 vLLM 0.7 进行部署。

2. **草稿模型来源。** 从 Red Hat Speculators 获取已对齐的 EAGLE-3 草稿头（或使用 SpecForge 自行训练）。将其加载到 vLLM 的推测解码配置中。

3. **基线数据。** 在启用推测解码前：记录批量大小 (batch) 为 1/8/32 时的吞吐量 (throughput)、p50/p99 延迟 (latency) 以及 GPU 利用率。发布基线报告。

4. **启用 EAGLE-3。** 切换配置并重新运行相同的基准测试。报告加速比、接受率以及 p99 尾部延迟 (tail-latency) 的变化量。

5. **P-EAGLE。** 启用并行推测；对比更深的草稿树与串行 EAGLE-3 的表现。报告 P-EAGLE 带来收益与造成损耗的拐点。

6. **领域流量 (Domain traffic)。** 在同一服务器上分别运行 ShareGPT、HumanEval 及特定领域流量。测量各数据分布下的接受率。识别草稿模型发生分布漂移 (drift) 的时机。

7. **第二个目标模型。** 在 Qwen3-Coder-30B MoE 上运行相同流程。草稿生成会更棘手（受 MoE 路由噪声 (routing noise) 影响）。报告结果。

8. **K8s HPA。** 在 Kubernetes 下部署，配置 HPA 跟踪 `queue_wait_ms` 指标。演示当负载增至三倍时的横向扩展 (scale-out) 效果。

9. **成本对比 (Cost comparison)。** 计算每百万输出 token 的成本 ($/1M tokens)，并与 Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 在相同评估集上的成本进行对比。发布报告。

## 使用方法 (Use It)

$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput

## 发布上线 (Ship It)

`` `outputs/skill-inference-server.md` `` 描述了本次交付物。包含经过实测的推理服务栈（serving stack）、投机解码（speculative decoding）实现、完整的基准测试报告以及 Kubernetes (K8s) 部署方案。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 相较于基线（baseline）的实测加速比 | 在两个模型上，质量匹配的情况下吞吐量（throughput）提升 2.5 倍以上 |
| 20 | 真实流量下的接受率（acceptance rate） | 按分布统计的接受率报告 |
| 20 | P99 尾部延迟（P99 tail-latency）控制 | 批处理大小（batch size）为 1/8/32 时，开启与关闭投机解码的 P99 延迟 |
| 20 | 运维（Ops） | K8s 部署、基于队列等待时间的水平自动扩缩容（HPA）、平滑发布 |
| 15 | 文档与方法论 | 清晰说明改动内容及原因 |
| **100** | | |

## 练习

1. 测量草稿模型（draft model）落后目标模型一个版本时的接受率下降情况（例如 Llama 3.3 到 3.4 的版本漂移）。构建相应的监控告警。

2. 实现 N-gram 回退机制（ngram-fallback）：当 EAGLE-3 的接受率低于设定阈值时，切换至 N-gram 草稿。报告可靠性提升情况。

3. 运行受控的混合专家模型（Mixture of Experts, MoE）实验：对比在相同 Qwen3-Coder-30B 模型中注入路由噪声（routing noise）与不注入的情况。测量草稿接受率的敏感度。

4. 扩展至 H200 (141 GB) 硬件。报告每个副本可承载的模型规模余量（headroom）提升情况，并评估是否能够服务未量化（unquantized）的 Llama 3.3 70B 模型。

5. 在相同的 H100 硬件上对 TensorRT-LLM 的投机解码进行基准测试。报告其相较于 vLLM 的优势所在。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Draft model | "Speculator" | 小型模型，为目标模型生成 N 个词元（token）以供验证 |
| EAGLE-3 | "2026 draft architecture" | 基于目标模型隐藏状态（hidden states）训练的草稿头（draft head）；接受率约 75% |
| P-EAGLE | "Parallel speculation" | 树状草稿分支，在目标模型单次前向传播中完成验证 |
| Acceptance rate | "Hit rate" | 无需重新采样即被接受的草稿词元比例 |
| Quantization | "FP8 / INT4" | 降低权重精度，以便在 GPU 显存（GPU memory）中容纳更大模型 |
| Queue wait | "HPA metric" | 请求在推理开始前于待处理队列中的等待时间 |
| Speculators hub | "Aligned drafts" | Red Hat Neural Magic 提供的针对常见开源模型的 EAGLE 草稿模型中心 |

## 延伸阅读

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — 参考推理服务栈
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 并行投机解码论文及集成方案
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 草稿头训练流水线
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 对齐的草稿模型中心
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — 厂商替代方案
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) — 商业架构参考
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — 核心方法论文
- [vLLM repository](https://github.com/vllm-project/vllm) — 代码与基准测试