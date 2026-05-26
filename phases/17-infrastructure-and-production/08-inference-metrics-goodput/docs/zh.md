# 推理指标（Inference Metrics）— TTFT、TPOT、ITL、Goodput 与 P99

> 四个指标决定了推理部署是否有效。首字延迟（Time To First Token, TTFT）包含预填充（Prefill）、排队和网络时间。单令牌输出时间（Time Per Output Token, TPOT，等同于令牌间延迟 Inter-Token Latency, ITL）是每个令牌的内存受限（Memory-bound）解码成本。端到端延迟（End-to-End Latency）等于 TTFT 加上 TPOT 乘以输出长度。吞吐量（Throughput）是整个集群每秒处理的令牌总数。但对产品而言真正重要的是有效吞吐量（Goodput）—— 即同时满足所有服务等级目标（Service Level Objective, SLO）的请求比例。高吞吐量但低有效吞吐量意味着你正在处理那些无法及时送达用户的令牌。2026 年在 TRT-LLM 上运行 Llama-3.1-8B-Instruct 的参考数据：平均 TTFT 为 162 毫秒，平均 TPOT 为 7.33 毫秒，平均端到端延迟为 1,093 毫秒。务必报告 P50、P90 和 P99 分位数（Percentiles）——绝不要只报告平均值（Mean）。同时注意测量陷阱：GenAI-Perf 在计算 ITL 时排除了 TTFT，而 LLMPerf 则将其包含在内；对于同一次运行，这两个工具给出的 TPOT 结果并不一致。

**Type:** 学习
**Languages:** Python（标准库，简易分位数计算器与有效吞吐量报告器）
**Prerequisites:** 第 17 阶段 · 04（vLLM 服务内部机制）
**Time:** 约 60 分钟

## 学习目标

- 准确定义 TTFT、TPOT、ITL、端到端延迟（End-to-End Latency, E2E）、吞吐量（Throughput）和有效吞吐量（Goodput），并指出它们各自衡量的系统组件。
- 解释为何平均值（Mean）不适用于大语言模型（Large Language Model, LLM）服务场景，以及如何解读 P50/P90/P99。
- 构建一个多约束服务等级目标（SLO）（例如：TTFT < 500 毫秒 且 TPOT < 15 毫秒 且 E2E < 2 秒），并据此计算有效吞吐量。
- 列举两个对同一次运行给出不同 TPOT 结果的基准测试工具，并解释其原因。

## 问题背景

“我们的吞吐量达到了每秒 15,000 个令牌。”那又怎样？如果 40% 的请求端到端延迟超过了 2 秒，用户就会直接放弃会话。仅凭吞吐量无法告诉你产品是否真正可用。

推理过程涉及多个延迟维度，且每个维度的瓶颈表现各不相同。预填充（Prefill）阶段受计算限制（Compute-bound），其耗时随提示词（Prompt）长度增加而增长。解码（Decode）阶段受内存限制，其耗时随批次大小（Batch Size）增加而变化。排队延迟属于运维调度问题。网络延迟则受物理距离制约。你需要为每个维度设定独立的指标，需要依赖分位数（Percentiles）进行评估，更需要一个能综合回答“用户是否获得了预期体验”的单一指标——这就是有效吞吐量。

## 核心概念

### TTFT — 首字延迟（Time To First Token）

`TTFT = queue_time + network_request + prefill_time`

当提示词（prompt）较长时，预填充（prefill）阶段占主导地位。在 H100 上运行 FP8 精度的 Llama-3.3-70B 模型时，一个 32k 长度的提示词仅纯预填充就需要约 800 毫秒。排队时间（queue time）反映了调度器在负载下的行为。网络请求耗时（network request）是包含 TLS 握手在内的物理传输时间。TTFT 是用户在接收到任何流式返回内容前所感知到的延迟。

### TPOT / ITL — 词间延迟（Inter-Token Latency）

同一指标有多种叫法。`TPOT`（Time Per Output Token，单输出词耗时）、`ITL`（Inter-Token Latency，词间延迟）、`decode latency per token`（单解码词延迟）——本质相同。它指的是首个词之后，连续流式输出词之间的时间间隔。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在相同的 Llama-3.3-70B H100 技术栈上，若启用分块预填充（chunked prefill），TPOT 平均值约为 7 毫秒。若未启用分块预填充，当相邻序列进行长预填充时，TPOT 可能飙升至 50 毫秒。请重点关注 P99 值，而非平均值。

### E2E 延迟（End-to-End Latency）

`E2E = TTFT + TPOT * output_tokens + network_response`

对于长输出（>500 个词），E2E 延迟主要由 TPOT 决定。对于提示词较长但输出较短的场景，E2E 延迟则主要由 TTFT 决定。报告 E2E 延迟时，应基于输出长度进行条件化统计。

### 吞吐量（Throughput）

`throughput = total_output_tokens / elapsed_time`

这是一个聚合指标。它能反映集群的整体效率，但无法体现单个请求的健康状况。

### 有效吞吐量（Goodput）—— 你真正应该关注的指标

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

服务等级目标（Service Level Objective, SLO）是一个多约束条件。只有当所有约束条件均满足时，一个请求才算“有效”。Goodput 即满足条件的请求占比。在 60% 的有效率下追求高吞吐量是失败的；而在 99% 的有效率下接受较低的吞吐量才是正确目标。

到 2026 年，Goodput 已成为 MLPerf Inference v6.0 提交报告中的核心指标，也是 AI 平台提供商内部服务等级协议（Service Level Agreement, SLA）追踪的标准。

### 为什么平均值是错误的统计指标

大语言模型（Large Language Model, LLM）的延迟分布呈右偏态（right-skewed）。在一个解码批次中，若存在一个进行长预填充的相邻序列，可能会导致 500 个词的 TPOT 约为 7 毫秒，而另外 20 个词的 TPOT 飙升至约 60 毫秒。此时平均 TPOT 为 9 毫秒，但 P99 TPOT 高达 65 毫秒。用户会频繁遭遇 P99 级别的延迟——这正是导致用户流失的原因。

务必报告三元组指标（P50、P90、P99）。为了优化用户体验，P99 才是你真正需要优化的目标。

### 参考数据 —— 2026 年 TRT-LLM 上的 Llama-3.1-8B-Instruct

- 平均 TTFT：162 毫秒
- 平均 TPOT：7.33 毫秒
- 平均 E2E 延迟：1,093 毫秒
- P99 TPOT：根据分块预填充配置的不同，在 10-25 毫秒之间波动。

这些是 NVIDIA 官方发布的参考基准数据。该数值会随模型规模（70B 模型约为 3-5 倍）、硬件（H100 与 B200 相比约差 3 倍）以及负载情况而变化。

### 测量陷阱

2026 年最常用的两款基准测试工具（benchmark tools）对同一次运行的 TPOT 计算结果存在分歧：

- **NVIDIA GenAI-Perf**：在计算 ITL 时排除 TTFT。ITL 从第 2 个词开始计算。
- **LLMPerf**：包含 TTFT。ITL 从第 1 个词开始计算。

假设某请求的 TTFT 为 500 毫秒，总解码耗时 700 毫秒，共输出 100 个词。GenAI-Perf 报告的 `ITL = 700/99 = 7.07 ms`，而 LLMPerf 报告的 `ITL = 1200/100 = 12.00 ms`。工具的选择会直接改变最终数值。

务必注明所使用的工具，并公开具体的计算定义。

### 构建 SLO

2026 年面向消费者的 70B 聊天模型，合理的 SLO 设定如下：

- TTFT P99 <= 800 毫秒。
- TPOT P99 <= 25 毫秒。
- 输出词数 <300 时，E2E P99 <= 3 秒。
- Goodput 目标 >= 99%。

企业级 SLO 通常会收紧 TTFT 要求（200-400 毫秒），并放宽 E2E 限制。关键在于将这些指标明确记录下来，全面测量这三项数据，并将 Goodput 作为单一综合指标进行追踪。

### 测量方法

- 使用真实流量或贴近现实的合成流量进行测试（例如使用 LLMPerf 并配置 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- 基准测试的并发目标（concurrency）应设定为峰值并发量的 2 倍。
- 执行 30-50 轮迭代，对合并后的样本数据计算百分位数。
- 发布结果时需附带工具名称、工具版本、模型、硬件、并发数及提示词分布信息。

## 使用它

`code/main.py` 是一个简易的有效吞吐量（goodput）计算器。它会生成一个合成的延迟分布（latency distribution），应用服务等级目标（SLO），并计算有效吞吐量。此外，它还展示了在同一请求轨迹（trace）上，GenAI-Perf 与 LLMPerf 在每输出令牌时间（TPOT）指标上的差异。

## 交付上线

本章节将生成 `outputs/skill-slo-goodput-gate.md` 文件。给定工作负载（workload）和 SLO，它会生成一份支持 CI/CD 的基准测试方案（benchmark recipe），该方案以有效吞吐量而非传统吞吐量（throughput）作为部署门禁的判定标准。

## 练习

1. 运行 `code/main.py`。生成一个带有 1% 尾部尖峰（tail spike）的分布。当将 P99 TPOT 从 30 ms 收紧至 15 ms 时，有效吞吐量会发生怎样的变化？
2. 某供应商宣称“在 H100 上运行 Llama 3.3 70B 可达 15,000 tok/s”。在采信该数据前，请列出三个需要追问的问题。
3. 为什么分块预填充（chunked prefill）能保护 P99 TPOT，却无法保护平均 TPOT？
4. 为语音助手构建一个面向消费者的 SLO（首个令牌是被听到而非被阅读的）。哪个指标对用户感知最明显？
5. 阅读 LLMPerf 的 README 和 GenAI-Perf 的文档。找出另外三个这两款工具定义或测量结果存在分歧的指标。

## 关键术语

| 术语 | 通常的说法 | 实际含义 |
|------|----------------|------------------------|
| TTFT | “首令牌时间” | 排队 + 网络 + 预填充；在长提示词场景下主要由预填充主导 |
| TPOT | “每输出令牌时间” | 首个令牌之后，受内存带宽限制的逐令牌解码成本 |
| ITL | “令牌间延迟” | 在大多数工具中与 TPOT 相同（并非全部——参见 GenAI-Perf） |
| E2E | “端到端” | TTFT + TPOT * output_len；在此基础上叠加响应侧的网络延迟 |
| Throughput（吞吐量） | “tok/s” | 集群效率；脱离延迟百分位数则毫无参考价值 |
| Goodput（有效吞吐量） | “SLO 达标率” | 同时满足所有 SLO 约束条件的请求比例 |
| P99 | “尾部延迟” | 百分之一的最差情况延迟；衡量用户体验的核心指标 |
| SLO 多约束 | “联合约束” | 三项延迟边界的逻辑与（AND）关系；只要违反其中任意一项，请求即判定为失败 |
| GenAI-Perf 与 LLMPerf | “工具陷阱” | 两款工具在 ITL 是否包含 TTFT 的定义上存在分歧 |

## 延伸阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的标准定义。
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 替代性定义与测量方案。
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 真实部署环境下的实测数据。
- [LLMPerf](https://github.com/ray-project/llmperf) — 基于 Ray 的开源基准测试工具。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 官方基准测试工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 业界公认的基于有效吞吐量的基准测试。