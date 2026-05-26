# LLM API 负载测试（Load Testing）—— 为什么 k6 和 Locust 会“撒谎”

> 传统负载测试工具并非为流式响应（Streaming Responses）、可变输出长度、Token 级指标（Token-level Metrics）或 GPU 饱和（GPU Saturation）而设计。大多数团队会踩中两个陷阱。GIL 陷阱（GIL Trap）：Locust 的 Token 级测量在 Python GIL（全局解释器锁）下运行分词（Tokenization），在高并发下会与请求生成竞争资源；分词积压会虚高报告的 Token 间延迟（Inter-token Latency）——瓶颈在于客户端而非服务端。提示词同质化陷阱（Prompt-uniformity Trap）：在循环中使用相同的提示词仅测试了 Token 分布上的一个点；而真实流量具有可变长度和多样的前缀匹配。LLMPerf 通过 `--mean-input-tokens` + `--stddev-input-tokens` 参数解决了这一问题。2026 年工具选型指南：专注 LLM 的工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于保障 Token 级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**——支持流式感知，通过 TestRun/PrivateLoadZone CRD 实现 Kubernetes 原生分布式部署，最适合 CI/CD 门禁（CI/CD Gates）；Vegeta 用于 Go 语言的恒定速率压测；Locust 2.43.3 仅配合 LLM-Locust 扩展方可支持流式测试。负载模式：稳态（Steady-state）、爬坡（Ramp）、尖峰（Spike，用于自动扩缩容测试）、浸泡（Soak，用于检测内存泄漏）。

**Type:** 构建
**Languages:** Python（标准库，简易真实提示词生成器 + 延迟收集器）
**Prerequisites:** 第 17 阶段 · 08（推理指标），第 17 阶段 · 03（GPU 自动扩缩容）
**Time:** 约 75 分钟

## 学习目标

- 解释导致通用负载测试工具在 LLM API 测试中失真的两种反模式（Anti-patterns）（GIL 陷阱、提示词同质化陷阱）。
- 根据特定目的选择工具：LLMPerf（基准测试运行）、k6 + 流式扩展（CI 门禁）、guidellm（大规模合成数据生成）、GenAI-Perf（NVIDIA 官方参考实现）。
- 设计四种负载模式（稳态、爬坡、尖峰、浸泡），并指出每种模式旨在捕获的故障模式。
- 使用输入 Token 的均值（Mean）与标准差（Stddev）构建真实的提示词分布，而非采用固定长度。

## 问题背景

你使用 k6 对 LLM 端点进行了 500 并发用户的负载测试。系统扛住了，你顺利上线。但在生产环境中，仅 200 个真实用户就让服务崩溃——P99 首字延迟（Time To First Token, TTFT）飙升，GPU 占用率打满。

问题出在两方面。首先，k6 发送了 500 个完全相同的提示词——你的请求合并（Request Coalescing）与前缀缓存（Prefix Caching）机制让你误以为系统正在处理 500 个并发解码，而实际上只处理了一个。其次，k6 无法像人类肉眼感知那样追踪流式响应中的 Token 间延迟；它只看到一个 HTTP 连接，而非以不同间隔到达的 500 个 Token。

LLM 的负载测试已自成一套独立学科。

## 核心概念

### GIL 陷阱（Locust）

Locust 基于 Python 运行，其客户端分词（tokenization）过程受全局解释器锁（GIL）限制。在高并发场景下，分词任务会在请求生成后排队。报告中的词元间延迟（inter-token latency）实际上包含了客户端的分词积压时间。你以为服务器响应慢，其实是测试框架（test harness）的瓶颈。

修复方案：使用 LLM-Locust 扩展将分词移至独立进程，或改用编译型语言编写的测试框架（如 k6，或使用 `tokenizers.rs` 的 LLMPerf）。

### 提示词单一性陷阱（prompt-uniformity trap）

所有主流的负载测试工具都只允许你配置单个提示词（prompt）。在 10,000 次循环测试中，每次发送的都是完全相同的提示词。服务器每次看到的都是相同的前缀——前缀缓存（prefix cache）命中率接近 100%，吞吐量数据看起来非常漂亮。

修复方案：从提示词分布中进行采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150` 参数——生成长度与内容各异的提示词。

### 四种负载模式

1. **稳态（steady-state）**——在 30-60 分钟内保持恒定的每秒请求数（RPS）。可捕获：基线性能回退。
2. **爬坡（ramp）**——在 15 分钟内将 RPS 从 0 线性提升至目标值。可捕获：容量临界点、预热异常。
3. **尖峰（spike）**——RPS 突然飙升至 3-10 倍并持续 2 分钟，随后恢复。可捕获：自动扩缩容延迟、队列饱和、冷启动影响。
4. **浸泡（soak）**——保持稳态运行 4-8 小时。可捕获：内存泄漏、连接池漂移、可观测性数据溢出。

### 2026 年工具图谱

**LLMPerf**（Anyscale）——基于 Python 但底层分词由 Rust 驱动。支持均值/标准差提示词生成。具备流式感知（streaming-aware）能力。性能测试的首选默认工具。

**NVIDIA GenAI-Perf**——NVIDIA 官方参考实现。采用 Triton 客户端；指标覆盖全面。需注意：其词元间延迟（Inter-Token Latency, ITL）未包含首词元时间（Time To First Token, TTFT），而 LLMPerf 则包含。针对同一服务器，两款工具计算出的单输出词元时间（Time Per Output Token, TPOT）会有所不同。

**LLM-Locust**（TrueFoundry）——Locust 的扩展插件，专门解决 GIL 陷阱。保留熟悉的 Locust 领域特定语言（DSL）并增加流式指标支持。

**guidellm**——面向大规模合成基准测试的工具。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本身（基于 Go 语言编译，无 GIL 限制）新增了流式感知指标。
- k6 Operator 利用 `TestRun` / `PrivateLoadZone` 自定义资源定义（CRD）实现 Kubernetes 原生的分布式测试。
- 最适合用于 CI/CD 门禁（gates）与服务等级协议（SLA）测试。

**Vegeta**——基于 Go 语言，比 k6 更轻量。提供恒定速率的 HTTP 压力饱和测试。虽非专为大语言模型（LLM）设计，但非常适合网关/限流测试。

**Locust 2.43.3 原版**——在 LLM 测试中存在 GIL 陷阱。仅在使用 LLM-Locust 扩展时才能规避。

### CI 中的 SLA 门禁

在拉取请求（PR）中运行 k6，配置如下：

- 在基线 RPS 下各执行 30-50 次迭代。
- 门禁指标：P50/P95 首词元时间（TTFT）、5xx 错误率 < 5%、TPOT 低于设定阈值。
- 一旦指标超标，立即中断构建流程。

### 构建真实的提示词分布

基于真实流量样本（如有）或公开数据集（例如用于对话的 ShareGPT 提示词、用于代码生成的 HumanEval）构建分布。将均值与标准差参数输入 LLMPerf。务必不惜一切代价避免“单提示词循环”测试。

### 关键数据备忘

- k6 Operator 1.0 GA 发布：2025 年 9 月。
- k6 v2026.1.0：支持流式感知指标。
- 典型 LLMPerf 测试：在并发数 X 下发送 100-1000 个请求。
- 典型 CI 门禁：每个 PR 执行 30-50 次迭代。
- 四种负载模式：稳态、爬坡、尖峰、浸泡。

## 立即使用

`code/main.py` 模拟了具有真实提示词分布（prompt distribution）的负载测试（load test），测量有效 TPOT（Time Per Output Token），并演示了均匀提示词陷阱（uniform-prompt trap）。

## 交付（Ship It）
本课程的产出文件为 `outputs/skill-load-test-plan.md`。根据工作负载（workload）和 SLA（Service Level Agreement），选择合适工具并设计四种负载模式。

## 练习
1. 运行 `code/main.py`。对比均匀分布与真实分布——差距出现在哪里？
2. 为 CI（Continuous Integration）门禁编写 k6 脚本：在 100 并发下，TTFT（Time To First Token）P95（第 95 百分位）< 800 ms，运行时长 5 分钟。
3. 你的浸泡测试（soak test）显示内存以 50 MB/小时的速度增长。列出三种可能的原因，以及用于区分它们的观测手段（instrumentation）。
4. 进行尖峰测试（spike test），从 10 RPS（Requests Per Second）突增至 100 RPS。如果已部署 Karpenter + vLLM production-stack（第 17 阶段 · 03 + 18），预期的恢复时间是多少？
5. GenAI-Perf 报告 TPOT=6ms；在同一服务器上，LLMPerf 报告 TPOT=11ms。请解释原因。

## 关键术语
| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| LLMPerf | "LLM 测试框架" | Anyscale 的基准测试工具，支持流式感知 |
| GenAI-Perf | "NVIDIA 工具" | NVIDIA 官方参考测试框架 |
| LLM-Locust | "面向 LLM 的 Locust" | Locust 扩展插件，修复了 GIL 陷阱（GIL trap） |
| guidellm | "合成基准测试" | 大规模合成负载测试工具 |
| k6 Operator | "K8s 版 k6" | 基于 CRD（Custom Resource Definition）的分布式 k6 控制器 |
| GIL 陷阱（GIL trap） | "Python 客户端开销" | 分词（tokenization）积压导致报告的延迟虚高 |
| 提示词均匀性陷阱（Prompt-uniformity trap） | "单提示词谎言" | 循环使用相同提示词会命中缓存，导致吞吐量虚高 |
| 稳态（Steady-state） | "恒定负载" | 持续 N 分钟的平稳 RPS |
| 爬坡（Ramp） | "线性上升" | 在指定时间内从 0 升至目标值 |
| 尖峰（Spike） | "突发测试" | 负载突然倍增后恢复原状 |
| 浸泡（Soak） | "长时测试" | 持续数小时以检测内存泄漏等问题 |

## 延伸阅读
- [TianPan — 大语言模型应用负载测试指南](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — 2026 年 LLM 负载测试](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — LLM 推理基准测试入门](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)