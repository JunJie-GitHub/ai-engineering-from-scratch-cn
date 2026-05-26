---
name: inference-server
description: 交付一个支持 EAGLE-3 或 P-EAGLE 草稿模型（draft model）、具备 Kubernetes 自动扩缩容（Kubernetes autoscaling）功能，并包含完整吞吐量/延迟/成本报告的投机解码（speculative decoding）推理服务器。
version: 1.0.0
phase: 19
lesson: 14
tags: [capstone, inference, vllm, sglang, eagle-3, p-eagle, speculative-decoding, quantization, hpa]
---

给定两个开源目标模型（Llama 3.3 70B 与 Qwen3-Coder-30B MoE 或 GPT-OSS-120B），构建并交付一套集成投机解码（speculative decoding）、模型量化（quantization）及 Kubernetes 自动扩缩容的生产级推理服务栈。公布实测的加速倍数与尾部延迟（tail-latency）指标。

构建计划：

1. 在 vLLM 0.7（或 SGLang 0.4）框架下部署目标模型，并启用 FP8 Marlin 量化。
2. 从 Red Hat Speculators 加载已对齐的 EAGLE-3 草稿模型（draft model）（或通过 SpecForge 自行训练）。
3. 基线数据：在不启用投机解码的情况下，记录 batch size 为 1/8/32 时的 tokens/s 吞吐量及 p50/p99 延迟。
4. 启用 EAGLE-3。重新运行相同的基准测试。报告加速比、接受率（acceptance rate）以及 p99 尾部延迟的变化量。
5. 启用 P-EAGLE 并行投机解码；报告树深度增加带来收益与产生损耗的拐点。
6. 跨不同数据分布运行基准测试：ShareGPT、HumanEval 及领域数据。公布接受率的漂移情况。
7. 在第二个目标模型（MoE 架构）上重复上述测试；识别草稿接受率对路由噪声（routing-noise）的敏感度。
8. 在 Kubernetes 上部署，并配置 HPA（Horizontal Pod Autoscaler）监控 `queue_wait_ms` 指标。演示当负载增至三倍时的横向扩容（scale-out）过程。
9. 在匹配的评估集上，对比每百万 token 成本（$/1M tokens）与 Anthropic Claude Sonnet 4.7 及 OpenAI GPT-5.4 的差异。

评估标准：

| 权重 | 评估标准 | 测量指标 |
|:-:|---|---|
| 25 | 相较于基线的实测加速比 | 在两个模型上保持质量一致的前提下，吞吐量提升 2.5 倍以上 |
| 20 | 真实流量下的接受率 | 按数据分布分别输出接受率报告 |
| 20 | P99 尾部延迟控制 | 启用与未启用投机解码时，batch size 为 1/8/32 的 p99 延迟 |
| 20 | 运维能力 | K8s 部署、基于队列等待时间的 HPA 配置、平滑发布、先排空后升级（drain-first upgrade） |
| 15 | 文档撰写与方法论 | 指标推导清晰，基线对比严格匹配 |

硬性否决项：

- 仅报告稳态吞吐量而未提供尾部延迟数据。
- 基于 CPU 使用率而非队列等待时间配置 HPA。在 GPU 饱和时将引发剧烈震荡（thrashing）。
- 忽略草稿模型与目标模型的版本对齐。版本漂移的草稿模型带来的开销甚至高于不使用投机解码。
- 成本对比中未扣除托管 API 的提示词缓存（prompt-caching）折扣。

拒绝规则：

- 若发布流程未包含连接排空（rollout drain）机制，则拒绝验收。在请求处理期间进行原地升级将直接导致不合格。
- 拒绝接受跨数据分布聚合的接受率报告。必须按数据分布分别提供。
- 若在 batch size=32 时声称投机解码具有优势，但未提供匹配的无投机解码基线数据，则拒绝认可。

输出要求：一个代码仓库，需包含 vLLM / SGLang 配置文件、EAGLE-3 草稿模型下载脚本、K8s 部署清单（manifests）、基于队列等待时间的 HPA 配置、针对 ShareGPT / HumanEval / 领域数据的基准测试框架、每百万 token 成本（$/1M tokens）对比表，以及一份技术文档。该文档需明确指出投机解码引入的三项尾部延迟性能回退（regressions）问题，并说明针对每项问题所采用的缓解措施（批次门控 batch gating、n-gram 回退 ngram fallback、量化参数微调 quantization tweak）及其修复效果。