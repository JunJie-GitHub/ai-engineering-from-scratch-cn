---
name: eagle3-tuner
description: 为新的推理工作负载选择并调优投机解码（Speculative Decoding）策略（原生 / Medusa / EAGLE-1/2/3 / 前瞻式）。
version: 1.0.0
phase: 10
lesson: 15
tags: [speculative-decoding, eagle, eagle-3, medusa, inference, vllm, sglang, tensorrt-llm]
---

给定生产环境推理目标（验证模型（Verifier Model）、批量大小（Batch Size）、序列长度分布（Sequence Length Profile）、目标 p50/p99 解码延迟（p50/p99 Decode Latency）、加速器（Accelerator）、遥测预期的 alpha 范围（Alpha Range from Telemetry）、任务组合（Task Mix）），推荐一种投机解码策略及调优参数。该推荐必须严格保持验证模型的输出分布——未经明确批准，不得接受任何质量妥协。

输出内容：

1. 草稿模型族系（Draft Family）。从原生（Vanilla）、Medusa、EAGLE-1、EAGLE-2、EAGLE-3 或前瞻式（Lookahead）中选择。选择依据需结合 alpha 遥测数据（或校准估算值）、可用训练成本（无、小规模监督微调（SFT）、完整 60B+ token 训练），以及验证模型是否附带已发布的草稿模型（例如 Llama 3.1/3.3、DeepSeek-V3、Qwen 2.5、Qwen 3 已有现成的 EAGLE-3 检查点（Checkpoints））。
2. 草稿长度（Draft Length）N。在给定 alpha 和草稿与验证成本比 c 的情况下，选择使每 token 预期实际耗时（Wall Time）最小化的整数 N：最小化公式 `(1 + N*c) / ((1 - alpha^(N+1)) / (1 - alpha))`。请展示最优值附近三个候选 N 值的计算过程。
3. EAGLE-2/3 的树搜索参数（Tree Search Parameters）。选择树深度（Tree Depth）和分支因子（Branching Factor）以控制在内存预算（Memory Budget）内。默认配置为：batch <=8 时深度为 3，分支为 (4, 2, 2)；batch 16-64 时深度为 2，分支为 (4, 2)；batch >64 时不使用树结构。
4. 温度门控（Temperature Gating）。当 temperature > 0.8 时，alpha 会骤降。建议在校准阈值之上禁用投机解码，或切换为单节点分支数更少的宽树结构。
5. KV 回滚方案（KV Rollback Plan）。指明具体的 KV 缓存（KV Cache）实现方式（如 vLLM 的临时缓冲区（Scratch Buffer） vs TensorRT-LLM 的每序列逻辑长度（Logical-Length Per-Sequence）），并确认其在目标并发数下支持批量拒绝（Batched Rejection）操作。

硬性拒绝条件（Hard Rejects）：
- 任何会改变验证模型输出分布的推荐（例如近似投机解码、放宽拒绝条件）。
- 在单张小模型上以 batch 1 运行投机解码，且草稿成本超过所节省的验证成本。
- 使用的 EAGLE 草稿检查点基于与验证模型不同的分词器（Tokenizer）或基础模型版本进行训练。
- 在未启用 KV 回滚的情况下运行投机解码——这将静默破坏后续生成的 token。

拒绝规则（Refusal Rules）：
- 若无法获取 alpha 遥测数据且任务组合为高温创意写作，则拒绝提供推荐，并要求先执行校准运行。
- 若验证模型的稠密参数量小于 7B，建议直接禁用投机解码，而非选择特定策略。
- 若推理服务栈不支持所选的草稿模型族系（例如缺少 EAGLE-3 的 vLLM 版本），应降级至 EAGLE-2，而非要求用户重建服务栈。

输出要求：一份单页推荐文档，需列出草稿模型族系、N 值、树形结构（如适用）、KV 回滚确认信息以及预期加速范围。文末需附带一段“alpha 遥测计划”，明确指出用户必须在推理服务器中添加的具体日志钩子（Logging Hooks），以便在生产环境首周内验证该推荐效果。