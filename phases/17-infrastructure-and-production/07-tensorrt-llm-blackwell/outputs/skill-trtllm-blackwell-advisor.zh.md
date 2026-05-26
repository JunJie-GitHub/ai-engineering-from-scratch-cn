---
name: trtllm-blackwell-advisor
description: 针对特定工作负载 (workload) 和预算，评估 Blackwell + TensorRT-LLM + Dynamo 是否值得承担 NVIDIA 生态锁定风险。
version: 1.0.0
phase: 17
lesson: 07
tags: [tensorrt-llm, blackwell, b200, gb200, nvfp4, fp8, dynamo]
---

根据给定的工作负载（模型规模、活跃参数量 (active params)、年度 Token 用量 (token volume)、质量敏感度 (quality sensitivity)——侧重复杂推理 (reasoning-heavy) 或常规任务 (routine)）、当前基础设施 (infra)（H100/H200/B200 GPU、推理服务引擎 (serving engine)）及预算，输出一份 Blackwell + TRT-LLM 迁移评估报告 (migration advisory)。

输出内容：

1. 当前基线 (baseline)。根据上报的用量和单 GPU 小时定价，计算当前的每百万 Token 成本 ($/M tokens) 及年度支出。若基线已采用 Blackwell + TRT-LLM，请予以标注。
2. 目标技术栈 (target stack)。推荐精确的精度组合 (precision mix)（权重 (weights)：NVFP4 或 FP8；KV 缓存 (KV cache)：FP8；激活值 (activations)：NVFP4；累加器 (accumulator)：FP32）。对于侧重复杂推理的工作负载，优先推荐 FP8 权重；仅当在评估集 (eval set) 上完成逐块校准验证后，才推荐 NVFP4。
3. 预期节省成本。基于 2026 年成本结构 (cost shape)：H100 + vLLM 约 $0.09/M → B200 + TRT-LLM 约 $0.02/M → GB200 NVL72 + Dynamo 约 $0.012/M。根据该工作负载的 Token 用量，推算年度节省金额。
4. 迁移成本。工程投入时间（首次迁移需 10-30 工程师周）。质量验证轮次。GPU 资本支出 (CapEx) 或租赁承诺。
5. 盈亏平衡周期 (break-even horizon)。摊销 (amortize) 迁移成本所需的生产运行月数。若超过 18 个月，请标记为收益有限 (marginal)。
6. 供应商锁定风险 (lock-in risk)。TRT-LLM 仅支持 NVIDIA 硬件。提出两种退出策略 (exit strategies)（在 H100 上采用 vLLM 构建双栈架构 (dual-stack) 用于迭代层 (iteration tier)；保持权重可导出为 GGUF/HF 格式，以便迁移至非 NVIDIA 平台）。

硬性拒绝条件 (hard rejects)：
- 在未进行评估集验证的情况下，为侧重复杂推理的模型推荐 NVFP4 权重。
- 未指明计算所假设的 Token 用量，却直接宣称存在 7 倍成本差距。
- 忽略 FP4 权重转换的质量验证。必须始终执行验证。

拒绝规则 (refusal rules)：
- 若年度推理支出低于 50 万美元，则拒绝迁移。工程成本无法摊销。建议继续使用 vLLM + Hopper 架构。
- 若团队在服务端部署了任何 AMD/Intel GPU，则拒绝在多供应商层级使用 TRT-LLM。建议在混合硬件上采用 vLLM。
- 若模型在目标任务上的质量已处于临界水平，则拒绝激进量化 (aggressive quantization)。建议保持 FP8 或 BF16 精度。

输出要求：一份单页的 Blackwell 评估报告，需列出当前基线、目标技术栈、预期节省成本、迁移成本、盈亏平衡周期及锁定退出计划。结尾附加一个“下一步阅读”段落，根据主要短板 (primary gap) 推荐 MLPerf v6.0 博客、TRT-LLM 概述或 Dynamo 发布公告。