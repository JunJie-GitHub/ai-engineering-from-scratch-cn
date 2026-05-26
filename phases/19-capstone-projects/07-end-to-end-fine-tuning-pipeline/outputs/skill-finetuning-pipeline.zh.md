---
name: 微调流水线
description: 运行一个可复现的从数据处理到监督微调（Supervised Fine-Tuning, SFT）再到直接偏好优化（Direct Preference Optimization, DPO）最终部署服务的微调流水线，包含消融实验、量化以及符合2026模型开放框架（Model Openness Framework, MOF）的模型卡片。
version: 1.0.0
phase: 19
lesson: 07
tags: [综合项目, 微调, axolotl, trl, dpo, grpo, vllm, eagle-3, mof]
---

给定一个基础模型（Base Model）（如 Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B）和一个特定任务的数据集，构建一个单命令流水线，以生成可服务的端点（Served Endpoint）和可复现的模型卡片（Model Card）。

构建计划：

1. 数据阶段：使用 Datatrove 进行去重，采用 Nemotron-CC 风格的质量过滤器，使用 Presidio 清理个人身份信息（Personally Identifiable Information, PII），并设置固定随机种子划分训练集/验证集。
2. 数据污染检查：针对 MMLU-Pro、MT-Bench-v2 和 RewardBench-2 运行 MinHashLSH 算法。若发现重叠则拒绝使用。
3. 监督微调（SFT）：使用 Axolotl v0.8，配合 ZeRO-3、Flash Attention 3 和序列打包（Packed Sequences）技术，在 8 张 H100 显卡上训练 2-3 个轮次（Epochs）。
4. 偏好调优：使用 TRL 0.15 进行直接偏好优化（DPO）（或采用带可验证奖励的群组相对策略优化（Group Relative Policy Optimization, GRPO））训练 1 个轮次，并进行 beta 参数扫描。
5. 量化：生成 GPTQ-INT4-Marlin、AWQ-INT4 和 GGUF-Q4_K_M 格式。
6. 服务部署：使用 vLLM 0.7 配合 EAGLE-3 投机解码（Speculative Decoding）（草稿头通过 Red Hat Speculators 或 SGLang SpecForge 实现）。在 Kubernetes（K8s）上部署，并根据队列等待时间配置水平自动扩缩容（Horizontal Pod Autoscaler, HPA）。
7. 评估：在基础模型/仅SFT/SFT+DPO/SFT+GRPO 各阶段，使用 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 进行评估。
8. 安全性：统计 Llama Guard 4 通过率，并配置 ShieldGemma-2 输出过滤器。
9. 模型卡片：遵循 2026 模型开放框架（MOF）标准，包含数据、训练、评估、安全性和可复现性等章节。

评估标准：

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 相较于基础模型的评估提升（Eval delta） | 在 MMLU-Pro、MT-Bench-v2 及特定任务基准测试上的实测增益 |
| 20 | 流水线可复现性 | 使用相同种子单命令重新运行，生成一致的哈希值 |
| 20 | 数据规范性 | 去重率、PII 清理覆盖率、污染检查通过 |
| 20 | 服务效率 | 批次大小为 1/8/32 时的 tokens/s、EAGLE-3 接受率、每百万 token 成本（$/1M tokens） |
| 15 | 模型卡片与安全评估 | 2026 MOF 完整度 + Llama Guard 4 通过率 |

硬性否决项：

- 跳过 MinHash 污染检查的流水线。将 MMLU-Pro 数据泄露至训练集是典型的评估作弊失败模式。
- 未附带随机种子或 YAML 配置文件的训练任务。可复现性是硬性要求。
- 未配置 EAGLE-3 或等效投机解码（Speculative Decoding）的服务部署。仅达到基准 tokens/s 已不符合 2026 年的标准。
- 缺失安全评估。每个微调模型交付时都必须附带 Llama Guard 4 通过率。

拒绝规则：

- 拒绝发布未附带 lm-eval-harness 提交哈希（Commit SHA）却声称基准测试分数的模型卡片。
- 拒绝在许可证禁止衍生模型的数据上进行微调。MOF 会对数据许可进行评级。
- 拒绝交付未在评估矩阵上测量质量损失的量化模型。

输出：一个代码仓库，包含流水线编排器、Llama 3.3 8B 及一个备选基础模型的 YAML 配置文件、SFT 和 DPO 的 W&B（Weights & Biases）运行日志、量化产物、服务化端点、三项基准测试的评估矩阵、安全评估报告、符合 2026 MOF 标准的模型卡片，以及一份关于你发现并修复的三个最严重数据规范性问题的说明文档。