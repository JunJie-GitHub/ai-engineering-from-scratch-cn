# 综合项目 07 — 端到端微调流水线（从数据到监督微调 (Supervised Fine-Tuning, SFT) 到直接偏好优化 (Direct Preference Optimization, DPO) 再到服务部署）

> 使用自有数据训练 8B 模型，基于自有偏好进行直接偏好优化 (DPO) 对齐，完成量化与投机解码 (Speculative Decoding)，并以可衡量的每百万 token 成本（$/1M tokens）提供服务。2026 年的开源技术栈包括：用于配置的 Axolotl v0.8、用于偏好优化的 TRL 0.15、用于快速迭代的 Unsloth、用于量化的 GPTQ/AWQ/GGUF，以及搭载 EAGLE-3 用于服务部署的 vLLM 0.7。本综合项目的目标是可复现地运行整条流水线——输入 YAML 配置，输出已部署的端点——并依据 2026 模型开放框架 (Model Openness Framework) 发布模型卡片。

**Type:** 综合项目
**Languages:** Python（流水线）、YAML（配置）、Bash（脚本）
**Prerequisites:** 阶段 2（机器学习）、阶段 3（深度学习）、阶段 7（transformers）、阶段 10（从零构建大语言模型）、阶段 11（大语言模型工程）、阶段 17（基础设施）、阶段 18（安全）
**Phases exercised:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**Time:** 35 小时

## 问题陈述

到 2026 年，每一支专业的 AI 团队都会随时备有一套微调流水线。这并非因为他们要发布前沿的基础模型，而是因为下游适配——领域监督微调 (SFT)、基于标注偏好的直接偏好优化 (DPO)、用于投机解码的蒸馏草稿模型，以及搭载 EAGLE-3 的服务部署——才是产生可量化收益的关键所在。Axolotl v0.8 负责处理多 GPU 的 SFT 配置，TRL 0.15 负责处理 DPO 和群体相对策略优化 (Group Relative Policy Optimization, GRPO)。Unsloth 能让你在单 GPU 上实现快速迭代，而搭载 EAGLE-3 的 vLLM 0.7 能在不损失质量的前提下将解码吞吐量提升 2-3 倍。工具链已经成熟；真正的工艺体现在 YAML 配置、数据治理规范以及严格的评估纪律上。

你将使用特定任务数据，对一个 8B 基础模型（Llama 3.3、Qwen3 或 Gemma 3）依次进行 SFT 和 DPO 训练，完成量化以服务部署，并基于 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 评估性能提升。你还需依据 2026 模型开放框架发布模型卡片。本项目的核心在于可复现性——仅需一条命令即可端到端重新运行整条流水线。

## 核心概念

该流水线包含五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC 风格分类器）、个人身份信息 (Personally Identifiable Information, PII) 擦除、划分规范检查（防止公共基准数据污染）。**监督微调 (SFT)**：Axolotl YAML 配置、在 8xH100 上运行零冗余优化器第3阶段 (ZeRO-3)、余弦学习率调度、序列打包 (Sequence Packing)、2-3 个训练轮次 (epoch)。**DPO 或 GRPO**：TRL 配置、1 个 epoch、偏好对采用人工标注或模型评判、beta 参数调优。**量化**：结合 GPTQ、AWQ 与 GGUF 以实现部署灵活性。**服务部署**：搭载 EAGLE-3 投机解码头 (Speculative Heads) 的 vLLM 0.7（或搭配 SpecForge 的 SGLang）、Kubernetes (K8s) 部署、基于队列等待时间的水平自动扩缩容 (Horizontal Pod Autoscaler, HPA)。

交付物为消融实验 (Ablation Studies) 结果：在三个特定任务基准上对比仅 SFT、SFT+DPO 与 SFT+GRPO 的效果。服务指标：batch size 为 1/8/32 时的 tokens/s、EAGLE-3 接受率、每百万 token 成本。安全评估：Llama Guard 4 通过率。模型卡片：偏见评估、可复现性随机种子、数据许可协议。

## 架构设计

raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)

## 技术栈

- 数据处理：使用 Datatrove 进行去重 (deduplication)，采用 Nemotron-CC 分类器进行质量筛选，利用 Presidio 进行 PII (个人身份信息) 脱敏
- 基座模型：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- 监督微调 (Supervised Fine-Tuning)：使用 Axolotl v0.8 框架，配合 ZeRO-3 (零冗余优化器)、Flash Attention 3 (注意力加速库) 与序列打包 (packed sequences) 技术
- 偏好对齐 (Preference Tuning)：使用 TRL 0.15 进行 DPO (直接偏好优化) 或 GRPO (组相对策略优化)；使用 Unsloth 进行单 GPU 迭代
- 模型量化 (Quantization)：通过 llama.cpp 生成 GPTQ (Marlin 内核)、AWQ (激活感知权重量化) 与 GGUF 格式
- 推理服务 (Serving)：使用 vLLM 0.7 配合 EAGLE-3 投机解码 (speculative decoding)（或 SGLang 0.4 + SpecForge）
- 模型评估 (Evaluation)：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估 (Safety Evaluation)：Llama Guard 4、ShieldGemma-2
- 基础设施 (Infrastructure)：Kubernetes (K8s) + NVIDIA device plugin，基于 queue-wait (队列等待时间) 指标配置 HPA (水平 Pod 自动扩缩容)
- 可观测性 (Observability)：训练阶段使用 W&B (Weights & Biases)，推理阶段使用 Langfuse

## 构建流程

1. **数据流水线 (Data Pipeline)。** 在原始语料库上运行 Datatrove 去重。应用 Nemotron-CC 风格的质量分类器。使用 Presidio 清洗 PII (个人身份信息)。使用固定随机种子划分训练集与验证集。
2. **数据污染检查 (Contamination Check)。** 针对每个验证集划分，计算其与 MMLU-Pro、MT-Bench-v2、RewardBench-2 测试集的 MinHash (最小哈希) 相似度。拒绝任何存在重叠的数据。
3. **Axolotl 监督微调 (SFT)。** 配置包含 ZeRO-3、FA3 (Flash Attention 3) 与序列打包的 YAML 文件。在 8xH100 集群上训练 2-3 个 epoch (训练轮次)。将日志记录至 W&B。
4. **TRL DPO / GRPO 偏好对齐。** 加载 SFT 检查点 (checkpoint)，在偏好数据对 (preference pairs) 上运行一个 epoch 的 DPO（或在数学/代码任务上使用可验证奖励运行 GRPO）。对 beta (DPO 温度参数) 进行网格搜索。
5. **模型量化 (Quantization)。** 生成三种量化版本：GPTQ-INT4-Marlin、AWQ-INT4 以及适用于 llama.cpp 的 GGUF-Q4_K_M。记录模型体积与标称吞吐量。
6. **基于投机解码的推理服务 (Serving with Speculative Decoding)。** 配置 vLLM 0.7，加载通过 Red Hat Speculators 训练的 EAGLE-3 draft heads (草稿头)。测量 batch size 为 1 / 8 / 32 时的接受率 (acceptance rate) 与尾部延迟 (tail latency)。在相同评估基准下，报告每百万 token 成本 ($/1M tokens) 并与 Anthropic / OpenAI 进行对比。
7. **评估矩阵 (Eval Matrix)。** 在基座模型、仅 SFT、SFT+DPO、SFT+GRPO 四个阶段分别运行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro 评估。生成对比表格。
8. **安全评估 (Safety Eval)。** 在开发集 (dev set) 上测试 Llama Guard 4 的通过率。配置 ShieldGemma-2 作为输出过滤器。
9. **模型卡片 (Model Card)。** 采用 MOF 2026 (模型开放框架) 模板：包含数据、训练、评估、安全、许可证、可复现性等章节，并附带 YAML 配置文件与 commit SHAs (提交哈希值)。

## 使用指南

$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF

## 交付上线

`outputs/skill-finetuning-pipeline.md` 描述了最终交付物。只需一条命令，即可驱动数据依次流经监督微调（Supervised Fine-Tuning, SFT）、直接偏好优化（Direct Preference Optimization, DPO）、量化（Quantization）、服务部署（Serving）与评估（Evaluation）阶段，并自动生成模型卡片（Model Card）及服务端点。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 相较于基线模型的评估提升 | 在目标任务（MMLU-Pro、MT-Bench-v2 及特定任务）上测得的增益 |
| 20 | 流水线可复现性 | 使用相同随机种子，单条命令即可端到端完整重跑 |
| 20 | 数据洁净度 | 去重率、个人身份信息（Personally Identifiable Information, PII）清洗覆盖率、数据污染检查（Contamination check）通过 |
| 20 | 服务部署效率 | 批次大小（batch size）为 1/8/32 时的 tokens/s、EAGLE-3 接受率、每百万 token 成本（$/1M tokens） |
| 15 | 模型卡片与安全评估 | 2026 模型开放框架（Model Openness Framework, MOF）完整度 + Llama Guard 4 通过率 |
| **100** | | |

## 练习

1. 在相同的特定任务基准测试上，分别运行仅 SFT、SFT+DPO 与 SFT+GRPO（Group Relative Policy Optimization，组相对策略优化）。报告哪种偏好优化方法胜出及其优势幅度。

2. 将 Llama 3.3 8B 替换为 Qwen3 14B。在质量对齐的前提下，测量每百万 token 的成本（$/1M tokens）。

3. 测量 EAGLE-3 在领域数据与通用 ShareGPT 数据上的接受率。报告其差值，并分析该差值对延迟预算（Latency budgets）的影响。

4. 注入 1% 的数据污染（将 MMLU-Pro 答案泄露至训练数据中）并重新运行评估。观察 MMLU-Pro 准确率出现不切实际的飙升。构建一个数据污染检查持续集成（Continuous Integration, CI）门禁以拦截此类问题。

5. 引入低秩自适应（Low-Rank Adaptation, LoRA）SFT 作为全量微调的替代方案。在显存占用降低 10 倍的条件下，测量两者之间的质量差距。

## 关键术语

| 术语 | 通俗叫法 | 实际含义 |
|------|-----------------|------------------------|
| Axolotl | “SFT 训练器” | 基于 YAML 配置的统一训练框架，支持 SFT、DPO 与知识蒸馏（Distillation） |
| TRL | “偏好调优器” | Hugging Face 推出的库，用于在大语言模型（Large Language Model, LLM）上执行 DPO、GRPO 与近端策略优化（Proximal Policy Optimization, PPO） |
| GRPO | “组相对策略优化” | DeepSeek R1 采用的强化学习（Reinforcement Learning, RL）方案，依赖可验证奖励（Verifiable rewards） |
| EAGLE-3 | “投机解码草稿” | 提前预测 N 个 token 的草稿头（Draft heads）；由 vLLM 使用目标模型进行验证 |
| MOF | “模型开放框架” | 2026 年标准，用于根据数据、代码和许可证对模型发布版本进行分级 |
| Contamination check | “划分洁净度” | 基于 MinHash 的检测方法，用于发现测试集数据是否泄露至训练集中 |
| Acceptance rate | “EAGLE / MTP 指标” | 目标模型接受的草稿 token 所占比例 |

## 延伸阅读

- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) — 监督微调（Supervised Fine-Tuning, SFT）/ 直接偏好优化（Direct Preference Optimization, DPO）参考训练器
- [TRL 文档](https://huggingface.co/docs/trl) — DPO 与组相对策略优化（Group Relative Policy Optimization, GRPO）参考实现
- [Unsloth](https://github.com/unslothai/unsloth) — 单 GPU 迭代参考实现
- [DeepSeek R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO 方法论
- [vLLM + EAGLE-3 文档](https://docs.vllm.ai) — 参考推理服务栈（Serving Stack）
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 替代性投机解码（Speculative Decoding）训练器
- [模型开放框架 2026（Model Openness Framework 2026）](https://isocpp.org/) — 开源发布分级标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 标准评估运行器（Evaluation Runner）