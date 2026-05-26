---
name: llm-pipeline-reviewer
description: 在投入数百万美元运行之前，审查端到端的大语言模型（Large Language Model, LLM）训练流水线清单（manifest）。
version: 1.0.0
phase: 10
lesson: 13
tags: [流水线, 训练, 清单, 评估门控, 成本, 回滚]
---

给定一份拟议的训练流水线清单（描述分词器（tokenizer）、数据、预训练（pre-training）、监督微调（Supervised Fine-Tuning, SFT）、对齐（alignment）、评估（evaluation）、量化（quantization）和部署（serving）阶段的 YAML 或 JSON 文件），请生成一份涵盖以下内容的审查报告：

1. 阶段图（Stage graph）。确认每个阶段都具有类型明确的输入和输出。指出缺失的依赖项、隐式状态，或任何直接使用裸目录而非具名制品哈希（artifact hash）的阶段。
2. 哈希链（Hash chain）。验证第 N 阶段的 output_hash 是否等于每个下游阶段 input_hashes 列表中的某一个。任何不匹配都意味着清单存在逻辑断裂，流水线不得启动。
3. 评估门控（Eval gate）。门控列表中的每个指标必须是数值型，并包含运算符、阈值和测量来源。拒绝任何主观的门控（如“看起来不错”）、无界的门控（无阈值）或在训练数据上进行测量的门控。
4. 回归防护（Regression guard）。新模型的核心基准测试（MMLU、MATH、HumanEval+、GPQA 或领域特定的等效测试）必须附带基线数据。没有基线的运行等同于无法检测性能回归的运行。
5. KL 散度预算（KL budget）。对齐阶段（RLHF、DPO、CAI、GRPO）必须声明相对于参考模型的累积 KL 散度（Kullback-Leibler divergence）上限。无界的 KL 散度意味着无限制的分布漂移。
6. 数据污染检查（Contamination check）。训练数据分片和评估集必须具有文档化的重叠检查机制（精确匹配或 13-gram）。要求的通过阈值：<0.1%。
7. 成本估算（Cost estimate）。提供每个阶段的运行前估算值及总计，并与预算门控进行对比。若估算值 > 预算，流水线将拒绝启动。
8. 回滚计划（Rollback plan）。针对每个阶段，明确失败时的具体操作：重新运行、回退至上一版本制品、修改输入并重新运行下游阶段。高成本阶段（如预训练）必须采用热检查点（warm checkpoint）策略。
9. 制品存储（Artifact store）。检查点、数据集、分词器和评估报告必须采用内容寻址（content-addressed，SHA-256）。采用文件名寻址的制品（如 "latest.pt"）将直接被拒绝。
10. 可观测性（Observability）。每个阶段必须输出结构化日志，包含追踪 ID（trace ID）、阶段名称、输入哈希、输出哈希、实际运行时间（wall clock）和成本。缺失追踪 ID 意味着运行结束后将无法进行事后调试。

导致审查中止的红旗警告（Red flags）：
- 门控缺失测量来源（针对没有任何阶段计算的指标设置门控）
- 某个阶段与下游阶段共享检查点（缺乏关注点分离）
- 对齐阶段未指定参考模型（缺乏 KL 散度计算的锚点）
- 采用大语言模型作为裁判（LLM-as-judge）的评估中，裁判模型与策略模型属于同一家族（存在数据污染风险）
- 成本估算超出预算 20% 以上
- 回滚计划仅包含“从头重新运行”

输出：一份两页的审查报告，针对每个门控给出 PASS/HOLD 结论，明确指出得出该结论的具体清单字段或缺失字段，以及将 HOLD 转为 PASS 所需的最小修改。