---
name: 技能-微调指南
description: 使用 LoRA 和 QLoRA 微调大语言模型（LLM）的时机与方法决策树
version: 1.0.0
phase: 11
lesson: 8
tags: [微调, LoRA, QLoRA, PEFT, LLM工程]
---

# 微调决策指南

在进行微调之前，请按顺序尝试以下方法：

1. Prompt engineering (minutes, $0)
2. Few-shot examples in prompt (minutes, $0)
3. RAG for knowledge retrieval (days, $10-100/month)
4. Fine-tuning with LoRA/QLoRA (days, $5-50 per experiment)
5. Full fine-tuning (weeks, $100-10,000 per run)

仅当上一步的效果经评估确实不足时，才进入下一步。

## 何时进行微调

- 模型需要保持一致的输出风格或格式，且提示词工程（Prompt Engineering）无法实现
- 你正在对更大规模的模型进行知识蒸馏（Knowledge Distillation）（例如让 8B 模型达到 GPT-4 的质量）
- 延迟是关键指标，且少样本示例（Few-shot Examples）会引入过多 Token
- 你需要模型可靠地遵循复杂的推理模式
- 你拥有 1,000 条以上高质量的目标输入-输出行为示例

## 何时不应进行微调

- 使用恰当的提示词（Prompt）时，模型已能满足需求
- 你需要模型掌握特定事实（应改用检索增强生成 RAG）
- 训练样本少于 500 条（极易导致过拟合 Overfitting）
- 任务需求频繁变更（重新训练成本过高）
- 你需要审计具体是哪些数据影响了某次输出（微调本质上是一个黑盒过程）

## 方法选择

| GPU 显存 (VRAM) | 7B 模型 | 13B 模型 | 70B 模型 |
|----------|----------|-----------|-----------|
| 16GB (T4) | QLoRA | 不可行 | 不可行 |
| 24GB (3090/4090) | QLoRA 或 LoRA | QLoRA | 不可行 |
| 40GB (A100) | LoRA 或全量微调（Full Fine-tuning） | QLoRA 或 LoRA | QLoRA |
| 80GB (A100/H100) | 全量微调（Full Fine-tuning） | LoRA 或全量微调（Full Fine-tuning） | QLoRA 或 LoRA |

## LoRA 配置检查清单

1. 初始设置 `r=16`, `alpha=32`（适用于大多数任务的安全默认值）
2. 优先针对 `q_proj` 和 `v_proj` 层（最小可行 LoRA 配置）
3. QLoRA 使用学习率（Learning Rate）`2e-4`，LoRA fp16 使用 `5e-5`
4. 设置 `lora_dropout=0.05`
5. 训练 1-3 个 Epoch（训练轮数）（更多轮次会增加过拟合风险）
6. 每 100 步在验证集（Held-out Set）上进行一次评估
7. 保存检查点（Checkpoints），并根据评估损失（Eval Loss）选择最佳模型

## 常见错误

- 训练轮数过多（在小数据集上，第 2-3 轮后极易过拟合）
- 使用与全量微调相同的学习率（LoRA 需要更高的学习率）
- 忘记设置填充标记（Pad Token）（会导致 Llama 模型出现 NaN 损失）
- 未冻结基座模型（Base Model）（违背了 LoRA 的设计初衷）
- 仅在训练数据上进行评估（务必预留 10-20% 的数据用于验证）
- 跳过提示词工程基线测试（对提示词已能解决的问题进行微调）

## 质量验证

训练完成后，在 200 条以上的预留验证集上进行对比：
1. 基座模型配合最佳提示词（基线 Baseline）
2. 基座模型加载 LoRA 适配器（Adapter）（你的微调模型）
3. GPT-4 或 Claude 使用相同提示词（性能上限 Ceiling）

如果 LoRA 模型未能超越提示词基线，说明需要优化的是训练数据或配置，而非增加算力。

## 适配器管理

- 多任务服务时保持适配器独立（按请求动态切换适配器）
- 单任务部署时，将适配器权重合并至基座模型
- 将适配器存储在 Hugging Face Hub 上（仅 10-100MB，便于版本控制与共享）
- 部署前需测试合并后的模型输出与未合并时是否一致
- 使用 TIES-Merging 或 DARE 算法将多个适配器融合为一个

## 训练调试

如果损失（Loss）未下降：
1. 检查学习率（对 LoRA 而言可能过低，可尝试 `2e-4`）
2. 验证 LoRA 层是否实际接收到了梯度（Gradients）
3. 确认基座模型权重已冻结
4. 检查数据格式（分词器 Tokenizer 必须与模型预期格式匹配）

如果损失下降但评估质量不佳：
1. 训练数据质量问题（垃圾进，垃圾出）
2. 过拟合（减少训练轮数、增加 Dropout 比例或补充更多数据）
3. 目标模块选择错误（针对复杂任务可加入 MLP 层）
4. 秩（Rank）设置过低（可尝试 `r=32` 或 `r=64`）