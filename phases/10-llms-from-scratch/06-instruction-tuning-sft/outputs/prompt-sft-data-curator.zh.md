---
name: SFT数据策展人提示词
description: 为监督微调（Supervised Fine-Tuning, SFT）设计和整理指令数据集
version: 1.0.0
phase: 10
lesson: 6
tags: [监督微调, 指令微调, 微调, 数据整理, 对齐]
---

# SFT（监督微调，Supervised Fine-Tuning）数据策展人

在为特定能力（代码生成、数学、对话、安全）设计指令微调（Instruction-Tuning）数据集时，请使用此框架来规划数据收集、定义质量标准并构建训练流水线（Training Pipeline）。

## 输入要求

请提供：
- **目标能力**（例如：“Python 代码生成”、“医疗问答”、“多轮对话”）
- **基座模型**（Base Model）（例如：Llama 3 8B、Mistral 7B、Qwen 2.5 72B）
- **预算**（标注工时、用于合成生成（Synthetic Generation）的 API 成本）
- **格式偏好**（Alpaca、ShareGPT、ChatML）

## 步骤 1：数据集设计

### 规模指南

| 质量级别 | 所需示例数量 | 预期效果 |
|--------------|----------------|------------------|
| 研究原型 | 1,000-5,000 | LIMA 级别：若示例由专家编写，其效果可与更大规模的数据集相媲美 |
| 生产环境 v1 | 10,000-50,000 | Stanford Alpaca 级别：在常见任务上具备扎实的指令遵循（Instruction Following）能力 |
| 生产环境 v2 | 50,000-200,000 | Vicuna/Llama 2 Chat 级别：稳健的多轮对话能力与领域覆盖度 |

质量永远优于数量。1,000 个由专家编写的示例（LIMA，2023 年 5 月）即可达到在 50,000+ 示例上训练的模型效果。请优先考虑：

1. **多样性** -- 覆盖目标能力的全部范围
2. **准确性** -- 每个回复必须在事实上正确无误
3. **清晰度** -- 回复应简洁且结构良好
4. **难度梯度** -- 包含简单、中等和困难的示例

### 多样性检查清单

对于通用助手：
- 开放式问题（20%）
- 事实性问答（20%）
- 创意写作（10%）
- 代码生成（15%）
- 推理与数学（15%）
- 文本摘要（10%）
- 带约束条件的指令遵循（10%）

针对特定领域的模型，请调整上述比例。例如，编程助手可能会将 60% 分配给代码生成，20% 分配给代码解释。

## 步骤 2：数据格式

### Alpaca 格式（单轮）

{
  "instruction": "Write a function that reverses a string in Python.",
  "input": "",
  "output": "def reverse_string(s):\n    return s[::-1]"
}

适用场景：单轮任务、简单的指令-回复对、快速原型开发。

### ShareGPT 格式（多轮）

{
  "conversations": [
    {"from": "system", "value": "You are a Python expert."},
    {"from": "human", "value": "How do I reverse a string?"},
    {"from": "gpt", "value": "Use slicing: s[::-1]"},
    {"from": "human", "value": "What about for a list?"},
    {"from": "gpt", "value": "Same syntax works: my_list[::-1]"}
  ]
}

适用场景：对话类应用、多轮上下文至关重要时。

### ChatML 格式（含特殊标记）

<|im_start|>system
You are a Python expert.<|im_end|>
<|im_start|>user
How do I reverse a string?<|im_end|>
<|im_start|>assistant
Use slicing: s[::-1]<|im_end|>

适用场景：针对原生使用 ChatML 的模型（如 Qwen、Yi）。

## 步骤 3：质量标准

### 单条示例检查

1. **回复相关性**：回复是否真正解答了指令？
2. **事实准确性**：所有声明是否可验证且正确？
3. **完整性**：回复是否全面涵盖了指令的要求？
4. **简洁性**：能否用更少的文字传达相同的信息？
5. **格式一致性**：回复是否符合预期的风格规范？

### 警示信号（拒绝该示例）

- 回复自相矛盾
- 回复包含有害内容且未进行安全拦截
- 回复存在事实或引用幻觉（Hallucination）
- 指令存在歧义且回复未作澄清
- 回复仅是对指令的简单改写或复述

### 数据集级别检查

- 来自单一来源/模板的样本比例不超过 5%
- 至少 80% 的回复词元 (Token) 具有实际意义（非填充内容）
- 平均回复长度为 50-200 个词元（避免过短或过长）
- 系统提示词 (System prompt) 多样性：至少包含 10 种不同的系统提示词

## 步骤 4：训练配置

| 参数 | 推荐范围 | 说明 |
|-----------|------------------|-------|
| 学习率 (Learning rate) | 1e-5 至 5e-5 | 模型越大取值越低（70B 模型使用 1e-5，7B 模型使用 5e-5） |
| 训练轮数 (Epochs) | 1-3 | 监控验证集损失 (Validation loss)，一旦出现上升迹象即停止训练 |
| 批次大小 (Batch size) | 32-128 | 若受 GPU 限制，可结合梯度累积 (Gradient accumulation) 进行扩展 |
| 预热 (Warmup) | 总步数的 0-5% | 重要性低于预训练阶段 |
| 权重衰减 (Weight decay) | 0.0-0.1 | 短期微调 (Fine-tuning) 任务中为可选项 |
| 损失掩码 (Loss masking) | 仅针对回复词元 | 对指令和系统提示词对应的词元进行掩码处理 |
| 预训练数据混合 (Pre-training data mixing) | 2-5% | 混合原始文本以防止灾难性遗忘 (Catastrophic forgetting) |

## 步骤 5：评估流程

训练完成后，需针对以下指标进行评估：

1. **指令遵循率 (Instruction following rate)**：模型针对测试提示词生成相关且完整回复的比例
2. **遗忘分数 (Forgetting score)**：与基座模型 (Base model) 相比，在预留通用文本语料库上的困惑度 (Perplexity)
3. **格式合规率 (Format compliance)**：符合预期对话格式的回复所占比例
4. **MT-Bench 或 AlpacaEval**：指令微调模型的标准基准测试 (Benchmark)
5. **领域特定评估 (Domain-specific eval)**：针对目标能力定制的评估方案

### 警告信号

- 第 1 轮后验证集损失上升：出现**过拟合 (Overfitting)**，需减少训练轮数或增加数据量
- 遗忘分数上升超过 15%：学习率过高或训练轮数过多
- 模型逐字复现训练样本：严重过拟合，需引入更多样化的数据
- 模型拒绝良性指令：在安全数据上过度训练，需重新平衡数据集