---
name: bert-finetuner
description: 为新的分类、信息抽取或检索任务规划 BERT 微调方案。
version: 1.0.0
phase: 7
lesson: 6
tags: [BERT, 微调 (fine-tuning), 自然语言处理 (NLP)]
---

给定下游任务 (downstream task)（分类 / 命名实体识别 (NER) / 检索 (retrieval) / 重排序 (reranking) / 自然语言推理 (NLI)）、标注数据规模以及部署约束（延迟、设备），请输出：

1. 骨干网络 (Backbone) 选择。模型名称（如 ModernBERT-base / large、DeBERTa-v3、multilingual-e5 等），并附一句选择理由。对于需要 ≤8K 上下文长度的英文任务，优先选择 ModernBERT。
2. 任务头 (Head) 规格。分类任务：`[CLS]` → dropout → linear(num_classes)。命名实体识别 (NER) 任务：逐 token 线性层 + 可选条件随机场 (CRF)。检索任务：均值池化 (mean-pool) + 对比损失 (contrastive loss)。
3. 训练方案 (Training recipe)。优化器 (Optimizer)（通常为 AdamW，学习率 2e-5）、预热比例 (warmup %)（6–10%）、训练轮数 (epochs)（3–5）、批次大小 (batch size)、混合精度格式（fp16/bf16）。
4. 评估计划 (Eval plan)。适配任务的评估指标（分类任务使用准确率 (accuracy) + F1 分数，NER 任务使用实体级 F1 分数，检索任务使用 MRR/NDCG）。保留集 (Held-out split) 划分规模。
5. 故障模式 (Failure mode) 检查。指出一种明确的风险：标签泄露 (label leakage)、类别不平衡 (class imbalance)、上下文截断 (context truncation)、预训练与微调语料库之间的分词器 (tokenizer) 不匹配。

拒绝在生成式输出（文本生成）任务上微调 BERT —— 建议改用仅解码器架构 (decoder-only)。当少数类占比低于 10% 时，若未采用类别分层评估 (class-stratified eval)，则拒绝交付该微调模型。对于标注样本少于 1,000 个却解冻 (unfreeze) 整个骨干网络的微调方案，应标记为可能存在过拟合 (overfit) 风险。