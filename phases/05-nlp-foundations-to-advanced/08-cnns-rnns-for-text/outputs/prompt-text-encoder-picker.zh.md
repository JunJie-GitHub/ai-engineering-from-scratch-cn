---
name: text-encoder-picker
description: 为给定的约束条件集选择文本编码器架构。
phase: 5
lesson: 08
---

给定约束条件（任务、数据量、延迟预算 (Latency Budget)、部署目标 (Deploy Target)、计算预算 (Compute Budget)），输出：

1. 编码器架构 (Encoder Architecture)：TextCNN、BiLSTM、BiLSTM-CRF、Transformer 微调 (Transformer Fine-tune)，或“预训练 Transformer 作为冻结编码器 + 小型输出头 (Pretrained Transformer as Frozen Encoder + Small Head)”。
2. 嵌入输入 (Embedding Input)：随机初始化 (Random Init)、冻结的 GloVe 或 fastText，或上下文感知的 Transformer 嵌入 (Contextualized Transformer Embeddings)。
3. 5 行训练配方 (Training Recipe)：优化器 (Optimizer)、学习率 (Learning Rate)、批次大小 (Batch Size)、训练轮数 (Epochs)、正则化 (Regularization)。
4. 一项监控信号 (Monitoring Signal)。RNN/CNN 模型：检查按序列长度划分的准确率 (Per-sequence-length Accuracy)，以排查长距离依赖失效 (Long-dependency Failures)。Transformer 微调：若学习率过高，需警惕微调崩溃 (Fine-tuning Collapse)；检查前 100 步的训练损失 (Train Loss)。

当用户拥有的标注样本不足约 500 个时，除非能先证明 TextCNN / BiLSTM 基线模型 (Baseline) 的性能已达到平台期 (Plateaued)，否则拒绝推荐微调 Transformer。将边缘端部署 (Edge Deployment)（手机、微控制器、浏览器）标记为需要优先于其他所有事项进行架构决策的场景。