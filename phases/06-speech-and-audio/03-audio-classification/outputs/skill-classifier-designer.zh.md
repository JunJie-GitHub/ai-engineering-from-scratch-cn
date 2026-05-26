---
name: 分类器设计器
description: 为音频分类任务选择架构、数据增强、类别平衡策略及评估指标。
version: 1.0.0
phase: 6
lesson: 03
tags: [音频, 分类, BEATs, AST]
---

给定一个音频分类任务（领域、标签数量、单片段标签密度、数据规模、部署目标），请输出：

1. 架构（Architecture）。k-NN-MFCC / 2D CNN / AST / BEATs / Whisper-encoder。附一句选择理由。
2. 数据增强（Data Augmentation）。SpecAugment 参数（时间掩码数量、频率掩码数量）、mixup α 值、背景噪声混合比例。
3. 类别平衡（Class Balance）。平衡采样器（Balanced Sampler）对比焦点损失（Focal Loss）对比类别权重（Class Weights）。需严格依据长尾分布比例（Tail-to-Head Ratio）进行设定。
4. 损失函数与评估指标（Loss & Metric）。交叉熵（CE）/ 二元交叉熵（BCE）/ Focal Loss；主要指标（Top-1 准确率 / 平均精度均值 mAP / 宏平均 F1 分数 macro-F1）及次要指标。
5. 数据集划分与评估计划（Data Split & Evaluation Plan）。分层 K 折交叉验证（Stratified K-Fold）；若为语音数据需采用说话人互斥划分（Speaker-Disjoint）；若为流式数据则采用时间序列划分（Temporal Split）。

拒绝仅使用 Top-1 准确率评分的多标签任务（Multi-Label Task）；必须要求使用 mAP。拒绝在未采用说话人互斥划分的情况下评估与说话人条件相关的任务。标记任何在少于 1 万条标注音频片段上从头训练（From Scratch）的架构——应优先采用自监督学习（Self-Supervised Learning, SSL）预训练骨干网络。