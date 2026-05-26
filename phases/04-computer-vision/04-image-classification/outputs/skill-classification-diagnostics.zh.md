---
name: skill-classification-diagnostics
description: 给定混淆矩阵 (Confusion Matrix) 和类别名称，揭示各类别的失效情况并提出最具影响力的单一修复方案
version: 1.0.0
phase: 4
lesson: 4
tags: [计算机视觉, 分类, 评估, 调试]
---

# 分类诊断 (Classification Diagnostics)

混淆矩阵的阅读视角。总体准确率 (Aggregate Accuracy) 告诉你分类器 (Classifier) 是否有效，而混淆矩阵则告诉你*它目前还不知道什么*。

## 适用场景

- 首次查看已训练分类器的验证集性能 (Validation Performance) 时。
- 在多次训练迭代之间，用于决定下一步调整方向。
- 模型上线前：验证关键类别是否存在静默失效 (Silent Failure)。
- 排查生产环境性能回退 (Regression) 问题，例如总体准确率下降了一个百分点，你需要查明具体原因。

## 输入参数

- `cm`：C×C 混淆矩阵（行表示真实标签，列表示预测标签）。
- `labels`：包含 C 个类别名称的列表，顺序需与矩阵一致。
- 可选参数 `class_priors`：各类别在训练集中的频率（默认值为 `cm` 的行和）。

## 执行步骤

1. **计算各类别指标 (Per-class Metrics)。** 若出现除以零的情况，则视为该类别指标未定义，报告为 `n/a`；切勿静默替换为 0。
   - precision_i = cm[i,i] / sum(cm[:, i])   （当该类别从未被预测时未定义）
   - recall_i    = cm[i,i] / sum(cm[i, :])   （当该类别无真实样本时未定义）
   - f1_i        = 2 * p * r / (p + r)        （当任一组成指标未定义时未定义）

2. **按 F1 分数 (F1 Score) 对表现最差的至多三个类别进行排序。** 若混淆矩阵中的类别少于三个，则按实际数量排序。排除所有指标均未定义的类别。

3. **找出每一行中非对角线 (Off-diagonal) 的最大值单元格**——即最常“窃取”该类别预测结果的其他类别。报告格式为 `真实类别 -> 预测类别`。

4. **对每个最差类别的失效模式 (Failure Mode) 进行分类。** 使用以下定量阈值以确保标签可复现：
   - `ambiguity`（歧义/混淆）—— 与另一类别存在双向混淆：同时满足 `cm[i,j] / sum(cm[i, :]) >= 0.15` 和 `cm[j,i] / sum(cm[j, :]) >= 0.15`。
   - `imbalance`（类别不平衡）—— 该类别的训练样本数量不足其主要混淆类别的 `0.5 倍`。
   - `label_noise`（标签噪声）—— `|precision_i - recall_i| >= 0.2`，且该类别不属于不平衡或歧义路径。
   - `systematic`（系统性误差）—— 没有任何单一混淆类别占该类别误差的比例超过 0.2；误差分散在三个或更多其他类别中。

5. **推荐最具影响力的单一后续操作**：
   - `ambiguity` -> 收集或合成具有判别性的样本，添加能保留区分特征的针对性数据增强 (Data Augmentation)。
   - `imbalance` -> 对少数类进行过采样 (Oversampling) 或应用类别加权损失 (Class-weighted Loss)。
   - `label_noise` -> 对该类别进行分层抽样 (Stratified Sampling) 审查；在进行任何其他更改前，先修正错误标签。
   - `systematic` -> 增加该类别的数据量，或在微调 (Fine-tuning) 时提高该类别损失的权重。

## 报告模板

[diagnostics]
  aggregate accuracy: X.XX
  macro F1:           X.XX

[top-3 worst classes]
  1. class <name>  F1 = X.XX  prec = X.XX  rec = X.XX
     top confusion: <name> -> <other>  (N cases)
     failure mode:  ambiguity | imbalance | label_noise | systematic
     action:        <one sentence>

  2. ...
  3. ...

[recommendation]
  single biggest lever: <one sentence naming the class and the fix>

## 规则

- 最多返回三个类别。数量过多会掩盖核心信号。
- 明确指出每个最差类别的主要混淆对象；切勿笼统总结为“与多个类别混淆”。
- 所有建议必须基于混淆矩阵的证据。在未指明具体类别的情况下，禁止给出“增加更多数据”等泛泛之谈。
- 当精确率 (Precision) 与召回率 (Recall) 差异超过 0.2 时，始终将标签噪声列为候选原因——经过训练后，真实类别的 P 和 R 通常趋于一致。