---
name: skill-ensemble-builder
description: 为您的问题选择合适的集成方法并进行配置
version: 1.0.0
phase: 2
lesson: 11
tags: [ensemble, bagging, boosting, random-forest, xgboost, stacking]
---

# 集成方法（Ensemble Methods）选择指南

集成方法通过组合多个模型，以产生优于任何单一模型的预测结果。核心问题始终是：该选择哪种集成方法，以及在何时使用？

## 决策清单

1. 当前模型的主要问题是什么？
   - 高方差（过拟合，Overfitting）：使用 Bagging（随机森林，Random Forest）
   - 高偏差（欠拟合，Underfitting）：使用 Boosting（梯度提升，Gradient Boosting；XGBoost）
   - 两者兼有，或追求最高精度：使用 Stacking（堆叠法）

2. 您拥有多少数据？
   - 少于 1,000 行：随机森林（鲁棒性强，不易配置错误）
   - 1,000 至 100,000 行：XGBoost 或 LightGBM（表格型数据的综合最佳选择）
   - 超过 100,000 行：LightGBM（最快的梯度提升算法，能很好地处理大规模数据）

3. 您可以投入多少调参时间？
   - 极少：使用默认参数的随机森林（几乎总是有效）
   - 适中：设置 `learning_rate=0.1` 的 XGBoost，并结合早停法（Early Stopping）调整 `n_estimators`
   - 充足：使用贝叶斯超参数搜索（Bayesian Hyperparameter Search）调优 LightGBM 或 XGBoost

4. 是否需要模型可解释性（Interpretability）？
   - 是：单棵决策树或结合特征重要性（Feature Importance）的小型随机森林
   - 部分：结合 SHAP 值（SHAP Values）的梯度提升模型
   - 否：堆叠法或深度集成（Deep Ensembles）

5. 数据是否包含大量噪声和异常值（Outliers）？
   - 是：随机森林（Bagging 对噪声具有鲁棒性）
   - 否：梯度提升（在干净数据上能进一步提升精度）

## 各方法适用场景

**随机森林（Random Forest / Bagging）**：稳妥的首选方案。在自助采样（Bootstrap Samples）上训练多棵树并取平均。在不增加偏差的情况下降低方差。在中等规模数据上几乎不可能过拟合。所需调参极少：设置 `n_estimators=100-500` 并保持其他默认值即可。

**AdaBoost**：采用样本重加权机制的序列提升算法。与简单基学习器（如决策树桩，Decision Stumps）配合效果良好。由于会提高误分类样本的权重，因此对异常值和噪声标签非常敏感。在实际应用中，已很大程度上被梯度提升算法取代。

**梯度提升（Gradient Boosting）**：将每棵新树拟合到当前集成模型的残差上。主要用于降低偏差。是处理表格型数据最强大的方法。需要调参：`learning_rate`、`n_estimators`、`max_depth`、`min_child_weight`、`subsample`。

**XGBoost**：引入正则化、二阶优化及系统级加速的梯度提升算法。原生支持缺失值处理。是 Kaggle 竞赛及表格数据生产环境机器学习的首选默认方案。

**LightGBM**：采用按叶生长（Leaf-wise Growth，而非按层生长）策略的梯度提升算法。在大规模数据集上比 XGBoost 更快。使用基于直方图的分裂算法。最适合超过 5 万行的数据集。

**CatBoost**：原生支持类别特征（Categorical Features）处理的梯度提升算法。无需进行独热编码（One-Hot Encoding）。当数据包含大量类别特征时表现优异。

**堆叠法（Stacking）**：在多个多样化基模型的预测结果上训练元学习器（Meta-learner）。适用于追求极致精度且计算资源充足的场景。务必通过交叉验证（Cross-Validation）生成基模型预测结果，以避免数据泄露（Data Leakage）。

**投票法（Voting）**：最简单的集成方法。包括硬投票（多数类胜出）或软投票（概率平均）。无需元学习器即可快速组合 2-3 个多样化模型。

## 常见误区

- 使用梯度提升时未启用早停法（若迭代轮数过多，必然导致过拟合）
- 将 `learning_rate` 设置过高（通常超过 0.3 会引发训练不稳定）
- 未对梯度提升的 `max_depth` 进行调优（默认无限制或过深的树极易过拟合）
- 堆叠法中使用的基模型类型完全相同（堆叠的核心在于模型多样性）
- 在噪声数据上使用 AdaBoost（每轮迭代中异常值的权重会不断升高）
- 期望随机森林能解决欠拟合问题（它降低的是方差，而非偏差）

## 各方法的调参优先级

**随机森林 (Random Forest)：**
1. n_estimators：100-500（数量越多通常效果不会更差，只是训练速度更慢）
2. max_depth：None（让树充分生长）或限制在 10-20 以提升速度
3. max_features：分类任务使用 "sqrt"，回归任务使用 "log2" 或 n/3

**XGBoost / LightGBM：**
1. learning_rate：0.01-0.3（若计算资源充足可训练更多树，则较低的值效果更好）
2. n_estimators：在验证集上使用早停法 (early stopping)，而非盲目猜测
3. max_depth：3-8（建议从 6 开始）
4. min_child_weight / min_data_in_leaf：1-20（值越大越能防止过拟合 (overfitting)）
5. subsample：0.7-1.0
6. colsample_bytree：0.7-1.0
7. reg_alpha (L1) 与 reg_lambda (L2)：0-10

## 快速参考

| 方法 | 主要降低 | 速度 | 调参工作量 | 适用场景 |
|--------|---------|-------|--------------|----------|
| 随机森林 | 方差 (Variance) | 快 | 低 | 噪声数据、快速建立基线 (baseline) |
| AdaBoost | 偏差 (Bias) | 快 | 低 | 简单基学习器 (base learners)、干净数据 |
| 梯度提升 (Gradient Boosting) | 偏差 | 中等 | 高 | 表格型数据 (tabular data)、算法竞赛 |
| XGBoost | 两者 | 快 | 高 | 生产环境表格型机器学习 |
| LightGBM | 两者 | 最快 | 高 | 大型数据集（5万行以上） |
| CatBoost | 两者 | 中等 | 中等 | 包含大量类别型特征 (categorical features) |
| 堆叠法 (Stacking) | 两者 | 慢 | 高 | 追求最高精度、模型多样性高 |
| 投票法 (Voting) | 方差 | 快 | 无 | 快速组合 2-3 个模型 |