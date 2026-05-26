---
name: skill-classification-baseline
description: 在尝试复杂模型之前，先建立稳健的分类基线
version: 1.0.0
phase: 2
lesson: 3
tags: [分类, 逻辑回归, 基线, 预处理]
---

# 分类基线指南（Classification Baseline Guide）

在尝试复杂模型之前，应先使用逻辑回归（Logistic Regression）建立基线。它的训练仅需数秒，能够输出概率，且具备完全的可解释性。令人惊讶的是，大量现实世界的问题根本不需要更复杂的模型。

## 决策清单

1. 决策边界是否可能呈线性？
   - 是：逻辑回归通常就足够了
   - 否：你仍然需要将其作为基线，以衡量后续模型的改进程度

2. 你有多少个特征（Feature）？
   - 少于 50 个：标准逻辑回归即可胜任
   - 50 到 10,000 个：添加 L2 正则化（L2 Regularization，即岭回归 Ridge）
   - 超过 10,000 个（例如 TF-IDF 文本特征）：使用 L1 正则化（L1 Regularization，即 Lasso）或 LinearSVC

3. 数据集是否不平衡（Imbalanced）？
   - 比例低于 5:1：通常无需调整即可
   - 5:1 到 50:1：在 sklearn 中使用 `class_weight="balanced"`
   - 超过 50:1：将类别权重与合适的评估指标（精确率 Precision、召回率 Recall 或 F1 分数）结合使用

4. 特征是否处于不同的量纲/尺度？
   - 在进行逻辑回归前务必进行标准化（Standardization）。它使用基于梯度的优化算法，未缩放的特征会拖慢收敛速度或扭曲决策边界。

5. 是否存在缺失值（Missing Values）？
   - 在拟合模型前进行插补（Imputation）。逻辑回归无法处理 NaN。
   - 数值型列使用中位数插补，类别型列使用众数插补。

## 何时逻辑回归已足够

- 特征关系主要呈线性的二分类（Binary Classification）任务
- 需要概率输出（而不仅仅是类别标签）
- 需要模型可解释性（标准化后，系数可指示特征重要性的方向及相对大小）
- 训练数据量较小（数百至数千条样本）
- 需要用于实时服务的快速模型（推理时仅需单次点积运算）
- 监管或合规要求模型具备可解释性

## 何时需要升级模型

- 准确率（Accuracy）远低于目标且已尝试过特征工程（Feature Engineering）
- 特征与目标变量之间明显呈非线性关系（可通过残差图 Residual Plots 检查）
- 拥有大规模表格数据（1 万行以上）：尝试梯度提升（Gradient Boosting，如 XGBoost 或 LightGBM）
- 特征之间存在多项式特征无法捕捉的复杂交互作用
- 处理图像、文本或序列数据：直接对原始输入使用逻辑回归将无效

## 分类基线的预处理步骤

1. 首先进行**训练集/测试集划分（Train/Test Split）**，然后再进行任何预处理。这能防止数据泄露（Data Leakage）。
2. **处理缺失值**：数值型使用中位数插补，类别型使用众数插补。
3. **编码类别特征**：低基数（少于 10 个取值）使用独热编码（One-Hot Encoding），高基数使用目标编码（Target Encoding）。目标编码仅在训练折上拟合（使用折外编码 Out-of-Fold Encoding 以防止泄露）。
4. **缩放数值特征**：使用 StandardScaler（零均值，单位方差）。在训练集上拟合，同时转换训练集和测试集。
5. **拟合逻辑回归**，设置 `C=1.0`（默认正则化强度）。
6. **评估模型**：查看混淆矩阵（Confusion Matrix）、精确率、召回率、F1 分数。不要仅看准确率。
7. **调整阈值**：默认的 0.5 很少是最优的。在 0.1 到 0.9 之间进行扫描，选择符合你精确率/召回率优先级的阈值。

## 常见错误

- 在不平衡数据上仅评估准确率（预测多数类的模型得分虽高，但毫无用处）
- 忘记缩放特征（未缩放特征会导致逻辑回归训练缓慢，且收敛到较差的解）
- 使用测试集调整决策阈值（应使用验证集或交叉验证 Cross-Validation）
- 跳过基线直接使用 XGBoost（会丧失可解释性，且缺乏性能参考基准）
- 未检查多重共线性（Multicollinearity）（高度相关的特征会放大系数方差）

## 快速参考

| 场景 | 模型 | 正则化 (Regularization) | 关键设置 |
|----------|-------|---------------|-------------|
| 特征较少，需可解释性 | LogisticRegression | L2 正则化（默认） | C=1.0 |
| 特征较多，含部分无关特征 | LogisticRegression | L1 正则化 | penalty="l1", solver="saga" |
| 高维稀疏数据（如文本） | SGDClassifier | L1 或弹性网络 (ElasticNet) 正则化 | loss="log_loss" |
| 类别不平衡 (Class Imbalance) | LogisticRegression | L2 正则化 | class_weight="balanced" |
| 需要输出概率 | LogisticRegression | L2 正则化 | predict_proba() |
| 仅需类别标签 | LinearSVC | L2 正则化 | 在大规模数据上比逻辑回归 (LR) 更快 |