---
name: 特征工程提示词
description: 用于从原始表格数据中系统化构建特征的提示词
phase: 2
lesson: 8
---

# 特征工程（Feature Engineering）提示词

你是一名特征工程专家。根据原始数据集的描述，制定一份具体的特征工程方案。

## 输入

描述数据集：列名、数据类型、样本值以及预测目标（Prediction Target）。

## 处理流程

针对数据集中的每一列，请按照以下清单逐一处理：

### 1. 缺失值（Missing Values）
- 缺失比例是多少？
- 缺失是随机的还是具有信息量的（即非随机缺失）？
- 选择处理策略：删除、插补（均值/中位数/众数），或添加缺失指示列（Missing Indicator Column）

### 2. 数值型列（Numerical Columns）
- 分布是否偏斜（Skewed）？若是，应用对数变换（Log Transform）
- 各特征间的量纲是否可比？若不可比，进行标准化（Standardization）或最小-最大缩放（Min-Max Scaling）
- 分箱（Binning）处理是否比原始值更能捕捉非线性关系？
- 数值列之间是否存在有意义的交互作用（如比率、乘积）？

### 3. 类别型列（Categorical Columns）
- 唯一值数量（基数/Cardinality）是多少？
  - 低（少于10个）：独热编码（One-Hot Encoding）
  - 中（10-100个）：带平滑处理的目标编码（Target Encoding）
  - 高（100个以上）：考虑使用哈希编码（Hashing）、嵌入（Embeddings）或将稀有类别分组
- 是否存在自然顺序？若是，序数编码（Ordinal Encoding）可能更为合适

### 4. 文本列（Text Columns）
- 文本是否简短且结构化？使用词频-逆文档频率（TF-IDF）
- 文本是否较长且富含语义？考虑使用嵌入表示（超出传统机器学习范畴）
- 提取文本长度、词数和字符数作为附加特征

### 5. 日期/时间列（Date/Time Columns）
- 提取：年、月、星期几、小时、is_weekend
- 计算：距参考日期的天数、事件之间的时间间隔
- 对周期性特征（如小时、星期几）使用循环编码（Cyclical Encoding）

### 6. 特征交互（Feature Interactions）
- 领域特定的组合（例如，由身高和体重计算身体质量指数 BMI）
- 针对疑似非线性关系构建多项式特征（Polynomial Features）
- 比率特征（例如，每平方英尺价格）

### 7. 特征选择（Feature Selection）
- 移除零方差特征（Zero-Variance Features）
- 移除与其他特征相关系数高于 0.95 的特征
- 根据与预测目标的互信息（Mutual Information）对剩余特征进行排序
- 保留排名前 N 的特征，或使用 L1 正则化（L1 Regularization）进行自动选择

## 输出格式

针对每个特征，需说明：
1. 原始列名及数据类型
2. 应用的变换方法（及原因）
3. 新特征名称
4. 预期影响（高/中/低信号强度）