---
name: skill-svd
description: 将奇异值分解（Singular Value Decomposition, SVD）应用于实际问题，包括数据压缩、降噪、推荐系统和最小二乘求解
phase: 1
lesson: 11
---

你是一位擅长将奇异值分解（Singular Value Decomposition, SVD）应用于实际工程问题的专家。当面对涉及矩阵、数据压缩、噪声、缺失数据或线性系统的任务时，请判断 SVD 是否为合适的工具，并确定如何应用它。

## 决策框架

### 步骤 1：识别问题类型

- **数据压缩 / 降维（Dimensionality Reduction）**：使用截断奇异值分解（Truncated SVD）。保留前 k 个奇异值。通过能量阈值（95% 是常见目标）或下游任务表现来选择 k。
- **降噪（Noise Reduction）**：计算完整 SVD。在奇异值谱中寻找间隙。在间隙处进行截断。该间隙用于区分信号与噪声。
- **缺失数据 / 推荐系统（Recommendation Systems）**：填充缺失项（使用行均值或零），计算 SVD，使用低秩重构。在生产环境中，请使用交替最小二乘法（Alternating Least Squares, ALS）或原生支持处理缺失数据的增量 SVD（Incremental SVD）。
- **最小二乘 / 伪逆（Pseudoinverse）**：计算 SVD。对非零奇异值求逆。将 V Sigma+ U^T 与目标向量相乘。比正规方程（Normal Equations）更稳定。
- **文本相似度 / 主题建模（Topic Modeling）**：构建词项-文档矩阵（Term-Document Matrix）。应用 SVD（即潜在语义分析/索引，Latent Semantic Analysis/Indexing, LSA/LSI）。将文档和词项投影到低秩空间。使用余弦相似度（Cosine Similarity）进行比较。
- **数值秩确定（Numerical Rank Determination）**：计算 SVD。统计高于阈值（相对于最大奇异值）的奇异值数量。这比行化简更可靠。
- **矩阵范数计算（Matrix Norm Computation）**：谱范数（Spectral Norm）= 最大奇异值。Frobenius 范数 = 奇异值平方和的平方根。核范数（Nuclear Norm）= 奇异值之和。
- **条件数（Condition Number）**：sigma_max / sigma_min。用于衡量系统对扰动的敏感程度。

### 步骤 2：选择合适的变体

| 场景 | 方法 | 原因 |
|-----------|--------|-----|
| 稠密矩阵，需要完整分解 | `np.linalg.svd(A)` / Julia 中的 `svd(A)` | 标准算法，数值稳定 |
| 仅需前 k 个分量 | `scipy.sparse.linalg.svds(A, k)` | 当 k 较小时，比完整 SVD 更快 |
| 稀疏矩阵 | `scipy.sparse.linalg.svds` | 高效处理稀疏存储格式 |
| 流式数据 | 增量 SVD / 在线 SVD | 无需从头重新计算即可更新分解结果 |
| 缺失数据（推荐系统） | ALS、Funk SVD 或 NMF | 标准 SVD 要求矩阵完整 |
| 超大规模矩阵（数百万行） | 随机化 SVD（`sklearn.utils.extmath.randomized_svd`） | O(mn log k) 优于 O(mn min(m,n)) |
| 对中心化数据进行主成分分析（Principal Component Analysis, PCA） | 对中心化数据矩阵进行 SVD | 等价于协方差矩阵的特征分解（Eigendecomposition），但数值更稳定 |

### 步骤 3：选择秩 k

- **能量阈值（Energy Threshold）**：计算累积能量 = sum(sigma_1^2 ... sigma_k^2) / sum(all sigma^2)。当能量超过 0.95（高保真任务可设为 0.99）时停止。
- **间隙检测（Gap Detection）**：绘制奇异值曲线。寻找急剧下降的拐点。该间隙标志着信号与噪声的分界线。
- **交叉验证（Cross-Validation）**：针对下游任务，遍历不同的 k 值，并在预留数据集上评估性能。
- **肘部法则（Elbow Method）**：绘制重构误差随 k 变化的曲线。肘部位置即为增加更多分量不再带来显著收益的临界点。
- **领域知识（Domain Knowledge）**：若已知数据包含 d 个潜在因子，则直接设定 k = d。

### 步骤 4：验证结果

- **重构误差（Reconstruction Error）**：计算 ||A - A_k|| / ||A||。若截断合理，该值应较小。
- **解释方差（Explained Variance）**：对于 PCA 或压缩任务，报告所捕获的总方差（能量）比例。
- **下游任务性能（Downstream Task Performance）**：若 SVD 作为预处理步骤，请测量端到端指标。
- **视觉检查（Visual Inspection）**：对于图像，直观对比原始图像与重构图像。对于推荐系统，将预测结果与已知评分进行核对。

## 常见错误

- 通过计算 A^T A 的特征分解（Eigendecomposition）来求解奇异值分解（SVD）。这会导致条件数（Condition Number）平方，从而损失数值精度（Numerical Precision）。请使用专用的 SVD 例程。
- 仅需前 k 个分量时却使用完整 SVD（Full SVD）。对于大型矩阵，请使用截断 SVD（Truncated SVD）或随机 SVD（Randomized SVD）。
- 直接对包含缺失值的矩阵应用 SVD。标准 SVD 要求矩阵完整。请改用矩阵补全（Matrix Completion）方法（如 ALS、Funk SVD）。
- 忽略数据预处理中的中心化（Centering）。对于主成分分析（PCA），在执行 SVD 前必须对数据进行中心化（减去均值）。若不进行中心化，第一个主成分将捕获数据的均值而非方差（Variance）。
- 过度截断。保留的奇异值（Singular Values）过少会丢失有效信号，保留过多则会引入噪声。建议使用能量阈值（Energy Thresholds）或交叉验证（Cross-Validation）来确定截断点。
- 混淆 SVD 与特征分解。SVD 适用于任意矩阵（任意形状、任意秩）。特征分解则要求矩阵为方阵且具备完整的特征向量集。对于对称半正定矩阵（Symmetric Positive Semi-definite Matrices），两者是等价的。

## 代码模式

### 快速压缩
U, S, Vt = np.linalg.svd(A, full_matrices=False)
k = np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.95) + 1
A_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

### 用于最小二乘的伪逆
U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.array([1/s if s > 1e-10 else 0 for s in S])
x = Vt.T @ np.diag(S_inv) @ U.T @ b

### 去噪
U, S, Vt = np.linalg.svd(noisy_data, full_matrices=False)
k = find_gap(S)
clean_data = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

### 大规模 PCA
from sklearn.utils.extmath import randomized_svd
U, S, Vt = randomized_svd(X_centered, n_components=50, random_state=42)
explained_variance = S**2 / (n_samples - 1)

## 何时不应使用 SVD

- 矩阵非常稀疏且仅需少量分量。请直接使用稀疏特征求解器（Sparse Eigensolvers）。
- 需要非负因子（如主题建模、光谱解混）。请改用非负矩阵分解（NMF）。
- 数据具有强烈的非线性结构，线性方法无法有效捕获。请使用自编码器（Autoencoders）或流形学习（Manifold Learning）。
- 需要对流数据进行实时更新且矩阵持续变化。请使用增量/在线 SVD（Incremental/Online SVD）或近似算法。
- 矩阵虽能装入内存但规模过大，导致即使使用随机 SVD 也过慢。可考虑矩阵草图（Matrix Sketching）方法或基于采样的策略。

## 计算成本

| 方法 | 时间复杂度 | 空间复杂度 |
|--------|------|-------|
| m x n 矩阵的完整 SVD | O(mn min(m,n)) | O(mn) |
| 截断 SVD（前 k 个） | O(mnk) | O((m+n)k) |
| 随机 SVD（前 k 个） | O(mn log k) | O((m+n)k) |
| 幂迭代（1 个向量） | O(mn * iters) | O(m+n) |

对于 10000 x 5000 的矩阵：
- 完整 SVD：约 2500 亿次运算
- 截断 SVD（k=50）：约 25 亿次运算
- 随机 SVD（k=50）：约 5 亿次运算

请根据实际的数据规模与精度要求选择合适的方法。