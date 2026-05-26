# 奇异值分解 (Singular Value Decomposition)

> 奇异值分解（SVD）是线性代数中的瑞士军刀。每个矩阵都存在对应的分解，每位数据科学家都需要掌握它。

**类型：** 实战构建
**编程语言：** Python, Julia
**前置要求：** 第一阶段，第 01 课（线性代数直觉）、第 02 课（向量与矩阵运算）、第 03 课（矩阵变换）
**预计耗时：** 约 120 分钟

## 学习目标

- 通过幂迭代法（Power Iteration）实现 SVD，并解释 U、Sigma 和 V^T 的几何意义
- 应用截断奇异值分解（Truncated SVD）进行图像压缩，并评估压缩率与重建误差之间的关系
- 通过 SVD 计算摩尔-彭若斯伪逆（Moore-Penrose Pseudoinverse），以求解超定最小二乘系统（Overdetermined Least-Squares Systems）
- 将 SVD 与主成分分析（Principal Component Analysis, PCA）、推荐系统（潜在因子）以及自然语言处理（Natural Language Processing, NLP）中的潜在语义分析（Latent Semantic Analysis）建立联系

## 问题背景

假设你有一个 1000x2000 的矩阵。它可能是用户对电影的评分数据，可能是文档-词频表，也可能是图像的像素值。你需要对其进行压缩、去噪、挖掘其中的隐藏结构，或者用它来求解最小二乘系统。然而，特征分解（Eigendecomposition）仅适用于方阵。即便如此，它还要求矩阵必须拥有一组完整的线性无关特征向量。

而 SVD 适用于任意矩阵。无论形状如何、秩是多少，均无需附加条件。它将矩阵分解为三个因子，从而揭示该矩阵对空间进行变换的几何本质。它是整个线性代数中最通用、也最实用的矩阵分解方法。

## 核心概念

### SVD 的几何意义

无论矩阵形状如何，它都依次执行三个操作：旋转、缩放、旋转。SVD 将这一分解过程显式地展现出来。

A = U * Sigma * V^T

      m x n     m x m    m x n    n x n
     (any)    (rotate)  (scale)  (rotate)

对于任意矩阵 A，SVD 将其分解为：
- V^T 在输入空间（n 维）中旋转向量
- Sigma 沿每个轴进行缩放（拉伸或压缩）
- U 将结果旋转到输出空间（m 维）

graph LR
    A["Input space (n-dim)\nData cloud\n(arbitrary orientation)"] -->|"V^T\n(rotate)"| B["Scaled space\nAligned with axes\nthen scaled by Sigma"]
    B -->|"U\n(rotate)"| C["Output space (m-dim)\nRotated to output\norientation"]

可以这样理解：你将一个矩阵交给 SVD，它会告诉你：“该矩阵首先通过 V^T 旋转输入球体，然后通过 Sigma 将其拉伸为椭球体，最后通过 U 旋转该椭球体。”奇异值（Singular Values）就是该椭球体各轴的长度。

### 完整分解

对于形状为 m x n 的矩阵 A：

A = U * Sigma * V^T

where:
  U     is m x m, orthogonal (U^T U = I)
  Sigma is m x n, diagonal (singular values on the diagonal)
  V     is n x n, orthogonal (V^T V = I)

The singular values sigma_1 >= sigma_2 >= ... >= sigma_r > 0
where r = rank(A)

U 的列称为左奇异向量（Left Singular Vectors）。V 的列称为右奇异向量（Right Singular Vectors）。Sigma 的对角线元素称为奇异值。它们始终为非负数，且通常按降序排列。

### 左奇异向量、奇异值与右奇异向量

SVD 的每个组成部分都有明确的几何意义。

**右奇异向量（V 的列）：** 它们构成了输入空间（R^n）的一组标准正交基（Orthonormal Basis）。它们是输入空间中的方向，矩阵会将这些方向映射为输出空间中的正交方向。可以将它们视为定义域的自然坐标系。

**奇异值（Sigma 的对角线）：** 它们是缩放因子。第 i 个奇异值告诉你矩阵沿第 i 个右奇异向量方向拉伸了多少倍。奇异值为零意味着矩阵将该方向完全压缩（坍缩）。

**左奇异向量（U 的列）：** 它们构成了输出空间（R^m）的一组标准正交基。第 i 个左奇异向量是第 i 个右奇异向量（经过缩放后）在输出空间中落点的方向。

它们之间的关系如下：

A * v_i = sigma_i * u_i

The matrix A takes the i-th right singular vector v_i,
scales it by sigma_i, and maps it to the i-th left singular vector u_i.

这为你提供了逐坐标理解任意矩阵作用的直观视角。

### 外积形式

SVD 可以表示为秩为 1 的矩阵之和：

A = sigma_1 * u_1 * v_1^T + sigma_2 * u_2 * v_2^T + ... + sigma_r * u_r * v_r^T

Each term sigma_i * u_i * v_i^T is a rank-1 matrix (an outer product).
The full matrix is the sum of r such matrices, where r is the rank.

这种形式是低秩近似（Low-Rank Approximation）的基础。每一项都增加一层结构。第一项捕获最重要的单一模式，第二项捕获次重要的模式，依此类推。截断该求和式，即可在任意给定秩下获得最优近似。

Rank-1 approx:    A_1 = sigma_1 * u_1 * v_1^T
                  (captures the dominant pattern)

Rank-2 approx:    A_2 = sigma_1 * u_1 * v_1^T + sigma_2 * u_2 * v_2^T
                  (captures the two most important patterns)

Rank-k approx:    A_k = sum of top k terms
                  (optimal by the Eckart-Young theorem)

### 与特征分解的关系

SVD 与特征分解（Eigendecomposition）有着深刻的联系。矩阵 A 的奇异值和向量直接来源于 A^T A 和 A A^T 的特征值与特征向量。

A^T A = V * Sigma^T * U^T * U * Sigma * V^T
      = V * Sigma^T * Sigma * V^T
      = V * D * V^T

where D = Sigma^T * Sigma is a diagonal matrix with sigma_i^2 on the diagonal.

So:
- The right singular vectors (V) are eigenvectors of A^T A
- The singular values squared (sigma_i^2) are eigenvalues of A^T A

Similarly:
A A^T = U * Sigma * V^T * V * Sigma^T * U^T
      = U * Sigma * Sigma^T * U^T

So:
- The left singular vectors (U) are eigenvectors of A A^T
- The eigenvalues of A A^T are also sigma_i^2

这一联系揭示了三点重要信息：
1. 奇异值始终为实数且非负（它们是半正定矩阵特征值的平方根）。
2. 虽然可以通过对 A^T A 进行特征分解来计算 SVD，但这会使条件数（Condition Number）平方，从而损失数值精度。专用的 SVD 算法会避免这一问题。
3. 当 A 为方阵且对称半正定时，SVD 与特征分解完全等价。

### 截断 SVD：低秩近似

Eckart-Young-Mirsky 定理指出，在弗罗贝尼乌斯范数（Frobenius Norm）和谱范数（Spectral Norm）下，对 A 的最佳秩 k 近似仅保留前 k 个最大奇异值及其对应的向量即可获得：

A_k = U_k * Sigma_k * V_k^T

where:
  U_k     is m x k  (first k columns of U)
  Sigma_k is k x k  (top-left k x k block of Sigma)
  V_k     is n x k  (first k columns of V)

Approximation error = sigma_{k+1}  (in spectral norm)
                    = sqrt(sigma_{k+1}^2 + ... + sigma_r^2)  (in Frobenius norm)

这不仅仅是“一个不错”的近似。它在数学上被证明是秩为 k 的最优近似。不存在其他秩为 k 的矩阵比它更接近 A。

| 组件 | 相对大小 | 在秩 3 近似中保留？ |
|-----------|-------------------|------------------------|
| sigma_1 | 最大 | 是 |
| sigma_2 | 较大 | 是 |
| sigma_3 | 中大 | 是 |
| sigma_4 | 中等 | 否（误差） |
| sigma_5 | 中小 | 否（误差） |
| sigma_6 | 较小 | 否（误差） |
| sigma_7 | 极小 | 否（误差） |
| sigma_8 | 微小 | 否（误差） |

保留前 3 个：A_3 捕获最大的三个奇异值。误差 = 剩余值（sigma_4 到 sigma_8）。

如果奇异值衰减迅速，较小的 k 就能捕获矩阵的大部分信息。如果衰减缓慢，则说明该矩阵不具备低秩结构。

### 基于 SVD 的图像压缩

灰度图像本质上是一个像素强度矩阵。一张 800x600 的图像包含 480,000 个数值。SVD 允许你用少得多的数据对其进行近似。

Original image: 800 x 600 = 480,000 values

SVD with rank k:
  U_k:      800 x k values
  Sigma_k:  k values
  V_k:      600 x k values
  Total:    k * (800 + 600 + 1) = k * 1401 values

  k=10:   14,010 values   (2.9% of original)
  k=50:   70,050 values  (14.6% of original)
  k=100: 140,100 values  (29.2% of original)

  The compression ratio improves as k gets smaller,
  but visual quality degrades.

核心洞察：自然图像的奇异值衰减非常迅速。前几个奇异值捕获了宏观结构（形状、梯度），而后续的奇异值则捕获细节和噪声。在秩 50 处截断，通常能生成一张与原图几乎无法区分的图像，同时节省 85% 的存储空间。

### SVD 在推荐系统中的应用

Netflix 大奖（Netflix Prize）使其广为人知。你拥有一个用户-电影评分矩阵，其中大部分条目是缺失的。

             Movie1  Movie2  Movie3  Movie4  Movie5
  User1      [  5      ?       3       ?       1  ]
  User2      [  ?      4       ?       2       ?  ]
  User3      [  3      ?       5       ?       ?  ]
  User4      [  ?      ?       ?       4       3  ]

  ? = unknown rating

核心思想：该评分矩阵具有低秩特性。用户的品味并非完全独立。存在少数几个潜在因子（Latent Factors）（如动作片 vs. 剧情片、老片 vs. 新片、烧脑 vs. 感官刺激）能够解释大部分偏好。

对（补全后的）评分矩阵进行 SVD 分解，可得到：
- U：潜在因子空间中的用户画像
- Sigma：每个潜在因子的重要性
- V^T：潜在因子空间中的电影画像

用户对某部电影的预测评分，即为其用户画像与电影画像的点积（经奇异值加权）。低秩近似用于填补缺失的条目。

在实际应用中，通常会使用 Simon Funk 的增量 SVD 或交替最小二乘法（ALS, Alternating Least Squares）等变体来直接处理缺失数据。但核心思想是一致的：通过 SVD 进行潜在因子分解。

### SVD 在自然语言处理中的应用：潜在语义分析

潜在语义分析（LSA, Latent Semantic Analysis），也称潜在语义索引（LSI, Latent Semantic Indexing），将 SVD 应用于词-文档矩阵（Term-Document Matrix）。

             Doc1   Doc2   Doc3   Doc4
  "cat"      [  3      0      1      0  ]
  "dog"      [  2      0      0      1  ]
  "fish"     [  0      4      1      0  ]
  "pet"      [  1      1      1      1  ]
  "ocean"    [  0      3      0      0  ]

After SVD with rank k=2:

  Each document becomes a point in 2D "concept space."
  Each term becomes a point in the same 2D space.
  Documents about similar topics cluster together.
  Terms with similar meanings cluster together.

  "cat" and "dog" end up near each other (land pets).
  "fish" and "ocean" end up near each other (water concepts).
  Doc1 and Doc3 cluster if they share similar topics.

LSA 是最早成功从原始文本中捕获语义相似性的方法之一。其原理在于，同义词倾向于出现在相似的文档中，因此 SVD 会将它们归入相同的潜在维度。现代词嵌入（Word Embeddings）技术（如 Word2Vec、GloVe）均可视为这一思想的延伸。

### 基于 SVD 的降噪

含噪数据的信号集中在较大的奇异值中，而噪声则散布于所有奇异值。截断操作可以去除噪声基底。

**纯净信号的奇异值：**

| 组件 | 大小 | 类型 |
|-----------|-----------|------|
| sigma_1 | 极大 | 信号 |
| sigma_2 | 大 | 信号 |
| sigma_3 | 中等 | 信号 |
| sigma_4 | 接近零 | 可忽略 |
| sigma_5 | 接近零 | 可忽略 |

**含噪信号的奇异值（噪声叠加至所有分量）：**

| 组件 | 大小 | 类型 |
|-----------|-----------|------|
| sigma_1 | 极大 | 信号 |
| sigma_2 | 大 | 信号 |
| sigma_3 | 中等 | 信号 |
| sigma_4 | 较小 | 噪声 |
| sigma_5 | 较小 | 噪声 |
| sigma_6 | 较小 | 噪声 |
| sigma_7 | 较小 | 噪声 |

graph TD
    A["All singular values"] --> B{"Clear gap?"}
    B -->|"Above gap"| C["Signal: keep these (top k)"]
    B -->|"Below gap"| D["Noise: discard these"]
    C --> E["Reconstruct with A_k to get denoised version"]

该方法广泛应用于信号处理、科学测量和数据清洗。只要矩阵受到加性噪声（Additive Noise）污染，截断 SVD 都是一种从原理上分离信号与噪声的有效手段。

### 基于 SVD 的伪逆

摩尔-彭若斯伪逆（Moore-Penrose Pseudoinverse）A+ 将矩阵求逆推广到了非方阵和奇异矩阵。SVD 使得计算伪逆变得非常简单。

If A = U * Sigma * V^T, then:

A+ = V * Sigma+ * U^T

where Sigma+ is formed by:
  1. Transpose Sigma (swap rows and columns)
  2. Replace each non-zero diagonal entry sigma_i with 1/sigma_i
  3. Leave zeros as zeros

For A (m x n):      A+ is (n x m)
For Sigma (m x n):  Sigma+ is (n x m)

伪逆用于求解最小二乘问题（Least-Squares Problems）。如果 Ax = b 没有精确解（超定方程组 Overdetermined System），那么 x = A+ b 即为最小二乘解（使 ||Ax - b|| 最小化）。

Overdetermined system (more equations than unknowns):

  [1  1]         [3]
  [2  1] x   =   [5]       No exact solution exists.
  [3  1]         [6]

  x_ls = A+ b = V * Sigma+ * U^T * b

  This gives the x that minimizes the sum of squared residuals.
  Same result as the normal equations (A^T A)^(-1) A^T b,
  but numerically more stable.

### 数值稳定性优势

计算 A^T A 的特征分解会使奇异值平方（A^T A 的特征值为 sigma_i^2）。这会导致条件数平方，从而放大数值误差。

Example:
  A has singular values [1000, 1, 0.001]
  Condition number of A: 1000 / 0.001 = 10^6

  A^T A has eigenvalues [10^6, 1, 10^{-6}]
  Condition number of A^T A: 10^6 / 10^{-6} = 10^{12}

  Computing SVD directly: works with condition number 10^6
  Computing via A^T A:     works with condition number 10^{12}
                           (6 extra digits of precision lost)

现代 SVD 算法（如 Golub-Kahan 双对角化）直接对 A 进行操作，从不显式构造 A^T A。这就是为什么你应该始终优先使用 `np.linalg.svd(A)` 而非 `np.linalg.eig(A.T @ A)`。

### 与主成分分析（PCA）的联系

主成分分析（PCA, Principal Component Analysis）本质上就是对中心化数据进行 SVD。这不是类比，而是完全相同的计算过程。

Given data matrix X (n_samples x n_features), centered (mean subtracted):

Covariance matrix: C = (1/(n-1)) * X^T X

PCA finds eigenvectors of C. But:

  X = U * Sigma * V^T    (SVD of X)

  X^T X = V * Sigma^2 * V^T

  C = (1/(n-1)) * V * Sigma^2 * V^T

So the principal components are exactly the right singular vectors V.
The explained variance for each component is sigma_i^2 / (n-1).

In sklearn, PCA is implemented using SVD, not eigendecomposition.
It is faster and more numerically stable.

这意味着你在第 10 课中学到的所有关于降维（Dimensionality Reduction）的知识，底层都是 SVD。PCA 是 SVD 在机器学习中最常见的应用。

## 构建

### 步骤 1：使用幂迭代法（Power Iteration）从零实现奇异值分解（Singular Value Decomposition, SVD）

核心思路：为求解最大奇异值（Singular Value）及其对应向量（Vector），可对 A^T A（或 A A^T）应用幂迭代法。随后对矩阵进行降阶处理（Deflation），并重复该过程以计算下一个奇异值。

import numpy as np

def power_iteration(M, num_iters=100):
    n = M.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iters):
        Mv = M @ v
        v = Mv / np.linalg.norm(Mv)

    eigenvalue = v @ M @ v
    return eigenvalue, v

def svd_from_scratch(A, k=None):
    m, n = A.shape
    if k is None:
        k = min(m, n)

    sigmas = []
    us = []
    vs = []

    A_residual = A.copy().astype(float)

    for _ in range(k):
        AtA = A_residual.T @ A_residual
        eigenvalue, v = power_iteration(AtA, num_iters=200)

        if eigenvalue < 1e-10:
            break

        sigma = np.sqrt(eigenvalue)
        u = A_residual @ v / sigma

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        A_residual = A_residual - sigma * np.outer(u, v)

    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))

    return U, S, V

### 步骤 2：测试并与 NumPy 进行对比

np.random.seed(42)
A = np.random.randn(5, 4)

U_ours, S_ours, V_ours = svd_from_scratch(A)
U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

print("Our singular values:", np.round(S_ours, 4))
print("NumPy singular values:", np.round(S_np, 4))

A_reconstructed = U_ours @ np.diag(S_ours) @ V_ours.T
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.8f}")

### 步骤 3：图像压缩（Image Compression）演示

def compress_image_svd(image_matrix, k):
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return compressed

image = np.random.seed(42)
rows, cols = 200, 300
image = np.random.randn(rows, cols)

for k in [1, 5, 10, 20, 50]:
    compressed = compress_image_svd(image, k)
    error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
    original_size = rows * cols
    compressed_size = k * (rows + cols + 1)
    ratio = compressed_size / original_size
    print(f"k={k:>3d}  error={error:.4f}  storage={ratio:.1%}")

### 步骤 4：降噪（Noise Reduction）处理

np.random.seed(42)
clean = np.outer(np.sin(np.linspace(0, 4*np.pi, 100)),
                 np.cos(np.linspace(0, 2*np.pi, 80)))
noise = 0.3 * np.random.randn(100, 80)
noisy = clean + noise

U, S, Vt = np.linalg.svd(noisy, full_matrices=False)
denoised = U[:, :5] @ np.diag(S[:5]) @ Vt[:5, :]

print(f"Noisy error:    {np.linalg.norm(noisy - clean):.4f}")
print(f"Denoised error: {np.linalg.norm(denoised - clean):.4f}")
print(f"Improvement:    {(1 - np.linalg.norm(denoised - clean) / np.linalg.norm(noisy - clean)):.1%}")

### 步骤 5：伪逆矩阵（Pseudoinverse）

A = np.array([[1, 1], [2, 1], [3, 1]], dtype=float)
b = np.array([3, 5, 6], dtype=float)

U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1.0 / S)
A_pinv = Vt.T @ S_inv @ U.T

x_svd = A_pinv @ b
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
x_pinv = np.linalg.pinv(A) @ b

print(f"SVD pseudoinverse solution:  {x_svd}")
print(f"np.linalg.lstsq solution:   {x_lstsq}")
print(f"np.linalg.pinv solution:    {x_pinv}")


## 实际应用

完整的可运行演示代码位于 `code/svd.py` 中。运行该脚本，即可查看奇异值分解（Singular Value Decomposition, SVD）在图像压缩、推荐系统（Recommendation Systems）、潜在语义分析（Latent Semantic Analysis）和降噪中的应用。

python svd.py

`code/svd.jl` 中的 Julia 版本演示了相同的概念，使用了 Julia 原生的 `svd()` 函数和 `LinearAlgebra` 包。

julia svd.jl

## 交付成果

本章节将产出：
- `outputs/skill-svd.md` - 一份技能指南，帮助你在实际项目中掌握何时以及如何应用 SVD

## 练习

1. 从零开始实现完整的 SVD，不使用幂迭代（Power Iteration）。改为计算 A^T A 的特征分解（Eigendecomposition）以获取 V 和奇异值（Singular Values），然后计算 U = A V Sigma^{-1}。将数值精度与你实现的幂迭代版本以及 NumPy 的结果进行对比。

2. 加载一张真实的灰度图像（或将其转换为灰度图）。分别在秩（Rank）为 1、5、10、25、50、100 时对其进行压缩。针对每个秩，计算压缩比和相对误差。找出图像在视觉上达到可接受程度时的秩。

3. 构建一个微型推荐系统。创建一个 10x8 的用户-电影评分矩阵（User-Movie Ratings Matrix），其中包含部分已知条目。使用行均值填充缺失条目。计算 SVD 并重构一个秩为 3 的近似矩阵。利用重构后的矩阵预测缺失的评分，并验证预测结果是否合理。

4. 创建一个 100x50 的文档-词项矩阵（Document-Term Matrix），包含 3 个合成主题。每个主题关联 5 个词项。添加噪声后应用 SVD，验证前 3 个奇异值是否显著大于其余值。将文档投影到三维潜在空间（Latent Space）中，检查同一主题的文档是否聚集在一起。

5. 生成一个无噪声的低秩矩阵（Low-Rank Matrix，秩为 3，尺寸 50x40），并添加不同水平的高斯噪声（Gaussian Noise）（sigma = 0.1, 0.5, 1.0, 2.0）。针对每个噪声水平，通过遍历 k 从 1 到 40 并测量相对于原始干净矩阵的重构误差，找到最优截断秩（Truncation Rank）。绘制最优 k 值随噪声水平变化的曲线。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 奇异值分解 (Singular Value Decomposition) | “分解任意矩阵” | 将矩阵 A 分解为 U Sigma V^T，其中 U 和 V 为正交矩阵，Sigma 为对角矩阵且对角线元素非负。适用于任意形状的任何矩阵。 |
| 奇异值 (Singular Value) | “该成分的重要性” | Sigma 的第 i 个对角线元素。衡量矩阵沿第 i 个主方向拉伸的程度。始终为非负数，并按降序排列。 |
| 左奇异向量 (Left Singular Vector) | “输出方向” | U 的一列。表示第 i 个右奇异向量（经 sigma_i 缩放后）在输出空间中映射到的方向。 |
| 右奇异向量 (Right Singular Vector) | “输入方向” | V 的一列。表示矩阵将输入空间中的方向映射到第 i 个左奇异向量（经 sigma_i 缩放后）的方向。 |
| 截断奇异值分解 (Truncated SVD) | “低秩近似” | 仅保留前 k 个最大的奇异值及其对应的向量。可生成原矩阵在数学上可证明的最优秩 k 近似（Eckart-Young 定理）。 |
| 秩 (Rank) | “真实维度” | 非零奇异值的数量。表明矩阵实际使用的独立方向数量。 |
| 伪逆 (Pseudoinverse) | “广义逆” | V Sigma+ U^T。对非零奇异值取倒数，零值保持不变。用于求解非方阵或奇异矩阵的最小二乘问题。 |
| 条件数 (Condition Number) | “对误差的敏感度” | sigma_max / sigma_min。条件数越大，意味着输入的微小变化会导致输出的巨大变化。SVD 能直接揭示这一特性。 |
| 隐因子 (Latent Factor) | “隐藏变量” | SVD 发现的低秩空间中的一个维度。在推荐系统中，隐因子可能对应题材偏好；在自然语言处理（NLP）中，可能对应某个主题。 |
| 弗罗贝尼乌斯范数 (Frobenius Norm) | “矩阵的整体规模” | 矩阵所有元素平方和的平方根。等于所有奇异值平方和的平方根。常用于衡量近似误差。 |
| Eckart-Young 定理 (Eckart-Young Theorem) | “SVD 提供最优压缩” | 对于任意目标秩 k，截断 SVD 能在所有可能的秩 k 矩阵中最小化近似误差。 |
| 幂迭代法 (Power Iteration) | “寻找最大特征向量” | 将随机向量反复与矩阵相乘并进行归一化。最终收敛于最大特征值对应的特征向量。它是许多 SVD 算法的基础构建模块。 |

## 进一步阅读

- [Gilbert Strang：《线性代数及其应用》第7章](https://math.mit.edu/~gs/linearalgebra/) - 对奇异值分解（Singular Value Decomposition, SVD）及其应用进行了详尽的阐述
- [3Blue1Brown：SVD 究竟是什么？](https://www.youtube.com/watch?v=vSczTbgc8Rc) - 提供 SVD 的几何直观理解
- [我们推荐奇异值分解](https://www.ams.org/publicoutreach/feature-column/fcarc-svd) - 美国数学学会提供的通俗易懂的概述
- [Netflix 推荐算法大奖赛与矩阵分解（Matrix Factorization）](https://sifter.org/~simon/journal/20061211.html) - Simon Funk 关于将 SVD 应用于推荐系统的原始博客文章
- [潜在语义分析（Latent Semantic Analysis）](https://en.wikipedia.org/wiki/Latent_semantic_analysis) - SVD 在自然语言处理（Natural Language Processing, NLP）领域的开创性应用
- [Trefethen 与 Bau 合著的《数值线性代数（Numerical Linear Algebra）》](https://people.maths.ox.ac.uk/trefethen/text.html) - 理解 SVD 算法及其数值特性的权威参考