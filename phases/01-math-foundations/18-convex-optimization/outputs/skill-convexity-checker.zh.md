---
name: 凸性检查器技能
description: 判断优化问题是否为凸问题，并选择合适的求解器
version: 1.0.0
phase: 1
lesson: 18
tags: [优化, 凸性, 求解器]
---

# 凸性检查器 (Convexity Checker)

如何验证优化问题 (Optimization Problem) 是否为凸问题，以及如何根据验证结果采取相应措施。

## 决策检查清单

1. 目标函数 (Objective Function) 是否为凸函数？（检查海森矩阵 (Hessian Matrix) 的半正定性 (Positive Semi-Definiteness)，或使用复合规则。）
2. 所有不等式约束是否均为 `g_i(x) <= 0` 的形式，且每个 `g_i` 均为凸函数？
3. 所有等式约束是否为仿射（线性）约束？
4. 若以上三项均为“是”，则该问题为凸问题。请使用具有收敛性保证的凸优化求解器 (Convex Solver)。
5. 若任意一项为“否”，则该问题为非凸问题。请使用随机梯度下降 (Stochastic Gradient Descent, SGD) 或 Adam 优化器，并接受局部最优解。

## 如何检验函数的凸性

| 检验方法 | 适用对象 | 具体操作 |
|---|---|---|
| 二阶导数 >= 0 | 标量函数 f(x) | 计算 f''(x)。若对所有 x 均有 f''(x) >= 0，则为凸函数。 |
| 海森矩阵半正定 (PSD) | 多元函数 f(x) | 计算 H(x)。若在所有位置的特征值均 >= 0，则为凸函数。 |
| 定义检验法 | 任意函数 | 对采样的 x, y, t，验证 f(tx + (1-t)y) <= t*f(x) + (1-t)*f(y) 是否成立。 |
| 复合规则 | 复合函数 | 参见下方的复合规则表。 |
| 限制在直线上 | 多元函数 f | 当且仅当对所有 x, v，函数 g(t) = f(x + tv) 关于 t 为凸函数时，f 为凸函数。 |

## 复合规则（保持凸性）

| 运算操作 | 结果 |
|---|---|
| f + g（两者均为凸函数） | 凸函数 |
| c * f（c > 0，f 为凸函数） | 凸函数 |
| max(f, g)（两者均为凸函数） | 凸函数 |
| f(Ax + b)（其中 f 为凸函数） | 凸函数 |
| g(f(x))（其中 g 为凸且单调不减，f 为凸函数） | 凸函数 |
| g(f(x))（其中 g 为凸且单调不增，f 为凹函数） | 凸函数 |
| 凸函数之和 | 凸函数 |
| 凸函数的逐点上确界 | 凸函数 |

## 常见机器学习目标函数：是否为凸？

| 目标函数 | 是否为凸？ | 原因 |
|---|---|---|
| MSE: (1/n) sum(y - Xw)^2 | 是 | 关于 w 为二次型，海森矩阵 = (2/n) X^T X 为半正定 |
| Logistic loss: sum(log(1 + exp(-y_i * w^T x_i))) | 是 | 凸函数之和（属于 log-sum-exp 族） |
| Hinge loss: sum(max(0, 1 - y_i * w^T x_i)) | 是 | 凸（线性）函数的最大值 |
| L2 regularization: lambda * \|\|w\|\|^2 | 是 | 二次型，海森矩阵 = 2*lambda*I |
| L1 regularization: lambda * \|\|w\|\|_1 | 是 | 绝对值之和（凸但不可微） |
| Ridge regression: MSE + L2 | 是 | 两个凸函数之和 |
| LASSO: MSE + L1 | 是 | 两个凸函数之和 |
| Elastic net: MSE + L1 + L2 | 是 | 凸函数之和 |
| SVM (primal): hinge + L2 | 是 | 凸函数之和 |
| Cross-entropy with softmax | 是（针对 logits） | Log-sum-exp 为凸函数 |
| Neural network (any loss) | 否 | 非线性激活函数导致非凸复合 |
| k-means objective | 否 | 包含离散的分配步骤 |
| Matrix factorization: \|\|X - UV^T\|\|^2 | 否 | 关于 U 和 V 为双线性 |
| GAN loss | 否 | 极小极大问题，对生成器而言非凸 |
| Contrastive loss (InfoNCE) | 否 | 包含负样本的指数比值的对数 |

## 基于凸性的求解器选择

| 问题类型 | 求解器 | 收敛性保证 |
|---|---|---|
| 凸、光滑、无约束 | 梯度下降法 (Gradient Descent) | O(1/k) 收敛至全局最小值 |
| 凸、光滑、无约束 | L-BFGS | 超线性收敛至全局最小值 |
| 凸、光滑、无约束 | 牛顿法 (Newton's Method) | 在最小值附近二次收敛（若海森矩阵可计算） |
| 凸、光滑、有约束 | 内点法 (Interior Point Method) | 多项式时间复杂度 |
| 凸、非光滑（L1） | 近端梯度法 (Proximal Gradient) / ISTA | O(1/k) 收敛至全局最小值 |
| 凸、非光滑（L1） | ADMM | 灵活，可处理约束 |
| 凸、二次型 | 共轭梯度法 (Conjugate Gradient) | n 步内精确收敛 |
| 非凸、光滑 | SGD / Adam | 收敛至局部最小值 |
| 非凸、光滑 | SGD + 重启策略 | 平均而言可获得更优的局部最小值 |
| 非凸、光滑 | 过参数化 + SGD | 倾向于平坦极小值，泛化性能好 |

## 常见错误

- 仅因损失函数（loss function）是凸的就假设整个优化问题是凸的。损失函数必须针对你所优化的参数呈凸性。交叉熵（cross-entropy）在逻辑值（logits）上是凸的，但从输入到逻辑值的完整神经网络映射是非凸的。
- 在非凸问题上使用牛顿法（Newton's method）。海森矩阵（Hessian matrix）可能包含负特征值（eigenvalues），导致算法朝向鞍点（saddle points）或极大值点移动，而非极小值点。
- 忽略了 L1 正则化（L1 regularization）会导致目标函数在零点处不可微。标准梯度下降法（gradient descent）在此情况下效果不佳，应改用近端梯度下降法（proximal gradient descent）或次梯度法（subgradient methods）。
- 通过构造 A^T A 导致条件数（condition number）平方级恶化。若需求解最小二乘问题（least-squares problem）且矩阵 A 呈病态（ill-conditioned），应使用 QR 分解（QR decomposition）或奇异值分解（SVD），而非正规方程（normal equations）。
- 未经验证便断言问题是非凸的。许多机器学习（machine learning）问题（如线性模型、支持向量机（SVM）、逻辑回归（logistic regression））本质上是凸的，使用更强大的求解器能带来显著收益。

## 快速测试：我的问题是凸的吗？

1. Write out the objective: minimize f(w) subject to constraints
2. For each term in f(w):
   - Is it quadratic with PSD matrix? -> Convex
   - Is it a norm? -> Convex
   - Is it log-sum-exp? -> Convex
   - Does it involve w nonlinearly (sigmoid(w), w1*w2)? -> Likely non-convex
3. Are all constraints linear or convex inequalities?
4. If ALL terms are convex and constraints are convex/linear -> problem is convex
5. If ANY term is non-convex -> problem is non-convex
