---
name: 技能-信息论
description: 将信息论概念应用于机器学习损失函数、模型评估与特征选择
version: 1.0.0
phase: 1
lesson: 9
tags: [信息论, 熵, 损失函数]
---

# 机器学习中的信息论 (Information Theory)

在机器学习系统中，何时使用熵 (Entropy)、交叉熵 (Cross-Entropy)、KL散度 (KL Divergence) 和互信息 (Mutual Information)。

## 决策清单

1. 衡量单一分布的不确定性？使用**熵**。
2. 衡量模型对真实标签的拟合程度？使用**交叉熵**（即你的分类损失）。
3. 衡量两个分布之间的距离？使用**KL散度**。
4. 检查两个变量是否相关？使用**互信息**。
5. 报告语言模型质量？使用**困惑度 (Perplexity)**（交叉熵的指数）。
6. 将一个模型蒸馏到另一个模型？最小化从教师模型到学生模型的**KL散度**。

## 何时使用各度量指标

| 度量指标 | 公式 | 使用场景 | 机器学习应用 |
|---|---|---|---|
| 熵 H(P) | -sum(p log p) | 该分布的不确定性有多大？ | 数据复杂度、最大熵模型 (Maximum Entropy Models) |
| 交叉熵 H(P,Q) | -sum(p log q) | 模型 Q 预测真实分布 P 的效果如何？ | 分类损失、语言模型损失 |
| KL散度 D(P\|\|Q) | sum(p log(p/q)) | P 和 Q 的差异有多大？ | 变分自编码器损失 (VAE Loss / ELBO)、知识蒸馏 (Knowledge Distillation)、基于人类反馈的强化学习 (RLHF) |
| 互信息 I(X;Y) | H(X) - H(X\|Y) | Y 能提供多少关于 X 的信息？ | 特征选择、表示学习 (Representation Learning) |
| 困惑度 | exp(H(P,Q)) or 2^H | 模型的困惑程度如何？ | 语言模型评估 |
| 条件熵 H(X\|Y) | -sum(p(x,y) log p(x\|y)) | 已知 Y 后 X 的剩余不确定性 | 特征信息量评估 |

## 核心关系

Cross-entropy  = Entropy + KL divergence
H(P, Q)        = H(P)   + D_KL(P || Q)

Since H(P) is constant during training:
  Minimizing cross-entropy = Minimizing KL divergence

Mutual information = Entropy - Conditional entropy
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)

Perplexity = exp(cross-entropy in nats)
           = 2^(cross-entropy in bits)

## 快速参考：公式与单位

| 公式 | 比特 (Bits, log base 2) | 奈特 (Nats, log base e) |
|---|---|---|
| 信息量 (Information): -log(p) | -log2(p) | -ln(p) |
| 熵 (Entropy): -sum(p log p) | bits | nats |
| 1 nat = | 1.4427 bits | 1 nat |
| PyTorch 默认值 | -- | nats |
| 信息论论文常用单位 | bits | -- |

## 数值解读

| 熵值 | 含义 |
|---|---|
| 0 | 确定性分布。某一结果的概率为 1。 |
| log(n) | 最大不确定性。n 个结果的均匀分布。 |
| 低 | 分布呈尖峰状。模型置信度高。 |
| 高 | 分布较平坦。模型不确定性高。 |

| 困惑度值 | 语言模型质量 |
|---|---|
| 1 | 完美预测（实践中几乎不会发生） |
| 10 | 平均而言，模型在约 10 个等概率的 token 中进行选择 |
| 50 | GPT-2 在标准基准测试上的水平 |
| < 10 | 在表征充分的领域达到最先进水平 (State-of-the-art) |

## 常见误区

- 计算 KL 散度时误认为其具有对称性。D_KL(P||Q) != D_KL(Q||P)。若需对称度量，请使用 Jensen-Shannon 散度 (Jensen-Shannon Divergence)：JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)，其中 M = 0.5*(P+Q)。
- 忘记独热编码 (One-hot) 标签下的交叉熵可简化为 -log(p_true_class)。当真实分布为独热编码时，无需对所有类别求和。
- 代码中使用以 2 为底的对数，但报告时却使用奈特 (Nats)（或反之）。PyTorch 默认使用自然对数。需乘以 log2(e) = 1.4427 才能将奈特转换为比特 (Bits)。
- 计算空事件或零概率事件的熵。惯例规定：0 * log(0) = 0，因为 lim(p->0) p*log(p) = 0。
- 跨不同词表大小比较困惑度。词表大小为 50k 且困惑度为 30 的模型，与词表大小为 10k 且困惑度为 30 的模型不具备直接可比性。

## 各概念在生产环境机器学习中的应用场景

| 概念 | 常见应用场景 |
|---|---|
| 交叉熵损失 (Cross-entropy loss) | 各类分类模型（如 nn.CrossEntropyLoss） |
| KL散度 (KL divergence) | VAE ELBO、PPO 裁剪、知识蒸馏 |
| 熵正则化 (Entropy regularization) | 强化学习中的探索奖励（熵值越高 = 探索越多） |
| 互信息 (Mutual information) | 特征选择、InfoNCE 损失（对比学习） |
| 困惑度 (Perplexity) | 语言模型基准测试（越低越好） |
| 标签平滑 (Label smoothing) | 使用软目标替代独热编码，降低交叉熵的过度自信倾向 |
| 温度缩放 (Temperature scaling) | 在 softmax 前将 logits 除以 T，控制输出分布的熵 |