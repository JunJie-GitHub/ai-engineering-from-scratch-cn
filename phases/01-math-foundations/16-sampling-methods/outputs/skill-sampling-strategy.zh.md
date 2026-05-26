---
name: 采样策略选择
description: 为生成、估计或推理选择合适的采样方法
version: 1.0.0
phase: 1
lesson: 16
tags: [采样, MCMC, 生成]
---

# 采样策略选择

如何为文本生成、贝叶斯推断（Bayesian Inference）、蒙特卡洛估计（Monte Carlo Estimation）及模型训练选择合适的采样方法。

## 决策检查清单

1. 你是要生成输出（如文本、图像），还是要估计某个量（如积分、期望值）？
2. 你能直接从目标分布（Target Distribution）中采样，还是只能评估其概率密度？
3. 目标分布是离散的还是连续的？
4. 样本空间的维度是多少？低维（< 5）、中维（5-100）还是高维（> 100）？
5. 你需要精确样本还是近似样本？
6. 你是否需要采样操作可导（即需要计算梯度）？

## 各方法适用场景

| 方法 | 适用场景 | 复杂度 | 是否精确？ |
|---|---|---|---|
| 直接采样（Direct Sampling） | 拥有累积分布函数（CDF）或可使用库函数 | 每个样本 O(1) | 是 |
| 逆累积分布函数法（Inverse CDF） | 已知闭式 CDF 逆函数（如指数分布、柯西分布） | 每个样本 O(1) | 是 |
| Box-Muller 变换 | 无需依赖库即可生成正态分布样本 | 每个样本 O(1) | 是 |
| 拒绝采样（Rejection Sampling） | 可评估目标概率密度函数（PDF），低维（1-3） | 每个样本 O(1/接受率) | 是 |
| 重要性采样（Importance Sampling） | 需要计算期望值，而非获取独立样本 | n 个样本 O(n) | 近似 |
| 分层采样（Stratified Sampling） | 蒙特卡洛估计，希望降低方差 | n 个样本 O(n) | 近似 |
| Metropolis-Hastings 算法 | 高维空间，可评估未归一化密度 | 每步 O(1) + 预热期（Burn-in） | 渐近精确 |
| Gibbs 采样（Gibbs Sampling） | 可从每个条件分布中直接采样 | 每次完整扫描 O(d) | 渐近精确 |
| HMC/NUTS | 高维连续空间，密度函数平滑 | 每步 O(L * d) | 渐近精确 |
| 温度采样（Temperature Sampling） | 大语言模型（LLM）文本生成，控制创造性 | 词表大小 V 对应 O(V) | 不适用 |
| Top-k 采样 | LLM 生成，剔除低概率词元 | O(V log k) | 不适用 |
| Top-p（核采样/Nucleus） | LLM 生成，自适应候选集 | O(V log V) | 不适用 |
| 重参数化技巧（Reparameterization） | 需要高斯采样可导（如变分自编码器 VAEs） | O(d) | 是 |
| Gumbel-Softmax | 需要类别分布采样可导 | k 个类别对应 O(k) | 近似 |

## 大语言模型（LLM）生成参数设置

| 使用场景 | 温度（Temperature） | Top-p | Top-k | 备注 |
|---|---|---|---|---|
| 事实性问答 | 0.0（贪婪解码） | -- | -- | 确定性输出，无随机性 |
| 代码生成 | 0.2-0.5 | 0.9 | -- | 创造性较低，连贯性高 |
| 日常对话 | 0.7 | 0.9 | -- | 平衡型 |
| 创意写作 | 0.9-1.2 | 0.95 | -- | 多样性较高 |
| 头脑风暴 | 1.0-1.5 | 0.95 | -- | 多样性最大化，可能损失连贯性 |

温度（Temperature）与 Top-p 可结合使用。应先应用温度参数（缩放对数几率/Logits），再进行 Top-p 过滤。

## 马尔可夫链蒙特卡洛（MCMC）方法选择

| 特性 | Metropolis-Hastings | Gibbs 采样 | HMC/NUTS |
|---|---|---|---|
| 维度 | 任意 | 任意（最佳 < 100） | 高维（100+） |
| 是否需要条件分布 | 否 | 是 | 否 |
| 是否需要梯度 | 否 | 否 | 是 |
| 接受率 | 调节至 ~23% | 始终 100% | 调节至 ~65% |
| 样本相关性 | 高（随机游走） | 中等 | 低 |
| 预热期（Burn-in） | 较长 | 中等 | 较短 |
| 最佳适用场景 | 探索空间、简单模型 | 共轭模型、贝叶斯网络 | 连续后验分布、深度概率模型 |

## 常见误区

- 在高维空间中使用拒绝采样 (Rejection Sampling)。接受率 (Acceptance Rate) 会随维度呈指数级下降。当维度超过 5 时，应切换至马尔可夫链蒙特卡洛 (Markov Chain Monte Carlo, MCMC)。
- 将 MCMC 提议分布 (Proposal Distribution) 的方差设置得过高或过低。方差过高：大多数提议被拒绝，马尔可夫链 (Markov Chain) 陷入停滞；方差过低：所有提议均被接受，链移动缓慢。对于随机游走 Metropolis-Hastings (Random Walk MH) 算法，目标接受率应设定在 23% 左右。
- 忽略预烧期 (Burn-in)。MCMC 生成的前 N 个样本会受初始值影响而产生偏差。应至少丢弃前 1000 步迭代结果（针对复杂分布需丢弃更多）。
- 使用与目标分布 (Target Distribution) 差异过大的提议分布进行重要性采样 (Importance Sampling)。这会导致少数样本获得极大权重，从而使估计结果不可靠。务必监控有效样本量 (Effective Sample Size, ESS)：ESS = (sum w_i)^2 / sum(w_i^2)。
- 在需要确定性输出 (Deterministic Output) 的任务（如分类、结构化信息提取）中设置温度参数 (Temperature) > 0。此类场景应改用贪心解码 (Greedy Decoding, T=0) 或束搜索 (Beam Search)。
- 未将 Top-p 采样 (Top-p Sampling) 与温度参数结合使用。仅调节温度无法剔除长尾分布中的无效词元 (Tokens)，而 Top-p 采样可以实现这一目的。
- 直接对标准采样操作进行反向传播 (Backpropagation)。针对连续分布（如高斯分布），应使用重参数化技巧 (Reparameterization Trick)；针对离散分布（如类别分布），应使用 Gumbel-Softmax。

## 快速参考：方差缩减技术 (Variance Reduction Techniques)

| 技术 | 工作原理 | 方差缩减效果 |
|---|---|---|
| 分层采样 (Stratified Sampling) | 将空间划分为若干层，对每层分别采样 | 始终 <= 标准蒙特卡洛 (Standard Monte Carlo) |
| 对偶变量法 (Antithetic Variates) | 同时使用 U 和 1-U | 适用于单调函数 |
| 控制变量法 (Control Variates) | 减去一个已知均值的变量 | 缩减幅度与相关性成正比 |
| 重要性采样 | 对来自更优提议分布的样本进行重新加权 | 取决于提议分布的质量 |
| 拉丁超立方采样 (Latin Hypercube Sampling) | 对每个维度独立进行分层 | 在高维空间中优于普通分层采样 |