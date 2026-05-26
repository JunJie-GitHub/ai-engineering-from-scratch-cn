# 大语言模型的差分隐私 (Differential Privacy)

> 差分随机梯度下降 (Differentially Private Stochastic Gradient Descent, DP-SGD) 仍是标准方案——注入噪声的梯度更新可提供形式化的 (ε, δ) 保证。其在计算、内存和模型效用 (utility) 方面的开销巨大；参数高效的差分隐私微调（低秩自适应 (Low-Rank Adaptation, LoRA) + DP-SGD）是 2025 年的主流配置（ACM 2025）。现有两类研究证据存在矛盾：基于金丝雀 (canary) 的成员推理 (Membership Inference)（Duan 等，2024）显示其对语言模型的攻击效果有限；而训练数据提取 (Training-Data Extraction)（Carlini 等，2021；Nasr 等，2025）却能恢复大量逐字记忆的内容。这一矛盾在 2025 年 3 月的研究（arXiv:2503.06808）中得到解释：差异源于测量对象的不同——人为插入的金丝雀数据与“最易提取”的真实数据。新型金丝雀设计实现了无需影子模型 (shadow models) 的基于损失 (loss-based) 的成员推理攻击 (MIA)，并首次对使用真实数据训练且具备实际差分隐私保证的大语言模型进行了有意义的差分隐私审计。替代方案包括：PMixED（arXiv:2403.15638）——通过在下一个词元 (token) 分布上采用混合专家 (Mixture of Experts) 架构，在推理阶段实现私有化预测；以及差分隐私合成数据生成 (DP synthetic data generation)（Google Research 2024）。新兴攻击：基于大语言模型反馈的差分隐私逆向攻击 (Differential Privacy Reversal via LLM Feedback) —— 置信度分数泄露 (confidence-score leakage)。

**Type:** 构建实践
**Languages:** Python（标准库，DP-SGD 噪声注入与 ε-δ 隐私预算会计 (ε-δ accountant) 演示）
**Prerequisites:** 阶段 01 · 09（信息论），阶段 10 · 01（大模型训练）
**Time:** 约 60 分钟

## 学习目标

- 定义 (ε, δ)-差分隐私 ((epsilon, delta)-differential privacy) 并阐述 DP-SGD 的标准流程。
- 解释 2024-2025 年间的研究分歧：金丝雀成员推理攻击与训练数据提取攻击为何呈现出不同的结论。
- 描述 PMixED 的原理，并说明为何推理阶段的私有化预测可作为差分隐私训练的替代方案。
- 描述基于大语言模型反馈的差分隐私逆向攻击 (Differential Privacy Reversal via LLM Feedback)。

## 问题背景

大语言模型具有记忆特性。Carlini 等人（2021）的研究表明，生产环境中的语言模型能够按需逐字复现训练文本。差分隐私 (Differential Privacy, DP) 是形式化的防御手段：通过训练使模型输出在数学上可证明对任意单个训练样本不敏感。2024-2025 年的证据表明，DP-SGD 是必要的，但实际部署的 ε 值可能并未与真实的威胁模型相匹配。

## 核心概念

### (ε, δ)-差分隐私（Differential Privacy）

若对于任意仅相差一条记录的两个数据集以及任意输出事件 S，随机算法 M 满足以下条件，则称其具备 (ε, δ)-差分隐私（(ε, δ)-DP）：
P(M(D) in S) <= e^ε * P(M(D') in S) + δ。

直观理解：输出分布足够接近（由 ε 参数化），使得除了以 δ 的概率外，无法可靠地推断出任何单个个体的贡献。

### 差分隐私随机梯度下降（DP-SGD）

Abadi 等人（2016）提出的标准流程：
1. 采样一个小批量（mini-batch）。
2. 计算逐样本梯度（per-example gradients）。
3. 将每个逐样本梯度裁剪至阈值 C。
4. 对裁剪后的梯度求和，并添加标准差为 σ * C 的高斯噪声。
5. 使用加噪后的梯度总和更新模型参数。

隐私开销（privacy cost）由隐私会计（accountant，如矩会计 Moments Accountant、Rényi DP 会计）进行追踪。大语言模型（LLM）文献中报告的 ε 值因威胁模型、数据敏感性和效用目标的不同而差异巨大；不存在普遍“安全”的默认 ε 值。已发表的案例在某些 LLM 训练设置中 ε 值大致在 1–10 之间，但这仅为示例说明，并非推荐默认值。通常，更低的 ε 需要添加更多噪声，并可能导致更大的效用损失（utility loss）。

### LoRA + DP-SGD

对前沿模型进行完整的 DP-SGD 训练成本过高。LoRA（Hu 等人，2022）将梯度更新限制在一个小型适配器（adapter）上，从而减少了逐样本梯度的存储开销。LoRA + DP-SGD 是 2025 年常见的配置。差分隐私保证仅适用于适配器部分；基础模型（base model）保持固定。

### 2024-2025 年的研究分歧

两条证据线：

- **金丝雀成员推理攻击（Canary MIA，Duan 等人 2024）**：在训练数据中插入唯一的“金丝雀”样本，测量成员推理攻击者（membership-inference attacker）是否能识别它们。在语言模型上的报告显示成功率有限，表明 MIA 较难实施。
- **训练数据提取（Training-data extraction，Carlini 2021，Nasr 等人 2025）**：使用前缀提示模型，测量其是否能逐字恢复训练数据中的文本。报告显示存在大量记忆现象，表明在相关意义上 MIA 较易实施。

2025 年 3 月的研究结论（arXiv:2503.06808）指出：两者衡量的是不同维度。MIA 针对插入的金丝雀样本提问“样本 e 是否在数据集 D 中？”。而数据提取提问“我能从 D 中恢复出什么？”。对隐私而言，真正关键的是“最易提取”的样本；金丝雀样本由于未经过可提取性优化，会低估这一风险。

新型金丝雀设计。无需影子模型（shadow models）的基于损失的 MIA。首次在真实数据上对 LLM 进行具有现实差分隐私保证的非平凡 DP 审计。

### DP 训练的替代方案

- **PMixED（arXiv:2403.15638）**：推理阶段的私有预测。针对下一词元分布（next-token distributions）采用专家混合（Mixture of Experts）架构；每个专家仅接触训练数据的一个分片（shard）；聚合时添加噪声以实现 DP。完全避免了 DP 训练。
- **DP 合成数据生成（Google Research 2024）**：使用 DP-SGD 对 LoRA 进行微调，采样生成合成数据，并在该合成数据上训练下游分类器。

两者均以改变威胁模型为代价，规避了完整 DP 训练带来的效用损失。

### 基于 LLM 反馈的差分隐私逆向攻击

2025 年新兴的攻击方式。利用 DP 训练模型的置信度分数（confidence scores）作为预言机（oracle）来重新识别个体。即使模型输出本身未发生泄露，置信度分布仍可能造成隐私泄露。

防御措施：不暴露置信度，或在暴露前对其进行截断/量化。这是超出 (ε, δ)-DP 训练之外的额外要求。

### 在 Phase 18 中的定位

第 20-21 课讲解偏差/公平性（bias/fairness）。第 22 课讲解隐私。第 23 课讲解通过水印技术（watermarking）实现的数据溯源（provenance）。第 27 课涵盖监管层面的数据溯源层。

## 使用方法

`code/main.py` 在一个玩具二分类数据集上模拟了差分隐私随机梯度下降（Differential Privacy Stochastic Gradient Descent, DP-SGD）。你可以遍历噪声乘数 σ 和裁剪范数 C，并跟踪 (ε, δ) 隐私预算与准确率损失。金丝雀攻击（Canary Attack）会插入一个独特的训练样本，并测量在对数损失测试（log-loss test）中，应用差分隐私（Differential Privacy, DP）前后能否检测到该样本。

## 交付成果

本模块将生成 `outputs/skill-dp-audit.md` 文件。针对语言模型部署中提出的差分隐私声明，该文件将审计以下内容：(ε, δ) 具体数值、所使用的隐私会计（Privacy Accountant）方法、成员推理攻击（Membership Inference Attack, MIA）评估协议，以及是否已评估置信度暴露向量（confidence-exposure vectors）。

## 练习

1. 运行 `code/main.py`。在 {0.5, 1.0, 2.0} 范围内遍历 σ，报告 (ε, δ) 与准确率之间的权衡关系。找出模型效用（utility）急剧下降的临界点。

2. 实现金丝雀样本插入与对数损失测试。测量在 σ = 1.0 时，应用 DP-SGD 前后的样本检测率。

3. 阅读 Nasr 等人（2025）关于训练数据提取的论文。为何在中等 ε 值下，数据提取的成功率并未显著下降？这对将 MIA 作为评估手段有何启示？

4. 设计一个完全在推理阶段运行的 PMixED（arXiv:2403.15638）部署方案。PMixED 解决了哪些 DP-SGD 未涵盖的威胁模型（threat model）？

5. 概述基于大语言模型反馈的差分隐私逆向攻击（DP Reversal via LLM Feedback）。设计一种限制置信度分数泄露的防御措施，并估算其部署成本。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| DP（差分隐私） | “(ε, δ)-差分隐私” | 形式化隐私保护：在相邻数据集发生变化时，输出分布保持高度相似 |
| DP-SGD（差分隐私随机梯度下降） | “注入噪声的 SGD” | 梯度裁剪 + 添加高斯噪声；标准的差分隐私训练算法 |
| LoRA + DP-SGD | “高效的隐私微调” | 在低秩适配器上应用 DP-SGD；2025 年主流配置 |
| MIA（成员推理攻击） | “成员推断” | 判定特定样本是否属于训练集的攻击方法 |
| Canary（金丝雀样本） | “插入的水印样本” | 用于量化差分隐私泄露程度的唯一性训练样本 |
| PMixED | “私有推理混合架构” | 在推理阶段通过专家混合模型处理下一个 token 分布，实现差分隐私 |
| DP Reversal（差分隐私逆向攻击） | “置信度泄露攻击” | 将模型置信度作为预言机（oracle）以实施重识别的攻击 |

## 延伸阅读

- [Abadi 等人 — DP-SGD (arXiv:1607.00133)](https://arxiv.org/abs/1607.00133) — 标准的差分隐私训练算法
- [Carlini 等人 — 提取训练数据 (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805) — 数据提取领域的奠基性论文
- [Duan 等人 — 大语言模型上的金丝雀 MIA (arXiv:2402.07841, 2024)](https://arxiv.org/abs/2402.07841) — 成功率受限的 MIA 研究
- [Kowalczyk 等人 — 审计大语言模型的差分隐私 (arXiv:2503.06808, 2025年3月)](https://arxiv.org/abs/2503.06808) — 化解相关张力与争议
- [PMixED (arXiv:2403.15638)](https://arxiv.org/abs/2403.15638) — 推理阶段的私有化预测