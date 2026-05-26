# 智能体共识与拜占庭容错 (Byzantine Fault Tolerance)

> 经典分布式系统拜占庭容错 (BFT) 与随机性大语言模型 (LLM) 的碰撞。2025至2026年间涌现出三个研究方向：**CP-WBFT** (arXiv:2511.10400) 通过置信度探测 (confidence probe) 为每次投票赋予权重；**DecentLLMs** (arXiv:2507.14928) 采用无领导者架构，通过并行工作节点提案与几何中位数聚合 (geometric-median aggregation) 进行决策；**WBFT** (arXiv:2505.05103) 将加权投票与层次结构聚类 (Hierarchical Structure Clustering) 相结合，以划分核心节点与边缘节点。《AI智能体能否达成共识？》(arXiv:2603.01213) 给出的客观实证结果表明，即便只是标量共识 (scalar agreement) 在当今也十分脆弱——单个欺骗性智能体就足以破坏智能体混合系统 (Mixture-of-Agents)。BFT 是必要条件，但并非充分条件。本课程将构建一个最小化的 BFT 协议，注入三种针对智能体的特定攻击（拜占庭式谎言 (byzantine lie)、阿谀奉承式从众 (sycophantic conformity)、相关性错误单一文化 (correlated-error monoculture)），并评估各共识变体如何应对这些攻击。

**类型：** 学习 + 构建
**编程语言：** Python (stdlib)
**前置知识：** 第16阶段 · 07（心智社会与辩论），第16阶段 · 13（共享内存）
**预计耗时：** 约75分钟

## 问题

假设你有 N 个大语言模型智能体，各自生成一个答案。它们之间存在分歧。多数投票却选出了错误答案，因为其中两个智能体存在相关性（共享相同的基础模型、训练数据和故障模式）。第三个智能体恰好以一种新颖的方式出错——导致所谓的“多数”实际上是虚假多数。

现在引入一个欺骗性智能体：它故意撒谎。或者引入一个阿谀型智能体：它总是附和最后一个发言者。在经典 BFT 中，假设拜占庭节点占比 `f < n/3` 且行为任意。但 2026 年的现实是，大语言模型节点即使诚实也具有随机性，跨模型存在相关性，且会相互影响输出。你不能将它们视为独立的伯努利投票者 (Bernoulli voters)。

经典 BFT（实用拜占庭容错 (PBFT)，1999）并非错误，而是不完整。它能处理任意比特翻转，但无法处理“三个诚实智能体因共享训练数据而产生相同幻觉 (hallucination)”的情况。本课程将以 PBFT 为基础，叠加三项 2025-2026 年的适应性改进。

## 概念

### 经典拜占庭容错（Byzantine Fault Tolerance, BFT）机制提供的保障

实用拜占庭容错（Practical Byzantine Fault Tolerance, Castro & Liskov, OSDI 1999）可容忍 `f < n/3` 个拜占庭节点。该协议包含三个阶段（预准备、准备、提交）和两种原语（签名消息、仲裁证书（Quorum Certificates））。它能在 `n >= 3f + 1` 个诚实或恶意节点中就单一值达成共识。

该机制提供的保障非常严格，但基于以下假设：

1. **故障相互独立。** 拜占庭节点之间不会协同作恶。
2. **诚实节点绝对诚实。** 诚实节点输出的正确性无需担忧；该协议仅用于消除分歧。
3. **问题存在真实答案（Ground Truth）。** 即使对错误的事实达成共识，协议依然视为成功。

大语言模型智能体（LLM Agents）打破了上述所有假设。运行相同基座模型的两个智能体会共享相同的缺陷。即便是“诚实”的 LLM 仍会产生幻觉（Hallucination）。而在面对模糊问题时，“真相”完全由智能体自行决定——不存在外部预言机（Oracle）进行验证。

### 针对 LLM 的三种特有攻击

**拜占庭谎言（Byzantine Lie）。** 某个智能体故意输出错误答案。只要满足 `f < n/3`，经典 BFT 即可处理此类情况。

**阿谀型从众（Sycophantic Conformity）。** 某个智能体在投票前会先阅读其他智能体的答案，并倾向于与最后发言者保持一致。这并非恶意行为，但会与“声音最大”的节点产生关联。经典 BFT 无法阻止此类行为，因为该智能体仍能通过所有签名验证。

**相关错误同质化（Correlated-Error Monoculture）。** 三个智能体共享同一个基座模型，它们会幻觉出相同的错误答案，导致多数派出错。经典 BFT 对此无能为力，因为这三个智能体是“诚实”地达成了一致。

### 2025-2026 年的应对方案

**CP-WBFT** (arXiv:2511.10400) —— 置信度探测加权拜占庭容错（Confidence-Probed Weighted BFT）。每个投票节点会为其答案附加一个置信度探测值（自报概率或由独立校准模型预测）。投票权重随置信度动态调整。在完全图拓扑中报告称 BFT 性能提升了 85.71%。缓解目标：阿谀型从众（从众节点对其主动提出的立场往往置信度较低）。

**DecentLLMs** (arXiv:2507.14928) —— 无领导者架构。工作智能体（Worker Agents）并行提出方案，评估智能体（Evaluator Agents）对方案进行打分，最终答案取所有打分位置的几何中位数（Geometric Median）。在 `f < n/2` 时具备鲁棒性。缓解目标：拜占庭谎言与相关错误（几何中位数对异常值具有鲁棒性，会向密集簇靠拢，而非偏向模型偏差的平均值）。

**WBFT** (arXiv:2505.05103) —— 基于层次结构聚类的加权拜占庭容错（Weighted BFT with Hierarchical Structure Clustering）。投票权重由回答质量与基于历史学习到的信任评分共同决定。将智能体聚类为核心（Core）与边缘（Edge）两组；核心智能体必须率先达成共识，边缘智能体随后跟随。缓解目标：可扩展性（核心共识规模小且速度快）以及部分缓解同质化问题（核心组可特意选择多样化模型）。

### 实证研究：“AI 智能体能达成共识吗？” (arXiv:2603.01213)

该论文在多个前沿模型上测量了标量一致性（Scalar Agreement，即 LLM 智能体就单一数值达成一致的程度）。研究结果令人不安：

- 即使在没有对抗者的情况下，LLM 智能体在许多基准测试中对标量问题的分歧率仍超过 30%。
- 仅一个采用欺骗性人设的智能体，就能将智能体混合（Mixture-of-Agents）共识从诚实基线拉偏 40 个百分点以上。
- 分歧率与模型多样性呈正相关——异构集成（Heterogeneous Ensembles）的分歧高于同构集成（优点：误差互不相关），但收敛漂移也更慢（缺点：达成共识所需时间更长）。

核心结论：BFT 提供了对齐输出的机制，但无法告诉你最终对齐的输出是否正确。需将其与验证机制（第 16 阶段 · 08 角色专业化）、多样性策略（第 16 阶段 · 15 辩论变体）以及评估智能体（第 16 阶段 · 24 基准测试）结合使用。

### 精简版核心协议

面向 LLM 智能体的最小化 BFT 轮次流程：

1. task arrives; each agent i produces answer a_i
2. each agent attaches confidence probe c_i in [0, 1]
3. aggregator collects (a_i, c_i) from all n agents
4. aggregator groups by semantic cluster (equivalent answers)
5. aggregator computes weight for each cluster C:
     w(C) = sum_{i in C} c_i
6. winner = cluster with max weight, if max > threshold * sum(c_i)
   else: retry or escalate
7. minority clusters logged with provenance for post-hoc audit

语义聚类（Semantic Clustering）步骤是针对 LLM 的特有设计。例如，“该研究报告为 4.2%”和“提升了 4.2%”这两个答案属于同一聚类。简单的字符串相等性检查会漏掉这种情况。在生产环境中，建议使用轻量级嵌入模型（Embedding Model）或显式规范化（Canonicalization）处理。

### 阈值调优

`threshold` 参数决定了何时接受结果、何时重试。设置过低：会接受弱势多数派的结果。设置过高：可能永远无法接受任何结果。经验取值范围：当 `n=5-7` 个智能体时约为 0.5-0.67，`n` 较小时需适当提高。若低于该阈值，则应升级至人工处理或切换至另一组智能体集成。

### 共识机制无效的场景

- **模糊问题。** 如果问题本身没有真实答案（Ground Truth），共识仅仅是一种主观意见。应如实界定。
- **复合型问题。** 例如“编写代码并解释它”——包含两个独立答案。应对每个部分分别进行投票。
- **对抗性多轮交互。** 如果智能体能够观察历史轮次并进行模仿（如 Du 2023 辩论研究所述），它们会开始无视事实真相而盲目达成一致。应限制交互轮次（通常为 2-3 轮）。

## 构建

`code/main.py` 实现了：

- `AgentVoter` —— 一种脚本化策略，返回（答案，置信度）。
- `MajorityVote` —— 经典多数投票（plurality）。
- `CPWBFT` —— 结合语义聚类（semantic clustering）的置信度加权投票。
- `DecentLLMs` —— 对评分提案进行几何中位数（geometric-median）聚合。
- `Scenario` —— 在三种攻击模式下运行每个聚合器。

实现的攻击模式：

1. `byzantine`：一个智能体以高置信度撒谎。
2. `sycophancy`：一个智能体复制它看到的第一个答案，并赋予匹配的置信度。
3. `monoculture`：三个智能体以中等置信度共享一个错误答案（相关误差，correlated error）。

运行：

python3 code/main.py

预期输出：一个（攻击模式，聚合器）-> 最终答案的表格，其中正确答案会被高亮显示。多数投票在单一文化（monoculture）攻击下会失效。CPWBFT 的置信度加权机制能缓解盲从（sycophancy）攻击。当同质化错误智能体数量不足总数一半时，DecentLLMs 的几何中位数聚合会将结果拉向诚实智能体集群。

## 使用

`outputs/skill-consensus-designer.md` 为多智能体集成（multi-agent ensemble）设计了一套共识协议（consensus protocol）：包含聚类方法、权重分配、阈值设定，以及未达阈值轮次的升级策略。

## 部署

在部署任何共识机制之前：

- **使用上述至少三种模式进行攻击测试**。你的协议应当以可预测的方式失败，而非静默失效。
- **记录每一个少数派集群**及其来源谱系。少数派集群是你检测相关误差的早期预警系统。
- **强制限制讨论轮数**。禁止“持续辩论直到达成一致”——这会助长盲从行为。
- **将一致性与正确性分离**。共识输出应交由验证器（verifier）处理；验证器需独立于智能体集成之外。
- **监控一致率**。一致率骤升意味着从众偏差（conformity bias）；骤降则意味着模型漂移（model drift）。

## 练习

1. 运行 `code/main.py`。验证多数投票在单一文化攻击下会失效，但当同质化攻击的置信度低于 0.7 时，CPWBFT 能部分缓解该问题。
2. 添加第四种攻击模式：**静默弃权** —— 一个智能体拒绝回答（“我不知道”）。每个聚合器应如何处理弃权情况？实现你的方案。
3. 将语义聚类从字符串规范化（string canonicalization）替换为嵌入相似度（embedding similarity）（可使用任意开源嵌入模型）。盲从攻击会发生什么变化？
4. 阅读 CP-WBFT 论文（arXiv:2511.10400）。实现置信度探针校准步骤（使用独立的校准模型检查每个智能体自报的置信度）。测量在单一文化场景下的准确率提升。
5. 阅读《Can AI Agents Agree?》（arXiv:2603.01213）。复现一个简化的标量一致性实验：三个智能体，一个标量问题，使用欺骗性角色提示词（deceptive-persona prompt）。CPWBFT 或 DecentLLMs 能否识别出该问题？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 拜占庭容错 (BFT) | “拜占庭容错” | Castro 与 Liskov 于 1999 年提出的共识协议，可容忍 `f < n/3` 的任意故障。 |
| 拜占庭故障 (Byzantine) | “任何不良行为” | 可能撒谎、丢弃消息或静默失败的节点——除安全崩溃（失效停止）外的任何异常行为。 |
| 置信度探测 (Confidence probe) | “你有多大把握？” | 附加在投票上的、由模型自报或校准器预测的概率值。 |
| 语义聚类 (Semantic clustering) | “答案相同，表述不同” | 在统计票数前，将语义等价的答案进行分组。 |
| 几何中位数 (Geometric median) | “鲁棒中心” | 使到所有样本点距离之和最小的点。与均值不同，它对异常值具有鲁棒性。 |
| 模型同质化 (Monoculture) | “同一模型，同一故障” | 当智能体共享训练数据或基础模型时，产生的相关性错误。 |
| 谄媚性从众 (Sycophantic conformity) | “附和声音最大者” | 智能体的投票会偏向于最先发言或声音最大的参与者。 |
| 核心/边缘架构 (Core/Edge) | “分层拜占庭容错” | WBFT 的拆分架构：先由小规模核心（Core）节点达成共识，边缘（Edge）节点随后跟进。以此控制延迟上限。 |

## 延伸阅读

- [Castro & Liskov — Practical Byzantine Fault Tolerance (OSDI 1999)](https://pmg.csail.mit.edu/papers/osdi99.pdf) —— 奠基之作
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) —— 基于置信度的投票加权
- [DecentLLMs — leaderless multi-agent consensus](https://arxiv.org/abs/2507.14928) —— 几何中位数聚合
- [WBFT — Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103) —— 核心/边缘拆分以控制延迟上限
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213) —— 标量共识的脆弱性与欺骗性人格攻击