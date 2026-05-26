# 谈判与议价 (Negotiation and Bargaining)

> 智能体 (Agent) 就资源、价格、任务分配及条款进行协商。2026 年的基准测试已给出明确结论：谈判竞技场 (NegotiationArena, arXiv:2402.05863) 表明，大语言模型 (Large Language Model, LLM) 可通过角色设定操控 (Persona Manipulation，如营造“绝望感”) 使收益提升约 20%；《衡量议价能力》(arXiv:2402.15813) 指出买方角色比卖方更难，且模型规模扩大并无助益——其提出的 **OG-Narrator**（确定性报价生成器 (Deterministic Offer Generator) + LLM 叙述器）将成交率从 26.67% 提升至 88.88%；大规模自主谈判竞赛 (arXiv:2503.06416) 运行了约 18 万次谈判，发现采用**思维链隐藏** (Chain-of-Thought-Concealing) 策略的智能体能通过向对手隐藏推理过程而获胜；Bhattacharya 等人（2025）基于哈佛谈判项目 (Harvard Negotiation Project) 指标的评估显示，Llama-3 最有效，Claude-3 最具攻击性，GPT-4 最公平。本课程将实现合同网协议 (Contract Net Protocol，FIPA 的前身，见第 02 课)，搭建基于 LLM 的买方/卖方智能体，运行类似 OG-Narrator 的分解流程，并测量各项结构选择对成交率的影响。

**类型：** 学习 + 构建
**编程语言：** Python（标准库）
**前置要求：** 第 16 阶段 · 02（FIPA-ACL 协议传承 (FIPA-ACL Heritage)），第 16 阶段 · 09（并行群体网络 (Parallel Swarm Networks)）
**预计耗时：** 约 75 分钟

## 问题

两个智能体需要就价格达成一致。若仅依赖纯语言提示词，2024-2026 年的大语言模型 (LLM) 的成交率低得惊人（在 arXiv:2402.15813 中严格参数化的议价场景下仅约 27%）。扩大模型规模无法解决此问题：GPT-4 在议价结构上并不比 GPT-3.5 更优；它只是更擅长议价的*语言表达*。

根本原因在于 LLM 混淆了两项任务——决定报价与叙述报价。OG-Narrator 将二者分离：由确定性报价生成器计算数值策略，LLM 仅负责叙述。成交率随之跃升至约 89%。

这印证了多智能体系统 (Multi-Agent System) 领域的一项经典结论：将底层机制与通信层解耦方能取胜。合同网协议 (Contract Net Protocol，FIPA, 1996; Smith, 1980) 是任务市场机制的参考标准。将 LLM 接入叙述模块，即可构建出现代化的 LLM 驱动型任务市场。

## 核心概念

### 合同网协议（Contract Net Protocol），一段话概括

史密斯（Smith）于1980年提出的合同网协议（Contract Net Protocol）：**管理者（manager）**广播**征求建议书（call for proposals, cfp）**；**竞标者（bidders）**回复包含其报价的**提议（propose）**消息；管理者选定中标者，并向其发送**接受提议（accept-proposal）**，向落选者发送**拒绝提议（reject-proposal）**。中标者执行任务。可选消息：**拒绝（refuse）**（竞标者拒绝参与提议）。国际智能体通信基金会（FIPA）将其标准化为 `fipa-contract-net` 交互协议。

### 为什么 OG-Narrator 能胜出

《测量语言模型的议价能力》（arXiv:2402.15813）指出：

- 大语言模型（LLM）经常违反议价规则（报出毫无逻辑的价格，无视对方的协议区间（ZOPA））。
- 锚定效应（anchoring）表现不佳（轻易接受糟糕的首次报价；反报价时采用象征性金额而非策略性金额）。
- 仅靠扩大模型规模无法解决这些问题。更大的模型只是用更合理的语言包装了同样严重的策略性错误。

OG-Narrator 的架构分解如下：

           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘

报价生成器采用经典的谈判策略：鲁宾斯坦议价模型（Rubinstein bargaining model）、蔡滕策略（Zeuthen strategy），或简单的价格以牙还牙策略（tit-for-tat）。LLM 仅负责生成叙述文本。最终消息包含确定性的价格以及自然语言包装。

成交率大幅提升的原因在于：
- 价格始终保持在议价区间内。
- 锚定报价基于策略而非情绪。
- LLM 专注于其擅长的领域：文本生成。

### NegotiationArena 的研究发现

arXiv:2402.05863 提供了该领域的标准基准测试。核心发现如下：

- LLM 通过扮演特定角色（例如“我必须在周五前卖掉它，非常着急”）可将收益提升约 20%——角色操纵是一种切实有效的策略。
- 公平/合作的智能体（agent）容易被对抗性智能体剥削；防御需要明确的反制姿态。
- 在约 40% 的基准测试场景中，对称配对会收敛至不公平的结果。

这并非说明“LLM 不擅长谈判”，而是表明“LLM 的谈判方式过于拟人化，连同人类易受剥削的弱点也一并继承了”。

### 思维链（Chain-of-Thought）的隐藏

大规模自主谈判竞赛（arXiv:2503.06416）基于多种 LLM 策略执行了约 18 万次谈判。获胜者成功向对手隐藏了其推理过程：

- 如果智能体将“我最多只出 75 美元；我的保留价格是 70 美元”打印到公开可见的草稿板（scratchpad）中，对手就能直接读取。
- 获胜者在私有环境中计算策略；输出通道仅包含报价和最低限度的必要叙述。

这呼应了经典博弈论（Aumann 1976 关于理性与信息的研究）：暴露你的私有估值会直接导致收益损失。LLM 缺乏这种直觉，会欣然在推理轨迹（reasoning traces）中打出自己的保留价格，而这些轨迹最终对对手可见。

工程实践要点：必须将私有草稿板上下文与公开消息上下文严格隔离。这不是可选项，而是必选项。

### Bhattacharya 等人（2025）——模型排名

基于哈佛谈判项目（Harvard Negotiation Project）的评估指标（原则性谈判、最佳替代方案（BATNA）尊重度、利益互惠性）：

- **Llama-3** 在达成交易方面最有效（成交率与收益最高）。
- **Claude-3** 是最具攻击性的谈判者（高锚定报价、后期才让步）。
- **GPT-4** 最为公平（不同配对间的收益方差最小）。

这只是 2025 年的阶段性快照。重点不在于 2026 年 4 月哪个模型会胜出，而在于不同的基础模型具有持久且稳定的谈判风格。异构集成（Heterogeneous ensembles，见第 15 课）正是将这种差异作为多样性来源加以利用。

### 基于合同网协议 + LLM 的任务分配

合同网协议在现代 LLM 多智能体（multi-agent）系统中的复用方式如下：

1. 管理者智能体（Manager agent）将任务拆解为若干单元。
2. 向工作智能体（worker agents）广播包含任务描述的 `cfp`。
3. 每个工作智能体返回报价：`(price, eta, confidence)`，其中价格可以是 token 数量、计算单元或美元。
4. 管理者选定中标者（根据任务需求可为单个或多个）并授予任务。
5. 未中标的智能体可自由参与其他任务的竞标。

该架构能轻松扩展至百个工作智能体以上，因为其协调机制是“广播-响应”模式，而非同步聊天。已在生产环境中应用：Microsoft Agent Framework 的编排模式，以及部分 LangGraph 实现。

### LLM-利益相关者交互式谈判

NeurIPS 2024（https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf）引入了具有**秘密评分（secret scores）**和**最低接受阈值（minimum-acceptance thresholds）**的多方可计分博弈。每位利益相关者（stakeholder）拥有私有效用函数；LLM 必须从消息中推断这些效用。这是将双方议价推广至 N 方联盟形成（coalition formation）的泛化模型。该研究对具有异构工作能力的生产级任务市场具有重要参考价值。

### 叙述与机制分离原则

纵观 2024-2026 年间的所有谈判基准测试，一条一致的工程准则是：

> 让 LLM 负责叙述。不要让 LLM 计算报价。

如果报价必须是数值（价格、预计完成时间（ETA）、数量），则应根据谈判状态确定性生成该数值，再由 LLM 生成包装文本。如果报价需要是提案结构（任务拆解、角色分配），可让 LLM 起草，但在发送前必须依据模式（schema）进行验证并执行约束检查。

## 构建它

`code/main.py` 实现了：

- `ContractNetManager`、`ContractNetTask`、`Bid` — 管理器与竞标者，广播招标请求（cfp），收集提案，授予合同。
- `og_narrator_bargain(state, rng)` — OG-Narrator 买方：采用确定性的齐森式让步（Zeuthen-style concession）策略，向中间点收敛。
- `seller_response(state, rng)` — 确定性的卖方还价策略（作为两种风格的结构性基准真值（ground truth））。
- `naive_llm_bargain(state, rng)` — 模拟纯大语言模型（LLM）议价者：选择的价格方差较高，且经常超出可能达成协议区间（ZOPA）。
- 测量指标：在 1000 次试验中统计成交率（deal rate），每次试验均重新采样保留价格（reservation prices）。

运行：

python3 code/main.py

预期输出：naive-LLM 成交率约为 65-75%；OG-Narrator 成交率约为 85-95%；这 15-25 个百分点的差距体现了将报价生成（offer-generation）与叙述（narration）解耦所带来的结构性优势。此外还包含一个合同网（Contract Net）任务市场分配示例，涉及三个竞标者和一个任务。

## 使用它

`outputs/skill-bargainer-designer.md` 设计了一套议价协议（bargaining protocol）：明确由谁生成报价（确定性算法或 LLM）、由谁负责叙述、私有草稿区（scratchpads）如何与公共消息隔离，以及如何监控成交率。

## 部署它

生产环境议价检查清单：

- **隔离草稿区。** 私有状态（private state）绝不能泄露给对方的上下文。这是不可妥协的底线。
- **确定性报价生成。** 价格、数量、预计到达时间（ETAs）：通过计算得出，而非依赖提示词（prompt）。
- **根据数据模式（schema）验证所有传入报价。** 在协议边界直接拒绝超出可能达成协议区间（ZOPA）的报价。
- **限制轮次。** 最多 3-5 轮；若陷入僵局，则升级至调解器（mediator）。
- **持续监控成交率与收益方差（payoff variance）。** 成交率下降是一个预警信号——通常源于提示词漂移（prompt drift）或对方侧的攻击。
- **记录所有被拒绝的提案**及其确定性依据。对于合同网管理器而言，落选的竞标者需要清楚了解原因。

## 练习

1. 运行 `code/main.py`。验证 OG-Narrator 在成交率上是否优于 naive-LLM。具体高出多少？
2. 实现**基于人设的收益优化**（arXiv:2402.05863）——买方仅在叙述环节采用“本周急需购买”的人设，报价生成器保持不变。成交率或收益是否会发生变化？
3. 实现思维链（chain-of-thought）**隐藏机制**：维护一个不传递给对方的私有草稿字符串。如果意外泄露会发生什么（可通过交换通道来模拟）？
4. 将合同网扩展为带有保留价（reserve price）的 N 竞标者拍卖。当所有报价均高于保留价时，管理器如何在最低价与最高质量之间进行权衡？你会选择哪种授标规则（award rule），为什么？
5. 阅读 Bhattacharya 等人（2025）关于哈佛谈判项目（Harvard Negotiation Project）指标的论文。实现两种不同风格（激进型 vs 公平型）的议价者。测量在对称与非对称配对下的收益方差。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 合同网协议 (Contract Net) | “任务市场” | Smith 1980, FIPA 1996。包含 cfp + propose + accept/reject。标准的任务市场机制。 |
| 可能达成协议区间 (ZOPA) | “可能达成协议的区域” | 买方最高出价与卖方最低要价之间的重叠部分。超出该区间的报价无法成交。 |
| 谈判协议最佳替代方案 (BATNA) | “谈判协议的最佳替代方案” | 交易失败时的备选方案。它决定了你的保留价格。 |
| 报价生成与叙述模块 (OG-Narrator) | “报价生成器 + 叙述器” | 架构解耦：确定性报价生成与大语言模型（LLM）叙述分离。 |
| 蔡滕策略 (Zeuthen Strategy) | “风险最小化让步” | 经典的报价生成策略，依据风险承受限度进行让步。 |
| 鲁宾斯坦讨价还价模型 (Rubinstein Bargaining) | “交替报价均衡” | 引入贴现因子的无限期讨价还价博弈论模型。 |
| 思维链隐藏 (CoT Concealment) | “隐藏你的推理过程” | arXiv:2503.06416 中的获胜方案保留了私有推理草稿（scratchpads）；公共通信通道仅展示最终报价。 |
| 角色设定操纵 (Persona Manipulation) | “情绪化姿态” | arXiv:2402.05863：通过注入绝望或紧迫感等角色人设，可获得约 20% 的收益提升。 |

## 延伸阅读

- [NegotiationArena](https://arxiv.org/abs/2402.05863) — 基准测试；关于角色设定操纵与利用的研究发现
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) — OG-Narrator 架构及“买方谈判难度高于卖方”的结论
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) — 约 18 万次谈判；思维链隐藏策略获胜
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) — 具有私有效用值的多方可计分博弈
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — 经典机制，发表于 IEEE Transactions on Computers