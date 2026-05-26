# 智能体经济、代币激励与声誉

> 长周期自主智能体 (Long-horizon autonomous agents，参考 METR 的 1 至 8 小时工作曲线) 需要具备经济主体性 (economic agency)。新兴的**五层技术栈 (5-layer stack)** 为：**去中心化物理基础设施网络 (DePIN)**（物理算力）→ **身份层 (Identity)**（W3C 去中心化标识符 (DIDs) + 声誉资本）→ **认知层 (Cognition)**（检索增强生成 (RAG) + 模型上下文协议 (MCP)）→ **结算层 (Settlement)**（账户抽象 (account abstraction)）→ **治理层 (Governance)**（智能体去中心化自治组织 (Agentic DAOs)）。已投入生产环境的智能体激励网络包括 **Bittensor**（TAO 子网奖励特定任务模型）、**Fetch.ai / ASI 联盟**（ASI-1 Mini 大语言模型 (LLM) + FET 代币）以及 **Gonka**（基于 Transformer 的工作量证明 (PoW)，将算力重新分配至具有生产力的 AI 任务）。学术研究方面：AAMAS 2025 的去中心化 LaMAS 采用**沙普利值贡献归因 (Shapley-value credit attribution)** 来公平奖励贡献智能体；Google Research 的《大语言模型机制设计》提出了在单调聚合 (monotone aggregation) 下采用第二价格支付的**代币拍卖 (token auctions)**。本课程将构建一个最小化智能体市场，将沙普利值贡献归因应用于多智能体流水线，并运行第二价格代币拍卖，从而使博弈论机制得以具体落地。

**Type:** 学习
**Languages:** Python (标准库)
**Prerequisites:** 第 16 阶段 · 16（谈判与博弈），第 16 阶段 · 09（并行群体网络）
**Time:** 约 75 分钟

## 问题

当智能体共同创造价值但需要单独获得奖励时，多智能体系统 (Multi-agent systems) 会变得复杂。经典机制——平均分配、最后贡献者独占——要么不公平，要么容易被博弈操纵。基于联盟的沙普利值 (Shapley values) 奖励机制在构造上是公平的，但计算成本高昂。2025-2026 年的文献推动了实用的近似方法：沙普利采样 (Shapley sampling)、单调聚合拍卖 (monotone aggregation auctions)，以及基于已确认贡献累积的链上声誉 (on-chain reputation)。

除了贡献归因 (credit attribution)，该领域已转向真正的经济智能体 (economic agents)：Bittensor TAO 奖励用于微调子网特定模型的挖矿算力，Fetch.ai/ASI 使用 FET 代币奖励 ASI-1 Mini 大语言模型的使用，Gonka 将 Transformer 工作量证明 (proof-of-work) 重新分配至具有生产力的 AI 任务。如今，能够自主交易的智能体已经存在；核心问题在于如何对齐激励 (align incentives)。

本课程将智能体经济 (agent economies) 视为一个特定的问题族——贡献归因、机制设计 (mechanism design) 与声誉——并使用最精简的数学推导来构建每一部分，以确保核心概念能够牢固掌握。

## 概念

### 五层智能体经济栈（Agent-Economy Stack）

1. **去中心化物理基础设施网络（DePIN，物理算力）**。提供 GPU、存储和带宽租赁的去中心化基础设施。包括 Bittensor 子网、Render Network 和 Akash。它并非专为智能体设计，但智能体会调用其资源。
2. **身份（Identity）**。W3C 去中心化标识符（Decentralized Identifiers, DIDs）为每个智能体提供独立于任何平台的持久化 ID。声誉将累积至该 DID。智能体网络协议（Agent Network Protocol, ANP）使用 DID 作为发现层。
3. **认知（Cognition）**。智能体的推理循环：大语言模型（LLM）+ 检索增强生成（RAG）+ 模型上下文协议（MCP）。其他层级均在此基础上构建。
4. **结算（Settlement）**。账户抽象（Account Abstraction, ERC-4337）允许智能体直接从自身余额支付 Gas 费，而无需持有 ETH。智能体可为服务、其他智能体或算力付费。
5. **治理（Governance）**。智能体 DAO（Agentic DAOs）：一种治理结构，人类*与*智能体均可对协议变更进行投票，且投票权与声誉挂钩。

并非所有生产系统都会完整采用这五层。Bittensor 使用了第 1、2 层，部分使用第 3、4 层，未使用第 5 层。OpenAI 智能体仅使用了第 3 层。该栈仅作为参考图谱，而非强制要求。

### Bittensor、Fetch.ai、Gonka —— 实际运行案例

**Bittensor (TAO)**。子网（Subnets）专注于特定任务（如语言建模、图像生成、预测）。矿工提交模型输出，验证者对其进行排名；基于质押权重的评分机制分配 TAO 奖励。每个子网拥有独立的评估体系。其经济学启示在于：为特定任务的输出质量付费，而非为消耗的算力付费。

**Fetch.ai / ASI 联盟**。ASI-1 Mini 大语言模型运行于 Fetch.ai 网络之上；用户需支付 FET 代币以获取推理服务。此处“智能体即对等节点（agents-as-peers）”的理念更为突出：Fetch 上的一个智能体可调用另一个智能体执行任务，并以 FET 支付费用。

**Gonka**。Transformer 工作量证明（Proof-of-Work, PoW）：其“工作量”指的是 Transformer 的前向传播（forward passes）。矿工通过运行具有已知正确输出（来自训练数据）的推理任务来赚取收益。这是一种资源生产型 PoW，而非基于哈希的 PoW。

截至 2026 年 4 月，这三者均已达到生产级标准。它们的收益分配机制各不相同：Bittensor 根据子网验证者的评估奖励质量；Fetch 根据付费用户衡量的效用进行奖励；Gonka 则奖励可验证的推理工作。

### 基于沙普利值（Shapley Value）的贡献度分配

三个智能体协作完成一项任务，最终输出得分为 0.8。各自的贡献度是多少？

沙普利值（Shapley Value）：满足四个公理（有效性、对称性、线性性、零贡献者）的唯一贡献分配方案。对于智能体 `i`：

shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))

其中 `S_i_O` 表示在排序 `O` 中位于 `i` 之前的智能体集合。实际操作中：枚举所有排列组合，记录每个智能体在每种排列中的边际贡献，然后取平均值。

当 N=3 时，共有 6 种排列。当 N=10 时，则高达 360 万种——因此在实践中，通常采用采样排序而非完全枚举。

### 用于聚合的第二价格拍卖（Second-Price Auction）

Google Research（《大语言模型的机制设计》）提出使用第二价格代币拍卖来聚合 LLM 输出。设定如下：N 个智能体各自提交一个补全（completion）方案；每个智能体对被选中都有一个私有估值。拍卖方选择估值最高的方案，但支付*第二高*的估值。在单调聚合条件下（价值取决于选择了哪个方案，而非投标数量），该机制是诚实的（truthful）——智能体会按其真实价值出价。

这对 LLM 系统的意义在于：你可以将补全任务外包给多个定价不同的智能体；拍卖机制能选出最优方案并公平支付，且智能体没有虚报价格的动机。

### 声誉资本（Reputation Capital）

与 DID 绑定的声誉分数通过已确认的贡献累积而成。一个简单的更新规则如下：

rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)

其中衰减因子 `alpha` 接近 1。声誉具有以下特点：
- **读取成本低**：便于路由决策（“将困难任务分配给高声誉智能体”）。
- **伪造成本高**：随时间累积且与 DID 绑定。
- **可被削减（Slashed）**：未通过验证的贡献将被扣除。

### AAMAS 2025 去中心化 LaMAS

LaMAS 提案（AAMAS 2025）结合了以下要素：DID 身份、基于沙普利值的贡献度分配以及简易拍卖机制。其核心主张是：将贡献度分配步骤去中心化，可使系统具备可审计性，并免疫单点操纵。

### 经济模型的失效场景

- **价格预言机操纵（Price Oracle Manipulation）**。如果贡献度函数存在漏洞，智能体就会利用它。每种机制都需要经过对抗性测试。
- **女巫攻击（Sybil Attacks）**。单个运营者创建 N 个虚假智能体以夸大自身贡献。DID 能延缓但无法完全阻止此类攻击；提高声誉伪造成本是主要的缓解手段。
- **验证成本**。贡献度分配的公平性完全取决于验证者。如果验证成本低廉（如使用小型 LLM），则容易被操纵；如果成本高昂（如人工评审），则系统难以扩展。
- **监管不确定性（Regulatory Overhang）**。智能体经济与金融监管存在交叉。截至 2026 年，Bittensor、Fetch 和 Gonka 在部分司法管辖区仍处于法律灰色地带。

### 智能体经济的适用场景

- **异构运营者的开放网络**。没有单一团队控制所有智能体。
- **输出可验证**。缺乏验证机制，贡献度分配就只是盲目猜测。
- **长周期工作流**。一次性任务无法从声誉累积中获益。
- **代币化支付在你的司法管辖区具备法律可行性**。

在封闭的企业系统中，经济模型会让位于更简单的分配方式（由管理者分配工作，指标内部化）。经济学文献主要适用于开放网络。

## 构建实现

`code/main.py` 实现了：

- `shapley(value_fn, agents)` — 针对小规模 N 值，通过枚举法进行精确的沙普利值 (Shapley value) 计算。
- `second_price_auction(bids)` — 真实机制 (truthful mechanism)；获胜者支付第二高报价。
- `Reputation` — 绑定去中心化标识符 (DID) 的声誉系统，具备指数衰减 (exponential decay) 与罚没 (slashing) 机制。
- 演示 1：三个智能体 (agent) 协作，通过精确的沙普利值分配贡献度。
- 演示 2：五个智能体竞标任务槽位；第二价格拍卖 (second-price auction) 选出获胜者并确定支付金额。
- 演示 3：将任务分配给具有异构声誉 (heterogeneous reputation) 的智能体，共进行 100 轮；基于声誉加权的路由 (reputation-weighted routing) 优于随机分配。

运行：

python3 code/main.py

预期输出：每个智能体的沙普利值；展示真实报价均衡 (truthful-bid equilibrium) 的拍卖结果；在预热 (warmup) 阶段后，基于声誉加权的路由相比随机分配展现出 10-20% 的质量提升。

## 实际应用

`outputs/skill-economy-designer.md` 设计了一个最小化的智能体经济 (agent economy)：包含身份层选择、贡献度归属机制 (credit attribution mechanism)、支付机制以及声誉规则。

## 发布实践

在 2026 年运行智能体经济：

- **从声誉系统起步，而非代币。** 声誉系统实现成本低且本身具备高价值；代币则会引入法律与经济层面的复杂性。
- **先验证，后奖励。** 绝不在缺乏独立验证步骤的情况下分配贡献度。自我报告的质量评估极易引发女巫博弈 (Sybil game)。
- **采用沙普利值采样，而非精确计算。** 对 100-1000 种排列顺序进行采样；精确枚举法无法扩展至大规模场景。
- **设定衰减上限与声誉下限。** 无限制的衰减会抹杀合法贡献者；衰减过慢则会持续奖励那些声誉虚高但已不活跃的智能体。
- **以对抗性方式审计机制。** 在开放网络前运行红队 (red-team) 测试场景。每种机制背后都有一套博弈论 (game theory) 逻辑；你的目标是主动发现漏洞，而非等待攻击者来利用。

## 练习

1. 运行 `code/main.py`。验证沙普利值之和是否等于总价值（效率公理 (efficiency axiom)）。修改价值函数 (value function)；观察沙普利值的分配是否按预期方向变化？
2. 实现沙普利值*采样*（基于 K 种排列顺序的蒙特卡洛方法 (Monte Carlo)）。K 值如何影响近似精度？在 N=4 时与精确计算结果进行对比。
3. 在拍卖前实现联盟组建 (coalition-forming) 步骤：智能体可合并为团队并以整体身份竞标。会形成哪些联盟？其结果是否比独立竞标更符合帕累托更优 (Pareto-better)？
4. 阅读 Google Research 关于机制设计 (mechanism design) 的文章。找出一个假设条件，若该条件被违反将破坏真实性 (truthfulness)。这种失效模式在大语言模型 (LLM) 场景下会呈现何种形态？
5. 阅读 AAMAS 2025 关于去中心化 LaMAS 的论文。在合成任务上针对 10 个智能体实现其沙普利值计算步骤。精确计算耗时多久？进行 100 次采样后，结果与精确值的接近程度如何？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 去中心化物理基础设施网络（DePIN） | “去中心化物理基础设施” | 通过代币激励的计算/存储/带宽资源。代表项目包括 Bittensor、Akash、Render。 |
| 去中心化标识符（DID） | “去中心化标识符” | W3C 制定的可移植身份规范。智能体（Agent）的声誉绑定至 DID，而非特定平台。 |
| 账户抽象（ERC-4337） | “账户抽象” | 可代付 Gas 费的合约账户，从而实现智能体支付功能。 |
| 沙普利值（Shapley value） | “公平的贡献度归属” | 满足有效性、对称性、线性性和零玩家性质的唯一分配方案。 |
| 第二价格拍卖（Second-price auction） | “维克里拍卖” | 诚实机制（Truthful mechanism）：胜出者支付第二高的出价。兼容单调聚合（Monotone aggregation）。 |
| 声誉资本（Reputation capital） | “累积质量评分” | 基于已确认贡献并绑定至 DID 的评分；随时间推移会衰减。 |
| 智能体 DAO（Agentic DAO） | “智能体与人类共同治理” | 将智能体投票者视为一等公民的去中心化自治组织（DAO），投票权与声誉挂钩。 |
| TAO / FET / GPU 积分 | “代币计价单位” | Bittensor 的 TAO、Fetch.ai 的 FET 以及各类 DePIN 代币。 |

## 延伸阅读

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — 2026 年关于五层智能体经济（Agent Economy）技术栈的综述
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — 结合单调聚合的代币拍卖机制
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — 基于沙普利值的贡献度归属
- [Bittensor TAO documentation](https://docs.bittensor.com/) — 子网（Subnet）架构与奖励分配机制
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini 大语言模型（LLM）与 FET 代币
- [W3C Decentralized Identifiers (DIDs) spec](https://www.w3.org/TR/did-core/) — 身份基础设施规范