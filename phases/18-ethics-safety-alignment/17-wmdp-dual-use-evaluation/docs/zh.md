# WMDP 与双重用途能力评估 (Dual-Use Capability Evaluation)

> Li 等人，《WMDP 基准测试：通过遗忘学习 (unlearning) 衡量并减少恶意使用》（ICML 2024，arXiv:2403.03218）。涵盖生物安全（1,520 题）、网络安全（2,225 题）和化学（412 题）的 4,157 道多项选择题。题目设定在“黄区”（yellow zone）——即直接促成性知识（proximate enabling knowledge），并经过多专家评审及 ITAR/EAR 法律合规性过滤。双重用途：作为双重用途能力（dual-use capability）的代理评估（proxy evaluation），以及遗忘学习基准测试（配套的 RMU 方法在降低 WMDP 表现的同时保留了通用能力）。2024-2025 年领域叙事：OpenAI/Anthropic 2024 年的早期评估报告称，相较于互联网搜索仅有“轻微提升”（mild uplift）；到 2025 年 4 月，OpenAI 的《准备框架 v2》（Preparedness Framework v2）指出模型已“处于能够切实帮助新手制造已知生物威胁的边缘”。Anthropic 的生物武器获取试验显示提升了 2.53 倍，但尚不足以排除 ASL-3 风险。

**类型：** 学习
**语言：** Python（标准库，适配 WMDP 的提升评估工具链 (uplift evaluation harness)）
**前置条件：** 第 18 阶段 · 第 16 阶段（红队工具 (red-team tooling)），第 14 阶段（智能体工程 (agent engineering)）
**时间：** 约 60 分钟

## 学习目标

- 描述 WMDP 的三个领域、题目数量及“黄区”过滤标准。
- 解释 RMU 方法，并说明为何 WMDP 同时作为评估基准与遗忘学习基准。
- 阐述 2024-2025 年的能力提升（uplift）叙事：“轻微提升” → “处于切实帮助的边缘” → “不足以排除 ASL-3”。
- 区分相对于新手的提升幅度与专家级的绝对能力。

## 问题定义

双重用途能力是各实验室前沿安全框架（第 18 课）下的核心测量难题。核心问题在于：模型 X 是否实质性地提升了新手在生物、化学或网络领域造成大规模伤害的能力？直接测量（要求模型实际生成有害内容）既违法也不符合伦理。代理测量（proxy measurement）需要一个模型无法拒绝的基准测试（以获取真实的能力数据），但其题目本身不能构成有害出版物。

## 核心概念

### “黄区”（yellow zone）

此类问题要求掌握有害过程的邻近性、赋能性知识，但并非直接的合成配方。例如“什么试剂能催化[已公开路径]的第4步？”，而非“我如何制造[危险化合物]？”。每道问题均经过多位领域专家审核，并经过筛选以符合《国际武器贸易条例》（ITAR）与《出口管理条例》（EAR）的出口管制合规要求。

共计 4,157 道问题：
- 生物安全（Biosecurity）：1,520
- 网络安全（Cybersecurity）：2,225
- 化学（Chemistry）：412

采用多项选择题格式。模型在未被要求提供任何协助的情况下作答；无需诱导有害行为即可测量其能力。

### 表示误导遗忘法（Representation Misdirection for Unlearning, RMU）

配套的机器遗忘（Unlearning）方法。应用于 LLaMa-2-7B 模型后，将大规模杀伤性武器代理基准（WMDP）得分降至接近随机水平，同时将大规模多任务语言理解基准（MMLU）及其他通用能力基准的分数波动控制在几个百分点以内。该已发表的方法已成为后续所有生物、化学与网络安全领域机器遗忘论文的基线。

### 2024-2025 年能力提升（uplift）叙事

分为三个阶段：

1. **2024年“温和提升”（mild uplift）。** OpenAI 和 Anthropic 早期的准备度/负责任扩展政策（RSP）评估报告显示，对于尝试生物相关任务的新手而言，模型相比互联网搜索仅有微弱优势。公开表述为：前沿模型确实能提供帮助，但并未比 Google 搜索带来实质性提升。

2. **2025年4月“临界点”（on the cusp）。** OpenAI 的《准备度框架 v2》（Preparedness Framework v2）指出，模型已“处于能够实质性帮助新手制造已知生物威胁的临界点”。这并非能力声明，而是警告该临界点已近在咫尺。

3. **Anthropic 2025年生物武器获取试验。** 针对新手参与者的对照研究，测量了其在获取阶段任务的相对成功率。报告显示提升倍数为 2.53 倍。该结果不足以排除自动化安全等级-3（ASL-3，见第18课）风险——Anthropic 负责任扩展政策（Responsible Scaling Policy）第 3 级的阈值已被达到或逼近。

### 新手相对提升 vs 专家绝对能力

一个关键区分：

- **新手相对提升（Novice-relative uplift）。** 模型能为非专家提供多大帮助？呈乘数效应。由于新手所知甚少，相对优势显得很高；即便是有限的信息也能带来帮助。
- **专家绝对能力（Expert-absolute capability）。** 模型在极限努力下能输出多少信息？专家能从中提取的信息远超新手。其绝对上限很高。

安全论证（第18课）同时针对这两点：“模型无法为新手提供足以执行任务的提升幅度”，且“专家无法从模型中提取尚未公开的信息”。

### 测量陷阱

WMDP 仅是能力代理指标（capability proxy），而非实际部署测量。在 WMDP 上得分高的模型，在实践中是否容易被新手利用，取决于以下因素：
- 诱导抗性（Elicitation resistance）：在不触发安全过滤器的情况下提取该能力的难度
- 隐性知识（Tacit knowledge）：需要湿实验（wet-lab）操作技能而非单纯信息的能力
- 执行壁垒（Execution barriers）：采购渠道、设备条件等

Anthropic 2025 年的生物武器获取试验在 WMDP 式能力评估之上增加了新手诱导层：它测量的是实际任务成功率，而非多项选择题能力。

### 在第 18 阶段中的定位

第 12-16 课聚焦于针对模型输出的攻防工具链。第 17 课为双重用途能力层（dual-use capability layer）——即前沿安全框架（第 18 课）所评估的测量维度。第 30 课则收束全篇，结合当前 2026 年网络/生物/化学/核领域的能力提升证据完成闭环。

## 使用它

`code/main.py` 构建了一个简易的 WMDP 风格评估框架 (WMDP-shaped evaluation harness)。使用模拟模型在按类别划分的问题上进行测试，并报告各领域的得分。一种简单的机器遗忘 (Machine Unlearning) 干预措施（将特定领域的表征 (Representation) 置零）会降低得分；你可以借此衡量其与通用能力 (General Capability) 之间的权衡。

## 交付它

本课时将生成 `outputs/skill-wmdp-eval.md`。针对一项关于双重用途能力 (Dual-use Capability) 的声明（“我们的模型对生物武器 (Bioweapons) 没有实质性帮助”），它会进行审查：运行了哪些基准测试、评估时采用了哪种拒绝路径 (Refusal Path)（原始补全 (Raw Completion) 还是策略门控 (Policy-gated)），以及是否有新手诱导研究 (Novice-elicitation Studies) 来补充多项选择题的结果。

## 练习

1. 运行 `code/main.py`。报告在简易机器遗忘步骤前后各领域的准确率。解释其与通用能力之间的权衡。

2. 为简易 WMDP 增加第四个领域（例如放射性领域）。在黄区 (Yellow Zone) 中指定两种示例性问题类型。解释为何设计此类问题比添加 MMLU 风格 (MMLU-shaped) 的问题更困难。

3. 阅读 WMDP 2024 第 5 节（RMU 方法论）。构思一种更简单的机器遗忘方法（例如，抑制处理领域内容的前 k 个神经元 (Top-k Neurons)），并描述其预期的通用能力损耗。

4. Anthropic 2025 的生物武器获取试验报告了 2.53 倍的性能提升 (Uplift)。描述可能导致该数值被高估的两个因素（新手样本量 (Novice Sample Size)、任务保真度 (Task Fidelity)）以及可能导致被低估的两个因素（诱导上限 (Elicitation Ceiling)、模型安全门控 (Model Safety Gating)）。

5. 阐明针对 ASL-3 的安全论证 (Safety Case) 除了通过 WMDP 机器遗忘测试外，还需要什么。至少列出两项互补的诱导研究 (Elicitation Studies)。

## 关键术语

| 术语 | 通常的说法 | 实际含义 |
|------|-----------------|------------------------|
| WMDP | “双重用途基准测试” | 涵盖生物/网络/化学领域的 4,157 道黄区多项选择题 |
| 黄区 (Yellow Zone) | “具备赋能作用但非合成步骤” | 紧邻有害能力的邻近知识，但不包含具体的合成配方 |
| RMU | “机器遗忘基线” | 表征误导遗忘 (Representation Misdirection for Unlearning)；降低 WMDP 得分，同时保留通用能力 |
| 新手相对提升 (Novice-relative Uplift) | “对非专家的帮助程度” | 相较于现状下的互联网搜索，为新手带来的倍数级优势 |
| 专家绝对能力 (Expert-absolute Capability) | “专家的能力上限” | 有动机的专家能从模型中提取的最大信息量 |
| 获取阶段任务 (Acquisition-phase Task) | “合成前的步骤” | 采购、设备、许可——危害路径中最早期的环节 |
| ITAR/EAR | “出口管制合规” | 限制发布特定赋能知识的法律框架 |

## 进一步阅读

- [Li 等人 — WMDP 基准测试（Weapons of Mass Destruction Proxy Benchmark）（arXiv:2403.03218，ICML 2024）](https://arxiv.org/abs/2403.03218) — 该基准测试与 RMU（Representation Model Unlearning）论文
- [OpenAI — 准备就绪框架（Preparedness Framework）v2（2025 年 4 月 15 日）](https://openai.com/index/updating-our-preparedness-framework/) — “处于临界点”（on the cusp）相关措辞
- [Anthropic — 负责任扩展政策（Responsible Scaling Policy）v3.0（2026 年 2 月）](https://www.anthropic.com/responsible-scaling-policy) — ASL-3（AI Safety Level 3）生物危害阈值（bio threshold）与获取试验（acquisition trial）结果
- [DeepMind — 前沿安全框架（Frontier Safety Framework）v3.0（2025 年 9 月）](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — 生物能力提升（bio-uplift）CCL（Critical Capability Level）