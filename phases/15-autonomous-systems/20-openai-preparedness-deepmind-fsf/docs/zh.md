# OpenAI 准备度框架（Preparedness Framework）与 DeepMind 前沿安全框架（Frontier Safety Framework）

> OpenAI 准备度框架 v2（2025年4月）引入了研究类别（Research Categories）——包括长程自主性（Long-range Autonomy）、藏拙策略（Sandbagging）、自主复制与适应（Autonomous Replication and Adaptation）、削弱安全护栏（Undermining Safeguards）——这些类别独立于追踪类别（Tracked Categories）。追踪类别会触发能力报告（Capabilities Reports）以及由安全顾问组（Safety Advisory Group）审查的安全措施报告（Safeguards Reports）。DeepMind 的前沿安全框架 v3（2025年9月发布，2026年4月17日新增追踪能力等级）将自主性整合至机器学习研发（ML R&D）与网络安全（Cyber）领域中（ML R&D 自主性等级 1 = 以具备竞争力的成本，完全自动化 AI 研发流水线，替代人类与 AI 工具的组合）。FSF v3 明确通过自动化监控来应对工具性推理（instrumental-reasoning）滥用所导致的欺骗性对齐（deceptive alignment）问题。需坦诚指出的是：PF v2 中的研究类别（包括长程自主性）并不会自动触发缓解措施（mitigations）；政策措辞仅为“潜在（potential）”。DeepMind 自身也指出，如果工具性推理能力持续增强，自动化监控“长期来看将不再足够”。

**类型：** 学习
**语言：** Python（标准库，三框架决策表差异对比工具）
**前置条件：** 第 15 阶段 · 第 19 阶段（Anthropic 责任扩展策略 RSP）
**耗时：** 约 45 分钟

## 问题背景

第 19 课详细研读了 Anthropic 的扩展策略（Scaling Policy）。本课通过研读 OpenAI 和 DeepMind 的对应文档，补全了这一全景图。这三份文档属于关联产物，旨在回答同一个核心问题——前沿实验室（Frontier Lab）应在何时暂停或限制模型发布（Pause or Gate a Model）——它们在少数关键类别上达成共识，但在具体细节上存在重要分歧。

共识在于：三者均将长程自主性标记为值得追踪的能力类别。三者均承认欺骗性行为（如对齐伪装 alignment faking、藏拙 sandbagging）是一类特定的风险。三者均设有内部审查机构。分歧在于：OpenAI 将类别划分为“追踪类”（强制缓解 Mandatory Mitigation）和“研究类”（无自动触发机制 Automatic Trigger）。DeepMind 则将自主性融入两个特定领域，而非单独命名。各实验室对类别的命名各不相同（如追踪类 vs 研究类、关键类 vs 中等类、一级 vs 二级）；能力被归入不同类别后，所产生的实际操作后果在各实验室之间也存在差异。

将这三份文档对照阅读是一项极具价值的练习。同一项能力在 Anthropic 可能属于“强制缓解”，在 OpenAI 可能是“仅监控不触发”，而在 DeepMind 则可能是“在特定领域内追踪”。政策立场（Policy Posture）的差异至关重要。

## 核心概念

### OpenAI 准备就绪框架 v2（2025年4月）

结构：

- **已追踪类别（Tracked Categories）**：触发能力报告（Capabilities Reports，说明模型具备的能力）与安全保障报告（Safeguards Reports，说明已落实的缓解措施）。在部署前需经安全顾问组（Safety Advisory Group）审查。
- **研究类别（Research Categories）**：实验室正在追踪但尚未承诺采取具体缓解措施的模型能力。包括长期自主性（Long-range Autonomy）、策略性低估（Sandbagging）、自主复制与适应（Autonomous Replication and Adaptation）、破坏安全保障（Undermining Safeguards）。

v2 版本的核心框架设定：研究类别不会自动触发缓解措施。政策措辞为“潜在”缓解措施。这是需要重点关注的差异点：一个类别被归入“已追踪”还是“研究”，决定了它是处于“受控（gated）”状态还是“观察（observed）”状态。

### DeepMind 前沿安全框架 v3（2025年9月；2026年4月17日新增已追踪能力等级）

结构：

- **关键能力等级（Critical Capability Levels, CCLs）**：跨越五个领域的能力阈值：网络（Cyber）、生物（Bio）、机器学习研发（ML R&D）、化学/生物/放射/核（CBRN）、自主性（Autonomy，已并入机器学习研发与网络领域）。
- **已追踪能力等级（Tracked Capability Levels）**：2026年4月新增的细化指标。具体示例：机器学习研发自主性等级 1 = 以具备竞争力的成本，完全自动化人工智能研发流水线（相较于“人类 + AI 工具”的组合）。
- **欺骗性对齐监控（Deceptive alignment monitoring）**：明确承诺针对工具性推理（instrumental-reasoning）滥用行为实施自动化监控。

其对自主性的界定方式与 OpenAI 不同。DeepMind 未将“自主性”保留为顶级领域，而是将其并入自主性可能造成危害的具体领域（机器学习研发与网络）。其核心论点是：脱离具体领域的自主性等同于无风险的能力；而反对观点则认为：跨领域的自主性是一种元风险（meta-risk），框架理应明确命名。

### 三大框架的共识

- 内部安全顾问组（Anthropic 称为 SAG，OpenAI 称为 SAG，DeepMind 为内部委员会）。针对高能力模型在部署前进行审查。
- 明确将欺骗性对齐（deceptive alignment）/ 对齐伪装（alignment faking）列为风险类别。
- 按既定周期发布固定文档（Anthropic：前沿安全路线图、风险报告；OpenAI：能力与安全保障报告；DeepMind：前沿安全框架更新周期）。
- 承认仅依赖监控的防御手段存在上限。DeepMind 明确指出：“自动化监控在长期内将不再足够。”

### 三大框架的分歧

- **Anthropic**：v3.0 版本中移除了暂停承诺；明确将 AI 研发-4（AI R&D-4）阈值设为下一道关卡。
- **OpenAI**：采用已追踪与研究类别的划分；研究类别（含长期自主性）不会自动触发关卡限制。
- **DeepMind**：将自主性并入其他领域；2026年4月通过已追踪能力等级增加细化维度。

### 策略性低估（Sandbagging）：使三大框架均复杂化的特定能力

策略性低估（Sandbagging，指模型在评估中故意表现低于实际水平）被归入 OpenAI 的研究类别。Anthropic 的负责任扩展政策（Responsible Scaling Policy, RSP）v3.0 通过评估上下文差距（Lesson 1）来应对该问题。DeepMind 则在其前沿安全框架（Frontier Safety Framework, FSF）v3 中通过欺骗性对齐监控来应对。

若模型在评估中策略性低估，所有框架的能力阈值都会被低估。框架的有效性完全取决于测量的准确性。正因如此，除了实验室自我评估外，外部测量（第21课，METR）与对抗性评估（adversarial evaluation）必不可少。

### 政策解读技巧

- **定位（Locate）**：你所关注的每一项能力都应在政策文件中找到对应条目。若找不到，则说明该政策未涵盖此项。
- **分类（Classify）**：该能力属于已追踪（触发缓解措施）还是研究（仅追踪但不触发）？OpenAI 对此有明确命名；Anthropic 与 DeepMind 也有各自的等效分类。
- **周期（Cadence）**：政策是按既定时间表更新，还是仅在特定事件发生后更新？既定周期更新机制更为可靠。
- **独立性（Independence）**：外部审查是强制要求还是可选项目？Anthropic 与 Apollo 及美国人工智能安全研究所（US AI Safety Institute）合作；OpenAI 与 METR 合作；DeepMind 则主要依赖内部安全顾问组。

## 使用方法

`code/main.py` 实现了一个小型决策表差异（diff）工具。给定某项能力（如自主性（autonomy）、欺骗性对齐（deceptive alignment）、研发自动化（R&D automation）、网络能力提升（cyber uplift）等），它会输出三项政策分别如何对该能力进行分类，以及会触发哪些缓解措施（mitigations）。该工具仅作为阅读辅助，并非政策制定工具。

## 交付输出

`outputs/skill-cross-policy-diff.md` 会生成针对特定能力的跨政策对比报告，并以三项框架作为参考基准。

## 练习

1. 运行 `code/main.py`。确认差异（diff）工具的输出结果与政策分类一致，至少针对两项你能够对照源文档进行验证的能力。

2. 完整阅读 OpenAI 准备度框架（Preparedness Framework）v2。识别其中的每一项研究类别（Research Category）。针对每一项，用一句话说明为何将其归入研究类别而非追踪类别（Tracked Category）。

3. 完整阅读 DeepMind 前沿安全框架（Frontier Safety Framework, FSF）v3，以及 2026 年 4 月的追踪能力等级（Tracked Capability Levels）更新。找出机器学习研发自主性等级 1（ML R&D autonomy level 1）的具体评估标准。你将如何从外部对其进行测量？

4. 策略性低效表现（Sandbagging）被归入 OpenAI 的研究类别中。设计一项评估，迫使存在策略性低效表现的模型暴露出其真实能力。请参考第 1 课中关于评估上下文博弈（eval-context-gaming）的讨论。

5. 针对某项特定能力（自行选择）对比这三项政策。指出你认为哪项政策的分类最为严谨，哪项最不严谨。请引用源文本作为依据。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| 准备度框架（Preparedness Framework） | “OpenAI 的规模扩展政策” | PF v2（2025 年 4 月）；追踪类别（Tracked）与研究类别（Research）的划分 |
| 追踪类别（Tracked Category） | “强制缓解措施” | 触发能力与安全保障报告；SAG 审查 |
| 研究类别（Research Category） | “仅监控” | 纳入追踪但无自动缓解措施；包含长期自主性（Long-range Autonomy） |
| 前沿安全框架（Frontier Safety Framework） | “DeepMind 的规模扩展政策” | FSF v3（2025 年 9 月）+ 追踪能力等级（Tracked Capability Levels，2026 年 4 月） |
| 关键能力等级（CCL） | “关键能力阈值” | DeepMind 按领域设定的阈值（网络、生物、机器学习研发、化生放核（CBRN）） |
| 机器学习研发自主性等级 1（ML R&D autonomy level 1） | “研发自动化” | 以具有竞争力的成本完全自动化 AI 研发流程 |
| 策略性低效表现（Sandbagging） | “战略性表现不佳” | 模型在评估中故意表现低于实际水平；归入 OpenAI 研究类别 |
| 工具性推理（Instrumental reasoning） | “手段-目的推理” | 关于如何实现目标的推理；DeepMind 的监控目标 |

## 延伸阅读

- [OpenAI — 更新我们的准备框架（Preparedness Framework）](https://openai.com/index/updating-our-preparedness-framework/) — v2 版本公告。
- [OpenAI — 准备框架 v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 完整文档。
- [DeepMind — 强化我们的前沿安全框架（Frontier Safety Framework）](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 版本公告。
- [DeepMind — 更新前沿安全框架（2026 年 4 月）](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — 新增追踪能力等级（Tracked Capability Levels）。
- [Gemini 3 Pro FSF 报告](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF 格式风险报告（Risk Report）示例。