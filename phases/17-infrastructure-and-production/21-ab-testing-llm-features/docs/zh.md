# 大语言模型（LLM）功能的 A/B 测试（A/B Testing）—— GrowthBook、Statsig 与“凭感觉评估”（Vibes）问题

> 传统的 A/B 测试并非为非确定性（Non-deterministic）大语言模型而设计。关键区别在于：离线评估（Evals）回答的是“模型能否完成任务？”，而 A/B 测试回答的是“用户是否在意？”。两者缺一不可；仅凭直觉检查（Vibe Checks）就上线的时代已经结束。2026 年需要测试的维度包括：提示词工程（Prompt Engineering，即措辞优化）、模型选择（如 GPT-4 对比 GPT-3.5 对比开源软件 OSS；在准确率、成本与延迟之间权衡）、生成参数（如温度 Temperature、Top-p 采样）。真实案例：某聊天机器人的奖励模型（Reward Model）变体使对话长度提升了 70%，留存率提升了 30%；Nextdoor AI 的邮件主题行实验在优化奖励函数（Reward Function）后，点击通过率（CTR）提升了 1%；可汗学院（Khan Academy）的 Khanmigo 在延迟与数学准确性的权衡轴上进行了迭代。平台选择：**Statsig**（于 2025 年 9 月被 OpenAI 以 11 亿美元收购）—— 支持序贯检验（Sequential Testing）、CUPED（利用预实验数据控制方差）、一体化平台。**GrowthBook** —— 开源、数据仓库原生（Warehouse-Native），内置贝叶斯（Bayesian）+ 频率学派（Frequentist）+ 序贯引擎，支持 CUPED、样本比率不匹配（SRM）检查，以及本杰明尼-霍赫伯格（Benjamini-Hochberg）与邦费罗尼（Bonferroni）多重比较校正。你的选择取决于对数据仓库 SQL（Warehouse-SQL）的偏好，以及“被 OpenAI 收购”这一事实是否会影响贵组织的决策。

**Type:** 学习
**Languages:** Python（标准库，简易序贯检验模拟器）
**Prerequisites:** 第 17 阶段 · 13（可观测性），第 17 阶段 · 20（渐进式部署）
**Time:** 约 60 分钟

## 学习目标

- 区分离线评估（“模型能否完成任务”）与 A/B 测试（“用户是否在意”）。
- 列举三个可测试维度（提示词、模型、参数），并为每个维度选择合适的指标。
- 解释 CUPED、序贯检验以及本杰明尼-霍赫伯格多重比较校正。
- 根据对数据仓库 SQL 的偏好以及企业对收购事件的态度，在 Statsig 与 GrowthBook 之间做出选择。

## 核心问题

你手动调整了系统提示词（System Prompt），感觉效果变好了，于是直接上线。结果转化率的变化只是随机噪声，你却开始怪罪指标。或者你上线了一个新模型，转化率毫无波动——是模型性能下降了，还是变化幅度太小以至于无法检测？你无从得知，因为你上线时根本没有做 A/B 测试。

离线评估只能回答模型在标注数据集上能否完成特定任务，却无法回答用户是否更喜欢该输出。只有受控的在线实验才能给出答案，且前提是该实验具备足够的统计功效（Statistical Power）、能够控制非确定性，并进行了多重比较校正。

## 核心概念

### 评估（Evals）与 A/B 测试（A/B tests）

**评估（Evals）**——离线进行，基于带标签的数据集，由评判器（评分标准、大语言模型作为评判器（LLM-as-judge）或人工）执行。核心问题：“在固定的数据分布下，输出是否正确、有帮助且安全？”

**A/B 测试（A/B tests）**——在线进行，面向真实用户，采用随机分组。核心问题：“新变体是否推动了关键的用户级指标？”

两者缺一不可。评估（Evals）用于在功能上线前捕捉性能退化（regressions）；A/B 测试则用于在上线后验证实际的产品影响。

### 测试内容

1. **提示词工程（Prompt engineering）**——措辞、系统提示词（system-prompt）结构、示例。指标：任务成功率、用户留存率、单次请求成本。
2. **模型选择（Model selection）**——GPT-4 对比 GPT-3.5-Turbo 对比 Llama-OSS。指标：任务准确率 + 单次请求成本 + P99 延迟（P99 latency）。多目标优化。
3. **生成参数（Generation parameters）**——温度（temperature）、top-p、max_tokens。指标：任务相关（输出多样性与确定性之间的权衡）。

### CUPED —— 方差缩减（Variance Reduction）

基于实验前数据的受控实验（Controlled-experiments Using Pre-Experiment Data）。在比较实验后数据前，先通过回归消除实验前期的方差。典型的方差缩减（variance reduction）幅度为 30%-70%。有效样本量（effective sample size）得以免费提升。

实现情况：Statsig 和 GrowthBook 均已内置支持。

### 序贯检验（Sequential Testing）

经典 A/B 测试假设样本量固定。序贯检验（sequential testing，“边看边决策”）能够在多次查看数据时控制假阳性率（false-positive rate）。始终有效的序贯方法（如 mSPRT（mSPRT）、Howard 置信序列（Howard's confidence sequences））允许你在结果显著领先时提前终止实验。

### 多重比较校正（Multiple-comparison Corrections）

在 95% 置信水平下运行 20 个 A/B 测试，仅凭概率就会产生一个假阳性结果。Bonferroni 校正（Bonferroni correction）会收紧每次测试的显著性水平（α）；Benjamini-Hochberg 方法（Benjamini-Hochberg procedure）则用于控制错误发现率（false-discovery rate）。GrowthBook 同时实现了这两种方法。

### SRM —— 样本比例不匹配（Sample Ratio Mismatch）

分配哈希（assignment hash）负责将用户随机分配到不同变体。如果预设的 50/50 分流实际变成了 47/53，说明系统存在异常——SRM 检查会对此进行标记。两个平台均实现了该检查。

### Statsig 对比 GrowthBook

**Statsig**：
- 于 2025 年 9 月被 OpenAI 以 11 亿美元收购。托管式 SaaS 服务。
- 支持序贯检验、CUPED 以及预留人群（held-out populations）。
- 一体化平台：功能开关（feature flags）+ 实验平台 + 可观测性（observability）。
- 最佳适用场景：团队希望使用开箱即用的捆绑产品，且不介意 OpenAI 的所有权背景。

**GrowthBook**：
- 开源（MIT 协议）；原生数仓架构（warehouse-native，直接读取 Snowflake/BigQuery/Redshift 数据）。
- 支持多种统计引擎：贝叶斯（Bayesian）、频率学派（Frequentist）、序贯检验（Sequential）。
- 内置 CUPED、SRM、Bonferroni 校正及 BH 校正。
- 支持自托管或托管云服务。
- 最佳适用场景：重度依赖数仓 SQL 的团队，数据团队掌控指标层，且偏好开源软件（OSS）。

### 非确定性（Non-determinism）使统计功效计算复杂化

相同的提示词会产生不同的输出。传统的统计功效（statistical power）计算假设观测值独立同分布（independent and identically distributed, IID）。由于大语言模型的非确定性，有效样本量会低于名义样本量。建议将所需样本量乘以 1.3~1.5 倍作为安全余量。

### 实际案例结果

- 聊天机器人奖励模型（reward model）变体：对话长度提升 70%，留存率提升 30%。
- Nextdoor 邮件主题行：优化奖励函数（reward-function）后，点击率（click-through rate, CTR）提升 1%。
- Khan Academy 的 Khanmigo：在延迟与数学准确率之间进行迭代权衡。

### 反模式：凭直觉发布（Shipping on vibes）

每位高级工程师都能说出几个因为“感觉更好”就未经 A/B 测试直接上线的功能。其中大多数都导致了产品指标下滑，而团队往往数月后才察觉。A/B 测试正是强制验证机制（forcing function）。

### 需要记住的关键数据

- Statsig 被 OpenAI 收购：11 亿美元，2025 年 9 月。
- GrowthBook：MIT 开源协议；支持贝叶斯 + 频率学派 + 序贯检验。
- CUPED 方差缩减幅度：30%~70%。
- 大语言模型非确定性 → 需增加 30%~50% 的样本量缓冲。

## 实践应用

`code/main.py` 模拟了包含固定边界（Fixed Boundaries）与序贯边界（Sequential Boundaries）的序贯 A/B 测试（Sequential A/B Test）。该脚本展示了序贯方法如何支持提前终止实验。

## 交付上线

本节将生成 `outputs/skill-ab-plan.md`。根据功能变更、工作负载与基线指标，自动推荐实验平台、特性门控（Feature Gates）及所需样本量。

## 练习题

1. 运行 `code/main.py`。若预期提升幅度（Lift）为 5%，基线转化率（Baseline Conversion）为 3%，要达到 80% 的统计功效（Statistical Power），需要多大的样本量？
2. 为一家受医疗法规监管且采用本地部署（On-Prem）的客户，在 Statsig 与 GrowthBook 之间做出选择。
3. 设计一项 A/B 测试，对比 GPT-4 与 GPT-3.5 在“单票解决成本”（Cost-per-resolved-ticket）上的表现。请明确主要指标（Primary Metric）、护栏指标（Guardrail Metric）与次要指标（Secondary Metric）。
4. 你的金丝雀发布（Canary Release）已通过，但 A/B 测试显示转化率下降了 1.2%。你会选择上线吗？请撰写升级上报标准（Escalation Criteria）。
5. 将 CUPED（控制变量法）应用于实验前周期（Pre-period）数据，其方差为实验后周期（Post-period）的 60%。计算有效样本量（Effective Sample Size）的提升幅度。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 评估 (Eval) | “离线测试” | 基于标注数据集的模型能力评估 |
| A/B 测试 (A/B Test) | “实验” | 面向真实用户的在线随机对照比较 |
| CUPED | “方差缩减” | 利用实验前周期数据进行回归分析以降低方差 |
| 序贯测试 (Sequential Test) | “允许随时查看的测试” | 始终有效的统计流程，支持提前终止实验 |
| 多重比较 (Multiple Comparison) | “族系误差” | 同时运行多项测试会放大假阳性率 |
| 邦费罗尼校正 (Bonferroni) | “严格校正” | 将显著性水平 α 除以测试次数 |
| 本杰明尼-霍赫伯格方法 (Benjamini-Hochberg) | “BH FDR” | 控制错误发现率（False Discovery Rate），保守性较低 |
| 样本比例不匹配 (SRM) | “分组异常” | 样本比例失衡；通常由流量分配缺陷引起 |
| Statsig | “OpenAI 旗下” | 商业级一体化平台，于 2025 年被收购 |
| GrowthBook | “开源那款” | 基于 MIT 协议、数仓原生（Warehouse-native）平台 |
| mSPRT（混合序贯概率比检验） | “序贯概率比检验” | 经典的序贯统计流程 |

## 延伸阅读

- [GrowthBook — 如何对 AI 进行 A/B 测试](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — 超越提示词：数据驱动的大语言模型优化](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig 与 GrowthBook 对比](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng 等人 — CUPED 方法](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — 置信序列](https://arxiv.org/abs/1810.08240)