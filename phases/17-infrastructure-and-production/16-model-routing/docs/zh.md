# 模型路由作为降本原语

> 动态路由调度器（Dynamic Broker）会评估每个请求（任务类型、Token 长度、嵌入向量相似度、置信度），将简单查询分发至低成本模型，并将复杂查询升级至前沿模型。该机制也称为模型级联（Model Cascading）。生产环境案例表明，在欧美地区的部署中，该方案可在保持同等质量（Iso-Quality）的前提下实现 20%~60% 的成本降低；对于高并发 SaaS 服务，路由效率提升 30% 即可转化为每年六位数的成本节约。2026 年的行业背景是，大语言模型（LLM）推理价格每年下降约 10 倍——GPT-4 级别 Token 的单价已从 2022 年底的 20 美元/百万降至 2026 年的约 0.40 美元/百万。这一降幅主要得益于更优的服务栈（Serving Stacks）（第 17 阶段 · 04-09），而非硬件迭代。模型路由（Model Routing）正是你在不牺牲产品质量的前提下，将推理降价转化为利润空间的关键手段。其典型故障模式为低成本模型漂移（Cheap-Model Drift）：路由策略将 40% 的流量导向较弱模型，导致推理任务质量下滑 3%~5%，且长达一个季度未被察觉。因此，必须通过在线质量指标（Online Quality Metrics）（而非仅依赖离线评估集 Offline Eval Sets）来设置路由闸门。

**类型：** 学习
**语言：** Python（标准库，玩具级级联路由模拟器）
**前置知识：** 第 17 阶段 · 01（托管 LLM 平台），第 17 阶段 · 19（AI 网关）
**预计耗时：** 约 60 分钟

## 学习目标

- 解释模型级联（Model Cascading）机制：优先使用低成本模型并进行置信度检查，在置信度较低时升级至更强模型。
- 列举四种路由信号（任务分类、提示词长度、与已知困难样本集的嵌入向量相似度、首轮推理的自评估置信度）。
- 在目标路由分流比例（Routing Split）与可接受的质量损耗容忍度（Quality Loss Tolerance）下，计算预期的混合成本（Blended Cost）。
- 指出用于捕捉低成本模型性能缓慢衰退（Cheap-Model Creep）的漂移监控指标（Drift-Monitoring Metric），即在线质量闸门（Online Quality Gate）。

## 问题背景

你的服务每月在 GPT-5 上的花费高达 8 万美元。数据分析显示，70% 的查询非常简单，例如“巴黎现在几点？”或“重写这句话”。一款 Haiku 级别模型（Haiku-Class Model）仅需 3% 的成本即可完美处理这些请求。剩余 30% 的查询则需要 GPT-5 的推理能力——涉及代码编写、数学计算或多步规划。

如果将 70% 的流量路由至低成本模型，30% 路由至高成本模型，你的账单将在保持同等产品质量的前提下下降约 65%。这就是模型路由的核心价值。其中的难点在于，如何构建路由调度器而不导致质量回退。

## 核心概念

### 四种路由信号 (Routing Signals)

1. **任务分类 (Task Classification)**：简单/复杂/代码生成/数学/对话。可采用基于规则的分类器、小型大语言模型（LLM，如 Haiku 级别，定价约 0.25 美元/百万 Token），或计算与已标注类别的嵌入向量（Embedding）相似度。输出：路由目标 = 廉价模型 / 均衡模型 / 前沿模型。

2. **提示词长度 (Prompt Length)**：超过 4K Token 的提示词通常需要前沿模型以保证连贯性。低于 500 Token 的提示词通常不需要。

3. **与已知困难集的嵌入相似度 (Embedding Similarity to Known-Hard Set)**：如果查询请求与已知困难类别的余弦相似度（Cosine Similarity）大于 0.88，则直接升级路由至前沿模型。

4. **首轮推理的自置信度 (Self-Confidence from First-Pass)**：先发送至廉价模型；如果模型的对数概率（Log-Probabilities）显示置信度低，或模型拒绝回答，或输出模棱两可的表述，则在前沿模型上重试。这会增加约 10% 流量的 P95 延迟（P95 Latency），但能为其余 90% 的流量节省 50% 以上的成本。

### 三种路由模式 (Routing Patterns)

**预路由 (Pre-Route)**（前置分类器）：增加约 5-10 毫秒延迟；整体速度最快。

**级联路由 (Cascade)**（优先使用廉价模型，低置信度时升级）：中位延迟约为原来的 1.2 倍（廉价模型运行加验证），升级请求的延迟约为 2 倍。提供最佳的质量底线。

**集成路由 (Ensemble Route)**（对抽样请求并行运行廉价模型与前沿模型，由奖励模型进行优选）：质量最高，成本也最高；仅用于关键的 A/B 测试。

### 实现方案 (Implementation)

AI 网关（AI Gateways，参见 Phase 17 · 19）提供路由功能。LiteLLM 提供包含回退（Fallback）和成本路由（Cost-Routing）的 `router` 配置。Portkey 提供防护（Guards）与路由功能。Kong AI Gateway 提供基于插件的路由。OpenRouter 的模型市场暴露了推荐 API。

开源方案：RouteLLM（LMSYS）、Not Diamond（商业版）、Prompt Mule。

### 2026 年价格曲线 (Price Curve)

| 模型类别 | 2022 年末 | 2026 年 | 变化 |
|-------------|-----------|------|--------|
| GPT-4 级别质量 | ~20 美元/百万 Token | ~0.40 美元/百万 Token | 成本降低 50 倍 |
| 前沿模型（GPT-5, Claude 4） | — | ~3-10 美元/百万 Token | 新增层级 |

大部分改进源于服务效率的提升——Phase 17 · 04-09 的核心经验已转化为服务提供商侧的成本下降。通过路由，你可以在应用层直接捕获这些收益，而无需等待所有用户迁移至廉价模型层级。

### 数据漂移才是真正风险 (Data Drift)

你的路由策略将 40% 的流量分配给廉价模型。六个月后，任务分布发生偏移（用户提问更专业、问题更长）。路由器并未察觉，因为其分类器是基于第一季度数据训练的。质量在悄然下降，却无人提出足够强烈的投诉。直到你在竞品基准测试中落败，才发现问题所在。

通过在线质量指标（Online Quality Metrics）管控路由：

- 各路由的用户点赞/点踩反馈。
- 针对各路由保留样本（5%）的自动化 LLM 裁判（LLM-Judge）评估。
- 升级率（Escalation Rate）：如果级联路由的向上升级比例超过 30%，说明廉价模型被过度路由。
- 各路由的拒绝回答率。

### 关键数据备忘

- 2026 年在同等质量（Iso-Quality）下的路由成本节省：案例研究显示为 20-60%。
- 2022-2026 年 LLM 价格降幅：综合年均降幅约 10 倍。
- GPT-4 级别模型 2022 年 vs 2026 年价格：~20 美元/百万 Token → ~0.40 美元/百万 Token。
- 级联路由延迟影响：中位延迟约 1.2 倍，升级请求延迟约 2 倍（约占 10% 流量）。

## 使用指南 (Use It)

`code/main.py` 在混合工作负载上模拟预路由、级联路由和集成路由。报告综合成本、质量损失及升级率。

## 部署上线 (Ship It)

本课时将生成 `outputs/skill-router-plan.md`。根据工作负载（workload）和质量预算（quality budget），选择路由模式（routing pattern）及相关信号（signals）。

## 练习

1. 运行 `code/main.py`。在何种准确率下限（accuracy floor）下，级联（cascade）的表现会优于预路由（pre-route）？
2. 你的用户群体中 30% 为企业客户（处理复杂查询），70% 为免费用户（处理简单查询）。请设计路由分流（routing split）策略。应使用何种在线指标（online metric）作为决策闸门？
3. 某路由策略导致质量下降 2%，但能节省 40% 的成本。这是否值得上线发布（ship）？答案取决于产品定位——请从正反两方面进行论述。
4. 利用 OpenAI / Anthropic API 返回的对数概率（logprobs）实现置信度检查（confidence check）。你初始设定的阈值（threshold）是多少？
5. 在六个月期间，升级率（escalation rate）从 8% 攀升至 22%。请诊断三个潜在原因，并分别给出对应的修复方案。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 模型路由（Model routing） | “成本代理（cost broker）” | 根据每个请求动态选择模型 |
| 模型级联（Model cascade） | “廉价优先，逐级升级（cheap-first escalate）” | 先运行低成本模型，置信度低时回退至前沿模型 |
| 预路由（Pre-route） | “先分类（classify first）” | 前置分类器；无需重新运行 |
| 集成路由（Ensemble route） | “并行择优（parallel pick）” | 并行运行多个模型，由奖励模型（reward model）选出最佳结果 |
| 升级率（Escalation rate） | “向上路由占比（uprouted %）” | 级联请求中触发升级的比例 |
| RouteLLM | “LMSYS 路由器（LMSYS router）” | 开源路由库 |
| Not Diamond | “商业路由器（commercial router）” | SaaS 模型路由产品 |
| 数据漂移（Drift） | “廉价模型隐性退化（cheap creep）” | 分布发生偏移但路由器未察觉 |
| 在线质量闸门（Online quality gate） | “实时检查（live check）” | 自动化 LLM 裁判（LLM-judge）对实时流量进行抽样评估 |

## 延伸阅读

- [AbhyashSuchi — 2026 年大语言模型路由最佳实践](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — 2026 年推理优化的崛起](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM 论文 / 代码](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — 模型路由](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — 具备路由原语（routing primitives）的多模型网关。