# 大语言模型的 FinOps（财务运营）—— 单位经济学与多租户归因

> 传统的 FinOps（财务运营）在大语言模型（LLM）支出面前已然失效。成本是按令牌（token）交易计算的，而非资源运行时间。标签无法直接映射——一次 API 调用是一笔交易，而非一项资产。工程决策（提示词设计、上下文窗口、输出长度）本质上就是财务决策。2026 年的实践指南要求在系统上线首日就部署三个归因（attribution）维度：按用户（`user_id`）用于席位定价与增购评估，按任务（`task_id` + `route`）用于产品功能面成本核算与优先级排序，按租户（`tenant_id`）用于单位经济学（unit economics）核算与续约决策。四个令牌层级——提示词（prompt）、工具调用（tool）、记忆上下文（memory）、响应（response）——若仅使用单一计费桶，将掩盖真实支出。多租户产品的管控阶梯（enforcement ladder）如下：按租户限流（rate limits）（预期峰值的 2-3 倍，返回明确的 429 状态码与 `retry-after` 头）；每日支出上限（spend cap）（合同上限的 1.5-3 倍；触发限流收紧与告警）；当支出 Z 分数（z-score）> 4 时触发熔断开关（kill switches）（自动暂停并呼叫值班人员）。归因模式包括：标签聚合（tag-and-aggregate）、遥测关联（telemetry-joiner，通过 trace-ID 关联账单；精度最高）、采样外推（sampling-and-extrapolation）、基于模型的分配（model-based allocation）、事件溯源（event-sourced）、实时流处理（real-time streaming）。核心单位指标（unit metric）应为：单次已解决查询的成本、单次生成产物的成本——而非每百万令牌成本（$/M tokens）。事后打标签（retroactive tagging）总会遗漏边缘情况；必须在请求创建时进行插桩（instrument）。

**Type:** 学习
**Languages:** Python（标准库，带熔断开关的简易成本归因模拟器）
**Prerequisites:** 第 17 阶段 · 13（可观测性），第 17 阶段 · 14（缓存）
**Time:** 约 60 分钟

## 学习目标

- 解释为何传统的 FinOps（标签 + 层级）在 LLM 支出面前失效，并指出三个新的归因维度。
- 列举四个令牌层级（提示词、工具调用、记忆上下文、响应），并说明单一计费桶为何会掩盖成本。
- 为多租户产品设计一套管控阶梯（限流 → 支出上限 → 熔断开关）。
- 选择单位指标（单次已解决查询/生成产物的成本），而非每百万令牌成本（$/M tokens）。

## 问题背景

账单显示 40,000 美元。但你不知道：
- 是哪个租户产生的支出。
- 是哪个产品功能驱动的。
- 是否有单个用户存在滥用行为。
- 罪魁祸首是提示词膨胀、工具调用，还是记忆上下文放大。

在云服务商侧使用标签聚合对云资源（如 EC2、S3）有效，因为标签会传播到明细账单中。但 LLM API 调用不会自动打标签——你必须在调用点显式标记用户/任务/租户，并贯穿整个流程。事后归因总会遗漏边缘情况。

## 核心概念

### 三个归因维度 (Attribution Dimensions)

**按用户** (`user_id`)：谁产生了多少成本。用于驱动席位定价、增购谈判，并识别高活跃用户。

**按任务** (`task_id` + `route`)：哪个产品界面/功能产生了多少成本。用于驱动功能优先级排序及下线高成本功能的决策。

**按租户** (`tenant_id`)：哪个客户具有盈利能力。用于驱动单位经济效益分析、续约定价及层级阈值设定。

在项目上线首日，就应在调用点 (Call Site) 对这三个维度进行埋点。事后追溯的效果总是更差。

### 四个 Token 层级 (Token Layers)

| 层级 | 示例 | 占总量的典型比例 |
|-------|---------|---------------------|
| 提示词 (Prompt) | system + user input | 40-60% |
| 工具调用 (Tool) | tool-call results fed back | 20-40% (agent workloads) |
| 记忆 (Memory) | prior conversation / retrieved docs | 10-30% |
| 响应 (Response) | model output | 10-30% |

将这四类混为一谈会导致优化工作陷入盲目。应在归因架构中将它们明确拆分。

### 管控阶梯 (Enforcement Ladder)

1. **按租户限流 (Rate Limit)**。设置为预期峰值的 2-3 倍。返回 429 状态码并附带 `Retry-After`。租户会感受到使用摩擦，但不会产生意外账单。

2. **按租户每日支出上限 (Daily Spend Cap)**。设置为合同上限的 1.5-3 倍。触发条件：收紧限流策略 + 向客户成功团队 (Customer Success) 发送告警。

3. **熔断开关 (Kill Switch)**。当支出 Z 分数 (Z-score) 相对于租户基线 > 4 时触发。自动暂停该租户服务；呼叫值班人员；升级至运维与客户成功团队。

### 归因模式 (Attribution Patterns)

- **打标与聚合 (Tag-and-aggregate)**：在元数据请求头中打上标签；后续进行聚合。实现简单；精度较粗。
- **遥测关联 (Telemetry joiner)**：通过 Trace ID 将调用链追踪数据与账单关联。精度最高。成熟团队的标准做法。
- **采样与外推 (Sampling + extrapolation)**：抽取 5-10% 的样本进行放大推算。适用于粗略估算成本，性价比高；但会遗漏长尾数据。
- **基于模型的分配 (Model-based allocation)**：使用回归分析推断成本驱动因素。适用于缺乏标签的历史遗留数据。
- **事件溯源 (Event-sourced)**：将成本作为流式事件（如 Kafka / Kinesis）处理。支持实时计算。
- **实时流处理 (Real-time streaming)**：仪表盘实现亚秒级更新。

### “每 X 成本”作为核心指标 (Unit Metric)

`$/M tokens` 是供应商话术。产品指标应关注：

- 每解决一个支持工单的成本。
- 每生成一篇文章的成本。
- 每成功完成一次智能体任务的成本。
- 每用户会话分钟的成本。

将成本与产品业务结果挂钩。否则优化工作将失去基准。

### 成本归因追踪数据结构 (Trace Shape)

trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false

在每次调用时输出。存储至数据湖 (Data Lake)。按维度进行聚合。该功能属于第 17 阶段 · 第 13 项可观测性栈 (Observability Stack) 的范畴。

### 复合节省技术栈 (Compounded-Savings Stack)

技术栈组合：缓存 (Cache) + 批处理 (Batch) + 路由 (Route) + 网关 (Gateway)。同时启用四项时：
- L2 缓存（第 17 阶段 · 第 14 项）：输入成本降低约 10 倍。
- 批处理（第 17 阶段 · 第 15 项）：成本减半（50% 折扣）。
- 路由至低成本模型（第 17 阶段 · 第 16 项）：成本降低 60%。
- 网关效率优化（第 17 阶段 · 第 19 项）：处理冗余请求与重试。

理想情况下的叠加效果：成本仅为原始基线的 5-10%。大多数团队仅启用 2-3 个杠杆；极少有团队能同时叠加全部四项。

### 需要牢记的关键数据

- 归因维度：按用户、按任务、按租户。
- 四个 Token 层级：提示词、工具调用、记忆、响应。
- 熔断开关：支出 Z 分数 > 4。
- 核心指标：每解决一次查询的成本，而非 `$/M tokens`。
- 叠加优化效果：有望将成本降至基线的 5-10%。

## 使用它

`code/main.py` 模拟了一个采用三级管控阶梯（three-tier enforcement ladder）的多租户大语言模型（LLM）服务。该脚本会注入一个滥用租户，并演示熔断开关（kill switch）的触发过程。

## 交付成果

本课时将生成 `outputs/skill-finops-plan.md` 文件。该文件会根据产品特性与业务规模，设计成本归因架构（attribution schema）与管控阶梯。

## 练习

1. 运行 `code/main.py`。熔断开关在 Z 分数（z-score）达到多少时触发？你如何设定该阈值？
2. 设计一个按租户、按任务划分的成本看板（cost dashboard）。你会优先构建哪 5 个视图？
3. 你最大的租户处于单位经济亏损（unit-economics-negative）状态。请提出三项干预措施，并按对客户的影响程度排序。
4. 计算某客服产品的单次已解决工单成本（cost per resolved ticket）：每张工单消耗 300 万 token，日均约 800 张工单，采用 GPT-5 缓存费率。
5. 论证回溯性标签（retroactive tagging）是否可行。在什么情况下可以接受？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 按用户归因（Per-user attribution） | “用户级成本” | 每次调用均打上 `user_id` 标签 |
| 按任务归因（Per-task attribution） | “功能成本” | 通过 `task_id` + `route` 标识产品功能面 |
| 按租户归因（Per-tenant attribution） | “客户成本” | 使用 `tenant_id`；驱动单位经济效益分析 |
| 四层 Token 结构（Four token layers） | “成本层级” | 提示词（prompt）+ 工具调用（tool）+ 记忆（memory）+ 响应（response） |
| 速率限制（Rate limit） | “429 防护” | 在网关层执行的单租户上限控制 |
| 每日支出上限（Daily spend cap） | “每日天花板” | 租户级预算，附带告警机制 |
| 熔断开关（Kill switch） | “自动暂停” | 支出 Z 分数 > 4 时触发自动暂停 |
| 单次解决成本（Cost per resolved） | “产品单元指标” | 成本与产品业务结果挂钩，而非单纯按 token 计算 |
| 遥测数据关联器（Telemetry joiner） | “追踪至计费” | 精度最高的成本归因模式 |
| 叠加优化（Stacked optimization） | “缓存+批处理+路由+网关” | 复合节省效果，可将成本降至基线的 ~5-10% |

## 延伸阅读

- [FinOps 基金会 — AI FinOps 概览](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps 学院 — 2026 年单位成本指南](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — 2026 年 LLM 智能体成本归因指南](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Azure OpenAI 中的托管 LLM](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)