# 大语言模型路由层（LLM Routing Layer）— LiteLLM, OpenRouter, Portkey

> 供应商锁定（Provider lock-in）成本高昂。不同的工具调用（tool-calling）工作负载适合不同的模型。路由网关（Routing gateway）提供统一的 API 接口、重试机制、故障转移（failover）、成本追踪和安全护栏（guardrails）。2026 年主导市场的三种架构类型为：LiteLLM（开源自托管）、OpenRouter（托管型 SaaS）、Portkey（生产级，于 2026 年 3 月开源）。本课程将明确决策标准，并逐步演示一个基于标准库（stdlib）的路由网关实现。

**类型：** 学习
**语言：** Python（标准库（stdlib），路由 + 故障转移 + 成本追踪器）
**前置条件：** 第 13 阶段 · 02（函数调用（function calling）），第 13 阶段 · 17（网关（gateways））
**时长：** 约 45 分钟

## 学习目标

- 区分自托管、托管型和生产级路由方案。
- 实现一个回退链（fallback chain），在提供商出现故障时按定义的优先级顺序进行重试。
- 跨提供商追踪单次请求的成本和 Token 使用量。
- 根据给定的生产环境约束条件，在 LiteLLM、OpenRouter 和 Portkey 之间做出选择。

## 问题背景

需要关注提供商路由的典型场景：

1. **成本。** Claude Sonnet 的价格是 Haiku 的 3 倍。对于任务分流（triage）场景，Haiku 已足够；对于内容综合（synthesis）场景，Sonnet 则物有所值。按请求进行路由。

2. **故障转移（Failover）。** OpenAI 出现一小时的故障。所有请求均失败。你希望在不重新部署的情况下自动回退（fallback）到 Anthropic。

3. **延迟（Latency）。** 实时聊天界面需要极快的首字生成时间（time-to-first-token）。批量摘要器则不需要。根据延迟服务等级协议（SLA）进行路由。

4. **合规性（Compliance）。** 欧盟用户的数据必须保留在欧盟区域内。按区域进行路由。

5. **实验测试（Experimentation）。** 在同一工作负载上对两个模型进行 A/B 测试。按测试分组（test bucket）进行路由。

为每个集成手动编写所有这些逻辑是重复且低效的。路由网关提供统一的 OpenAI 兼容 API，并自动处理其余所有逻辑。

## 核心概念

### OpenAI 兼容的代理接口形态 (OpenAI-compatible proxy shape)

所有系统都采用 OpenAI 兼容的接口格式。路由网关 (routing gateway) 暴露 `/v1/chat/completions` 端点，接收 OpenAI 数据模式 (schema)，并在内部代理请求至 Anthropic / Gemini / Cohere / Ollama 或任何其他后端。客户端无需关心底层实现。

### 模型别名 (Model aliases)

你的代码不再硬编码 `claude-3-5-sonnet-20251022`，而是使用 `our_smart_model`。网关负责将别名映射到实际模型。当 Anthropic 发布 Claude 4 时，你只需在服务端更新别名映射，业务代码完全无需修改。

### 故障回退链 (Fallback chains)

primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse

网关通过配置文件定义此逻辑。重试次数会计入预算限制，从而防止回退级联导致成本失控。

### 语义缓存 (Semantic caching)

相同或高度相似的提示词 (prompts) 将直接命中缓存，而无需请求底层提供商。在重复的智能体循环 (agent loops) 中，此举可节省 30% 至 60% 的成本。缓存键基于向量嵌入 (embeddings) 生成，高度相似的提示词会共享同一个缓存槽位。

### 安全护栏 (Guardrails)

网关层级：

- **个人身份信息脱敏 (PII redaction)。** 在发送提示词前，通过正则表达式或机器学习模型进行过滤。
- **策略违规拦截。** 拒绝包含禁止内容的提示词。
- **输出过滤。** 清理模型生成结果，防止数据泄露。

Portkey 和 Kong 均内置了预设的安全护栏策略。LiteLLM 则将其作为可选功能。

### 基于密钥的速率限制 (Per-key rate limits)

一个 API 密钥对应一个团队。基于密钥的预算限制可防止单个团队耗尽共享配额。大多数网关均支持此功能。

### 自托管与托管服务的权衡 (Self-hosted vs managed trade-offs)

| 考量因素 | LiteLLM（自托管） | OpenRouter（托管服务） | Portkey（生产级） |
|--------|----------------------|----------------------|----------------------|
| 代码 | 开源，Python 编写 | 托管 SaaS | 开源（2026年3月）+ 托管 |
| 部署 | 自行部署代理 | 注册账号即可 | 两者皆可 |
| 支持提供商 | 100+ | 300+ | 100+ |
| 计费方式 | 使用自有密钥 | OpenRouter 积分 | 使用自有密钥 |
| 可观测性 | OpenTelemetry | 控制台面板 | 完整 OTel + PII 脱敏 |
| 适用场景 | 需要完全掌控权的团队 | 快速原型开发 | 需满足合规要求的生产环境 |

如果你拥有站点可靠性工程 (SRE) 团队且重视数据主权，LiteLLM 是最佳选择。如果你希望单一订阅且无需维护基础设施，OpenRouter 更胜一筹。如果你需要开箱即用的安全护栏与合规支持，Portkey 则是首选。

### 成本追踪 (Cost tracking)

每个请求均携带 `provider`、`model`、`input_tokens` 和 `output_tokens` 字段。将其乘以网关维护的定价表中的单模型单 Token 价格，即可计算成本。支持按用户/团队/项目进行聚合统计。

### MCP 与路由集成 (MCP plus routing)

网关可同时路由大语言模型 (LLM) 调用与模型上下文协议 (MCP) 采样请求。当采样请求的 `modelPreferences` 指定了特定模型时，网关会将其转换并路由至正确的后端。这也是第 13 阶段 · 第 17 课（MCP 网关）与本课的路由网关有时会合并为单一服务的原因。

### 路由策略 (Routing strategies)

- **静态优先级。** 按列表顺序优先调用，出错时触发回退。
- **负载均衡。** 采用轮询或加权分配。
- **成本感知。** 在满足延迟与质量要求的前提下，选择成本最低的模型。
- **延迟感知。** 选择过去 N 分钟内响应最快的模型。
- **任务感知。** 通过提示词分类器，将编码任务路由至特定模型，将摘要任务路由至另一模型。

## 使用它

`code/main.py` 仅用约 150 行代码实现了一个路由网关 (routing gateway)：接收符合 OpenAI 格式的请求，将其转换为面向各提供商的存根 (stubs)，执行按优先级排序的故障转移链 (fallback chain)，跟踪单次请求的成本，并对输入数据执行 PII (个人身份信息) 脱敏 (PII redaction) 处理。请通过以下三种场景运行该脚本：正常请求、主提供商宕机触发故障转移、脱敏机制成功拦截 PII 泄露。

重点关注：

- `ROUTES` 字典：别名 (alias) 映射到按优先级排序的具体提供商列表。
- 故障转移循环 (fallback loop)：在遇到 5xx 状态码时自动重试。
- 成本跟踪器 (cost tracker)：将 Token 使用量乘以各模型的单价。
- PII 脱敏器 (PII redactor)：在转发请求前，清除符合社会安全号码 (SSN) 格式的模式。

## 部署上线

本实践将生成 `outputs/skill-routing-config-designer.md` 文件。根据给定的工作负载特征（延迟、成本、合规性），该技能模块会自动选择 LiteLLM / OpenRouter / Portkey，并生成相应的路由配置。

## 练习

1. 运行 `code/main.py`。触发宕机场景；确认故障转移已切换至第二提供商，且成本核算准确无误。
2. 添加语义缓存 (semantic caching)：以提示词 (prompt) 的 SHA256 哈希值作为查找键；缓存命中时直接返回结果。测量重复调用时的成本节省情况。
3. 添加提示词分类器 (prompt classifier)：将包含“code ...”的提示词路由至偏向智能的别名，将包含“summarize ...”的提示词路由至偏向速度的别名。
4. 设计按团队划分的预算机制：为每个团队设置月度支出上限；网关在达到上限后拒绝新请求。选择一种执行粒度（按单次请求或按时间窗口）。
5. 并排阅读 LiteLLM、OpenRouter 和 Portkey 的官方文档。分别指出这三者各自独有、而其他两者不具备的一项功能。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|------------------------|--------|
| 路由网关 (Routing gateway) | “LLM 代理” | 位于多个提供商前方的统一 API 接口层 |
| OpenAI 兼容 (OpenAI-compatible) | “支持 OpenAI 格式” | 接收 `/v1/chat/completions` 格式的请求，并将其转换为任意后端可识别的格式 |
| 模型别名 (Model alias) | “our_smart_model” | 代码中使用的名称，由网关映射到具体的模型 |
| 故障转移链 (Fallback chain) | “重试列表” | 失败时按顺序尝试的提供商列表 |
| 语义缓存 (Semantic caching) | “提示词嵌入缓存” | 以提示词的嵌入向量 (embedding) 作为键；语义相近的请求可共享缓存命中 |
| 护栏机制 (Guardrails) | “输入/输出过滤器” | 脱敏 PII，拦截违反策略的内容 |
| 按密钥限流 (Per-key rate limit) | “团队预算” | 绑定到特定 API 密钥的配额限制 |
| 成本跟踪 (Cost tracking) | “单次请求花费” | 汇总各模型的 Token 使用量乘以对应单价 |
| LiteLLM | “开源代理” | 可自托管的开源路由网关 |
| OpenRouter | “托管型 SaaS” | 提供托管服务且采用积分计费的网关 |
| Portkey | “生产级方案” | 开源与托管结合，内置护栏机制 |

## 延伸阅读

- [LiteLLM — 文档](https://docs.litellm.ai/) — 自托管路由网关（Self-hosted Routing Gateway）
- [OpenRouter — 快速入门](https://openrouter.ai/docs/quickstart) — 托管路由 SaaS（Managed Routing SaaS）
- [Portkey — 文档](https://portkey.ai/docs) — 带安全护栏（Guardrails）的生产级路由
- [TrueFoundry — LiteLLM 对比 OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — 决策指南
- [Relayplane — 2026 年大语言模型（LLM）网关对比](https://relayplane.com/blog/llm-gateway-comparison-2026) — 供应商调研