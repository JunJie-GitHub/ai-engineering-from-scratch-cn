---
name: tool-registry
description: 构建具备 JSON Schema 验证、并行分发（parallel dispatch）和可观测性（observability）的生产级工具目录与注册表。
version: 1.0.0
phase: 14
lesson: 06
tags: [函数调用, 工具, 模式, 验证, BFCL, 并行工具]
---

给定一个任务领域，生成一个工具目录，使智能体（agent）能够在 BFCL V4 的评估维度（axes）（自主性、多轮对话、实时、非实时、幻觉）上可靠地使用。

产出内容：

1. 工具定义（Tool definitions）。针对每个工具：`name`（采用 snake_case 命名法）、`description`（明确告知模型何时应使用以及何时不应使用该工具）、带有类型化属性的 JSON Schema 输入、必填字段、适用时的枚举值（enums）、数值型字段的最小/最大值、各工具超时时间、各工具沙箱策略（sandbox policy）（文件系统访问面、网络权限、内存上限）。
2. 描述质量检查。对每个描述进行审查：“该描述是否明确告知模型在何种情况下应优先选择此工具而非其他工具？”如果两个工具的描述存在重叠，则拒绝并重新编写。
3. 并行分发计划（Parallel-dispatch plan）。针对每个实际任务，识别哪些工具调用是独立的（可并行执行），哪些必须按顺序执行。输出预期的分发图（dispatch graph）。
4. 验证策略（Validation policy）。枚举检查、类型强制转换规则（type coercion）（例如“接受字符串形式的整数，拒绝字符串形式的浮点数”）、必填字段强制校验。每次失败均返回结构化的观测字符串（structured observation string），绝不向主循环抛出异常。
5. 可观测性（Observability）。每个工具都会发出一个 OpenTelemetry GenAI `tool_call` 跨度（span），并附带属性 `gen_ai.tool.name`、`gen_ai.tool.call.id`、`gen_ai.tool.call.arguments`、`gen_ai.tool.call.result`（当内容策略要求时，采用引用形式而非内联形式）。

硬性拒绝条件（Hard rejects）：

- 通用的 shell/命令执行工具。予以拒绝，并拆分为具体的动词操作（如 `git_status`、`fs_read`、`npm_test`）。
- 参数具有封闭值集时缺失枚举值（enums）。枚举验证是捕获模型行为漂移（drift）成本最低的方式。
- 两个不同工具使用相同的描述。模型无法在它们之间做出可靠的选择。
- `description` 仅包含工具名称（如“将两个数字相加”）。必须包含何时优先选择该工具而非替代方案的说明。
- 未设置超时时间。每个工具调用都必须设定时间上限。

拒绝规则（Refusal rules）：

- 如果单个智能体（agent）的工具列表超过 30 个，则予以拒绝，并建议采用子智能体委派（subagent delegation）机制（参见第 17 课）。
- 如果任何工具在未设置确认关卡（confirmation gate）的情况下执行破坏性操作，则予以拒绝，并指引至第 09 课（权限与沙箱机制）。
- 如果任务涉及计算机操作（computer use）（点击、输入、截图），则予以拒绝，并指引至第 21 课——这属于具有基于视觉的操作（vision-based actions）的独立工具形态。

输出内容：一个可直接粘贴至 Anthropic / OpenAI / Gemini SDK 调用中的 JSON 工具目录、一张分发图（dispatch-graph）示意图、一份验证策略文档，以及注册表应通过的 BFCL 风格微型评估（mini-eval）。

结尾附上“下一步阅读”指引：第 09 课（沙箱机制）、第 23 课（OTel GenAI 跨度）或第 30 课（评估驱动）。