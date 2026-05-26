# OpenAI Agents SDK：移交（Handoffs）、护栏（Guardrails）与追踪（Tracing）

> OpenAI Agents SDK 是基于 Responses API 构建的轻量级多智能体（Multi-Agent）框架。包含五大核心原语（Primitives）：智能体（Agent）、移交（Handoff）、护栏（Guardrail）、会话（Session）与追踪（Tracing）。移交操作被建模为名为 `transfer_to_<agent>` 的工具。护栏会在输入或输出阶段触发拦截。追踪功能默认开启。

**类型：** 学习 + 实战
**语言：** Python（标准库）
**前置知识：** 第 14 阶段 · 01（智能体循环 Agent Loop），第 14 阶段 · 06（工具使用 Tool Use）
**预计耗时：** 约 75 分钟

## 学习目标

- 列举 OpenAI Agents SDK 的五大核心原语。
- 解释移交机制：为何将其建模为工具、模型所见的命名格式，以及上下文如何传递。
- 区分输入护栏、输出护栏与工具护栏；解释 `run_in_parallel` 与阻塞模式的区别。
- 使用标准库实现一个包含移交、护栏与跨度式追踪（Span-style Tracing）的运行时环境。

## 问题背景

无法清晰委派任务的智能体，最终往往会将所有内容塞进同一个提示词（Prompt）中。缺乏护栏的智能体可能会泄露个人身份信息（PII）、输出违反策略的内容，或陷入无限循环。OpenAI 的 SDK 将三大核心原语进行了标准化封装，从而使多智能体协作变得切实可行。

## 核心概念

### 五大基础原语 (Primitives)

1. **智能体 (Agent)。** 大语言模型 (LLM) + 指令 + 工具 + 转交 (Handoff)。
2. **转交 (Handoff)。** 将任务委托给另一个智能体。在模型中表现为一个名为 `transfer_to_<agent_name>` 的工具。
3. **护栏 (Guardrail)。** 对输入（仅限首个智能体）、输出（仅限末尾智能体）或工具调用（针对每个函数工具）进行验证。
4. **会话 (Session)。** 跨轮次自动维护对话历史。
5. **追踪 (Tracing)。** 内置跨度 (Span)，用于记录大语言模型生成、工具调用、转交和护栏操作。

### 将转交作为工具

模型会在其工具列表中看到 `transfer_to_billing_agent`。调用该工具会向运行时 (Runtime) 发出信号，以执行以下操作：

1. 复制对话上下文（或通过 `nest_handoff_history` 测试版功能进行折叠）。
2. 使用目标智能体的指令对其进行初始化。
3. 交由目标智能体继续执行运行流程。

这实质上是将主管模式 (Supervisor Pattern)（第 13 课 / 第 28 课）进行了产品化封装。

### 护栏

包含三种类型：

- **输入护栏 (Input guardrails)。** 作用于首个智能体的输入。在任何大语言模型调用之前，拦截不安全或超出范围的请求。
- **输出护栏 (Output guardrails)。** 作用于末尾智能体的输出。捕获个人身份信息 (PII) 泄露、策略违规或格式错误的响应。
- **工具护栏 (Tool guardrails)。** 针对每个函数工具运行。验证参数、检查权限并审计执行情况。

运行模式：

- **并行模式 (Parallel)**（默认）。护栏大语言模型与主大语言模型并行运行。可降低尾部延迟 (Tail Latency)。若触发拦截，主大语言模型的工作将被丢弃（造成 Token 浪费）。
- **阻塞模式 (Blocking)**（`run_in_parallel=False`）。护栏大语言模型优先运行。若触发拦截，则不会在主调用上浪费任何 Token。

触发拦截时会抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered` 异常。

### 追踪

默认开启。每次大语言模型生成、工具调用、转交和护栏操作都会发出一个跨度 (Span)。设置 `OPENAI_AGENTS_DISABLE_TRACING=1` 可退出追踪。通过 `add_trace_processor(processor)` 可将跨度数据同步分发至您自己的后端与 OpenAI 的后端。

### 会话

`Session` 将对话历史存储在指定后端（如 SQLite、Redis 或自定义后端）。调用 `Runner.run(agent, input, session=session)` 会自动加载历史并追加新内容。

### 该模式的常见陷阱

- **转交漂移 (Handoff drift)。** 智能体 A 转交给智能体 B，B 又转回给 A。建议添加跳数计数器 (Hop Counter) 进行限制。
- **护栏绕过 (Guardrail bypass)。** 工具护栏仅对函数工具生效；内置工具（如文件读取器、网页抓取器）需要单独配置策略。
- **过度追踪 (Over-tracing)。** 跨度中可能包含敏感内容。建议结合 OpenTelemetry (OTel) 生成式 AI 内容捕获规则（第 23 课）——将内容外部存储，仅通过 ID 引用。

## 动手构建

`code/main.py` 使用标准库实现了该 SDK 的核心架构：

- `Agent`、`FunctionTool`、`Handoff`（作为具备转交语义的函数工具）。
- `Runner`，包含输入/输出/工具护栏、转交分发逻辑以及跳数计数器。
- 一个简单的跨度发射器，用于展示追踪结构。
- 一个分诊智能体 (Triage Agent)，根据用户查询将任务转交至计费或支持模块；并在特定输入上触发护栏拦截。

运行方式：

python3 code/main.py

追踪结果展示了两次成功的转交、一次输入护栏拦截，以及一个与真实 SDK 输出结构一致的跨度树。

## 投入使用

- 适用于以 OpenAI 为核心产品的 **OpenAI Agents SDK**。
- 适用于以 Claude 为核心产品的 **Claude Agent SDK**（第 17 课）。
- 当你需要显式状态（explicit state）和持久化恢复（durable resume）时使用 **LangGraph**（第 13 课）。
- 当你需要精确控制（如语音、多提供商、联邦部署）时采用**自定义方案**。

## 交付上线

`outputs/skill-agents-sdk-scaffold.md` 用于搭建一个 Agents SDK 应用脚手架，包含分流代理（triage agent）、交接（handoff）、输入/输出/工具护栏（guardrail）、会话存储（session store）以及追踪处理器（trace processor）。

## 练习

1. 添加交接跳数计数器（handoff hop counter）：在 N 次交接后拒绝请求。追踪该行为。
2. 将 `nest_handoff_history` 实现为一个可选项——在交接前将历史消息折叠为一条摘要。
3. 编写一个阻塞式输出护栏（blocking output guardrail）。对比会触发该护栏的提示词与正常通过的提示词在延迟上的差异。
4. 将 `add_trace_processor` 连接到 JSON 日志记录器。每个跨度（span）会输出何种数据结构？
5. 阅读 SDK 文档。将你的标准库（stdlib）玩具项目迁移至 `openai-agents-python`。你之前的建模有哪些错误？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 代理（Agent） | “大语言模型 + 指令” | SDK 中的代理类型；负责管理工具和交接 |
| 交接（Handoff） | “转移” | 模型调用的工具，用于将任务委托给另一个代理 |
| 护栏（Guardrail） | “策略检查” | 对输入/输出/工具调用进行的验证 |
| 触发器（Tripwire） | “护栏触发” | 护栏拒绝时抛出的异常 |
| 会话（Session） | “历史存储” | 在多次运行之间持久化的对话记忆 |
| 追踪（Tracing） | “跨度（Spans）” | 针对大语言模型、工具、交接和护栏的内置可观测性 |
| 阻塞式护栏（Blocking guardrail） | “顺序检查” | 护栏优先运行；触发时不会浪费 Token |
| 并行护栏（Parallel guardrail） | “并发检查” | 护栏并行运行；延迟更低，但触发时会浪费 Token |

## 延伸阅读

- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 基础组件（primitive）、交接、护栏、追踪
- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude 风格的对应实现
- [Anthropic：构建高效代理](https://www.anthropic.com/research/building-effective-agents) — 何时才真正需要引入交接机制
- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Agents SDK 跨度所映射的标准