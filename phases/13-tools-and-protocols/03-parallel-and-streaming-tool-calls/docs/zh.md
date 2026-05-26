# 并行工具调用与工具流式处理

> 将三次独立的天气查询串行执行 (serialized) 需要三次往返通信 (round trips)。若并行运行，总耗时将缩短至最慢的那次调用。目前所有前沿模型提供商均支持在单轮对话中发出多个工具调用。收益是实实在在的，但底层架构 (plumbing) 却颇为微妙。本教程将涵盖这两部分内容：并行扇出 (fan-out) 与流式参数重组 (streamed-argument reassembly)，并重点剖析 ID 关联陷阱 (id-correlation trap)。

**类型：** 实战开发
**语言：** Python（标准库、线程池 + 流式处理框架）
**前置知识：** 第 13 阶段 · 02（函数调用深入解析）
**时长：** 约 75 分钟

## 学习目标

- 解释为何存在 `parallel_tool_calls: true` 参数，以及何时应将其禁用。
- 在并行扇出 (fan-out) 过程中，将流式传输的参数片段正确关联到对应的工具调用 ID (tool-call id)。
- 将不完整的 `arguments` 字符串重组为完整的 JSON，且避免过早解析。
- 运行一个三城市天气基准测试，直观对比串行与并行调用的延迟差异。

## 核心问题

若不采用并行调用，智能体 (agent) 在回答“班加罗尔、东京和苏黎世的天气如何”时，会按以下流程执行：

user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer

这涉及三次 LLM 往返通信，且每次还需承担执行器 (executor) 的延迟。总耗时约为理想实际耗时 (wall-clock time) 的 4 倍。

采用并行调用时：

user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer

仅需一次 LLM 往返通信。执行器耗时取决于三者中的最大值，而非总和。在 OpenAI、Anthropic 和 Gemini 上的生产环境基准测试表明，针对扇出型工作负载，实际耗时可缩短 60% 至 70%。

代价在于关联逻辑的复杂性。当这三个调用无序完成时，返回结果必须携带匹配的 `tool_call_id`，以便模型能够正确对齐。在流式返回结果时，你必须在执行前将零散的参数片段组装成完整的 JSON。Gemini 3 引入唯一 ID 的部分原因，正是为了解决实际应用中两个并行调用同一工具时无法区分的难题。

## 核心概念

### 启用并行

- **OpenAI。** 默认开启 `parallel_tool_calls: true`。设置为 `false` 可强制串行执行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false` 启用并行（Claude 3.5 及以上版本默认开启）。设置为 `true` 可强制串行执行。
- **Gemini。** 始终支持并行；将 `tool_config.function_calling_config.mode = "AUTO"` 交由模型自行决定。

当工具之间存在顺序依赖（例如先 `create_file` 后 `write_file`）、某次调用的输出需作为另一次调用的输入，或限流器 (Rate Limiter) 无法处理扇出 (Fan-out) 请求时，应禁用并行。

### ID 关联

模型发出的每次调用都带有一个 `id`。宿主程序 (Host) 返回的每个结果都必须包含相同的 `id`。若缺少此标识，结果将无法明确对应。

- **OpenAI。** 每个 tool 角色的消息中均包含 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` 块中均包含 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 中均包含 `id`（Gemini 3 及以上版本；Gemini 2 通过名称匹配，但在同名并行调用时会失效）。

### 并发执行调用

宿主程序会在独立的线程、协程或远程工作节点上运行每次调用的执行器。最简单的测试框架 (Harness) 使用线程池；生产环境通常使用 `asyncio` 配合 `asyncio.gather` 或结构化并发 (Structured Concurrency)。完成顺序是不可预测的——`id` 才是唯一的标识符。

一个常见错误：按照调用列表的顺序而非实际完成顺序返回结果。这通常能正常工作，因为模型只关心 `tool_call_id`，但如果结果丢失或重复，乱序提交会使调试更加困难。建议按照实际完成顺序返回结果，并附带明确的 `id`。

### 流式工具调用

当模型以流式输出时，`arguments`（参数）会分片到达。三个并行调用的独立数据块流会在网络传输中交错。你需要为每个 `id` 维护一个累加器 (Accumulator)。

各提供商的数据结构如下：

- **OpenAI。** 每个数据块对应 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。数据块携带 `index`（在调用列表中的位置）。你需要按索引进行累加，在首次出现时读取 `id`，并在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流事件以 `message_start` 开始，随后每个块对应一个 `content_block_start`，类型为 `tool_use`（包含 id、name 和空的 input）。`content_block_delta` 事件携带 `input_json_delta` 数据块。`content_block_stop` 用于关闭每个块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上版本）会发射带有 `functionCallId` 的数据块，使调用能够清晰交错。在 Gemini 3 之前，流式输出每次仅返回一个完整的调用。

### 部分 JSON 与过早解析陷阱

在 `arguments` 完整之前，绝不能对其进行解析。类似 `{"city": "Beng` 的部分 JSON 是无效的，会引发异常。正确的触发时机是提供商的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop` 或 Gemini 的流结束事件。只有在此之后才应尝试 `json.loads`。更稳健的做法是使用增量 JSON 解析器 (Incremental JSON Parser)，在结构完整时逐步产出事件；OpenAI 的流式指南推荐此方法，以支持显示实时“思考中”指示器的用户体验。单纯依靠大括号计数来判断完整性并不可靠（引号内或转义内容中的大括号会导致误判），仅应将其作为非正式的调试启发式手段。

### 乱序完成

call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third

宿主程序的回复仍必须引用对应的 `id`：

[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]

对于 OpenAI 或 Anthropic，回复中的顺序不影响正确性。只要 `id` 匹配，Gemini 也接受任意顺序。

### 基准测试：串行与并行对比

`code/main.py` 中的测试框架模拟了三个延迟分别为 400、600 和 800 毫秒的执行器。串行执行总耗时为 1800 毫秒。并行执行耗时为 max(400, 600, 800) = 800 毫秒。这种差异是固定的而非成比例的，因此随着工具数量的增加，节省的时间也会相应增长。

实际应用中的注意事项：并行调用会对下游 API 造成压力。向限流服务发起 10 路扇出请求将会失败。第 13 阶段 · 第 17 节涵盖了网关级背压 (Backpressure) 机制；重试语义 (Retry Semantics) 计划在后续阶段实现。

### 流式扇出的实际耗时 (Wall-Clock)

如果模型本身采用流式输出，你可以在某个调用的参数完整后立即开始执行，而无需等待所有调用全部完成。这是 OpenAI 文档中提及的一项优化，但并非所有 SDK 都对外暴露了此功能。本课的测试框架实现了该逻辑：一旦模拟的流式输出产生完整的参数对象，宿主程序便会立即触发该调用。

## 使用指南

`code/main.py` 包含两部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 分别以串行和并行方式运行三个模拟的天气查询调用，并打印实际耗时（wall-clock time）。第二部分重放一个模拟的流式响应——将三个并行调用的 `arguments` 数据块交错混合在单一数据流中——并使用 `StreamAccumulator` 按 ID 进行重组。全程不涉及大语言模型（LLM）或网络请求，仅聚焦于数据重组逻辑。

重点关注：

- 串行计时器耗时为 1.8 秒。在相同的模拟延迟下，并行计时器耗时为 0.8 秒。
- 累加器（accumulator）通过按 ID 缓冲乱序到达的数据块来处理乱序问题，仅在每个调用的 JSON 数据完整时才进行解析。
- 执行器（executor）会在某个 ID 的参数确定后立即启动，而无需等待所有数据流结束。

## 交付上线

本课时将生成 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表（tool registry），该技能模块会审计哪些工具适合并行调用、哪些存在顺序依赖，以及哪些可能触发下游的速率限制（rate limits）——最终返回一个更新后的注册表，并为每个工具添加 `parallel_safe` 标志。

## 练习

1. 运行 `code/main.py` 并调整模拟延迟。验证并行与串行的耗时比是否近似于 `max/sum`（实际运行结果会因线程调度、序列化及测试框架开销而略微偏离理想值）。在何种延迟分布下，并行化将不再具有显著优势？

2. 扩展累加器以处理“调用在流式传输中途被取消”的情况：丢弃其缓冲区并发出 `cancelled` 事件。哪家提供商明确记录了此情况？请查阅 Anthropic 的 `content_block_stop` 语义以及 OpenAI 的 `finish_reason: "length"` 行为。

3. 将线程池替换为 `asyncio.gather`。对两者进行基准测试（benchmark）。由于上下文切换（context-switch）成本更低，你应该会看到异步方案带来小幅性能提升，但这仅在执行器执行真实 I/O 操作时成立。

4. 挑选两个不应并行执行的工具（例如先 `create_file` 后 `write_file`）。在注册表中添加一个 `ordering_dependency` 依赖图，并基于该图控制并行扇出（fan-out）。这是实现依赖感知调度（dependency-aware scheduling）的最小必要机制，未来的智能体工程（agent-engineering）阶段将对此进行规范化。

5. 阅读 OpenAI 的并行函数调用（parallel-function-calling）章节以及 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 建议禁用并行化的唯一现实工具类型。（提示：对同一资源产生实质性变更的操作。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 并行工具调用 (Parallel Tool Calls) | “单轮扇出” | 模型在单条助手消息中同时发出多个工具调用请求 |
| `parallel_tool_calls` | “OpenAI 的开关” | 用于启用或禁用多工具调用请求的发送 |
| `disable_parallel_tool_use` | “Anthropic 的反向开关” | 用于关闭并行模式的标志；默认情况下并行功能已启用 |
| 工具调用 ID (Tool Call ID) | “关联句柄” | 每次调用的唯一标识符，返回结果时必须原样回传该标识符以作匹配 |
| 累加器 (Accumulator) | “流缓冲区” | 按调用 ID 维护的字符串缓冲区，用于逐步拼接 `arguments` 的分块数据 |
| 乱序完成 (Out-of-order Completion) | “谁快谁先” | 并行调用以不可预测的顺序完成；ID 是关联各调用结果的纽带 |
| 依赖图 (Dependency Graph) | “顺序约束” | 某些工具的输出需作为其他工具的输入，此类场景无法并行执行 |
| 过早解析陷阱 (Parse-early Trap) | “JSON.parse 异常” | 尝试解析尚未接收完整的 `arguments` 字符串导致解析失败 |
| `streamFunctionCallArguments` | “Gemini 3 的特性” | 为每次调用分配唯一 ID 的流式参数分块传输机制 |
| 按完成顺序回复 (Completion-order Reply) | “不等全部完成” | 结果一旦返回即刻响应，通过 ID 进行键值匹配 |

## 延伸阅读

- [OpenAI — 并行函数调用](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为及退出并行的标志
- [Anthropic — 工具使用：实现工具调用](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 参数与结果批处理
- [Google — Gemini 函数调用并行部分](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 中基于 ID 关联的并行调用
- [OpenAI — 使用工具的流式响应](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 流式传输中参数分块的重组
- [Anthropic — 流式消息](https://docs.anthropic.com/en/api/messages-streaming) — 包含 `input_json_delta` 的 `content_block_delta` 事件