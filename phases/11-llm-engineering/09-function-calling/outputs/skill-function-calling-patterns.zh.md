---
name: 函数调用模式
description: 在生产环境中实现函数调用的决策框架——工具设计、错误处理、安全性及提供商模式
version: 1.0.0
phase: 11
lesson: 09
tags: [函数调用, 工具使用, 智能体, MCP, 安全性, OpenAI, Anthropic]
---

# 函数调用（Function Calling）模式

在构建使用工具的大语言模型（Large Language Model, LLM）应用时，请应用此决策框架。

## 何时使用函数调用

**在以下情况使用函数调用：**
- 模型需要实时数据（如天气、股票价格、数据库查询）
- 任务需要产生副作用（如发送电子邮件、创建记录、部署代码）
- 模型必须根据用户意图在多个操作中进行选择
- 你正在构建与外部系统交互的智能体（Agent）

**在以下情况改用结构化输出（Structured Outputs）：**
- 你需要从文本中提取数据（无需外部调用）
- 输出即为最终产物，而非中间步骤
- 你只有一个数据模式（Schema），而非多个工具供选择

**在以下情况结合使用两者：**
- 模型调用工具后，将工具返回的结果结构化为特定的输出格式

## 工具设计指南

1. **一个工具，一个操作。** 名为 `manage_database` 且同时处理查询、插入、更新和删除的工具过于宽泛。应拆分为 `query_records`、`insert_record`、`update_record`。工具越具体，模型的选择效果越好。

2. **描述即提示词（Prompt）。** 模型通过阅读工具描述来决定选择。撰写时应像给初级开发者编写说明一样。不仅要说明工具的功能，还要包含它的返回值。

3. **使用枚举（Enum）进行约束。** 如果参数只有 3-10 个有效值，请使用枚举。除非加以约束，否则模型会随意生成字符串——例如 "celsius"、"Celsius"、"C"、"metric"。

4. **工具越少越好。** GPT-4o 能很好地处理 5-10 个工具。当工具超过 20 个时，选择准确率会下降。超过 50 个时，预计会出现 10-15% 的错误选择。应将相关功能分组或使用路由层（Routing Layer）。

5. **必填即真正必填。** 仅当工具在缺少该参数时完全无法运行时，才将其标记为必填。带有合理默认值的可选参数能减少工具调用失败的情况。

## 各提供商特定模式

### OpenAI（GPT-4o、o3、GPT-4o-mini）

tools=[{"type": "function", "function": {"name": ..., "parameters": ...}}]
tool_choice="auto"       # model decides
tool_choice="required"   # must call at least one tool
tool_choice={"type": "function", "function": {"name": "specific_tool"}}

- 支持并行工具调用（单次响应中包含多个 `tool_calls`）
- 必须将工具调用 ID 与结果一同传回
- `gpt-4o-mini` 成本低 10 倍，且能很好地处理简单的工具路由
- 结构化输出模式可与工具参数配合使用，以确保严格遵循数据模式

### Anthropic（Claude 3.5 Sonnet、Claude 4 Opus）

tools=[{"name": ..., "description": ..., "input_schema": ...}]
tool_choice={"type": "auto"}     # model decides
tool_choice={"type": "any"}      # must call at least one tool
tool_choice={"type": "tool", "name": "specific_tool"}

- 工具调用以 `type: "tool_use"` 的内容块形式出现
- 结果需放入带有 `type: "tool_result"` 的用户消息中
- 字段名为 `input_schema` 而非 `parameters`（常见的迁移错误）
- 支持单次响应中包含多次工具调用

### Google（Gemini 2.0 Flash、Gemini 2.0 Pro）

function_declarations=[{"name": ..., "description": ..., "parameters": ...}]
function_calling_config={"mode": "AUTO"}   # or "ANY" or "NONE"

- 在顶层使用 `function_declarations`
- 结果通过 `function_response` 部分返回
- 支持并行函数调用

### 开源模型（Llama 3、Hermes、Qwen）

- 缺乏标准化格式——因模型和服务框架 (serving framework) 而异
- Hermes 格式（NousResearch）是最常见的微调 (fine-tuned) 惯例
- vLLM 为支持的模型提供兼容 OpenAI 的工具调用 (tool calling) 功能
- Ollama 支持兼容模型的基础工具调用
- 在生产环境部署前测试工具选择准确率——在伯克利函数调用排行榜 (Berkeley Function Calling Leaderboard) 上，开源模型的准确率比 GPT-4o 低 15-30%

## 错误处理模式

### 返回结构化错误 (structured errors)

{"error": true, "message": "City 'Toky' not found. Did you mean 'Tokyo'?", "code": "NOT_FOUND", "suggestions": ["Tokyo"]}

包含可操作的信息。“未找到”这种提示很差。“未找到，您是否指的是 X？”则更好。模型会利用错误信息进行自我纠正。

### 重试策略

1. 工具调用因可纠正的错误（如拼写错误、错误的枚举值）而失败
2. 将错误作为工具结果返回给模型
3. 模型进行调整并重试
4. 每次工具调用最多重试 3 次
5. 连续失败 3 次后，将错误返回给用户

### 超时处理

为所有工具执行设置超时时间。30 秒是一个合理的默认值。如果工具超时，应返回结构化的超时错误，以便模型能够告知用户，而不是让程序挂起。

## 安全检查清单

| 检查项 | 原因 | 方法 |
|-------|-----|-----|
| 函数白名单 | 防止任意代码执行 | 仅注册用户所需的工具 |
| 验证参数类型 | 防止类型混淆攻击 (type confusion attacks) | 执行前检查参数类型 |
| 清理字符串参数 | 防止注入攻击 | 拒绝或转义特殊字符 |
| 参数化数据库查询 | 防止 SQL 注入 | 切勿直接传入模型生成的 SQL |
| 过滤工具结果 | 防止数据泄露 | 移除 API 密钥、个人身份信息 (PII) 及内部错误 |
| 限制工具调用频率 | 防止失控循环 | 每次对话最多调用 10-20 次 |
| 记录所有工具调用 | 建立审计追踪 | 存储工具名称、参数、结果和时间戳 |
| 阻止路径遍历 | 防止文件系统越权访问 | 在文件工具中拒绝 `..` 和绝对路径 |
| 沙盒化 (sandbox) 代码执行 | 防止系统级访问 | 使用容器或受限内置函数 |
| 验证返回数据大小 | 防止上下文塞满 (context stuffing) | 截断超过 10KB 的结果 |

## 性能优化

- **并行调用：** 当模型请求多个独立的工具时，使用 `asyncio.gather()` 或 `concurrent.futures` 并发执行它们
- **缓存：** 在同一会话中对相同参数的工具结果进行缓存（例如天气数据在 60 秒内不会变化）
- **流式传输：** 在获取工具结果的同时，流式输出模型的最终响应
- **工具剪枝 (tool pruning)：** 如果上下文窗口紧张，仅包含与当前查询相关的工具定义（可使用分类器进行过滤）
- **使用小模型进行路由 (routing)：** 使用 `gpt-4o-mini` 或 `claude-3-5-haiku` 进行工具选择，然后将结果传递给更强大的模型进行综合处理

## 常见失败模式

| 失败现象 | 原因 | 解决方案 |
|---------|-------|-----|
| 选错工具 | 描述模糊 | 使用特定的触发词重写描述 |
| 缺少必需参数 | 模型遗漏了参数 | 在参数描述中添加清晰的示例 |
| 工具调用无限循环 | 模型反复调用同一工具 | 设置最大迭代次数（5-10 次）并检测重复调用 |
| 参数幻觉 (hallucinated arguments) | 模型编造了看似合理但错误的值 | 使用枚举类型，并对照已知值进行验证 |
| 工具结果过大 | API 返回了 100KB 的数据 | 在反馈给模型前进行截断或摘要 |
| 模型忽略工具结果 | 结果格式令人困惑 | 返回字段名清晰的规范 JSON |