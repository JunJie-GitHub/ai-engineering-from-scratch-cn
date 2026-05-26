# 内存块与休眠期计算（Letta）

> MemGPT 于 2024 年更名为 Letta。2026 年的版本演进引入了两项新设计：模型可直接编辑的离散功能内存块（discrete functional memory blocks），以及在主智能体（primary agent）空闲时异步整合内存的休眠期智能体（sleep-time agent）。这正是突破单次对话限制、实现内存扩展的关键路径。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 07（MemGPT）
**耗时：** 约 75 分钟

## 学习目标

- 指出 Letta 使用的三层内存架构：核心内存（core memory）、检索记忆（recall memory）与归档记忆（archival memory），并说明各自的作用。
- 解释内存块模式（memory-block pattern）：将人类块（Human block）、人设块（Persona block）和用户自定义块作为一等类型化对象（first-class typed objects）。
- 描述休眠期计算（sleep-time compute）的概念，说明其为何不处于关键路径（critical path）上，以及为何它能运行比主智能体更强大的模型。
- 实现一个脚本化的双智能体循环（two-agent loop），其中主智能体负责响应请求，休眠期智能体则在对话轮次之间整合内存块。

## 问题所在

MemGPT（第 07 课）解决了虚拟内存控制流（virtual-memory control flow）的问题。但在实际生产环境中，又暴露出三大难题：

1. **延迟（Latency）。** 每次内存操作都位于关键路径（critical path）上。如果智能体必须在用户等待时执行剪枝、摘要或数据对齐，尾部延迟（tail latency）将急剧上升。
2. **内存腐化（Memory rot）。** 写入操作不断累积，相互矛盾的事实长期留存，导致检索系统被陈旧内容淹没。
3. **结构丢失（Structure loss）。** 扁平化的归档存储无法表达“人类块始终保留在提示词中；人设块始终保留在提示词中；任务块随会话切换”这类逻辑。

Letta（letta.com）是 2026 年的重构版本。内存块使结构显式化；休眠期计算则将整合工作移出关键路径。

## 核心概念

### 三个层级

| 层级 | 范围 | 存储位置 | 写入方式 |
|------|-------|----------------|------------|
| 核心层 (Core) | 始终可见 | 位于主提示词 (Prompt) 内部 | 智能体 (Agent) 工具调用 (Tool Call) + 休眠期重写 |
| 回忆层 (Recall) | 对话历史 | 可检索 | 自动轮次日志记录 |
| 归档层 (Archival) | 任意事实 | 向量存储 (Vector) + 键值存储 (KV) + 图存储 (Graph) | 智能体工具调用 + 休眠期数据摄入 |

核心层是 MemGPT 的核心组件。回忆层是对话缓冲区及其被逐出的尾部数据。归档层是外部存储。这种划分解决了 MemGPT 原有双层架构的过度耦合问题。

### 记忆块 (Memory Blocks)

记忆块是核心层中经过类型定义、持久化且可编辑的片段。最初的 MemGPT 论文定义了两个：

- **人类块 (Human Block)** — 关于用户的事实（姓名、角色、偏好、目标）。
- **角色块 (Persona Block)** — 智能体的自我认知（身份、语气、约束条件）。

Letta 将其泛化为任意用户自定义的记忆块：用于当前目标的 `Task` 块、用于代码库事实的 `Project` 块、用于硬性约束的 `Safety` 块。每个记忆块都包含 `id`、`label`、`value`、`limit`（字符上限）和 `description`（以便模型知道何时对其进行编辑）。

记忆块可通过工具接口进行编辑：

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` — 压缩接近容量上限的记忆块。

### 休眠期计算 (Sleep-time Compute)

2025 年 Letta 新增的功能：在后台、非关键路径 (Critical Path) 上运行第二个智能体。休眠期智能体负责处理对话记录和代码库上下文，将 `learned_context` 写入共享记忆块，并对归档记录进行整合或失效处理。

由此衍生的特性包括：

- **无延迟成本。** 主响应无需等待内存操作。
- **允许使用更强模型。** 休眠期智能体可以使用成本更高、速度更慢的模型，因为它不受延迟限制。
- **自然的整合窗口。** 在用户无需等待时，进行去重、摘要生成以及矛盾事实的失效处理。

这种架构契合人类的工作方式：你执行任务，然后休息，长期记忆在夜间沉淀巩固。

### Letta V1 与原生推理 (Native Reasoning)

Letta V1（`letta_v1_agent`，2026）弃用了 `send_message`/心跳机制以及内联的 `Thought:` 标记，转而采用原生推理。OpenAI 的 Responses API 和 Anthropic 支持扩展思考的 Messages API 会在独立通道上输出推理过程，并在对话轮次间传递（生产环境中跨提供商加密传输）。控制循环 (Control Loop) 依然采用 ReAct 模式。思维轨迹是结构化的，而非提示词形态。

### 该模式的常见陷阱

- **记忆块膨胀 (Block Bloat)。** 无限调用 `block_append` 会迅速触及上限。应在写入操作前接入记忆块摘要器，防止超出容量限制。
- **静默漂移 (Silent Drift)。** 休眠期智能体重写了记忆块，但主智能体毫无察觉。应为记忆块引入版本控制，并在轨迹中展示差异。
- **污染整合 (Poisoned Consolidation)。** 休眠期智能体将攻击者可触及的内容处理进核心层。第 27 课的原则同样适用于休眠期接口。

## 动手构建

`code/main.py` 实现了：

- `Block`（块）—— `id`、`label`、`value`、`limit`、`description`。
- `BlockStore`（块存储）—— CRUD（增删改查）操作 + `near_limit(label)` 辅助函数。
- 两个脚本化智能体（Scripted Agents）—— `PrimaryAgent`（主智能体）负责处理单轮对话，`SleepTimeAgent`（休眠期智能体）负责在对话轮次之间进行信息整合。
- 一段执行轨迹（Trace），展示了包含块写入的三轮对话，以及一次休眠期处理（Sleep-time Pass），该处理会对块进行摘要并标记过时事实为无效。

运行方式：

python3 code/main.py

运行日志（Transcript）清晰地展示了职责分离：主轮次响应迅速，直接生成原始写入记录；休眠期处理则负责数据压缩与清理。

## 使用方式

- **Letta** (letta.com) 提供官方参考实现。支持自托管或托管云服务。
- **Claude Agent SDK skills**（技能）作为块状知识（Block-shaped Knowledge）—— 技能是一种已命名、带版本、可检索的指令块，智能体可按需加载。
- **自定义构建**适用于希望对存储后端拥有完全控制权的团队。建议遵循 Letta API 契约（API Contract），以便后续迁移。

## 部署与交付

`outputs/skill-memory-blocks.md` 可生成适配任意运行时的 Letta 风格块系统，内置休眠期钩子（Sleep-time Hooks），并包含安全规则与引用链路配置。

## 练习

1. 添加一个 `block_summarize` 工具，当 `near_limit` 返回 `true` 时，用模型生成的摘要替换块的值。应设置何种触发阈值，才能在最小化摘要调用次数的同时避免块溢出？
2. 在归档存储（Archival）中实现休眠期去重：若两条记录的文本 Token 重叠率超过 90%，则将其合并为一条。此操作仅限在休眠期处理中执行，绝不可放在关键路径（Critical Path）上。
3. 为块添加版本控制。每次写入时记录旧值及差异（Diff）。暴露 `block_history(label)` 接口，以便运维人员排查“智能体为何忘记了 X"等问题。
4. 将休眠期智能体视为不可信写入方。当它们修改 Persona（角色设定）或 Safety（安全）块时，提交前必须经过第二个智能体的审核。
5. 将示例迁移至使用 Letta API（`letta_v1_agent`）。块结构（Block Schema）会发生哪些变化？原生推理（Native Reasoning）又将如何改变执行轨迹的形态？

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 记忆块（Memory block） | “可编辑的提示词片段” | 核心记忆（Core Memory）中类型明确、持久化且可由大语言模型（LLM）编辑的片段 |
| 用户块（Human block） | “用户记忆” | 关于用户的事实信息，固定存储在核心层 |
| 角色块（Persona block） | “智能体身份” | 自我认知、语气风格、行为约束等，固定存储在核心层 |
| 休眠期计算（Sleep-time compute） | “异步记忆处理” | 第二个智能体在关键路径之外执行的数据整合工作 |
| 核心/召回/归档（Core / Recall / Archival） | “存储层级” | 三层记忆划分：始终可见层 / 对话上下文层 / 外部存储层 |
| 块限制（Block limit） | “容量上限” | 每个块的字符数上限；达到上限将强制触发摘要 |
| 原生推理（Native reasoning） | “思考通道” | 模型提供商级别的推理输出，而非提示词层面的 `Thought:` 标签 |
| 习得上下文（Learned context） | “休眠期输出” | 休眠期智能体写入共享块中的事实信息 |

## 延伸阅读

- [Letta 博客：Memory Blocks](https://www.letta.com/blog/memory-blocks) — 块模式（block pattern）
- [Letta 博客：Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — 异步整合（async consolidation）
- [Letta：重构智能体循环（Agent Loop）](https://www.letta.com/blog/letta-v1-agent) — 原生推理重写（native reasoning rewrite）
- [Packer 等人，MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — 起源（origin）