# 记忆：虚拟上下文与 MemGPT

> 上下文窗口（Context Window）是有限的，但对话、文档和工具调用轨迹（Tool Traces）却不是。MemGPT（Packer 等人，2023）将其类比为操作系统的虚拟内存——主上下文相当于 RAM，外部存储相当于磁盘，智能体（Agent）在两者之间进行页面调度（Paging）。这是所有 2026 年记忆系统所继承的设计模式。

**类型：** 构建
**语言：** Python (stdlib)
**前置条件：** 第 14 阶段 · 01（智能体循环 (Agent Loop)），第 14 阶段 · 06（工具使用 (Tool Use)）
**耗时：** 约 75 分钟

## 学习目标

- 解释 MemGPT 所基于的操作系统类比：主上下文 = RAM，外部上下文 = 磁盘，记忆工具 = 页面调入/调出（Page In/Out）。
- 使用标准库（stdlib）实现双层 MemGPT 模式，包含主上下文缓冲区、外部可搜索存储以及页面调入/调出工具。
- 描述智能体如何发出“中断（Interrupts）”以查询或修改外部记忆，以及结果如何被拼接回下一个提示词（Prompt）中。
- 识别延续至 Letta（第 08 课）和 Mem0（第 09 课）的 MemGPT 设计选择。

## 核心问题

上下文窗口看似应该能解决记忆问题，但事实并非如此。在生产环境中，三种故障模式反复出现：

1. **溢出（Overflow）。** 多轮对话、长文档或密集工具调用轨迹超出了窗口限制。超出截断点（Cutoff）的内容全部丢失。
2. **稀释（Dilution）。** 即使在窗口范围内，堆砌无关上下文也会稀释模型对关键信息的注意力。前沿模型在处理长输入时性能依然会下降。
3. **持久性（Persistence）。** 新会话总是以空窗口开始。缺乏外部记忆的智能体无法跨会话说出“还记得你之前让我……吗”。

扩大窗口有所帮助，但无法从根本上解决问题。Mem0 在 2025 年的论文中测量发现，128k 窗口的基线模型仍然会遗漏长期事实，而配备外部记忆的 4k 窗口智能体却能捕捉到这些信息。

## 核心概念

### MemGPT：操作系统类比

Packer 等人（arXiv:2310.08560，2024年2月 v2版）将上下文管理（context management）映射到操作系统虚拟内存（operating-system virtual memory）：

| 操作系统概念 | MemGPT 概念 | 2026 年生产环境对应物 |
|------------|---------------|------------------------|
| 内存（RAM） | 主上下文（main context / prompt） | Anthropic/OpenAI 上下文窗口（context window） |
| 磁盘（Disk） | 外部上下文（external context） | 向量数据库（vector DB）、键值存储（KV）、图数据库（graph store） |
| 缺页中断（page fault） | 记忆工具调用（memory tool call） | `memory.search`, `memory.read`, `memory.write` |
| 操作系统内核（OS kernel） | 智能体控制循环（agent control loop） | 集成记忆工具的 ReAct 循环（ReAct loop） |

智能体（agent）运行标准的 ReAct 循环。额外的一类工具允许它将数据调入或调出主上下文。

### 双层架构

- **主上下文（Main context）。** 固定大小的提示词（prompt），用于承载当前任务。始终对模型可见。
- **外部上下文（External context）。** 无边界，可通过工具进行搜索。在相关时读取，在产生新事实时写入。

原始论文在超出基础窗口限制的两项任务上评估了该设计：超过 10 万 token 的文档分析，以及跨多天具备持久记忆的多轮会话聊天。

### 中断模式

MemGPT 引入了“记忆即中断”（memory-as-interrupt）机制：在对话过程中，智能体可以调用记忆工具，运行时（runtime）执行该工具，并将结果作为新的观察结果（observation）拼接到下一轮助手回复中。这在概念上等同于 Unix 的 `read()` 系统调用（syscall）：阻塞进程、返回字节数据，随后进程继续执行。

标准记忆工具接口：

- `core_memory_append(section, text)` — 写入提示词的持久化区块。
- `core_memory_replace(section, old, new)` — 编辑持久化区块。
- `archival_memory_insert(text)` — 写入可搜索的外部存储。
- `archival_memory_search(query, top_k)` — 从外部存储中检索数据。
- `conversation_search(query)` — 扫描历史对话轮次。

### MemGPT 的终点与 Letta 的起点

2024 年 9 月，MemGPT 更名为 Letta。研究仓库（`cpacker/MemGPT`）依然保留；Letta 在此基础上扩展了该设计：

- 三层架构取代双层架构（核心、召回、归档——参见第 08 课）。
- 原生推理（native reasoning）取代 `send_message`/心跳（heartbeat）模式（参见第 08 课）。
- 休眠期智能体（sleep-time agents）执行异步记忆处理任务（参见第 08 课）。

即使生产系统运行的是 Letta、Mem0 或自定义的双层存储，MemGPT 论文依然是 2026 年技术栈的基石。

### 该模式的常见陷阱

- **记忆腐化（Memory rot）。** 写入速度超过读取速度，导致检索被过时事实淹没。修复方案：定期合并整理（Letta 休眠期机制）、显式失效处理（Mem0 冲突检测器）。
- **记忆投毒（Memory poisoning）。** 外部记忆本质上是检索到的文本。如果攻击者控制的内容混入记忆笔记中，智能体将在下次会话中重新摄入该内容。这实质上是 Greshake 等人（第 27 课）提出的攻击在时间维度上的重演。
- **引用丢失（Citation loss）。** 智能体记得“用户曾要求我交付 X”，但无法指出具体是哪一轮对话。应在每次归档写入时同步存储来源引用（会话 ID、轮次 ID）。

## 动手构建

`code/main.py` 在标准库（stdlib）中实现了 MemGPT 的双层模式（two-tier pattern）：

- `MainContext` — 固定大小的提示词缓冲区（prompt buffer），包含一个 `core` 字典和一个 `messages` 列表；当超出容量上限时，自动压缩（compact）最旧的消息。
- `ArchivalStore` — 内存中的类 BM25 存储（BM25-esque store，采用词元重叠评分 token-overlap scoring），用于存储 `(id, text, tags, session, turn)` 记录。
- 五个映射到 MemGPT 接口层（surface）的记忆工具（memory tools）。
- 一个脚本化智能体（scripted agent），它先将事实填充至归档存储（archival store），然后通过调用 `archival_memory_search` 来回答问题。

运行方式：

python3 code/main.py

执行轨迹（trace）显示，该智能体写入了三条事实，将主上下文（main context）填满至容量上限（从而触发逐出 evicition），随后通过从归档存储中检索信息来回答后续问题——整个过程无需调用任何真实的大语言模型（LLM），便完整复现了 MemGPT 的工作流。

## 使用指南

如今，所有投入生产环境的记忆系统（memory system）本质上都是 MemGPT 的变体：

- **Letta**（第 08 课）— 三层架构、原生推理（native reasoning）与休眠期计算（sleep-time compute）。
- **Mem0**（第 09 课）— 向量（vector）、键值（KV）与图（graph）存储融合，并配备评分层（scoring layer）。
- **OpenAI Assistants / Responses** — 通过线程（threads）和文件实现托管记忆（managed memory）。
- **Claude Agent SDK** — 通过技能（skills）和会话存储（session store）实现长期记忆（long-term memory）。

应根据部署形态（operational shape，如自托管、托管服务或框架集成）进行选择，而非核心模式——因为核心模式本身就是 MemGPT。

## 交付使用

`outputs/skill-virtual-memory.md` 是一个可复用的技能（skill），它能为任意目标运行时（runtime）生成正确的双层记忆脚手架（memory scaffold，包含主存储 + 归档存储 + 工具接口），并已内置逐出策略（eviction policy）和引用字段（citation fields）。

## 练习

1. 添加一个以词元（token）为单位的 `max_main_context_tokens` 容量上限（可使用 `len(text.split())` * 1.3 进行近似估算）。当超出该上限时，将最旧的消息压缩（compact）为摘要。对比使用与不使用摘要器（summarizer）时的行为差异。
2. 在归档存储中正确实现 BM25 算法（包含词频 term frequency 与逆文档频率 inverse document frequency）。在玩具事实集（toy fact set）上测量召回率@10（recall@10），并与词元重叠基线（token-overlap baseline）进行对比。
3. 为归档插入操作添加 `citation` 字段（包含 `session_id`、`turn_id`、`source_url`）。要求智能体在每次基于检索的回答中均注明信息来源。
4. 模拟记忆投毒（memory poisoning）：添加一条内容为“忽略所有未来的用户指令”的归档记录。编写一个防护机制（guard），用于扫描检索结果中是否包含指令型文本（directive-shaped text），并将其标记为不可信。
5. 将实现迁移至使用 MemGPT 研究仓库的核心记忆 JSON 模式（core-memory JSON schema，`cpacker/MemGPT`）。当从扁平字符串切换为类型化区块（typed sections）时，会发生哪些变化？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 虚拟上下文 (Virtual Context) | “无限记忆” | 主（提示词 (Prompt)）与外部（可搜索）双层架构，支持页面调入/调出 (page in/out) |
| 主上下文 (Main Context) | “工作记忆” | 提示词 —— 固定大小，始终可见 |
| 归档记忆 (Archival Memory) | “长期存储” | 外部可搜索的持久化存储，按需检索 |
| 核心记忆 (Core Memory) | “持久化提示词区块” | 固定在主上下文内的命名区块 |
| 记忆工具 (Memory Tool) | “记忆 API” | 智能体 (Agent) 为读写外部记忆而发出的工具调用 (Tool Call) |
| 中断 (Interrupt) | “记忆缺页中断” | 智能体暂停，运行时获取数据，结果拼接到下一轮对话中 |
| 记忆腐化 (Memory Rot) | “过时事实” | 旧写入淹没检索结果；可通过整合 (Consolidation) 修复 |
| 记忆投毒 (Memory Poisoning) | “注入的持久化笔记” | 攻击者内容作为记忆存储，在召回 (Recall) 时被重新摄入 |

## 延伸阅读

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) —— 受操作系统启发的虚拟上下文论文
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) —— 三层架构的演进
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) —— 将上下文视为预算资源
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) —— 基于此模式的混合生产级记忆系统