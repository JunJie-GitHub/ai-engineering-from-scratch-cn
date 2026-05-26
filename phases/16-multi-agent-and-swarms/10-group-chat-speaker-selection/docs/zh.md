# 群聊与发言者选择

> AutoGen GroupChat 与 AG2 GroupChat 在 N 个智能体（Agent）之间共享同一场对话；选择器函数（Selector Function，如大语言模型 LLM、轮询 Round-Robin 或自定义逻辑）负责决定下一位发言者。这是涌现式多智能体对话（Emergent Multi-Agent Conversation）的典型范式——智能体无需在静态图（Static Graph）中预设自身角色，它们仅对共享消息池（Shared Pool）做出响应。AG2 分支保留了 AutoGen v0.2 中 GroupChat 的语义；而 AutoGen v0.4 则将其重写为事件驱动的参与者模型（Event-Driven Actor Model）。微软于 2026 年 2 月将 AutoGen 转入维护模式（Maintenance Mode），并将其与 Semantic Kernel 合并为 Microsoft Agent Framework（2026 年 2 月发布候选版）。GroupChat 原语（Primitive）在 AG2 与 Microsoft Agent Framework 中均得以保留——一次学习，处处可用。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 第 16 阶段 · 04（原语模型）
**耗时：** 约 60 分钟

## 问题背景

当工作流（Workflow）已知且固定时，静态图（Static Graph，如 LangGraph）表现优异。但真实的对话并非静态：有时编码者需要询问评审者，有时需要咨询研究员，有时又需要对接撰稿人。若将每一种可能的交接路径都硬编码（Hardcoding），将导致图结构中的边（Edge）数量呈爆炸式增长。你真正需要的是*智能体对共享消息池做出响应*，并由某个函数动态决定下一位发言者。

这正是 AutoGen GroupChat 的核心机制。

## 核心概念

### 结构布局

              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"

每个智能体（Agent）都能看到所有消息。在每一轮对话中，都会调用一个选择器函数（Selector Function）来决定下一个发言者。

### 三种选择器类型

**轮询模式（Round-robin）。** 固定循环。具有确定性。复杂度随智能体数量 N 线性增长，但忽略上下文——即使当前话题是法律审查，编码智能体仍会获得发言权。

**大语言模型选择（LLM-selected）。** 调用大语言模型（LLM）读取最近的消息池，并返回最佳的下个发言者。具备上下文感知能力但速度较慢：每一轮都会增加一次 LLM 调用。这是 AutoGen 的默认模式。

**自定义模式（Custom）。** 使用任意逻辑的 Python 函数。典型用法：基于 LLM 选择并附加回退规则（例如，“编码智能体发言后，始终将发言权交给验证智能体”）。

### ConversableAgent API

agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})

`GroupChatManager` 负责维护选择器。当一个智能体完成一轮发言后，管理器会调用选择器，由其返回下一个智能体。该循环将持续进行，直到满足终止条件。

### 终止条件

三种常见模式：

- **最大轮数（Max rounds）。** 对总对话轮数设置硬性上限。
- **“TERMINATE”标记（"TERMINATE" token）。** 智能体可以发送一个哨兵消息（Sentinel Message）；管理器在检测到该消息时停止对话。
- **目标达成检查（Goal-reached check）。** 每轮运行一个轻量级验证器，在任务完成时终止对话。

### AutoGen 与 AG2 的分叉及微软智能体框架的合并

2025 年初，微软开始围绕事件驱动型参与者模型（Event-driven Actor Model）对 AutoGen（v0.4）进行重大重写。社区将 AutoGen v0.2 的 GroupChat 语义分叉（Fork）为 AG2，以保留早期使用者已集成的 API。

2026 年 2 月，微软宣布 AutoGen 将进入维护模式，其事件驱动型参与者模型将合并至 **Microsoft Agent Framework**（2026 年 2 月发布候选版，现已与 Semantic Kernel 合并）。GroupChat 概念在两条技术路线中均得以保留，但实现细节有所不同。对于需要兼容 v0.2 的代码，AG2 是首选的上游项目。

### GroupChat 的适用场景

- **涌现式对话（Emergent conversations）。** 你不想预先硬编码所有可能的下一位发言者。
- **角色混合任务（Role-mixing tasks）。** 编码智能体询问研究智能体，研究智能体询问档案智能体，档案智能体再反问编码智能体。流程并非有向无环图（DAG）。
- **探索性问题求解（Exploratory problem-solving）。** 类似于“头脑风暴会议”，而非“流水线作业”。

### GroupChat 的失效场景

- **严格确定性要求（Strict determinism）。** LLM 选择器可能表现不一致。相同的提示词（Prompt），不同次运行可能选出不同的下一位发言者。
- **阿谀级联效应（Sycophancy cascades）。** 智能体会倾向于顺从发言最自信的一方。需通过明确的反向提示词（Counter-prompt）进行干预。
- **上下文膨胀（Context bloat）。** 每个智能体都会读取所有消息；10 轮之后上下文会变得极其庞大。可使用投影机制（Projections，见第 15 课）来限定视图范围。
- **热点发言者（Hot speakers）。** 某个智能体因选择器偏好其专长领域而主导对话。可在选择器中引入发言平衡机制（Speaker balance）作为特性。

### 群聊模式 vs 主管模式

底层原语（Primitives）相同，默认配置不同：

- 主管模式（Supervisor）：一个智能体负责规划，其他智能体负责执行。选择器逻辑为“询问规划器下一步该做什么”。
- 群聊模式（Group chat）：所有智能体地位平等；选择器是基于共享消息池运行的函数。

两者均使用第 04 课介绍的四种原语。群聊模式默认采用大语言模型选择编排（LLM-selected orchestration）和全量共享状态（Full-pool shared state）。

## 构建它

`code/main.py` 使用标准库（stdlib）从零实现了一个群聊（GroupChat）。系统包含三个智能体（agent）（coder、reviewer、manager），提供轮询（round-robin）与大语言模型选择（LLM-selected）两种变体，并在检测到 `TERMINATE` 令牌（token）时终止对话。

该演示程序会打印对话记录，以及两种变体下选择器（selector）的决策轨迹。

运行：

python3 code/main.py

## 使用它

`outputs/skill-groupchat-selector.md` 为特定任务配置群聊选择器——支持轮询、大语言模型选择或自定义模式，并指定选择器的输入参数（如最近消息、智能体专长、轮次计数等）。

## 部署上线

检查清单：

- **最大轮次上限（Max rounds cap）。** 必须设置。常规任务建议设为 10-20 轮。
- **发言均衡指标（Speaker-balance metric）。** 跟踪每个智能体的发言轮次；当不均衡程度超过阈值时触发告警。
- **终止令牌（Termination token）。** 使用 `TERMINATE` 或专用的验证智能体（verifier agent）。
- **投影或受限内存（Projection or scoped memory）。** 在约 10 条消息后，建议为每个智能体仅提供受限视图（scoped view），以防止上下文膨胀（context bloat）。
- **选择器日志（Selector logging）。** 对于大语言模型选择变体，需同时记录选择器的输入及其最终选择。否则将无法进行调试。

## 练习

1. 运行 `code/main.py`。对比轮询与大语言模型选择模式下的对话过程。在每种模式下，哪个智能体占据主导地位？
2. 在选择器中添加“每个智能体最大发言次数（max-speaks-per-agent）”规则。这会对对话记录产生什么影响？
3. 实现目标达成终止机制：当 reviewer 返回 "approved" 时停止对话。在达到最大轮次上限前，该机制触发的频率如何？
4. 阅读 AutoGen 稳定版文档中关于群聊的部分（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。找出 `GroupChatManager` 使用的默认选择器。
5. 查阅 AG2 仓库（https://github.com/ag2ai/ag2），对比其 v0.2 版本的群聊与 v0.4 的事件驱动（event-driven）版本。v0.4 具体增强了哪些特性（吞吐量、容错性、可组合性）？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 群聊（GroupChat） | “多个智能体在一个聊天室里” | 共享消息池 + 选择器函数。AutoGen / AG2 的基础组件。 |
| 发言者选择（Speaker selection） | “下一个谁说话” | 决定下一个发言智能体的函数。支持轮询、大语言模型选择或自定义。 |
| 群聊管理器（GroupChatManager） | “会议主持人” | AutoGen 组件，负责持有选择器并循环控制发言轮次。 |
| 可对话智能体（ConversableAgent） | “基础智能体” | AutoGen 基类；指能够发送和接收消息的智能体。 |
| 终止令牌（Termination token） | “停止词” | 用于结束对话的哨兵字符串（通常为 `TERMINATE`）。 |
| 热点发言者（Hot speaker） | “某个智能体垄断发言” | 一种故障模式，指选择器持续选中同一个智能体。 |
| 上下文膨胀（Context bloat） | “消息池无限增长” | 每个智能体都会读取所有历史消息，导致上下文随轮次增加而不断膨胀。 |
| 投影（Projection） | “受限视图” | 针对特定角色提供的共享消息池局部视图，用于防止上下文膨胀。 |

## 延伸阅读

- [AutoGen 群聊文档](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 参考实现 (Reference Implementation)
- [AG2 仓库](https://github.com/ag2ai/ag2) — 社区维护的 AutoGen v0.2 后续版本
- [Microsoft Agent Framework 文档](https://microsoft.github.io/agent-framework/) — 整合后的继任框架，2026 年 2 月发布候选版 (Release Candidate)
- [AutoGen v0.4 发布说明](https://microsoft.github.io/autogen/stable/) — 事件驱动 (Event-Driven) 参与者模型 (Actor Model) 重写详情