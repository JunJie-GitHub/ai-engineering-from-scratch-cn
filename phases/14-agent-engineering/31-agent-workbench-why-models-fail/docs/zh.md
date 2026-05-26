# 智能体工作台（Agent Workbench）工程：为何能力强大的模型依然会失败

> 仅凭能力强大的模型是不够的。可靠的智能体（Agent）需要一个工作台（Workbench）：指令、状态、作用域、反馈、验证、审查和交接。剥离这些要素后，即便是前沿模型（Frontier Model）产出的成果也无法安全交付。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**前置条件：** 第 14 阶段 · 01（智能体循环 Agent Loop），第 14 阶段 · 26（失败模式 Failure Modes）
**耗时：** 约 45 分钟

## 学习目标

- 将模型能力与执行可靠性区分开来。
- 列举决定智能体能否交付的七个工作台层面（Workbench Surfaces）。
- 在小型代码库（Repository）任务中，对比仅使用提示词（Prompt）的运行与工作台引导的运行。
- 生成一份失败模式（Failure Mode）报告，将每个缺失的工作台层面与其引发的故障表现对应起来。

## 问题所在

你将一个前沿模型放入真实的代码库中，要求它添加输入验证。它打开了四个文件，编写了看似合理的代码，宣布成功并停止运行。你运行测试，结果两个失败。第三个被修改的文件与验证毫无关系。没有任何记录表明智能体做了哪些假设、首先尝试了什么，或者还有哪些工作未完成。

模型对 Python 的理解并没有错，错的是它对“工作”的认知。它根本不知道怎样才算完成、允许在哪些位置写入代码、哪些测试具有权威性，以及下一个会话该如何接续工作。

这不是模型缺陷，而是工作台缺陷。智能体周围的环境缺少了将一次性生成转化为可靠、可恢复工程实践的关键组件。

## 核心概念

工作台（Workbench）是任务执行期间封装模型（Model）的运行环境。它包含七个层面（Surfaces）：

| 层面 | 承载内容 | 缺失时的故障表现 |
|---------|-----------------|----------------------|
| 指令（Instructions） | 启动规则、禁止操作、完成定义 | 智能体（Agent）自行猜测“交付”的含义 |
| 状态（State） | 当前任务、已修改文件、阻塞项、下一步动作 | 每次会话都从零开始 |
| 作用域（Scope） | 允许访问的文件、禁止访问的文件、验收标准 | 修改泄漏到无关代码中 |
| 反馈（Feedback） | 捕获到循环中的真实命令输出 | 智能体在遇到 400 错误时仍宣告成功 |
| 验证（Verification） | 测试、代码检查（Lint）、冒烟测试、作用域检查 | “看起来没问题”的代码直接合入主分支 |
| 审查（Review） | 由不同角色进行的二次复核 | 构建者自己批改自己的作业 |
| 交接（Handoff） | 变更内容、变更原因、遗留事项 | 下一个会话重新摸索一切 |

工作台独立于模型。你可以更换模型而保留这些层面，但无法在替换层面的同时保持可靠性。

flowchart LR
  Task[Task] --> Scope[Scope Contract]
  Scope --> State[Repo Memory]
  State --> Agent[Agent Loop]
  Agent --> Feedback[Runtime Feedback]
  Feedback --> Verify[Verification Gate]
  Verify --> Review[Reviewer]
  Review --> Handoff[Handoff]
  Handoff --> State

循环的闭环基于状态文件，而非聊天记录。聊天内容是易失的，代码仓库才是事实记录系统（System of Record）。

### 工作台与提示词工程（Prompt Engineering）的对比

提示词告诉模型在当前轮次中你想要什么。工作台则告诉模型如何在多轮对话和跨会话中开展工作。大多数智能体失败的案例，本质上是工作台故障披上了提示词工程的外衣。

### 工作台与框架（Framework）的对比

框架为你提供运行时环境（Runtime）（如 LangGraph、AutoGen、Agents SDK）。工作台则为智能体在该运行时内部提供具体的工作场所。两者缺一不可。本迷你专题聚焦于后者。

### 从基础原语（Primitives）出发，而非供应商分类体系

目前关于“编排工程（Harness Engineering）”的论述层出不穷。Addy Osmani、OpenAI、Anthropic、LangChain、Martin Fowler、MongoDB、HumanLayer、Augment Code、Thoughtworks、walkinglabs 的 awesome 列表，以及 Medium 和 Hacker News 上持续不断的文章都在探讨这一概念。各方对于“harness”的边界、涵盖范围以及应使用的术语存在分歧。我们无需选边站队。这七个层面属于用户体验（UX）层；而在每个工作台之下，支撑任何可靠后端的都是同一套分布式系统（Distributed Systems）基础原语。

暂时抛开“智能体”这个标签。一次智能体运行本质上是跨越时间、进程和机器的计算过程。要使其可靠运行，你需要任何生产级系统都必需的基础原语。

| 原语（Primitive） | 定义 | 为智能体承载的内容 |
|-----------|------------|------------------------------|
| 函数（Function） | 类型化处理器。尽可能保持纯函数特性。拥有其输入和输出。 | 工具调用、规则检查、验证步骤、模型调用 |
| 工作进程（Worker） | 长期运行的进程，拥有一个或多个函数及其生命周期 | 构建者、审查者、验证器、MCP 服务器 |
| 触发器（Trigger） | 调用函数的事件源 | 智能体循环节拍、HTTP 请求、队列消息、定时任务（Cron）、文件变更、钩子（Hook） |
| 运行时（Runtime） | 决定在何处运行、使用何种超时和资源限制的边界 | Claude Code 的进程、LangGraph 的运行时、工作进程容器 |
| HTTP / RPC | 调用方与工作进程之间的通信链路 | 工具调用协议、MCP 请求、模型 API |
| 队列（Queue） | 触发器与工作进程之间的持久化缓冲区；支持背压（Back-pressure）、重试和幂等性（Idempotency） | 任务看板、反馈日志、审查收件箱 |
| 会话持久化（Session Persistence） | 在崩溃、重启或模型更换后仍能存活的状态 | `agent_state.json`、检查点（Checkpoints）、键值存储（KV Stores）、代码仓库本身 |
| 授权策略（Authorization Policy） | 谁可以在何种作用域下调用哪个函数 | 允许/禁止的文件、审批边界、MCP 能力列表 |

现在，将工作台的七个层面映射到这些原语上。

- **指令（Instructions）** — 策略（Policy）+ 函数元数据。规则即检查项（函数）。路由器（`AGENTS.md`）是附加在运行时启动阶段的策略。
- **状态（State）** — 会话持久化。运行时在每一步都会读取的键值存储。可以是文件、KV 或数据库；重要的是持久化语义，而非存储后端。
- **作用域（Scope）** — 针对每个任务的授权策略。允许/禁止的通配符模式（Globs）构成访问控制列表（ACL）。需要审批的环节构成权限格（Permission Lattice）。
- **反馈（Feedback）** — 写入队列的调用日志。每次 Shell 调用都是一条记录，具备持久化和可重放特性。
- **验证（Verification）** — 一个函数。对输入具有确定性。在任务关闭时触发。采用故障安全（Fails Closed）设计。
- **审查（Review）** — 一个独立的工作进程，对构建产物拥有只读授权，对审查报告拥有只写授权。
- **交接（Handoff）** — 由会话结束触发器发出的持久化记录。下一个会话的启动触发器会读取它。

智能体循环本身就是一个工作进程，它消费事件（用户消息、工具结果、定时器节拍）、调用函数（模型，随后是模型选择的工具）、写入记录（状态、反馈），并发出触发器（验证、审查、交接）。这并不神秘，其形态与作业处理器（Job Processor）完全一致。

### 流行模式的原语映射

每种流行的编排模式都可以归结为这八种原语。对照表如下。

| 供应商或社区模式 | 实际本质 |
|------------------------------|--------------------|
| Ralph Loop（Claude Code、Codex、agentic_harness 书籍）— 当智能体试图提前停止时，将原始意图重新注入全新的上下文窗口 | 一个触发器，以干净的上下文重新将任务入队；会话持久化负责延续目标 |
| 计划/执行/验证（PEV） | 三个工作进程，各司其职，通过状态和阶段间的队列进行通信 |
| 编排与计算分离（OpenAI Agents SDK，2026年4月）— 将控制面与执行面拆分 | 控制面/数据面（Control-plane / Data-plane）架构的重述。该概念比“智能体”标签早出现数十年 |
| 开放智能体通行证（OAP，2026年3月）— 在执行前根据声明式策略对每次工具调用进行签名和审计 | 由前置动作工作进程强制执行的授权策略，附带带签名的审计队列 |
| 指南与传感器（Birgitta Böckeler / Thoughtworks）— 前馈规则 + 反馈可观测性 | 授权策略 + 验证函数 + 可观测性追踪（Observability Traces） |
| 渐进式压缩，5阶段（Claude Code 逆向工程，2026年4月） | 一个状态管理工作进程，以类似定时任务的方式在会话持久化上运行，以将其控制在预算范围内 |
| 钩子/中间件（LangChain、Claude Code）— 拦截模型和工具调用 | 包裹在运行时调用路径周围的触发器 + 函数 |
| 渐进式披露的 Markdown 技能（Anthropic、Flue） | 一个函数注册表，其中函数元数据按需（Just-in-Time）加载到上下文中 |
| 沙盒智能体（Codex、Sandcastle、Vercel Sandbox） | 计算面（Compute Plane）：具备隔离文件系统、网络和生命周期的运行时 |
| MCP 服务器 | 通过稳定 RPC 暴露函数的工作进程，以能力列表作为授权依据 |

表中的每一项，都是智能体社区重新发现了一个在分布式系统中早已命名的原语，并为其赋予了新名称。这些标签对营销很有用，但作为工程术语则毫无价值。

### 实际数据说明了什么

“编排优于模型”的主张如今已有数据支撑。了解这些数据很有必要，因为它们也是反驳“只需等待更聪明的模型”这一论调的唯一诚实依据。

- Terminal Bench 2.0 — 模型不变，仅更改编排方式，就将一个编程智能体从 30 名开外提升至第五名（LangChain，《Anatomy of an Agent Harness》）。
- Vercel — 删除了智能体 80% 的工具；成功率从 80% 跃升至 100%（MongoDB）。
- Harvey — 仅通过优化编排，法律智能体的准确率就翻了一倍多（MongoDB）。
- 88% 的企业级 AI 智能体项目未能投入生产。失败原因主要集中在运行时层面，而非推理能力（preprints.org，《Harness Engineering for Language Agents》，2026年3月）。
- 一项针对三个主流开源框架的 2025 年基准测试报告显示，任务完成率约为 50%；在长上下文条件下，WebAgent 的完成率从 40-50% 暴跌至 10% 以下，主要原因在于无限循环和目标丢失（2026 年初的多篇分析文章均有广泛报道）。

结论并非“编排永远胜出”。模型确实会随着时间的推移吸收这些编排技巧。真正的结论是：当下，承担核心负荷的工程设计位于模型之外而非模型之内，而承载这些负荷的原语，正是任何生产级系统一直以来所必需的。

### 供应商论述的局限之处

这部分内容无需客气。

- LangChain 的《Anatomy of an Agent Harness》列举了十一个组件——提示词、工具、钩子、沙盒、编排、记忆、技能、子智能体以及运行时的“傻瓜循环”。它未提及队列、作为部署单元的工作进程、触发器语义、作为独立关注点的会话持久化，或授权策略。它将编排视为一个可配置的对象，而非一个可部署的系统。
- Addy Osmani 的《Agent Harness Engineering》确立了 `Agent = Model + Harness` 的框架和棘轮模式（Ratchet Pattern），但未阐明编排究竟由什么构成。读起来更像是一种立场声明，而非技术规范。
- Anthropic 和 OpenAI 在层面设计上挖掘最深，但始终局限于各自的运行时内部。2026年4月 Agents SDK 中宣布的“编排与计算分离”，是首个明确支持控制面/数据面拆分的供应商文档。这是一个基础原语概念，而非新发明。
- 《agentic_harness》一书将编排视为配置对象（Jaymin West 的《Agentic Engineering》第 6 章），书中最有力的一句话是“编排是智能体系统中的主要安全边界”。这不过是授权策略的另一种说法。
- Hacker News 上的讨论也反复指向同一结论。2026年4月的帖子《The agent harness belongs outside the sandbox》主张，编排应“更像一个位于一切之外的虚拟机管理程序（Hypervisor），根据上下文和用户授权访问”。这再次印证了授权策略应作为独立平面存在。

你无需反对上述任何观点，也能看出其中的差距。他们只是在描述一个已存在系统的用户体验层面。而我们是在构建系统本身。当系统构建得当时，七个层面会自然地从原语中衍生出来。当系统构建不当时，无论怎样打磨 `AGENTS.md` 也无法弥补缺失的队列。

因此，当你在别处听到“编排工程”时，请将其翻译为基础原语。提示词和规则即策略与函数。脚手架即运行时。护栏即授权与验证。钩子即触发器。记忆即会话持久化。Ralph Loop 即重新入队。子智能体即工作进程。沙盒即计算面。词汇在变，工程本质不变。工作台是面向智能体的用户体验层；而能在下一次供应商概念重构中存活下来的“编排”，本质上就是正确连接在一起的函数、工作进程、触发器、运行时、队列、持久化和策略。

## 构建它

`code/main.py` 会两次运行一个小型仓库任务。第一次仅使用提示词（prompt），第二次则接入七个交互面（Surface）。模型与任务均保持不变。该脚本会统计失败运行中缺失了哪些交互面，并输出一份故障模式（failure-mode）报告。

该仓库任务特意设计得很小：为一个单文件的 FastAPI 风格处理器添加输入验证，并编写一个能通过的测试。

运行方式：

python3 code/main.py

输出结果：两次运行的并排日志、总结仅提示词运行情况的 `failure_modes.json` 文件，以及针对工作台（workbench）运行的一行结论。

这里的代理（agent）只是一个基于规则的微型存根（stub）；重点在于交互面，而非模型。在本迷你课程的后续部分，你将把每个交互面重构为真实、可复用的工件（artifact）。

## 实际应用

在现实世界中，工作台交互面已经存在于三个地方，即使人们并不这么称呼它们：

- **Claude Code、Codex、Cursor。** `AGENTS.md` 和 `CLAUDE.md` 是指令面（instructions surface）。斜杠命令（slash commands）界定作用域（scope）。钩子（hooks）用于验证（verification）。
- **LangGraph、OpenAI Agents SDK。** 检查点（checkpoints）和会话存储（session stores）构成状态面（state surface）。交接（handoffs）即交接面（handoff surface）。
- **真实仓库上的 CI（持续集成）。** 测试、代码检查（lint）和类型检查（type-check）属于验证环节。PR 模板用于交接。CODEOWNERS 文件负责审查（review）。

工作台工程（Workbench engineering）是一门将这些交互面显式化并实现可复用的学科，而不是让每个团队都去重新摸索它们。

## 交付上线

`outputs/skill-workbench-audit.md` 是一个可移植的技能（skill），用于审计现有仓库中的七个工作台交互面，并报告哪些缺失、哪些不完整、哪些状态良好。将其放置在任意代理配置旁，它就能告诉你优先修复什么。

## 练习

1. 选择一个你已经在运行代理的仓库。为七个交互面打分，从 0（缺失）到 2（状态良好）。你最薄弱的交互面是哪个？
2. 扩展 `main.py`，使仅提示词的运行也能生成虚假的“成功”声明。验证验证关卡（verification gate）本应能捕获它。
3. 为你自己的产品添加第八个交互面。论证它为何不会归并到现有的七个交互面中。
4. 使用另一个会产生幻觉（hallucinates）出额外文件写入操作的存根代理重新运行脚本。哪个交互面会最先捕获它？
5. 将第 14 阶段 · 26 中提到的五种行业常见故障模式映射到这七个交互面上。每个交互面旨在吸收哪种模式？

## 关键术语

| 术语 | 通常的说法 | 实际含义 |
|------|----------------|------------------------|
| 工作台（Workbench） | “环境配置” | 围绕模型构建的工程化交互面，用于确保工作可靠 |
| 交互面（Surface） | “一份文档”或“一个脚本” | 代理在每一轮对话中读取或写入的具名、机器可读输入 |
| 记录系统（System of record） | “笔记” | 当聊天记录丢失时，代理视为事实依据的文件 |
| 完成定义（Definition of done） | “验收标准” | 代理无法伪造的、基于文件的客观检查清单 |
| 工作台审计（Workbench audit） | “仓库就绪检查” | 在开始工作前遍历七个交互面，标记缺失组件的过程 |

## 延伸阅读

将这些内容视为参考数据点，而非权威定论。每一篇都只提供了局部的分类体系。在决定是否采纳某个概念之前，请先将其还原为底层原语（primitive）（如函数（function）、工作进程（worker）、触发器（trigger）、运行时（runtime）、HTTP/RPC、队列（queue）、持久化（persistence）、策略（policy））。

厂商视角：

- [Addy Osmani, Agent Harness Engineering](https://addyosmani.com/blog/agent-harness-engineering/) — `Agent = Model + Harness` 与棘轮模式（ratchet pattern）；对基础设施的论述较少
- [LangChain, The Anatomy of an Agent Harness](https://blog.langchain.com/the-anatomy-of-an-agent-harness/) — 包含十一个组件：提示词（prompts）、工具（tools）、钩子（hooks）、编排（orchestration）、沙箱（sandboxes）、记忆（memory）、技能（skills）、子智能体（subagents）、运行时（runtime）；未涵盖队列、部署与授权（authz）
- [OpenAI, Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/) — Codex 团队对其运行时周边接口层面的看法
- [OpenAI, Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/) — 将智能体循环（agent loop）简化为基于函数调用的 `while` 循环
- [Anthropic, Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) — 特定运行时内面向长周期任务的接口设计
- [Anthropic, Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps) — 实战设计笔记
- [LangChain Deep Agents harness capabilities](https://docs.langchain.com/oss/python/deepagents/harness) — 运行时配置接口

具备可操作细节的实战文章：

- [Martin Fowler / Birgitta Böckeler, Harness engineering for coding agent users](https://martinfowler.com/articles/harness-engineering.html) — 指南（前馈 feedforward）+ 传感器（反馈 feedback）；最清晰的控制理论（control theory）框架
- [HumanLayer, Skill Issue: Harness Engineering for Coding Agents](https://www.humanlayer.dev/blog/skill-issue-harness-engineering-for-coding-agents) — “这不是模型问题，而是配置问题”
- [MongoDB, The Agent Harness: Why the LLM Is the Smallest Part of Your Agent System](https://www.mongodb.com/company/blog/technical/agent-harness-why-llm-is-smallest-part-of-your-agent-system) — 实际成效：Vercel 从 80% 提升至 100%，Harvey 准确率翻倍，Terminal Bench 排名从 Top 30 跃升至 Top 5
- [Augment Code, Harness Engineering for AI Coding Agents](https://www.augmentcode.com/guides/harness-engineering-ai-coding-agents) — 以约束优先的逐步解析
- [Sequoia podcast, Harrison Chase on Context Engineering Long-Horizon Agents](https://sequoiacap.com/podcast/context-engineering-our-way-to-long-horizon-agents-langchains-harrison-chase/) — 强调运行时考量优于模型考量

书籍、论文与参考实现：

- [Jaymin West, Agentic Engineering — Chapter 6: Harnesses](https://www.jayminwest.com/agentic-engineering-book/6-harnesses) — 专著级论述，将控制框架（Harness）视为核心安全边界
- [preprints.org, Harness Engineering for Language Agents (March 2026)](https://www.preprints.org/manuscript/202603.1756) — 学术视角，将其框架化为控制 / 代理能力 / 运行时
- [walkinglabs/awesome-harness-engineering](https://github.com/walkinglabs/awesome-harness-engineering) — 精选阅读清单，涵盖上下文、评估、可观测性（observability）、编排
- [ai-boost/awesome-harness-engineering](https://github.com/ai-boost/awesome-harness-engineering) — 另一份精选清单（工具、评估、记忆、MCP、权限）
- [andrewgarst/agentic_harness](https://github.com/andrewgarst/agentic_harness) — 具备生产就绪能力的参考实现，内置基于 Redis 的记忆模块与评估套件
- [HKUDS/OpenHarness](https://github.com/HKUDS/OpenHarness) — 开源智能体控制框架，内置个人智能体

值得阅读的 Hacker News 讨论串（重点在于分歧而非共识）：

- [HN: Effective harnesses for long-running agents](https://news.ycombinator.com/item?id=46081704)
- [HN: Improving 15 LLMs at Coding in One Afternoon. Only the Harness Changed](https://news.ycombinator.com/item?id=46988596)
- [HN: The agent harness belongs outside the sandbox](https://news.ycombinator.com/item?id=47990675) — 主张将授权（authorization）作为独立的控制平面

本课程内的交叉引用：

- Phase 14 · 23 — OpenTelemetry GenAI 规范：传感器文献所指向的可观测性层
- Phase 14 · 26 — 故障模式目录：七个接口层面旨在吸收的故障类型
- Phase 14 · 27 — 提示词注入（prompt injection）防御：位于授权策略原语层面
- Phase 14 · 29 — 生产运行时（队列 queue、事件 event、定时任务 cron）：本课所述原语在部署环境中的实际载体