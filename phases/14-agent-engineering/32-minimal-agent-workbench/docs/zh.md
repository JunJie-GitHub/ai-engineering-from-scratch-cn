# 最小化智能体工作台（Minimal Agent Workbench）

> 最实用的工作台（Workbench）仅需三个文件：根指令路由（Root Instructions Router）、状态文件（State File）和任务看板（Task Board）。其他所有内容都构建在这三者之上。如果一个代码仓库（Repo）连这三个文件都承载不了，任何模型也救不了它。

**类型：** 构建
**语言：** Python (stdlib)
**前置条件：** 第 14 阶段 · 31（为何能力强大的模型仍会失败）
**耗时：** 约 45 分钟

## 学习目标

- 明确构成最小可行工作台（Minimum Viable Workbench）的三个核心文件。
- 解释为何简短的根路由文件优于冗长且单一的 `AGENTS.md`。
- 构建一个状态文件，使智能体（Agent）能在每次决策时读取，并在操作结束时写入。
- 构建一个任务看板，使其能够在不依赖聊天历史的情况下，跨多个会话（Session）持续工作。

## 问题背景

大多数团队在搭建工作台时，往往会编写一份长达 3000 行的 `AGENTS.md` 便草草了事。模型加载该文件后，会忽略无法总结的部分，并且依然会在那些它一贯容易出错的环节失败。

你需要反其道而行之。一个精简的根文件，仅在相关时将智能体路由至更深层的文件。一份持久化的状态，供智能体在行动前读取、行动后写入。一个任务看板，清晰标明哪些任务正在进行、哪些受阻、以及下一步该做什么。

三个文件，各司其职。每一个都具备足够的机器可读性，以便日后演进为真正的系统。

## 核心概念

flowchart LR
  Agent[Agent Loop] --> Router[AGENTS.md]
  Router --> State[agent_state.json]
  Router --> Board[task_board.json]
  State --> Agent
  Board --> Agent

### AGENTS.md 是路由器，而非操作手册

一份优秀的 `AGENTS.md` 应当简短。它负责将智能体指向：

- 状态文件（当前所处位置）。
- 任务看板（剩余工作）。
- 更深层的规则（位于 `docs/agent-rules.md` 下）。
- 验证命令（如何确认工作已生效）。

任何更长的内容都应放入更深层的文档中，仅在需要时加载。冗长的手册会被忽略，而简短的路由器则会被遵循。

### agent_state.json 是单一事实来源（System of Record）

状态文件承载着：当前活跃的任务 ID、已修改的文件、做出的假设、阻塞项以及下一步行动。智能体在每次决策轮次中都会读取它。下一个会话将直接读取该文件，而非重放聊天记录。

状态必须保存在文件中，因为聊天历史并不可靠。会话会终止，对话会被截断，但文件不会。

### task_board.json 是任务队列

任务看板记录着每个任务的状态：`todo` | `in_progress` | `done` | `blocked`。当状态为空时，智能体将从该队列中拉取任务；当你想了解智能体是否按计划推进时，也需查阅此队列。

看板上的每个任务都包含 ID、目标、负责人（`builder`、`reviewer` 或 `human`）以及验收标准。看板刻意保持精简：当它大到超出单屏显示时，说明你面临的是规划问题，而非看板本身的问题。

### 三个文件是底线，而非上限

后续课程将引入范围契约（Scope Contracts）、反馈运行器（Feedback Runners）、验证关卡（Verification Gates）、审查清单（Reviewer Checklists）以及交接包（Handoff Packets）。而此处的三个文件，正是所有这些高级功能所依赖的基础。

## 动手构建

`code/main.py` 将最小化工作台（workbench）写入一个空仓库，并演示了智能体（agent）的单次交互轮次（turn），该轮次会：

1. 读取 `agent_state.json`。
2. 若状态为空，则从 `task_board.json` 中拉取下一个任务。
3. 在作用域（scope）内创建或更新单个文件。
4. 回写更新后的状态。

运行方式：

python3 code/main.py

该脚本会在自身同级目录下创建 `workdir/`，生成上述三个文件，执行一轮交互，并打印差异（diff）。重新运行该脚本，即可观察第二轮交互如何接续第一轮的结果继续执行。

## 实际应用

在生产级智能体（agent）产品中，同样的三个文件会以不同的名称出现：

- **Claude Code：** 使用 `AGENTS.md` 或 `CLAUDE.md` 作为路由（router）配置，采用 `.claude/state.json` 风格的存储来保存状态，并通过钩子（hooks）管理任务板。
- **Codex / Cursor：** 使用工作区规则（workspace rules）作为路由，会话内存（session memory）用于状态存储，聊天侧边栏中的排队任务充当任务板。
- **自定义 Python 智能体：** 就是你刚刚编写的相同文件。

名称虽变，底层结构（shape）不变。

## 实际生产环境中的模式

当在此最小化工作台之上叠加三种模式时，它便能经受住真实单体仓库（monorepo）的考验。这三种模式相互独立；请根据仓库的实际需求进行选择。

**采用“就近优先”原则的嵌套 `AGENTS.md`。** OpenAI 在其主仓库中发布了 88 个 `AGENTS.md` 文件，每个子组件对应一个。Codex、Cursor、Claude Code 和 Copilot 都会从当前工作文件向仓库根目录回溯，并将沿途找到的所有 `AGENTS.md` 拼接起来。子目录中的文件会扩展根目录文件的内容。Codex 额外引入了 `AGENTS.override.md` 用于替换而非扩展；该覆盖机制是 Codex 特有的，在跨工具协作时应避免使用。Augment Code 的实测数据点明了关键所在：编写得当的 `AGENTS.md` 文件能带来相当于从 Haiku 模型升级到 Opus 模型的质量飞跃；而编写糟糕的文件甚至会导致输出质量不如完全没有该文件。

**必须摒弃的反模式（anti-patterns），即便它们看似提供了全面覆盖。** 相互冲突的指令会悄无声息地将智能体从交互模式降级为贪婪模式（greedy mode）（ICLR 2026 AMBIG-SWE 论文显示：问题解决率从 48.8% 降至 28%）；请使用数字明确优先级，而非平铺直叙地堆叠指令。缺乏强制执行命令的不可验证风格规范（例如“遵循 Google Python 风格指南”）会让智能体自行编造合规行为；每条风格规范都应搭配具体的代码检查（lint）命令。将风格规范置于命令之前会掩盖验证路径；应遵循“命令优先，风格置后”的原则。为人类而非智能体编写文档会浪费上下文预算（context budget）；简洁本身就是一项核心特性。

**跨工具符号链接（symlinks）。** 通过单一根文件配合符号链接（`ln -s AGENTS.md CLAUDE.md`、`ln -s AGENTS.md .github/copilot-instructions.md`、`ln -s AGENTS.md .cursorrules`），可确保所有编程智能体共享同一事实来源（source of truth）。Nx 的 `nx ai-setup` 命令可通过单一配置文件，自动在 Claude Code、Cursor、Copilot、Gemini、Codex 和 OpenCode 之间完成此设置。

## 交付上线

`outputs/skill-minimal-workbench.md` 会为任何新仓库生成包含三个文件的工作台（workbench）：一个针对项目定制的路由文件（router）`AGENTS.md`、包含正确键值的 `agent_state.json`，以及预置了当前待办列表（backlog）的 `task_board.json`。

## 练习

1. 在 `agent_state.json` 中添加 `last_run` 时间戳。除非操作员（operator）确认，否则若文件超过 24 小时未更新，则拒绝运行。
2. 在任务板中添加 `priority` 字段，并修改拉取器（puller）逻辑，使其始终优先选择优先级最高的 `todo` 任务。
3. 将 `task_board.json` 迁移为 JSON Lines 格式，使每个任务独占一行，以便在版本控制中生成清晰的差异对比（diffs）。
4. 编写 `lint_workbench.py` 脚本，当 `AGENTS.md` 超过 80 行或引用了不存在的文件时，脚本执行失败。
5. 评估这三个文件中丢失哪一个造成的损失最大，并阐述理由。

## 关键术语

| 术语 | 常见叫法 | 实际含义 |
|------|----------|----------|
| 路由文件（Router） | `AGENTS.md` | 简短的根文件，用于指引智能体（agent）访问更深层的文档与文件 |
| 状态文件（State file） | “笔记” | 机器可读的记录，用于标记智能体当前进度，每轮交互均会更新 |
| 任务板（Task board） | “待办列表” | 包含状态、负责人及验收条件的 JSON 任务队列 |
| 记录系统（System of record） | “事实来源” | 当聊天记录丢失时，工作台将其视为权威依据的文件 |

## 延伸阅读

- [agents.md — the open spec](https://agents.md/) — 已被 Cursor、Codex、Claude Code、Copilot、Gemini、OpenCode 采用
- [Augment Code, A good AGENTS.md is a model upgrade. A bad one is worse than no docs at all](https://www.augmentcode.com/blog/how-to-write-good-agents-dot-md-files) — 经实测的质量提升数据
- [Blake Crosley, AGENTS.md Patterns: What Actually Changes Agent Behavior](https://blakecrosley.com/blog/agents-md-patterns) — 经验证有效的模式与无效的模式
- [Datadog Frontend, Steering AI Agents in Monorepos with AGENTS.md](https://dev.to/datadog-frontend-dev/steering-ai-agents-in-monorepos-with-agentsmd-13g0) — 实践中的嵌套优先级规则
- [Nx Blog, Teach Your AI Agent How to Work in a Monorepo](https://nx.dev/blog/nx-ai-agent-skills) — 跨六款工具的单一源文件生成方案
- [The Prompt Shelf, AGENTS.md Best Practices: Structure, Scope, and Real Examples](https://thepromptshelf.dev/blog/agents-md-best-practices/) — 经得起评审的章节排序结构
- [Anthropic, Claude Code subagents and session store](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/sub-agents)
- Phase 14 · 31 — 该最小化配置所能吸收的故障模式（failure modes）
- Phase 14 · 34 — 本课所预览的持久化状态模式（state schema）