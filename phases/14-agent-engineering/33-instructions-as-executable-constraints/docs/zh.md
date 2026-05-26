# 将智能体（Agent）指令转化为可执行约束

> 以自然语言编写的指令仅是愿望，而以约束形式编写的指令则是测试。工作台（Workbench）会将每条规则转化为智能体可在运行时检查、且审查者事后可验证的内容。

**类型：** 构建
**语言：** Python (stdlib)
**前置条件：** 第 14 阶段 · 32（最小化工作台）
**耗时：** 约 50 分钟

## 学习目标

- 将路由说明（Routing Prose）与操作规则（Operational Rules）分离。
- 将启动规则、禁止操作、完成定义（Definition of Done）、不确定性处理及审批边界表达为机器可检查的约束。
- 实现一个规则检查器（Rule Checker），用于根据规则集对每次运行进行评分。
- 使规则集易于进行差异对比（Diff-friendly），以便审查者清晰查看变更内容。

## 问题背景

典型的 `AGENTS.md` 读起来就像入职培训文档。它告诉智能体要“小心谨慎”、“充分测试”以及“不确定时提问”。然而三天后，智能体却提交了没有任何测试的变更，写入了禁止访问的目录，并且从未提问，因为它根本不知道界限在哪里。

当指令具备可操作性时，它们才真正强大；若仅停留在愿景层面，则显得苍白无力。解决之道在于编写工作台可解析、审查者可评分的规则。

## 核心概念

规则应置于 `docs/agent-rules.md` 文件中，与简短的根路由 (root router) 保持分离。每条规则均包含名称、类别和检查逻辑 (check)。

flowchart LR
  Router[AGENTS.md] --> Rules[docs/agent-rules.md]
  Rules --> Checker[rule_checker.py]
  Checker --> Report[rule_report.json]
  Report --> Reviewer[Reviewer]

### 涵盖大多数规则的五大类别

| 类别 | 规则解答的问题 | 示例 |
|----------|---------------------------|---------|
| 启动条件 (Startup) | 工作开始前必须满足什么条件？ | “状态文件存在且为最新” |
| 禁止事项 (Forbidden) | 绝对不允许发生什么？ | “禁止编辑 `scripts/release.sh`” |
| 完成定义 (Definition of done) | 如何证明任务已完成？ | “pytest 退出码为 0 且验收测试通过” |
| 不确定性处理 (Uncertainty) | 智能体 (agent) 不确定时该怎么做？ | “提出疑问备注而非盲目猜测” |
| 审批要求 (Approval) | 哪些操作需要人工审批？ | “任何新增依赖项、任何生产环境写入” |

若某条规则无法归入上述五类之一，通常意味着它应拆分为两条规则。请强制进行拆分。

### 规则具备机器可读性

每条规则包含一个标识符 (slug)、类别、单行描述，以及一个指向 `rule_checker.py` 中具体函数的 `check` 字段。添加规则即意味着添加对应的检查逻辑；检查器 (checker) 将随工作台 (workbench) 的扩展而不断演进。

### 规则便于版本差异对比

规则在单个 Markdown 文件中以标题为单位逐条存放。重命名操作在差异对比 (diff) 中清晰可见。新增规则置于对应类别的顶部。过时的规则应直接删除而非注释掉，因为工作台才是唯一事实来源 (source of truth)，而非记录团队上个季度想法的聊天日志。

### 规则与框架护栏的对比

框架护栏 (framework guardrails)（如 OpenAI Agents SDK 护栏、LangGraph 中断机制）在运行时 (runtime) 层面强制执行规则。本课程中的规则集则是这些护栏所实现的、可供人类阅读与审查的契约。两者缺一不可：运行时负责在单轮交互 (turn) 中捕获违规行为，而规则集则用于验证运行时是否在执行正确的操作。

## 动手构建

`code/main.py` 包含以下内容：

- 用于解析 `agent-rules.md` 并将规则加载至数据类 (dataclass) 的解析器。
- `rule_checker.py` 中的检查函数，每个 `check` 引用对应一个函数。
- 一个演示智能体运行流程，其中故意违反两条规则，并通过检查流程将其捕获。

运行方式：

python3 code/main.py

输出结果：解析后的规则集、运行轨迹 (run trace)、每条规则的通过/失败状态，以及保存在脚本同目录下的 `rule_report.json` 文件。

## 实际生产环境中的模式

有三种模式决定了规则集是能持续一个季度，还是在一周内迅速失效。

**编写时标记严重级别（Severity Tagging）。** 每条规则都携带 `severity` 字段：`block`、`warn` 或 `info`。检查器（Checker）会报告所有三种级别；运行时（Runtime）仅在 `block` 级别时拒绝执行。大多数团队在初期会夸大严重级别，然后在交付期限压力下悄悄降低它；在编写时标记严重级别能强制在前期完成校准。将其与验证关卡（Verification Gate）（第 14 阶段 · 38）配合使用，该关卡会将任何对 `block` 规则的覆盖操作签名记录到 `overrides.jsonl` 审计日志中。

**将规则过期作为强制机制（Forcing Function）。** 每条规则都带有 `expires_at` 日期（默认为编写后 90 天）。当一条未过期的规则连续 60 天零违规时，检查器会发出警告；在下一次季度评审中，团队需决定是保留该规则、将其降级为 `info`，还是直接删除。Cloudflare 的生产环境 AI 代码审查数据（2026 年 4 月，30 天内跨 5,169 个仓库运行了 131,246 次审查）显示，设置了明确过期时间的规则集每个仓库的规则数保持在 30 条以内；而未设置的规则集则膨胀至 80 条以上，且其中大多数从未触发过。

**以 Markdown 为源文件，以 JSON 为缓存。** `agent-rules.md` 是人工编写的源文件；`agent-rules.lock.json` 是检查器在热路径（Hot Path）中读取的缓存文件。该锁文件由提交前钩子（Pre-commit Hook）重新生成。Markdown 的差异（Diff）便于审查；JSON 解析则被排除在每次交互轮次（Turn）之外。其结构与 `package.json` / `package-lock.json` 以及 `Cargo.toml` / `Cargo.lock` 相同。

## 实际应用

在生产环境中：

- Claude Code、Codex 和 Cursor 会在会话开始时读取规则，并在拒绝执行操作时引用它们。检查器会在持续集成（CI）中重新运行这些规则，以捕获静默漂移（Silent Drift）。
- OpenAI Agents SDK 的护栏（Guardrails）将相同的检查注册为输入和输出护栏。Markdown 是文档层面的载体，而 SDK 是运行时的载体。
- 当运行中的节点违反规则时，LangGraph 的中断机制（Interrupts）会被触发。中断处理程序会读取该规则，向人类请求确认，然后恢复执行。

该规则集之所以能在上述三者之间无缝移植，是因为它本质上只是 Markdown 文本加上函数名称。

## 交付使用

`outputs/skill-rule-set-builder.md` 会访谈项目负责人，将其现有的自然语言指令分类到五个类别中，并输出版本化的 `agent-rules.md` 以及一个检查器骨架代码（Checker Stub）。

## 练习

1. 如果你的产品确实需要，可以添加第六个类别。请论证它为何不会归并到现有的五个类别中。
2. 扩展检查器功能，使规则能够携带严重级别（`block`、`warn`、`info`），并让报告据此进行聚合。
3. 将检查器接入 CI：如果最新一次智能体（Agent）运行中触发了 `block` 级别的规则失败，则使构建失败。
4. 为每条规则添加“过期（Expiry）”字段。若规则在 90 天内未触发检查失败，则将其列入待评审状态。
5. 找一个真实的 `AGENTS.md` 文件，并将其重写为五类规则。其中有多少行是具备可操作性的？有多少行仅是理想化的描述？

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------------|------------------------|
| 运行规则 (Operational rule) | “一条真正的指令” | 工作台 (workbench) 可在运行时进行检查的规则 |
| 倡导型规则 (Aspirational rule) | “多加小心” | 无检查机制的规则；应予以删除或升级 |
| 完成定义 (Definition of Done) | “验收通过” | 以文件为客观依据的任务完成证明 |
| 阻断级别 (Block severity) | “硬性规定” | 违反该规则将中止运行；未经操作员干预无法静默处理 |
| 规则过期 (Rule expiry) | “陈旧规则清理” | 连续 N 天未触发失败的规则将被列入待退役名单 |

## 延伸阅读

- [OpenAI Agents SDK 护栏 (guardrails)](https://platform.openai.com/docs/guides/agents-sdk/guardrails)
- [LangGraph 中断机制 (interrupts)](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [Anthropic，构建高效智能体 (Building Effective Agents)](https://www.anthropic.com/research/building-effective-agents)
- [Rick Hightower，Agent RuleZ：一种确定性策略引擎](https://medium.com/@richardhightower/agent-rulez-a-deterministic-policy-engine-for-ai-coding-agents-9489e0561edf) — 生产环境中的阻断/警告/提示级别 (block/warn/info severity)
- [Cloudflare，大规模编排 AI 代码审查](https://blog.cloudflare.com/ai-code-review/) — 13.1 万次审查运行经验与规则组合实践
- [microservices.io，GenAI 开发平台 — 第一部分：护栏](https://microservices.io/post/architecture/2026/03/09/genai-development-platform-part-1-development-guardrails.html) — 规则与持续集成 (CI) 之间的纵深防御 (defense in depth)
- [类型检查合规性：确定性护栏 (Type-Checked Compliance: Deterministic Guardrails) (arXiv 2604.01483)](https://arxiv.org/pdf/2604.01483) — 以 Lean 4 作为“规则即检查 (rule-as-check)”的上限
- [logi-cmd/agent-guardrails](https://github.com/logi-cmd/agent-guardrails) — 合并门禁 (merge-gate) 实现：作用域、变异测试与违规配额 (violation budgets)
- 第 14 阶段 · 32 — 该规则集可接入的最小工作台
- 第 14 阶段 · 38 — 消费规则报告的验证门禁 (verification gate)
- 第 14 阶段 · 39 — 对规则合规性进行评分的审查智能体 (reviewer agent)