# 智能体 (Agent) 初始化脚本

> 每次冷启动 (Cold Start) 的会话都会产生额外的开销。智能体不得不重复读取相同的文件、重试相同的探测，并重新摸索相同的路径。初始化脚本只需一次性完成这些工作，并将结果持久化 (Persist) 到状态中。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** Phase 14 · 32 (Minimal Workbench), Phase 14 · 34 (Repo Memory)
**耗时：** 约45分钟

## 学习目标

- 明确智能体在每次会话中绝不应重复执行的工作。
- 构建一个确定性 (Deterministic) 的初始化脚本，用于探测运行时环境、依赖项及仓库健康状况。
- 持久化探测结果，使智能体直接读取该结果而非重新执行检查。
- 在初始化失败时，实现快速、显式的报错，并提供统一的排查入口。

## 问题背景

开启一个会话。智能体需要猜测 Python 版本，猜测测试命令，为了找到入口点反复列出仓库根目录五次，尝试导入未安装的包，并询问用户配置文件的位置。等到它真正开始进行代码修改时，已经消耗了上万 token 在那些本应由单个脚本完成的初始化工作上。

解决方案是编写一个初始化脚本，在智能体执行任何其他操作之前运行，并生成一个 `init_report.json` 文件供智能体在启动时读取。

## 核心概念

flowchart TD
  Start[Session Start] --> Init[init_agent.py]
  Init --> Probes[probe runtime / deps / paths / env / tests]
  Probes --> Report[init_report.json]
  Report --> Decision{healthy?}
  Decision -- yes --> Agent[Agent Loop]
  Decision -- no --> Halt[fail loud, halt, surface to human]

### 初始化脚本的探测内容

| 探测项 | 重要性 |
|-------|----------------|
| 运行时版本 | Python 或 Node 版本错误会导致隐蔽的版本不兼容 Bug |
| 依赖项可用性 | 后期发现缺失依赖的修复成本是现在提前捕获的十倍 |
| 测试命令 | 智能体必须知晓如何验证；若命令缺失，则工作台已处于损坏状态 |
| 仓库路径 | 硬编码路径容易漂移失效；需一次性解析并固定 |
| 环境变量 | 缺失 `OPENAI_API_KEY` 是明确的故障面，而非难以排查的运行时问题 |
| 状态与看板新鲜度 | 崩溃会话遗留的陈旧状态极易引发隐患 |
| 最近已知良好提交 | 作为会话结束时交接差异 (Diff) 的基准锚点 |

### 快速失败、显式报错、统一排查

任何探测失败都意味着立即中止并向人类开发者报告。绝不存在“让智能体自己搞定”的情况。初始化脚本的核心目的就是在工作台损坏时拒绝启动。

### 幂等性 (Idempotent)

连续运行两次。第二次运行除了更新时间戳外，应不产生任何实际副作用 (No-op)。正是这种幂等性使得你可以将该脚本无缝集成到持续集成 (CI)、Git 钩子或任务前置斜杠命令中。

### 初始化脚本与启动规则的区别

规则（Phase 14 · 33）描述了执行操作前必须满足的条件。而初始化脚本则是确保这些规则能够被实际验证的工具。没有初始化脚本的规则只会沦为一句“小心点”；而没有规则的初始化脚本则只会生成一份精美的失败报告。

## 开始构建

`code/main.py` 实现了 `init_agent.py`：

- 五项探针（probes）：Python 版本、通过 `importlib.util.find_spec` 验证的依赖项、测试命令的可解析性、必需的环境变量、状态文件的新鲜度。
- 每个探针返回 `(name, status, detail)`。
- 脚本会将完整的探针结果写入 `init_report.json`，若任何阻塞级（block-severity）探针失败，则以非零状态码退出。

运行方式：

python3 code/main.py

该脚本会打印探针结果表，生成 `init_report.json`。若一切顺利（happy path），则以零状态码退出；若出现失败，则以非零状态码退出并列出失败的探针。

## 实际生产环境中的模式

以下三种模式能将真正实用的初始化脚本与流于形式的“过场”（ceremony）区分开来。

**最后已知良好提交锚定（Last-known-good commit anchoring）。** 将当前提交与上次成功合并时生成的 `LKG` 文件进行比对。若代码差异超出预设阈值（默认 50 个文件），则拒绝启动，并要求人工审批新基线。Cloudflare 的 AI 代码审查（AI Code Review）正是采用此模式来限定审查代理（reviewer agents）的作用范围：每次审查会话均锚定于同一个最后已知良好状态，从而避免偏差在多次会话中不断累积。

**带生存时间（TTL）的锁文件（Lock files with TTL）。** 在首次成功通过探针检查后，生成 `prereqs.lock` 文件。后续运行在 N 小时（默认 24 小时）内将信任该锁文件，并跳过高开销的探针检查。初始化脚本会优先读取该锁文件；若文件未过期且依赖清单的哈希值匹配，则直接短路（short-circuits）跳过后续检查。这与 Docker 用于镜像层缓存的模式一致：幂等探针 + 内容哈希 = 跳过。

**热路径（hot path）中无网络、无大语言模型（LLM）、无意外。** 初始化探针属于确定性的底层管道逻辑。若某个探针需要调用 LLM 来分类故障，或访问外部服务来验证许可证，那它就不再是探针，而是一个工作流。如果在试运行（dry run）中某个探针耗时超过三秒，应将其视为工作台坏味道（workbench smell），要么将其移出初始化阶段，要么对其结果进行缓存。

## 使用方式

在生产环境中：

- **Claude Code 钩子（hooks）。** `pre-task` 钩子会调用初始化脚本，若脚本执行失败则拒绝启动代理。
- **GitHub Actions。** `setup-agent` 作业负责运行初始化脚本；代理作业依赖于该作业。
- **Docker 入口点（entrypoint）。** 代理容器在执行代理运行时之前会先运行初始化脚本；失败时日志会直接输出。

该初始化脚本具有良好的可移植性，因为它不依赖任何特定框架。Bash、Make 或任务文件均可对其进行封装。

## 交付部署

`outputs/skill-init-script.md` 会扫描项目结构，将环境配置工作拆解为各项探针，并输出项目专属的 `init_agent.py` 脚本及配套的 CI 工作流，确保该脚本在任何代理步骤执行前优先运行。

## 练习

1. 添加一个探针（probe），用于对比当前提交（commit）与最近已知良好提交（last-known-good commit）之间的差异，若变更文件超过 50 个则拒绝启动。
2. 配置脚本以生成 `prereqs.lock` 文件，若该锁文件（lock file）的创建时间超过七天，则拒绝启动。
3. 添加 `--fix` 参数，用于自动安装缺失的开发依赖（dev dependencies），但在未经批准的情况下绝不修改运行时依赖（runtime dependencies）。
4. 将探针（probe）从硬编码函数迁移至 YAML 注册表（YAML registry）。请论证此设计权衡（trade-off）的理由。
5. 为每个探针（probe）设置时间预算（timing budget）。运行时间超过三秒的探针即构成工作台坏味道（workbench smell）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 探针（Probe） | “检查” | 返回 `(name, status, detail)` 的确定性函数 |
| 初始化报告（Init report） | “设置输出” | 与状态文件并列写入的 JSON，包含探针结果 |
| 幂等（Idempotent） | “可安全重跑” | 连续两次运行生成的报告除时间戳外完全一致 |
| 显式失败（Fail loud） | “不要吞掉错误” | 立即中止并向人工暴露问题；禁止静默回退 |
| 初始化开销（Setup tax） | “引导成本” | 智能体（agent）在每次会话中为重新发现显而易见信息所消耗的 Token |

## 延伸阅读

- [Anthropic，长周期智能体的有效管控框架](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [GitHub Actions，用于环境设置的复合动作](https://docs.github.com/en/actions/sharing-automations/creating-actions/creating-a-composite-action)
- [microservices.io，GenAI 开发平台：护栏机制](https://microservices.io/post/architecture/2026/03/09/genai-development-platform-part-1-development-guardrails.html) — 将 pre-commit 与 CI 检查作为初始化环节
- [Augment Code，如何构建你的 AGENTS.md（2026）](https://www.augmentcode.com/guides/how-to-build-agents-md) — 初始化预期
- [Codex Blog，Codex CLI 上下文压缩](https://codex.danielvaughan.com/2026/03/31/codex-cli-context-compaction-architecture/) — 将会话启动视为具备上下文压缩感知能力的初始化
- Phase 14 · 33 — 本脚本启用的规则集
- Phase 14 · 34 — 本脚本初始化的状态文件
- Phase 14 · 38 — 初始化脚本所供给的验证关卡
- Phase 14 · 40 — 消费初始化报告中“最近已知良好”状态的交接环节