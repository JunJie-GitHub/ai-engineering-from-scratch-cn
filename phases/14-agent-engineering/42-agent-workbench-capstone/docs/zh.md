# 综合实战项目（Capstone）：交付可复用的智能体（Agent）工作台工具包

> 本迷你课程（mini-track）的终点是一个可直接放入任意代码仓库（repo）的工具包。十一节课中涉及的工作台组件（surfaces）被整合进一个目录，只需执行 `cp -r`，次日智能体即可稳定运行。本次综合实战的产出物，正是本课程的核心交付成果。

**类型：** 构建
**语言：** Python（标准库）
**前置要求：** 阶段 14 · 31 至 14 · 41
**预计耗时：** 约 75 分钟

## 学习目标

- 将七个工作台组件打包至一个即插即用的目录中。
- 锁定模式定义（schemas）、脚本和模板，确保新仓库获得一个已知可靠的基线。
- 添加一个独立的安装脚本，以幂等（idempotent）方式部署该工具包。
- 明确工具包的包含与排除项，并为每一项的取舍提供合理依据。

## 问题背景

如果工作台仅存在于 Google 文档、聊天记录和三个记不全的脚本中，那么它每个季度都将被迫重建。解决之道在于使用带版本控制的工具包：一个包含工作台组件、模式定义、脚本以及一键安装程序的仓库或目录。

完成本课后，你将在磁盘上生成 `outputs/agent-workbench-pack/` 目录，并配备一个 `bin/install.sh` 脚本，用于将其部署至任意目标仓库。

## 核心概念

flowchart TD
  Pack[agent-workbench-pack/] --> Docs[AGENTS.md + docs/]
  Pack --> Schemas[schemas/]
  Pack --> Scripts[scripts/]
  Pack --> Bin[bin/install.sh]
  Bin --> Repo[target repo]
  Repo --> Surfaces[all seven workbench surfaces wired]

### 工具包目录结构

outputs/agent-workbench-pack/
├── AGENTS.md
├── docs/
│   ├── agent-rules.md
│   ├── reliability-policy.md
│   ├── handoff-protocol.md
│   └── reviewer-rubric.md
├── schemas/
│   ├── agent_state.schema.json
│   ├── task_board.schema.json
│   └── scope_contract.schema.json
├── scripts/
│   ├── init_agent.py
│   ├── run_with_feedback.py
│   ├── verify_agent.py
│   └── generate_handoff.py
├── bin/
│   └── install.sh
└── README.md

### 包含与排除原则

包含项：

- 组件模式定义。它们是交互契约。
- 上述四个脚本。它们构成运行时环境。
- 四份文档。它们定义了规则与评估标准。

排除项：

- 项目专属任务。任务应归属于目标仓库的看板，而非工具包内。
- 第三方供应商 SDK 调用。该工具包保持框架无关（framework-agnostic）特性。
- 入职引导文案。工具包应与团队现有的入职流程并列存放，而非嵌入其中。

### 安装程序

一个简短的 `bin/install.sh`（或 `bin/install.py`）脚本：

1. 若未提供 `--force` 参数，则拒绝覆盖已存在的工具包。
2. 将工具包复制至目标仓库。
3. 若存在 `.github/workflows/` 目录，则自动配置持续集成（CI）流水线。
4. 打印后续步骤：填充任务看板、设置验收命令、运行初始化脚本。

### 版本控制

工具包内包含一个 `VERSION` 文件。涉及模式定义变更或需要数据迁移的脚本更新将提升主版本号（major）。仅文档修改则提升补丁版本号（patch）。目标仓库的 `agent_state.json` 会记录其初始化时所基于的工具包版本。

## 开始构建

`code/main.py` 会将该工具包（pack）组装至课程目录旁的 `outputs/agent-workbench-pack/` 路径下，初始内容已预置本迷你课程系列（mini-track）前序课程中的模式（schemas）与脚本，以及你此前编写的文档。

运行方式如下：

python3 code/main.py

该脚本会复制并锁定各个配置表面（surfaces），写入 README 文件，打印工具包目录树，并以状态码 0 正常退出。重复执行该脚本具有幂等性（idempotent）。

## 实际生产环境中的模式

只有能够经受住分支派生（forks）、版本更新以及上游不兼容变更的考验，工具包才具备实际价值。以下四种模式可确保其稳定运行。

**`VERSION` 是契约，而非营销噱头。** 主版本（Major）升级需要进行状态迁移（state migration）。次版本（Minor）升级需要重新运行检查器（checker）。补丁版本（Patch）仅涉及文档更新。安装程序会在每次安装时将 `.workbench-version` 写入目标仓库；若目标仓库的锁定文件（lock）与工具包的 `VERSION` 不一致，`lint_pack.py` 将拒绝发布。这正是 `npm`、`Cargo` 和 `pyproject.toml` 能够在十年技术更迭中保持稳定的原因；智能体（agents）的开发并不会改变这些底层规则。

**跨工具分发的单一数据源。** Nx 提供了一个 `nx ai-setup` 命令，能够基于单一配置文件生成 `AGENTS.md`、`CLAUDE.md`、`.cursor/rules/`、`.github/copilot-instructions.md` 以及一个 MCP 服务器。该工具包也应遵循此原则：安装程序会创建符号链接（`ln -s AGENTS.md CLAUDE.md`），从而将单一事实来源（single source of truth）同步分发至所有编程智能体。为了优先适配某一特定工具而派生（fork）工具包，是一种典型的失败模式（failure mode）。

**在存在实质性状态时拒绝执行的 `uninstall.sh`。** 卸载工具包时，绝不能删除用户的 `agent_state.json`、`task_board.json` 或 `outputs/` 目录。卸载程序会移除模式文件、脚本、文档以及 `AGENTS.md`（提供 `--keep-agents-md` 参数以供选择保留），并且若状态文件存在任何未提交的更改，卸载程序将拒绝继续执行。状态数据归用户所有，工具包无权处置。

**技能可发布化。采用 SkillKit 风格的分发机制。** 该工具包以 SkillKit 技能（skill）的形式发布：执行 `skillkit install agent-workbench-pack` 命令，即可从单一数据源将其部署至 32 个 AI 智能体。工具包仓库是事实来源，SkillKit 则是分发渠道。供应商锁定（vendor lock-in）随之瓦解；七个配置表面保持统一。

## 使用方式

该工具包可通过以下三种途径交付：

- **作为直接置入仓库的目录。** 执行 `cp -r outputs/agent-workbench-pack /path/to/repo`。
- **作为公共模板仓库。** 派生（fork）并进行自定义，通过 `VERSION` 控制版本漂移（drift）。
- **作为 SkillKit 技能。** 集成至你的智能体产品中，仅需一条命令即可完成部署。

工具包是配方（recipe），每次安装都是一次实例化部署。

## 交付与发布

`outputs/skill-workbench-pack.md` 会生成针对特定项目调优的工具包：规则根据团队历史进行精细化打磨，作用域通配符（scope globs）与仓库结构精准匹配，评估维度（rubric dimensions）额外扩展了一个特定领域的条目。

## 练习

1. 确定哪份可选的第五份文档应被纳入标准包（canonical pack）。请说明筛选理由。
2. 使用 Python 重写安装脚本（installer），并添加 `--dry-run` 参数。对比其与 Bash 版本在易用性（ergonomics）上的差异。
3. 新增 `bin/uninstall.sh` 脚本以安全卸载该工具包（pack）。若状态文件包含非平凡历史记录（non-trivial history），则应拒绝执行。请明确界定“非平凡”的具体标准。
4. 新增 `lint_pack.py` 脚本，当工具包内容与 `VERSION` 文件发生偏离（drift）时触发失败。将其集成至该工具包专属仓库的持续集成（CI）流水线中。
5. 编写从手工搭建的工作台（workbench）迁移至该工具包的运维手册（runbook）。请规划出能将停机时间降至最低的操作顺序。

## 关键术语

| 术语 | 通俗叫法 | 实际含义 |
|------|----------|----------|
| 工作台工具包（Workbench pack） | “入门套件” | 包含全部七个功能面（surfaces）的版本化目录 |
| 安装脚本（Installer） | “安装脚本” | 以幂等（idempotent）方式部署工具包的 `bin/install.sh` |
| 工具包版本（Pack version） | “VERSION” | 架构或脚本变更时升级主版本号，仅文档更新时升级补丁版本号 |
| 即插即用工具包（Drop-in pack） | “cp -r 即用” | 首日即可直接运行，无需针对每个仓库进行定制配置 |
| 可派生模板（Forkable template） | “GitHub 模板” | 可通过 GitHub 的“Use this template”功能克隆的公共仓库 |

## 扩展阅读

- 第 14 · 31 至 14 · 41 阶段 — 该工具包集成的所有功能面（surfaces）
- [SkillKit](https://github.com/rohitg00/skillkit) — 在 32 个 AI 智能体（AI agents）上安装此技能
- [Nx Blog, Teach Your AI Agent How to Work in a Monorepo](https://nx.dev/blog/nx-ai-agent-skills) — 跨六款工具的单一源生成器（single-source generator）
- [agents.md — the open spec](https://agents.md/) — 你的工具包路由器（router）必须实现的内容
- [HKUDS/OpenHarness](https://github.com/HKUDS/OpenHarness) — 等效工具包的参考实现（reference implementation）
- [andrewgarst/agentic_harness](https://github.com/andrewgarst/agentic_harness) — 基于 Redis 的参考实现，附带评估套件（eval suite）
- [Augment Code, A good AGENTS.md is a model upgrade](https://www.augmentcode.com/blog/how-to-write-good-agents-dot-md-files) — 工具包文档的质量基准
- [Anthropic, Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Anthropic, Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- 第 14 · 30 阶段 — 消耗该工具包验证关卡（verification gate）的评估驱动型智能体开发（eval-driven agent development）
- 第 14 · 41 阶段 — 该工具包旨在优化的前后对比基准（before/after benchmark）