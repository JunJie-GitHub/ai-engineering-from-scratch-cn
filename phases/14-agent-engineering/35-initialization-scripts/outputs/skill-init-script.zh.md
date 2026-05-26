---
name: init-script
description: 对项目进行访谈，并生成包含五个探针（probes）的确定性 `init_agent.py` 脚本，以及配套的 CI 工作流（CI workflow）；若任一探针失败，则拒绝启动代理（agent）。
version: 1.0.0
phase: 14
lesson: 35
tags: [初始化, 探针, CI, 工作台, 显式失败]
---

给定代码仓库（repository）、代理产品及其依赖面（dependency surface），生成项目专属的初始化脚本与 CI 配置（CI wiring）。

产出内容：

1. `tools/init_agent.py`，包含以下探针：运行时版本（runtime version）、已列出的依赖项、测试命令可解析性（test command resolvability）、必需的环境变量（environment variables）、状态文件时效性（state file freshness）。
2. `init_report.json` 数据模式（schema）文档，与脚本并列存放。每个探针返回 `(name, status: pass|warn|fail, detail)`。
3. `.github/workflows/agent-init.yml`（或等效配置），用于运行该脚本，并在任何严重性为失败的探针触发时阻断代理任务。
4. 一个 `pre-task` 钩子脚本，供代理运行时（agent runtime）在每次会话启动前调用。
5. `docs/init.md` 中的文档，列出每个探针、其严重级别以及故障修复方法。

硬性拒绝条件（Hard rejects）：

- 未设置超时机制的网络调用探针。初始化过程必须快速且支持离线安全（offline-safe）运行。
- 需要调用大语言模型（LLM）的探针。初始化属于确定性底层管线（deterministic plumbing）。
- 被包装器（wrapper）吞没的非零退出码。显式报错（Fail loud）正是其核心目的。
- 缺乏幂等性（idempotency）却修改状态的探针。连续两次运行必须生成除时间戳外完全相同的报告。

拒绝规则（Refusal rules）：

- 若项目缺少测试命令，则拒绝交付该脚本。改为将此缺失项记录至工作台审计（workbench audit）中。
- 若环境变量列表包含脚本将打印的敏感信息（secrets），则拒绝执行并强制脱敏（redaction）。初始化报告绝不应携带任何敏感信息。
- 若探针在试运行（dry run）中耗时超过三秒，需在交付前暴露该耗时问题。耗时过长的探针会使初始化沦为形式主义。

输出结构：

<repo>/
├── tools/
│   ├── init_agent.py
│   └── pre_task.sh
├── docs/
│   └── init.md
└── .github/
    └── workflows/
        └── agent-init.yml

结尾附上“下一步阅读”指引，指向：

- 第 36 课：了解利用初始化报告中 `repo_paths` 的按任务范围契约（per-task scope contract）。
- 第 37 课：了解消费已解析测试命令的运行时反馈循环（runtime feedback loop）。
- 第 38 课：了解依赖探针通过的验证门禁（verification gate）。