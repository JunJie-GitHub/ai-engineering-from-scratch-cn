# 综合实战项目 16 — GitHub Issue-to-PR 自主智能体 (Autonomous Agent)

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 与 Google Jules 均呈现出相同的 2026 年产品形态：为 Issue 打上标签，即可获取 PR。在云沙箱 (cloud sandbox) 中运行智能体，验证测试通过后，提交附带决策依据 (rationale) 的待评审 PR。其中的难点在于自动复现仓库的构建环境、防止凭证泄露、实施按仓库维度的预算控制，以及确保智能体无法执行强制推送 (force-push)。本综合实战项目将构建自托管版本，并在成本与测试通过率方面与托管替代方案进行对比。

**Type:** 综合实战项目
**Languages:** Python（智能体）、TypeScript（GitHub App）、YAML（Actions）
**Prerequisites:** 第 11 阶段（大语言模型工程）、第 13 阶段（工具）、第 14 阶段（智能体）、第 15 阶段（自主化）、第 17 阶段（基础设施）
**Phases exercised:** P11 · P13 · P14 · P15 · P17
**Time:** 30 小时

## 问题描述

异步云端编程智能体 (async cloud coding agent) 与交互式编程智能体 (interactive coding agent)（见综合实战项目 01）属于不同的产品类别。其用户体验 (UX) 核心在于 GitHub 标签。当你为 Issue 添加 `@agent fix this` 标签后，工作节点将在云沙箱中启动，克隆仓库、运行测试、编辑文件、进行验证，并最终创建一个 PR，其正文中会包含智能体的决策依据。全程无需交互式循环，也无需终端操作。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 以及 Factory Droids 等产品均朝着这一方向演进。

具体的工程挑战包括：环境复现（智能体必须在无缓存开发镜像的情况下从零构建仓库）、不稳定测试 (flaky tests)（需重新运行或隔离处理）、凭证权限控制 (credential scoping)（使用具备最小细粒度权限的 GitHub App）、按仓库每日预算限制，以及禁止强制推送策略。本综合实战项目将针对测试通过率、成本及安全性，与托管替代方案进行量化对比。

## 核心概念

触发器为 GitHub Webhook（问题标签或 PR 评论）。调度器（Dispatcher）将任务入队至 ECS Fargate 或 Lambda。工作节点（Worker）将代码仓库（Repository）拉取至 Daytona 或 E2B 沙箱（Sandbox）中，并使用根据仓库推断出的通用 Dockerfile（基于编程语言与框架）。智能体（Agent）针对 Claude Opus 4.7 或 GPT-5.4-Codex 运行 mini-swe-agent 或 SWE-agent v2 循环。其迭代流程为：读取代码、提出修复方案、应用补丁、运行测试。

验证（Verification）是流程的门禁步骤。在打开拉取请求（PR）之前，完整的持续集成（CI）必须在沙箱中通过。系统会计算代码覆盖率增量（Coverage delta）；若下降幅度超过设定阈值，PR 仍会创建，但会被自动标记为 `needs-review`。智能体会将修复依据作为 PR 描述发布，并附带一个 `@agent` 讨论串，供审查者在后续跟进时进行交互。

安全性通过两个不同的 GitHub 层面进行作用域控制：该应用提供具有 `workflows: read` 权限以及受限的仓库内容/PR 作用域的短期安装令牌（Installation token）；分支保护（Branch protection，而非应用权限）强制执行“禁止直接写入 `main` 分支”和“禁止强制推送（Force-push）”——该应用绝不会被加入绕过列表（Bypass list）。由于 GitHub App 原生并不支持针对 `.github/workflows` 路径的只读访问权限，因此智能体必须在文件编辑的允许列表（Allow-list）中，于工作节点层面强制执行此限制。每个仓库每日的预算上限由调度器强制执行（例如：每个仓库每天最多创建 5 个 PR，每个 PR 预算 20 美元）。

## Architecture

GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups

## Stack

- 触发器（Trigger）：具有细粒度令牌（Fine-grained token）的 GitHub App；通过 Lambda 或 Fly.io 接收 Webhook
- 工作节点（Worker）：ECS Fargate 任务（或 GitHub Actions 自托管运行器（Self-hosted runner））
- 沙箱（Sandbox）：每个任务使用 Daytona 开发容器（Devcontainer）或 E2B 沙箱
- 智能体循环（Agent loop）：基于 Claude Opus 4.7 / GPT-5.4-Codex 的 mini-swe-agent 基线或 SWE-agent v2
- 检索（Retrieval）：tree-sitter 仓库映射（Repo-map） + ripgrep
- 验证（Verification）：沙箱内完整 CI + 代码覆盖率增量门禁（Coverage delta gate）
- 可观测性（Observability）：Langfuse，每个 PR 的追踪（Trace）归档链接附于 PR 描述中
- 预算（Budget）：每个仓库每日美元上限；每个仓库每日最大 PR 数量

## 动手构建

1. **GitHub 应用（GitHub App）**。细粒度安装令牌（Fine-grained installation token）：issues 读写、pull_requests 写入、contents 读写、workflows 读取。分支保护（Branch protection，唯一具备此能力的机制）强制执行“禁止直接向 `main` 推送”和“禁止强制推送（force-push）”；该应用未被加入绕过列表（bypass list）。由于 GitHub App 权限不支持路径级作用域，工作器（Worker）会对拟提交的代码差异（diff）执行白名单检查，强制“禁止在 `.github/workflows` 目录下写入”。

2. **Webhook 接收器（Webhook receiver）**。Lambda 函数接收 Issue 标签或 PR 评论的 Webhook 请求。按 `@agent fix this` 标签进行过滤，并将任务入队至 SQS（Simple Queue Service）。

3. **调度器（Dispatcher）**。从 SQS 中拉取任务。强制执行单仓库每日预算限制。启动一个 ECS Fargate 任务，传入仓库 URL、Issue 正文以及一个全新的 Daytona 沙箱（Daytona sandbox）。

4. **环境推断（Environment inference）**。自动检测编程语言（Python、Node、Go、Rust）及包管理器（uv、pnpm、go mod、cargo）。若项目中不存在 `Dockerfile`，则动态生成一个。

5. **Agent 循环（Agent loop）**。采用 mini-swe-agent 或 SWE-agent v2，搭配 Claude Opus 4.7 模型。可用工具：`ripgrep`、`tree-sitter repo-map`、`read_file`、`edit_file`、`run_tests`、`git`。硬性限制：成本上限 20 美元、物理时间（wall-clock）上限 30 分钟、Agent 交互轮次上限 30 轮。

6. **验证（Verification）**。循环结束后，在沙箱内运行完整的测试套件。通过 `jacoco` / `coverage.py` 计算代码覆盖率差异（coverage delta）。若持续集成（CI）失败：立即中止，不创建 PR。若覆盖率下降超过 2%：创建 PR 并附加 `needs-review` 标签。

7. **PR 提交（PR posting）**。推送 Agent 分支。通过 GitHub API 创建 PR，附带以下信息：标题、修改理由、差异摘要、追踪链接（trace URL）、成本及交互轮次。

8. **凭证安全（Credential hygiene）**。工作器使用短期有效的 GitHub App 安装令牌运行。日志在归档前会经过脱敏处理，清除敏感信息。

9. **评估（Evaluation）**。使用 30 个难度各异的内部种子 Issue。评估指标包括：通过率、PR 质量（差异大小、代码规范、覆盖率）、成本及延迟。在相同 Issue 集上，与 Cursor Background Agents 和 AWS Remote SWE Agents 进行横向对比。

## 使用方式

# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review

## 交付说明

`outputs/skill-issue-to-pr.md` 为最终交付物。该方案由 GitHub App 与异步云工作器组成，能够将带有特定标签的 Issue 自动转化为可供审查的 PR，并在过程中严格控制成本与凭证权限范围。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 30 个 Issue 的通过率 | 端到端成功（CI 通过 + 覆盖率达标） |
| 20 | PR 质量 | 差异大小、覆盖率变化、代码规范符合度 |
| 20 | 单个已解决 Issue 的成本与延迟 | 每个 PR 的美元成本与物理时间 |
| 20 | 安全性 | 作用域受限的令牌、单仓库预算、禁止强制推送、凭证安全 |
| 15 | 运维人员体验（Operator UX） | 修改理由注释、重试机制、@提及跟进 |
| **100** | | |

## 练习

1. 添加“修复不稳定测试（fix flaky test）”模式：使用标签 `@agent stabilize-flake TestX` 在沙盒（sandbox）内运行该测试 50 次，并提出能使其稳定的最小化修改方案。

2. 在三个共同问题上对比成本与 Cursor 后台代理（Cursor Background Agents）的表现。报告各工具在哪些场景下更具优势。

3. 实现预算仪表盘（budget dashboard）：显示每个仓库每日成本及每个用户成本。出现异常时触发告警。

4. 构建“试运行（dry-run）”模式：在不运行持续集成（CI）的情况下创建草稿拉取请求（draft PR），以便评审人员以较低成本审查方案。

5. 添加保留策略（retention policy）：超过 7 天未合并的 PR 分支将被自动删除。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GitHub App | “限定范围的机器人身份” | 具备细粒度权限（fine-grained permissions）与短期安装令牌（installation token）的应用 |
| 异步云代理（Async cloud agent） | “后台代理” | 在云沙盒（cloud sandbox）中运行的非交互式工作进程（non-interactive worker），而非终端进程 |
| 环境推断（Environment inference） | “Dockerfile 生成” | 检测编程语言与包管理器（package manager），若缺失则自动生成 Dockerfile |
| 验证（Verification） | “沙盒内持续集成” | 在创建拉取请求前，于工作进程内部运行完整测试套件（test suite） |
| 覆盖率差值（Coverage delta） | “覆盖率保持” | 从基准分支（base branch）到代理分支（agent branch）的测试覆盖率百分比变化 |
| 单仓库预算（Per-repo budget） | “每日上限” | 在调度器（dispatcher）处执行的美元金额与 PR 数量上限 |
| 修改依据（Rationale） | “PR 正文说明” | 代理对修改内容及原因的总结；必须包含在 PR 正文中 |

## 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 异步云代理的标准参考实现
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — 命令行界面（CLI）参考
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商业替代方案
- [OpenAI Codex (cloud)](https://openai.com/codex) — 托管型竞品
- [Google Jules](https://jules.google) — Google 的托管版本
- [Factory Droids](https://www.factory.ai) — 另一商业参考方案
- [GitHub App 文档](https://docs.github.com/en/apps) — 限定范围的机器人身份
- [Daytona 云沙盒](https://daytona.io) — 参考沙盒实现