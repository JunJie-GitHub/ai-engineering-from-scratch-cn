---
name: issue-to-pr
description: 构建一个在云沙箱 (Cloud Sandbox) 中运行的异步 GitHub Issue 转 PR 代理 (Async Agent)，用于复现构建、验证测试，并在严格的单仓库预算限制内提交可供审查的 PR。
version: 1.0.0
phase: 19
lesson: 16
tags: [综合项目, 异步代理, github, fargate, daytona, swe-bench, 预算控制, 安全机制]
---

针对包含标记为 `@agent fix this` 的 Issue 的 GitHub 仓库，部署一个自托管的云代理 (Agent)，将每个带标签的 Issue 转化为具备受限凭证 (Scoped Credentials) 和成本上限的、可供审查的 PR。

构建计划：

1. 配置具有细粒度令牌 (Fine-grained Token) 的 GitHub App：Issue 读写、PR 写入、内容读写、工作流读取。禁止强制推送 (Force-push)。对 `main` 分支启用分支保护 (Branch Protection) 以防止直接写入。
2. Webhook 接收器 (Webhook Receiver)（Lambda 或 Fly.io）过滤标签/PR 评论事件，并将其入队至 SQS。
3. 调度器 (Dispatcher) 强制执行单仓库每日金额与 PR 数量上限；为每个获准的任务启动一个 ECS Fargate 任务。
4. 环境推断 (Environment Inference)：从仓库内容中检测编程语言、包管理器及运行时环境。若缺失，则动态生成 (Synthesize) Dockerfile。
5. 为每个任务分配 Daytona 或 E2B 沙箱 (Sandbox)。将仓库克隆至全新的 `git worktree` 及代理分支中。
6. 代理循环 (Agent Loop)（基于 Claude Opus 4.7 或 GPT-5.4-Codex 运行的 mini-swe-agent 或 SWE-agent v2）。工具集：ripgrep、tree-sitter repo-map、read_file、edit_file、run_tests、git。限制：20 美元、30 轮交互、30 分钟。
7. 验证：在沙箱内运行完整 CI；通过 jacoco / coverage.py 计算覆盖率差异 (Coverage Delta)；若差异低于 -2% 则标记 `needs-review`；若 CI 失败则中止。
8. 通过 GitHub API 提交 PR，附带决策依据、差异摘要、追踪 URL、成本及交互轮数。
9. 可观测性 (Observability)：为每个 PR 生成 Langfuse 追踪记录；日志脱敏以清除密钥；提供单仓库预算控制面板。
10. 评估：在 30 个预设的内部 Issue 上进行测试；在包含 3 个共同 Issue 的子集上，与 Cursor Background Agents 和 AWS Remote SWE Agents 进行对比。

评估标准：

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 30 个 Issue 的通过率 | 端到端成功（CI 通过 + 覆盖率达标） |
| 20 | PR 质量 | 代码差异大小、覆盖率变化、代码风格一致性 |
| 20 | 单个已解决 Issue 的成本与延迟 | 单 PR 成本（美元）与单 PR 实际耗时 |
| 20 | 安全性 | 受限令牌、单仓库预算、禁止强制推送、凭证管理规范 |
| 15 | 操作员用户体验 (Operator UX) | 决策依据注释、重试机制支持、@提及跟进 |

硬性否决项：

- 任何具备强制推送能力的代理。直接一票否决。
- 跳过预算检查的调度器。无限循环是典型的故障模式。
- 未在沙箱内通过完整 CI 就提交的 PR。
- 包含未脱敏令牌或个人身份信息 (PII) 的追踪归档文件。

拒绝规则：

- 若 `main` 分支未启用分支保护，则拒绝安装。
- 若未设置单仓库每日预算（金额与 PR 数量），则拒绝运行。
- 拒绝自动重试失败的任务；所有重试均需人工重新添加标签触发。

输出：一个包含 GitHub App、Webhook 接收器、调度器与预算账本、Fargate 任务定义、沙箱生命周期管理器、mini-swe-agent 循环、30 个 Issue 的评估运行记录、与 Cursor Background Agents 及 AWS Remote SWE Agents 的横向对比报告，以及一份总结文档（需列出排名前三的构建推断 (Build Inference) 失败案例，以及针对每项失败所采取的 Dockerfile 动态生成 (Dockerfile Synthesis) 优化方案）。