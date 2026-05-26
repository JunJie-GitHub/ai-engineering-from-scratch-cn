---
name: migration-agent
description: 构建一个仓库级代码迁移代理（agent），将确定性转换规则（deterministic recipes）与代理回退循环（agent fallback loop）相结合，通过 MigrationBench 基准测试，并发布失败分类体系（failure taxonomy）。
version: 1.0.0
phase: 19
lesson: 09
tags: [结业项目, 代码迁移, openrewrite, libcst, migrationbench, 代理, 沙箱]
---

给定一个 Java 8 或 Python 2 仓库，生成一个已迁移的分支（升级至 Java 17 或 Python 3.12），确保测试套件全部通过（green）且代码覆盖率（coverage）回退最小。在包含 50 个仓库的 MigrationBench 子集上进行评估。

构建计划：

1. 确定性处理阶段：首先使用 OpenRewrite（Java）或 libcst（Python）执行机械式代码重写。将其提交为“配方（recipe）”提交，确保差异（diff）清晰整洁。
2. Daytona 沙箱（sandbox）：预装目标运行时环境；按分支构建；以只读方式挂载源代码。
3. 代理循环（agent loop）：基于 Claude Opus 4.7 + GPT-5.4-Codex，使用 LangGraph 或 OpenAI Agents SDK。工具集：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。对失败进行分类（依赖、语法、测试、构建工具），应用针对性修复并重新运行。
4. 预算上限：30 分钟、8 美元、20 轮交互。触发任一上限即终止流程，并将当前差异（diff）归档至 `budget_exhausted` 目录下。
5. 测试与覆盖率门禁（gate）：构建通过且测试全部通过；代码覆盖率下降幅度不得超过 2%。
6. 发起拉取请求（PR）：包含配方提交、代理提交以及总结性评论。
7. 失败分类体系（failure taxonomy）：为每个仓库打上以下标签之一：`{dep_upgrade_required, build_tool_drift, custom_annotation, test_flake, syntax_edge_case, budget_exhausted, coverage_regression}`。
8. 在 MigrationBench 的 50 个仓库上运行；发布各类别的通过率、单仓库成本及覆盖率保持情况；与仅使用确定性工具的基线（baseline）进行对比。

评估标准：

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 个仓库子集的 pass@1 指标 |
| 20 | 测试覆盖率保持情况 | 与基础分支相比的平均覆盖率差值 |
| 20 | 单仓库迁移成本 | 成功运行案例的平均单仓库成本（美元） |
| 20 | 代理与确定性工具集成度 | OpenRewrite 处理的修复比例与代理处理的比例对比 |
| 15 | 失败分析报告 | 分类体系的完整性及附带示例 |

硬性否决项（Hard rejects）：

- 跳过确定性处理阶段的流水线。OpenRewrite 处理机械式重写（占比 70-80%）的成本更低，且比任何代理都更可靠。
- 将覆盖率回退超过 2% 的情况视为通过。
- 将机械式重写与代理生成的更改合并到同一个提交中的 PR。必须将两者分离。
- 未在相同的 50 个仓库上运行匹配的“仅确定性工具”基线，就直接报告通过率。

拒绝规则（Refusal rules）：

- 拒绝将已迁移的分支强制推送（force-push）覆盖基础分支。必须始终创建新分支并发起 PR。
- 拒绝发起在沙箱中持续集成（CI）状态未转为绿色（通过）的 PR。
- 拒绝在未获得明确修改授权的企业仓库上运行。

输出成果：一个包含双层迁移流水线（two-layer migration pipeline）的仓库、50 个仓库的 MigrationBench 运行日志、失败分类体系（failure taxonomy）仪表盘、匹配的仅确定性工具基线运行结果，以及一份针对三种最常见失败类别的分析报告，并说明能够消除各类别的配方（recipe）变更方案。