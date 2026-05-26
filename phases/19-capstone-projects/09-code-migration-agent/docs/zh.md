# 综合实战项目 09 — 代码迁移智能体（仓库级语言/运行时升级）

> Amazon 的 MigrationBench（Java 8 至 17）与 Google 的 App Engine Py2 到 Py3 迁移工具设定了 2026 年的行业基准。Moderne 的 OpenRewrite 能够大规模执行确定性的抽象语法树（Abstract Syntax Tree, AST）重写。Grit 则采用代码修改（codemod）风格的领域特定语言（Domain-Specific Language, DSL）解决同一问题。生产级架构将二者结合：以确定性底层（Deterministic Substrate）确保重写安全，叠加智能体（Agent）层处理模糊场景，配合按分支隔离的构建沙箱，以及在拉取请求（Pull Request, PR）提交前自动通过的测试套件。本项目的目标是完成 50 个真实仓库的迁移，并公布通过率与失败分类体系。

**类型：** 综合实战项目
**编程语言：** Python（智能体）、Java / Python（目标代码）、TypeScript（仪表盘）
**前置要求：** 第 5 阶段（自然语言处理, NLP）、第 7 阶段（Transformer 架构）、第 11 阶段（大语言模型工程）、第 13 阶段（工具链）、第 14 阶段（智能体）、第 15 阶段（自主系统）、第 17 阶段（基础设施）
**涉及阶段：** P5 · P7 · P11 · P13 · P14 · P15 · P17
**预计耗时：** 30 小时

## 问题背景

大规模代码迁移是 2026 年编程智能体（Coding Agent）最清晰的生产级应用场景之一。评估标准明确（迁移后测试套件是否通过？），商业价值显著（Java 8 全量迁移是一项需要投入大量人力的工程），且基准测试公开（MigrationBench 的 50 个仓库子集）。Moderne 的 OpenRewrite 负责处理确定性部分。智能体层则负责处理 OpenRewrite 规则配方（Recipes）无法覆盖的场景：模糊重写、构建系统版本漂移、长尾语法特性以及传递性依赖破坏。

你将构建一个智能体，接收 Java 8 仓库（或 Python 2 仓库），并输出持续集成（Continuous Integration, CI）状态为绿色的迁移分支。你需要统计通过率、测试覆盖率保持情况、单仓库迁移成本，并构建失败分类体系。通过与仅依赖确定性工具的基线进行对比，你将明确智能体的核心价值究竟体现在何处。

## 核心概念

该流水线包含两个层级。**确定性底层**（Java 使用 OpenRewrite，Python 使用 libcst）安全地执行大部分机械性重写：导入语句、方法签名、空安全编辑、try-with-resources 语法替换以及废弃 API 替换。该层运行速度快，且能生成可审计的代码差异（diff）。**智能体层**（基于 Claude Opus 4.7 和 GPT-5.4-Codex 的 OpenAI Agents SDK 或 LangGraph）处理配方无法解决的场景：构建文件升级（Maven/Gradle/pyproject）、传递性依赖冲突、测试随机失败（Flaky Tests）以及自定义注解。

每个仓库将分配一个预装目标运行时的 Daytona 沙箱环境。智能体按以下流程迭代：执行构建、分类失败原因、应用修复、重新运行。硬性限制：单仓库耗时不超过 30 分钟，成本不超过 8 美元，智能体交互轮次不超过 20 次。若所有测试通过且覆盖率未下降，该分支将自动发起拉取请求（PR）。否则，该仓库将被归入相应的失败类别，并附带证据记录。

失败分类体系是本项目的主要交付物。在 50 个仓库中，哪些环节出了问题？是传递性依赖？自定义注解？构建工具版本？还是与迁移无关的测试随机失败？每个类别将统计出现次数，并提供一个典型代码差异示例。未来的配方开发者可优先针对排名前三的类别进行优化。

## 架构设计

目标仓库 (target repo)
      |
      v
OpenRewrite / libcst 确定性配方 (deterministic recipes)
   (安全、快速、可审计，覆盖约 70-80% 的修复)
      |
      v
每个分支的 Daytona 沙箱 (sandbox)
      |
      v
智能体循环 (agent loop) (Claude Opus 4.7 / GPT-5.4-Codex):
   - 运行构建 -> 捕获失败
   - 分类失败原因 (构建、测试、代码检查)
   - 应用修复 (补丁或重试配方)
   - 重新运行
   - 预算限制：30 分钟，8 美元，20 轮交互
      |
      v
测试与覆盖率差异门禁 (coverage delta gate)
      |
      v (通过)
提交拉取请求 (open PR)
      |
      v (失败)
按失败类别归档并附加复现用例 (attach repro)

## 技术栈 (Stack)

- 确定性底层 (deterministic substrate)：OpenRewrite (Java) 或 libcst (Python)
- 智能体 (Agent)：基于 Claude Opus 4.7 + GPT-5.4-Codex 的 OpenAI Agents SDK 或 LangGraph
- 沙箱 (Sandbox)：每个分支独立的 Daytona 开发容器 (devcontainers)，预装目标运行时 (runtime) (Java 17 / Python 3.12)
- 构建系统 (Build systems)：Maven、Gradle、uv (Python)
- 基准测试 (Benchmarks)：Amazon MigrationBench 50 个仓库子集 (Java 8 至 17)、Google App Engine Py2 至 Py3 迁移仓库
- 测试框架 (test harness)：并行运行器，通过 Jacoco (Java) 或 coverage.py (Python) 统计覆盖率
- 可观测性 (Observability)：Langfuse + 每个仓库的追踪包 (trace bundle)，包含每次差异代码块 (diff chunk)
- 仪表盘 (Dashboard)：失败分类体系 (failure taxonomy) 仪表盘，显示各类别数量及典型差异示例 (exemplar diffs)

## 构建流程 (Build It)

1. **配方执行阶段 (Recipe pass)。** 首先运行 OpenRewrite (Java) 或 libcst (Python) 配方。捕获 70-80% 的机械化迁移任务。将其提交为“配方”提交记录。

2. **构建试跑 (Build trial)。** 在 Daytona 沙箱中：安装目标运行时，执行构建。若通过（绿灯），则跳至测试阶段；若失败（红灯），则交由智能体处理。

3. **智能体循环 (Agent loop)。** 使用 LangGraph 集成工具：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。智能体对失败原因进行分类（依赖、语法、测试、构建工具），并应用针对性修复。重新运行。

4. **预算上限 (Budget caps)。** 每个仓库限制 30 分钟物理时间 (wall-clock)、8 美元成本、20 轮智能体交互。任何一项超标即中止流程，并将当前代码差异归档至 `budget_exhausted` 类别。

5. **测试与覆盖率门禁 (Test + coverage gate)。** 构建通过后，运行测试套件。将覆盖率与基准仓库进行对比。若覆盖率下降超过 2%，则归档至 `coverage_regression` 类别。

6. **提交拉取请求 (PR open)。** 成功后，推送分支，提交拉取请求 (PR)，附带代码差异以及配方应用情况与智能体生成提交记录的摘要。

7. **失败分类体系 (Failure taxonomy)。** 针对每个失败的仓库，打上类别标签：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建可视化仪表盘。

8. **50 仓库批量运行 (50-repo run)。** 在 MigrationBench 子集上执行全流程。报告各类别通过率、单仓库成本、覆盖率保持情况，并与仅使用确定性配方的基线 (baseline) 进行对比。

## 使用示例 (Use It)

$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4

## 部署上线 (Ship It)

`outputs/skill-migration-agent.md` 是本次任务的交付物。给定一个代码仓库，该流程会先执行确定性配方（deterministic recipes），随后进入智能体循环（agent loop），最终生成一个通过所有测试的迁移分支（green migrated branch），或将该仓库归档至特定的分类体系（taxonomy class）中。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 个仓库子集的 pass@1 指标 |
| 20 | 测试覆盖率保持（Test-coverage preservation） | 相较于基准分支的平均覆盖率差值 |
| 20 | 单个仓库迁移成本 | 成功运行时的单仓库美元成本（$/repo） |
| 20 | 智能体与确定性工具集成（Agent / deterministic-tool integration） | OpenRewrite 处理的修复占比与智能体生成修复的对比 |
| 15 | 失败分析报告 | 包含示例的分类体系完整度 |
| **100** | | |

## 练习

1. 仅使用 OpenRewrite（不启用智能体）运行迁移流水线。将其通过率与完整流水线进行对比。找出仅靠智能体就能产生差异（即解决问题）的具体案例。
2. 实现“代码规范检查（lint-clean）”：迁移完成后，运行代码风格检查工具（Java 使用 spotless，Python 使用 ruff）。若出现新的 lint 错误，则使 PR 失败。统计“覆盖率保持但代码风格退化”的比例。
3. 添加“最小化差异（minimal-diff）”优化器：在智能体生成的分支通过测试后，进行第二轮扫描以剔除不必要的变更。报告差异文件大小的缩减比例。
4. 扩展至第三个迁移场景：从 Node 18 升级至 Node 22。复用沙箱封装逻辑；将配方层替换为自定义的代码修改脚本（codemod）。
5. 将“首次构建成功耗时（time-to-first-green-build, TTFGB）”作为用户体验指标进行测量。目标：p50（中位数）低于 10 分钟。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 确定性基座（Deterministic substrate） | “配方引擎” | OpenRewrite / libcst：具备安全保障的声明式抽象语法树（AST）重写工具 |
| 代码修改脚本（Codemod） | “代码修改程序” | 一种以机械化方式变更源代码的重写规则 |
| 构建漂移（Build drift） | “工具版本偏差” | 主版本更新间 Maven / Gradle / uv 产生的细微行为差异 |
| 失败分类（Failure class） | “分类桶” | 仓库迁移失败的具体标注原因：依赖、语法、测试、构建工具、预算 |
| 覆盖率差值（Coverage delta） | “覆盖率保持” | 从基准分支到迁移分支的测试覆盖率百分比变化 |
| 智能体回合（Agent turn） | “工具调用轮次” | 智能体循环中的一次“规划 -> 执行 -> 观察”周期 |
| 预算耗尽（Budget exhaustion） | “触及上限” | 仓库在 30 分钟 / 8 美元 / 20 轮次的限制内耗尽资源且仍未通过测试 |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026 年标准基准测试（benchmark）
- [Moderne.io OpenRewrite 平台](https://www.moderne.io) — 确定性基底（deterministic substrate）参考
- [OpenRewrite 文档](https://docs.openrewrite.org) — 迁移规则（recipe）编写
- [Grit.io](https://www.grit.io) — 替代性代码转换（codemod）领域特定语言（DSL）
- [OpenAI 沙盒迁移指南](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 参考
- [Google App Engine Py2 至 Py3 迁移工具](https://cloud.google.com/appengine) — 替代性迁移基准测试（migration benchmark）
- [libcst](https://github.com/Instagram/LibCST) — Python 确定性基底（deterministic substrate）
- [Daytona 沙盒](https://daytona.io) — 分支级沙盒（per-branch sandbox）参考