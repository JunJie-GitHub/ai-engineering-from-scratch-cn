# 综合项目 10 — 多智能体软件工程团队

> SWE-AF 的工厂架构（factory architecture）、MetaGPT 的基于角色的提示词（role-based prompting）、AutoGen 0.4 的类型化 Actor 图（typed actor graph）、Cognition 的 Devin 以及 Factory 的 Droids，最终都收敛于同一种 2026 年的形态：由架构师制定计划，N 名编码员在并行工作树（parallel worktrees）中协同开发，审查员负责把关，测试员进行验证。并行工作树将实际耗时（wall-clock）转化为吞吐量（throughput）。共享状态（shared state）与交接协议（handoff protocols）则成为系统的故障面（failure surface）。本综合项目的目标是构建这样一个团队，在 SWE-bench Pro 上进行评估，并报告哪些交接环节会失效及其发生频率。

**Type:** 综合项目
**Languages:** Python / TypeScript（智能体），Shell（工作树脚本）
**Prerequisites:** 第 11 阶段（大语言模型工程），第 13 阶段（工具），第 14 阶段（智能体），第 15 阶段（自主系统），第 16 阶段（多智能体），第 17 阶段（基础设施）
**Phases exercised:** P11 · P13 · P14 · P15 · P16 · P17
**Time:** 40 小时

## 问题描述

单智能体编码框架（single-agent coding harnesses）在处理大型任务时会触及瓶颈。这并非因为单个智能体能力不足，而是 20 万 token 的上下文（context）无法同时容纳架构规划、四个并行的代码库切片、审查员意见以及测试输出。多智能体工厂模式将问题拆解：架构师负责规划，编码员在并行工作树中负责实现，审查员负责把关，测试员负责验证。SWE-AF 的“工厂”架构、MetaGPT 的角色机制、AutoGen 的类型化 Actor 图——这三种框架描述的其实是同一种形态。

系统的故障面在于交接环节。架构师规划了编码员无法实现的内容；编码员生成了相互冲突的代码差异（diffs）；审查员批准了由幻觉（hallucinated）产生的修复方案；测试员与仍在编写代码的编码员发生竞态冲突（race）。你将构建这样一个团队，在 50 个 SWE-bench Pro 问题上运行测试，追踪每一次交接过程，并发布事后复盘报告（post-mortem）。

## 核心概念

角色即类型化智能体 (typed agents)。**Architect**（Claude Opus 4.7）负责阅读问题 (issue)，编写计划，并将其拆分为具有明确接口 (explicit interfaces) 的子任务。**Coders**（Claude Sonnet 4.7，N 个并行实例，每个实例运行于独立的 `git worktree` + Daytona 沙箱中）独立实现子任务。**Reviewer**（GPT-5.4）阅读合并后的代码差异 (merged diff)，并决定批准或要求具体修改。**Tester**（Gemini 2.5 Pro）在隔离环境中运行测试套件，并附带构建产物 (artifacts) 报告通过/失败结果。

通信通过共享任务看板 (shared task board)（基于文件或 Redis）进行。每个角色仅处理其权限范围内的任务。任务交接 (handoffs) 采用基于 A2A 协议类型化的消息 (A2A-protocol-typed messages)。协调机制需关注：合并冲突解决 (merge-conflict resolution，由协调员角色或自动三路合并 (three-way merge) 处理)、共享状态同步 (shared-state synchronization，编码员启动后计划即冻结，重新规划视为独立事件)，以及评审员把关机制 (reviewer gatekeeping，评审员不得批准自身修改或由其提议的变更)。

Token 消耗膨胀 (Token amplification) 是其中的隐性成本。每个角色边界都会增加摘要提示词 (summary prompts) 和交接上下文 (handoff context)。原本单智能体 (single-agent) 运行 40 轮的任务，在四个角色间流转后总轮数将增至 160 轮。评估标准 (rubric) 特别权衡了 Token 效率与单智能体基线的对比，因为核心问题并非“多智能体 (multi-agent) 能否工作”，而是“它能否在单位成本上胜出”。

## Architecture

GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder

## Stack

- 编排 (Orchestration)：LangGraph，结合共享状态与按智能体划分的子图 (per-agent sub-graphs)
- 消息传递 (Messaging)：A2A 协议 (Google 2025)，用于传输类型化的智能体间消息 (typed inter-agent messages)
- 模型 (Models)：Opus 4.7（Architect）、Sonnet 4.7（Coders）、GPT-5.4（Reviewer）、Gemini 2.5 Pro（Tester）
- 工作树隔离 (Worktree isolation)：每位 Coder 独立执行 `git worktree add` + Daytona 沙箱
- 合并协调器 (Merge coordinator)：自定义三路合并 (three-way merge) + 大语言模型 (LLM) 介入的冲突解决
- 评估 (Eval)：SWE-bench Pro（50 个 Issue）、SWE-AF 场景、HumanEval++（用于单元测试）
- 可观测性 (Observability)：Langfuse，采用带角色标签的调用链跨度 (role-tagged spans) 及按智能体维度的 Token 核算
- 部署 (Deployment)：K8s，每个角色部署为独立的 Deployment，并根据任务积压量 (backlog) 配置 HPA（水平 Pod 自动扩缩容）

## 构建实施

1. **任务看板 (Task Board)。** 基于文件的 JSONL (JSON Lines) 格式，包含带类型的消息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。智能体 (Agents) 通过订阅标签进行通信。

2. **架构师 (Architect)。** 读取 GitHub Issue，使用计划模板调用 Opus 4.7 模型，该模板要求明确定义子任务接口（涉及的文件、公共函数、测试影响）。随后发出一个包含子任务有向无环图 (DAG) 的 `plan_request` 消息。

3. **编码员 (Coders)。** N 个并行工作节点，各自从看板领取一个子任务。每个节点会新建一个 `git worktree add` 分支并配置 Daytona 沙箱 (Sandbox) 环境。完成子任务实现后，发出 `diff_ready` 消息，附带代码补丁 (patch) 与测试变更差异。

4. **合并协调器 (Merge Coordinator)。** 待所有编码员完成后，将 N 个分支三路合并 (three-way merge) 至暂存分支。仅在出现文件级重叠时，才由大语言模型 (LLM) 介入进行冲突解决。

5. **审查员 (Reviewer)。** GPT-5.4 读取合并后的代码差异 (diff)。该模型无法批准由其自身生成的差异。审查完成后，发出 `approved`（无操作 (no-op)）或 `review_feedback` 消息，后者包含具体的修改要求，并自动路由回对应的编码员。

6. **测试员 (Tester)。** Gemini 2.5 Pro 在隔离沙箱中运行测试套件，并捕获相关产物 (artifacts)。随后发出 `test_passed` 或 `test_failed` 消息，附带堆栈跟踪 (stacktraces)。若测试失败，流程将回退至负责该失败子任务的编码员。

7. **交接核算 (Handoff Accounting)。** 每条跨越角色边界的消息都会在 Langfuse 中记录为一个追踪跨度 (span)，包含负载大小与所用模型信息。计算每个子任务的 Token 放大率（公式：`coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens`）。

8. **评估 (Evaluation)。** 在 50 个 SWE-bench Pro 任务上运行测试。将 pass@1 指标与单问题解决成本 ($-per-solved-issue) 与单智能体基线 (Single-Agent Baseline)（在单个工作树中运行一个 Sonnet 4.7）进行对比。

9. **事后复盘 (Post-mortem)。** 针对每个失败的任务，定位发生断裂的交接环节（如计划过于模糊、合并冲突、审查员误批、测试用例不稳定 (flake) 等）。生成交接失败分布直方图 (Histogram)。

## 使用方法

$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m

## 发布

`outputs/skill-multi-agent-team.md` 是最终的交付物。给定问题（Issue）URL 和并行度级别后，该团队将生成一个可直接合并的拉取请求（Pull Request, PR），并附带按角色统计的 Token 消耗明细。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 匹配 50 个问题子集，计算 pass@1 |
| 20 | 并行加速比（Parallel speedup） | 实际运行时间（Wall-clock time）与单智能体基线对比 |
| 20 | 审查质量 | 注入缺陷探测中的误批率（False-approval rate） |
| 20 | Token 效率 | 每个已解决问题的总 Token 消耗与单智能体对比 |
| 15 | 协同工程（Coordination engineering） | 合并冲突解决情况、交接失败直方图 |
| **100** | | |

## 练习

1. 在运行中途向代码差异（diff）中注入一个明显的缺陷（例如在主逻辑前额外添加 `return None`）。测量审查者的误批率。调整审查者提示词（prompt），直到误批率降至 5% 以下。

2. 将编码角色缩减为两人（架构师 + 编码员 + 审查员 + 测试员，其中编码员按顺序执行两个子任务）。对比实际运行时间与通过率。

3. 用单写入者约束（single-writer constraint）替代合并协调器（Merge coordinator）（即各子任务操作互不相交的文件集）。测量架构师的规划负担。

4. 将审查模型从 GPT-5.4 替换为 Claude Opus 4.7。测量误批率与 Token 成本的变化量（delta）。

5. 增加第五个角色：文档撰写员（使用 Haiku 4.5）。审查完成后，该角色生成变更日志（changelog）条目。评估文档质量是否足以证明额外 Token 消耗的合理性。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 并行工作树（Parallel worktree） | “隔离分支” | 通过 `git worktree add` 为每位编码员生成独立的工作树 |
| 任务看板（Task board） | “共享消息总线” | 智能体订阅的类型化消息的文件或 Redis 存储 |
| 任务交接（Handoff） | “角色边界” | 任何从一个角色上下文传递到另一个角色上下文的消息 |
| Token 放大率（Token amplification） | “多智能体开销” | 所有角色的总 Token 消耗 / 完成同一任务的单智能体 Token 消耗 |
| A2A 协议（A2A protocol） | “智能体间通信” | Google 2025 年发布的类型化智能体间消息规范 |
| 合并协调器（Merge coordinator） | “集成器” | 执行三路合并（three-way merge）并调解冲突的组件 |
| 误批（False approval） | “审查者幻觉” | 审查者批准了包含已知缺陷的代码差异（diff） |

## 延伸阅读

- [SWE-AF 工厂架构](https://github.com/Agent-Field/SWE-AF) — 2026 年多智能体工厂参考实现
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — 基于角色的多智能体框架
- [AutoGen v0.4](https://github.com/microsoft/autogen) — 微软的类型化 Actor 框架
- [Cognition AI (Devin)](https://cognition.ai) — 参考产品
- [Factory Droids](https://www.factory.ai) — 替代参考产品
- [Google A2A 协议](https://developers.google.com/agent-to-agent) — 智能体间消息传递规范
- [git worktree 文档](https://git-scm.com/docs/git-worktree) — 隔离底层机制
- [SWE-bench Pro](https://www.swebench.com) — 评估基准目标