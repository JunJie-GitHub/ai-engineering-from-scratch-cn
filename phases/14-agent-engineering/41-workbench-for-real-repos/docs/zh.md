# 在真实代码库中使用工作台

> 如果无法经受真实代码库的检验，前面十一课关于工作台界面（surfaces）的内容将毫无价值。本课将在一个小型示例应用上运行同一任务两次：仅提示词模式（prompt-only）与工作台引导模式（workbench-guided）。让数据说话。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** 阶段 14 · 32 至 14 · 40
**耗时：** 约 60 分钟

## 学习目标

- 在小型应用上整合七个工作台界面（surfaces）。
- 运行同一任务两次（仅提示词模式与工作台引导模式），并测量五项结果指标。
- 阅读前后对比报告，判断哪些界面带来了最大的效能提升。
- 针对“我的模型已经足够好”的质疑，为工作台的价值进行辩护。

## 问题背景

针对玩具任务的演示毫无说服力。只有当工作台能够处理真实仓库中的真实任务，并以更少的失败率、更少的回滚（reverts）以及生成可供下一次会话使用的交付包（packet）成功上线时，其价值才得以体现。

本课将提供这样一个贴近真实的代码库，并通过两条流水线（pipelines）运行同一任务。最终生成的前后对比报告，足以让持怀疑态度的人信服。

## 核心概念

flowchart TD
  Task[Task: validate /signup and add tests] --> A[Prompt-only run]
  Task --> B[Workbench-guided run]
  A --> M[Measure: 5 outcomes]
  B --> M
  M --> Report[before-after-report.md]

### 示例应用

位于 `sample_app/` 目录下的一个极简 FastAPI 风格处理器：

- `app.py`：包含 `/signup` 端点（尚未添加验证逻辑）。
- `test_app.py`：包含一个正常路径（happy-path）测试用例。
- `README.md` 和 `scripts/release.sh`：作为“禁止修改区域”的诱饵文件。

### 任务目标

> 为 `/signup` 添加输入验证：拒绝长度小于 8 个字符的密码，返回 422 状态码及类型化的错误响应体（typed error envelope）。添加一个测试用例以验证新行为。

### 两条流水线

仅提示词模式：

1. 阅读 README。
2. 阅读 `app.py`。
3. 编辑文件。
4. 声明完成。

工作台引导模式：

1. 运行初始化脚本（第 35 课）。
2. 阅读范围契约（scope contract，第 36 课）。
3. 读取状态（第 34 课）。
4. 仅编辑允许范围内的文件。
5. 通过反馈运行器（feedback runner）执行验收命令（第 37 课）。
6. 运行验证关卡（verification gate，第 38 课）。
7. 运行审查器（reviewer，第 39 课）。
8. 生成交接文档（handoff，第 40 课）。

### 测量的五项结果指标

| 结果指标 | 重要性说明 |
|---------|----------------|
| `tests_actually_run` | 大多数“测试通过”的声明无法验证 |
| `acceptance_met` | 证明目标达成的测试必须是实际执行过的测试 |
| `files_outside_scope` | 范围蔓延（scope creep）是导致失败的主要隐性原因 |
| `handoff_quality` | 下一次会话将为此付出代价或从中受益 |
| `reviewer_total` | 在验证关卡之上的定性评估 |

## 开始构建

`code/main.py` 针对同一个示例应用测试夹具（fixture）编排了两条流水线。两条流水线均为纯脚本运行（无大语言模型（LLM）介入循环），因此测量结果具备可复现性。该脚本会将对比结果输出至 `before-after-report.md` 和 `comparison.json`。

运行方式：

python3 code/main.py

输出结果：包含各流水线执行结果的终端表格、保存在脚本同目录下的 Markdown 报告，以及供后续绘制图表使用的 JSON 文件。

## 业界生产实践模式

质疑者常问：“工作台（workbench）究竟能带来多大帮助？”2026 年的数据本身已胜过千言万语。

**同一模型在 Terminal Bench 上从 Top-30 跃升至 Top-5。** LangChain 发布的《智能体控制架构剖析》（*Anatomy of an Agent Harness*，2026 年 4 月）指出：仅通过优化控制架构（harness），一个编程智能体在 Terminal Bench 2.0 上的排名便从 30 名开外跃升至第五。模型相同，工具界面（surfaces）不同，排名差距达 25 位。

**Vercel 通过删减工具将成功率从 80% 提升至 100%。** Vercel 报告称，在移除其智能体 80% 的工具后，任务成功率从 80% 飙升至 100%。工具界面更精简，作用域更聚焦，失败路径大幅减少。精简策略（负空间）反而胜出。

**Harvey 仅凭控制架构优化实现准确率翻倍。** 法律智能体仅通过优化控制架构（harness），准确率便提升了一倍以上，底层模型未作任何更改。

**88% 的企业级 AI 智能体项目未能投入生产。** preprints.org 发布的《语言智能体控制架构工程》（*Harness Engineering for Language Agents*，2026 年 3 月）论文指出，失败根源在于运行时（runtime）而非推理能力：状态陈旧、重试机制脆弱、上下文过度膨胀，以及缺乏从中间错误中有效恢复的能力。

**长上下文崩溃（Long-context collapse）。** WebAgent 的基线成功率为 40-50%，但在长上下文条件下会骤降至 10% 以下，主要诱因是无限循环与目标丢失。Ralph Loop 与交接数据包（handoff packet）的设计初衷正是为了缓冲此类问题。

**假阴性（False negatives）依然存在。** 单步事实性任务、单行代码检查（lints）、格式化运行，以及任何模型已逐字记忆的内容——这些任务仅靠提示词（prompt-only）运行反而更快。基准测试应如实枚举此类场景，以免工作台被误认为“过度设计”。

核心结论并非“控制架构（harness）永远制胜”。模型确实会随时间推移吸收这些架构技巧。真正的结论是：在当下，工程负载主要集中在七个工具界面（surfaces）上，数据已充分证明了这一点。

## 使用场景

当遇到以下情况时，本案例可作为你引用的核心依据：

- 有人质疑为何每个拉取请求（PR）都必须附带 `agent-rules.md` 与作用域契约（scope contract）。
- 团队希望“仅在本迭代”跳过验证关卡。
- 新智能体产品上线，你需要一套可移植的基准测试来验证其是否真正提升了效率。

数据本身的说服力远超文字解释。

## 交付部署

`outputs/skill-workbench-benchmark.md` 是一套可移植的评估控制架构（evaluation harness）。它可针对项目自身的示例应用，将任意智能体产品置于两条流水线中运行，并输出五项核心指标结果。

## 练习

1. 增加第六个评估指标：首次有效编辑耗时 (time-to-first-meaningful-edit)。如何对其进行清晰准确的测量？
2. 在你的代码库中选取一个真实的次日任务 (second-day task) 运行对比测试。工作台 (workbench) 的数据在哪些环节会出现偏差或不及预期？
3. 增加一次“假阴性” (false negative) 排查：找出那些仅靠提示词 (prompt-only) 反而更快、且工作台开销构成实际成本的任务。即便如此，仍需论证保留工作台的合理性。
4. 将脚本化的“智能体” (agent) 替换为真实的大语言模型 (LLM) 调用。哪些评估指标的结果会变得更加不稳定（噪声增加）？
5. 撰写一份面向非工程师的一页纸摘要。经过精简后，哪些核心内容得以保留？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 示例应用 (Sample app) | “玩具级仓库” (Toy repo) | 规模虽小但足够真实，能够完整覆盖全部七个交互面 (surfaces) |
| 流水线 (Pipeline) | “工作流” (Workflow) | 智能体遵循的、按顺序执行的交互面读写序列 |
| 前后对比报告 (Before/after report) | “证据/凭据” (The receipts) | 提交给质疑者的交付物 |
| 假阴性 (False negative) | “工作台大材小用” (Workbench overkill) | 仅用提示词反而更快的任务；如实列举此类情况具有重要参考价值 |
| 工作台基准测试 (Workbench benchmark) | “可靠性评分” (Reliability score) | 可在你的代码库中运行对比测试的可移植测试框架 (harness) |

## 延伸阅读

- [LangChain, The Anatomy of an Agent Harness](https://blog.langchain.com/the-anatomy-of-an-agent-harness/) — Terminal Bench 从 Top-30 跃升至 Top-5 的实证数据
- [MongoDB, The Agent Harness: Why the LLM Is the Smallest Part of Your Agent System](https://www.mongodb.com/company/blog/technical/agent-harness-why-llm-is-smallest-part-of-your-agent-system) — Vercel 与 Harvey 的实际指标
- [preprints.org, Harness Engineering for Language Agents](https://www.preprints.org/manuscript/202603.1756) — 88% 的企业级失败率及运行时根本原因分析
- [HN: Improving 15 LLMs at Coding in One Afternoon. Only the Harness Changed](https://news.ycombinator.com/item?id=46988596) — 在 15 个模型上成功复现
- [Cloudflare, Orchestrating AI Code Review at Scale](https://blog.cloudflare.com/ai-code-review/) — 生产环境 30 天内完成 13.1 万次代码审查运行
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- 第 14 阶段 · 32 至 14 · 40 — 本课程端到端覆盖的交互面
- 第 14 阶段 · 19 — SWE-bench、GAIA、AgentBench 作为本课程补充的宏观基准测试 (macro benchmarks)
- 第 14 阶段 · 30 — 同一测试框架可接入的评估驱动型智能体开发 (eval-driven agent development)