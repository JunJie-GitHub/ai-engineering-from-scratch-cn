# 验证门禁（Verification Gate）

> 智能体（Agent）无权自行将工作标记为完成。验证门禁会读取范围契约（Scope Contract）、反馈日志（Feedback Log）、规则报告（Rule Report）以及差异文件（Diff），并回答一个核心问题：该任务是否真正完成？如果门禁判定为否，无论对话记录如何，该任务均视为未完成。

**类型：** 构建（Build）
**语言：** Python（标准库）
**前置条件：** 第14阶段 · 33（规则），第14阶段 · 36（范围），第14阶段 · 37（反馈）
**耗时：** 约55分钟

## 学习目标

- 将验证门禁定义为基于工作台产物（Workbench Artifacts）的确定性函数。
- 将规则报告、范围报告、反馈记录与差异文件整合为单一判定结果。
- 生成 `verification_report.json` 文件，供审查智能体（Reviewer Agent）与持续集成（CI）系统共同读取。
- 遇到任何阻塞级（Block-severity）失败时，一律拒绝推进任务，绝无例外。

## 问题背景

智能体往往过于轻易地宣告成功。主要存在三种典型的失败模式：

- “看起来没问题。”模型仅阅读了自身生成的差异文件，便自行判定为正确。
- “测试已通过。”语气十分笃定，却没有任何实际运行测试的记录。
- “已满足验收标准。”对验收标准的解读过于宽松，以至于“只要看起来像完成了就行”。

工作台的应对方案是设立一个统一的验证门禁，通过读取智能体已产出的工件来做出最终裁决。该门禁具备确定性，已纳入版本控制，并直接接入持续集成（CI）流水线。智能体无法对其施加任何干扰。

## 核心概念

flowchart TD
  Diff[Diff] --> Gate[verify_agent.py]
  Scope[scope_report.json] --> Gate
  Rules[rule_report.json] --> Gate
  Feedback[feedback_record.jsonl] --> Gate
  Gate --> Verdict[verification_report.json]
  Verdict --> Pass{passed?}
  Pass -- yes --> Review[Reviewer Agent]
  Pass -- no --> Refuse[refuse done + surface to human]

### 门控 (Gate) 检查项

| 检查项 | 来源产物 (Artifact) | 严重级别 |
|-------|-----------------|----------|
| 所有验收命令均已执行 | `feedback_record.jsonl` | 阻断 (block) |
| 所有验收命令退出码为零 | `feedback_record.jsonl` | 阻断 (block) |
| 范围检查无禁止写入操作 | `scope_report.json` | 阻断 (block) |
| 范围检查无越界写入操作 | `scope_report.json` | 阻断 (block) 或 警告 (warn) |
| 所有阻断级别规则均通过 | `rule_report.json` | 阻断 (block) |
| 反馈中无 `null` 退出码 | `feedback_record.jsonl` | 阻断 (block) |
| 修改的文件与 `scope.allowed_files` 匹配 | 两者 | 警告 (warn) |

警告 (warn) 级别的发现会标注在判定结果中；阻断 (block) 级别的发现将阻止 `passed: true` 的生成。

### 确定性而非概率性

对于相同的产物 (Artifact) 集合，门控 (Gate) 每次必须输出相同的判定结果。禁止使用大语言模型 (LLM) 进行判定。大语言模型 (LLM) 判定应归属于审查端（第 14 阶段 · 39），其目标是定性评估，而非状态判定。

### 单一报告，单一输出路径

每个任务结束时，门控 (Gate) 会生成一个 `verification_report.json`，并写入 `outputs/verification/<task_id>.json` 路径下。持续集成 (CI) 系统也读取该路径。若多个门控 (Gate) 使用不同路径，将导致单一事实来源 (Source of Truth) 分叉。

### 严格拒绝，无例外

阻断 (block) 级别的发现不能被智能体 (Agent) 覆盖。它们只能由人工覆盖，并需记录 `override_reason` 和 `overridden_by` 用户 ID。该覆盖操作属于已签名的变更，而非智能体 (Agent) 的决策。

## 构建实现

`code/main.py` 实现了：

- 每个输入产物 (Artifact) 的加载器，均在本地使用桩代码 (Stub) 模拟，以确保本教程自包含。
- 一个纯函数 (Pure Function) `verify(task_id, artifacts) -> VerdictReport`。
- 一个打印器，用于展示各项检查结果及最终的通过/失败状态。
- 包含三种任务场景的演示：完全通过、范围蔓延 (Scope Creep)、缺失验收条件。

运行方式：

python3 code/main.py

输出：三份判定报告，均保存在脚本同级目录下。

## 实际生产环境中的模式

四种模式将验证门（verification gate）从“又一个代码检查任务”提升为“决定性关卡”。

**纵深防御（Defense-in-depth），而非单一关卡。** 提交前钩子（Pre-commit hook）→ 持续集成状态检查（CI status check）→ 工具调用前授权钩子（pre-tool authz hook）→ 合并前关卡（pre-merge gate）。每一层均具备确定性，因此某一层的失效会被下一层捕获。microservices.io 2026 年 3 月的操作手册明确指出：提交前钩子不可绕过，因为与模型端技能（model-side skill）不同，它不依赖于智能体（agent）遵循指令。验证门部署在持续集成（CI）/ 合并前层。

**确定性检查防御，模型评判仅处理细微差异。** Anthropic 2026 年的混合规范（Hybrid Norm）配对方案：可验证奖励（verifiable rewards，如单元测试、模式检查、退出码）用于回答“代码是否解决了问题？”；大语言模型评分量表（LLM rubrics）用于回答“代码是否易读、安全、符合规范？”验证门负责执行第一类检查；审查员（Phase 14 · 39）负责执行第二类检查。混合两者会导致评估信号失效。

**带签名的覆盖日志，而非 Slack 讨论串。** 每次覆盖（override）操作均会在 `outputs/verification/overrides.jsonl` 中追加一行记录，包含：时间戳、问题代码、原因、签名用户、当前 HEAD 提交。运行时会拒绝任何缺少签名的覆盖请求；审计轨迹（audit trail）由 Git 进行版本跟踪。这正是真正的覆盖策略与“覆盖表演”（override theater）之间的分水岭。

**将覆盖率底线（coverage floor）作为核心检查项。** `coverage_report.json` 为 `coverage_floor`（默认 80%）检查提供输入数据。若实测覆盖率低于该底线，或较上一次合并的底线下降超过 1 个百分点，验证门将判定失败。若缺少此项检查，智能体可能会静默删除未通过的测试用例，导致验证报告持续显示为绿色（通过状态）。

**`--strict` 模式将警告升级为阻断。** 针对发布分支、阻塞发布的拉取请求（ship-blocking PRs）或事后故障排查（post-incident triage），`--strict` 会将所有警告转为硬性失败（hard fail）。该标志按分支选择性启用（opt-in by branch），而非全局默认，因为“处处严格”会严重拖慢日常开发流程。

## 使用方式

生产环境模式：

- **持续集成（CI）步骤。** `verify_agent` 任务会针对智能体的最终产物运行验证门。若未返回 `passed: true`，合并保护机制将拒绝合并。
- **交接前钩子（Pre-handoff hook）。** 智能体运行环境在生成交接文档前会调用验证门。未获绿色（通过）裁决，则不执行交接。
- **人工排查。** 当智能体报告成功但人工存疑时，运维人员将查阅该报告。

验证门是工作台（workbench）流程中的决定性关卡。所有其他交互界面均位于其上游。

## 交付部署

`outputs/skill-verification-gate.md` 将验证门接入具体项目：明确哪些验收命令为其提供输入、哪些规则属于阻断级别、允许哪些越权写入操作，以及覆盖审计日志的存储方式。

## 练习

1. 添加 `coverage_floor`（覆盖率下限）检查：测试命令必须生成覆盖率至少达到 80% 的报告。确定由哪个构建产物（artifact）承载该下限要求。
2. 支持 `--strict`（严格）模式，该模式会将所有 `warn`（警告）升级为 `block`（阻断）。记录应将严格模式作为默认选项的适用场景。
3. 使验证门禁（gate）除输出 JSON 外，还能生成 Markdown 摘要。论证摘要中应包含哪些字段及其设计依据。
4. 添加 `time_since_last_human_touch`（距上次人工操作时间）检查：在人工按键操作后 60 秒内编辑的任何文件，均可豁免越界标记（off-scope flags）。
5. 在实际产品的智能体（agent）代码差异（diff）上运行该门禁。统计其中有多少是真实检测结果（findings），多少是噪声（noise）？该门禁在哪些方面还需要进一步演进？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| Verification gate（验证门禁） | “拦截检查” | 基于工作台构建产物（workbench artifacts）运行的确定性函数，用于输出通过/失败判定 |
| Block severity（阻断严重级别） | “硬性失败” | 会阻止 `passed: true` 的检测结果，且必须经过签名豁免（signed override）才能放行 |
| Override log（豁免日志） | “放行原因” | 包含原因和用户 ID 的签名记录，供审查流程审计 |
| Acceptance command（验收命令） | “完成凭证” | 退出码为 0 即代表任务 `done`（完成）的 Shell 命令 |
| One report path（单一报告路径） | “唯一事实来源” | `outputs/verification/<task_id>.json`，供 CI 系统与人工共同读取 |

## 延伸阅读

- [Anthropic，面向长周期应用开发的测试框架（Harness）设计](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- [OpenAI Agents SDK 护栏（guardrails）](https://platform.openai.com/docs/guides/agents-sdk/guardrails)
- [microservices.io，GenAI 开发平台：护栏（guardrails）](https://microservices.io/post/architecture/2026/03/09/genai-development-platform-part-1-development-guardrails.html) —— pre-commit 与 CI 之间的纵深防御（defense in depth）
- [ICMD，2026 年智能体 AI 运维（Agentic AI Ops）实战指南](https://icmd.app/article/the-2026-playbook-for-agentic-ai-ops-guardrails-costs-and-reliability-at-scale-1776661990431) —— 审批门禁阶梯（approval-gate ladder）（草稿 → 审批 → 低于阈值自动通过）
- [类型检查合规性：确定性护栏（arXiv 2604.01483）](https://arxiv.org/pdf/2604.01483) —— 将 Lean 4 作为确定性门禁（deterministic gating）的上限
- [logi-cmd/agent-guardrails —— 合并门禁规范（merge gate spec）](https://github.com/logi-cmd/agent-guardrails) —— 范围（scope）与变异测试（mutation-testing）门禁
- [Guardrails AI 与 MLflow 集成](https://guardrailsai.com/blog/guardrails-mlflow) —— 将确定性验证器（deterministic validators）作为 CI 评分器
- [Akira，面向智能体系统的实时护栏（guardrails）](https://www.akira.ai/blog/real-time-guardrails-agentic-systems) —— 工具调用前/后（pre/post-tool）门禁
- 第 14 阶段 · 27 —— 提示词注入（prompt injection）防御（该门禁的对抗性配对）
- 第 14 阶段 · 36 —— 该门禁强制执行的范围契约（scope contract）
- 第 14 阶段 · 37 —— 该门禁进行评分的反馈日志（feedback log）
- 第 14 阶段 · 39 —— 该门禁交接的审查智能体（reviewer agent）