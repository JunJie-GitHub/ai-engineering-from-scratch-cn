# 工具使用与函数调用

> Toolformer（Schick 等，2023）开创了自监督工具标注（self-supervised tool annotation）。伯克利函数调用排行榜 V4（Berkeley Function Calling Leaderboard V4，Patil 等，2025）设定了 2026 年的评估基准：40% 智能体（agentic）任务、30% 多轮（multi-turn）交互、10% 实时（live）场景、10% 非实时（non-live）场景以及 10% 幻觉（hallucination）测试。单轮（single-turn）调用已不再是难题，但记忆（memory）、动态决策（dynamic decision-making）与长程工具链（long-horizon tool chains）仍有待突破。

**Type:** 构建（Build）
**Languages:** Python（标准库 stdlib）
**Prerequisites:** 第 14 阶段 · 01（智能体循环 Agent Loop），第 13 阶段 · 01（函数调用深度解析 Function Calling Deep Dive）
**Time:** 约 60 分钟

## 学习目标

- 解释 Toolformer 的自监督训练信号（self-supervised training signal）：仅当工具执行能降低下一词元损失（next-token loss）时，才保留工具标注。
- 列举 BFCL V4 的五个评估类别及其各自的衡量指标。
- 使用标准库实现一个工具注册表（tool registry），包含模式验证（schema validation）、参数强制转换（argument coercion）和执行沙箱（execution sandboxing）。
- 剖析 2026 年的三大开放性问题：长程工具链编排（long-horizon tool chaining）、动态决策（dynamic decision-making）与记忆机制（memory）。

## 问题定义

早期的工具使用关注的是：模型能否预测出正确的函数调用？而现代工具使用则追问：模型能否在具备记忆能力、部分可观测性（partial observability）以及工具故障恢复机制的前提下，连续编排 40 步工具调用，且不产生调用不存在工具的幻觉？

Toolformer 确立了基线（baseline）：模型能够通过自监督学习掌握何时调用工具。BFCL V4 则定义了 2026 年的评估目标。两者之间的差距，正是生产级智能体（production agents）实际所处的应用空间。

## 核心概念

### Toolformer (Schick 等人，NeurIPS 2023)

核心思想：让模型利用候选 API 调用为其自身的预训练语料库（pretraining corpus）添加标注。针对每个候选调用执行相应操作，仅当引入工具结果能够降低下一个词元（token）的损失（loss）时，才保留该标注。随后在筛选后的语料库上进行微调（fine-tuning）。

涵盖工具：计算器、问答系统、搜索引擎、翻译器、日历。其自监督信号（self-supervision signal）纯粹基于工具是否有助于文本预测——无需人工标注。

规模效应结果：工具使用能力随模型规模扩大而涌现。较小模型会因工具标注而性能受损；较大模型则能从中获益。这就是为什么 2026 年的前沿模型（frontier models）已内置强大的工具使用能力，而大多数 7B 模型仍需经过显式的工具使用微调才能稳定可靠。

### Berkeley Function Calling Leaderboard V4 (Patil 等人，ICML 2025)

BFCL 是 2026 年事实上的评估基准（de facto evaluation）。V4 版本构成如下：

- **Agentic (40%)** — 完整的智能体（agent）轨迹：记忆、多轮交互与动态决策。
- **Multi-Turn (30%)** — 包含工具链的交互式对话。
- **Live (10%)** — 用户提交的真实提示词（prompt）（数据分布更难）。
- **Non-Live (10%)** — 合成测试用例。
- **Hallucination (10%)** — 检测何时不应调用任何工具。

V3 引入了基于状态的评估（state-based evaluation）：在执行一系列工具调用后，检查 API 的实际状态（例如“文件是否已创建？”），而非仅匹配工具调用的抽象语法树（AST）。V4 新增了网络搜索、记忆和格式敏感性类别。

2026 年的关键发现：单轮函数调用（single-turn function calling）已基本解决。失败主要集中在记忆（跨轮次携带上下文）、动态决策（根据先前结果选择工具）、长程链条（20 步以上后出现漂移）以及幻觉检测（在没有合适工具时拒绝调用）。

### 工具模式（Tool schema）

每个服务提供商（provider）都有其模式定义。它们在细节上有所不同，但整体结构一致：

name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)

Anthropic 直接使用 `input_schema`。OpenAI 使用 `function.parameters`。两者均接受 JSON Schema。描述字段承载关键信息（load-bearing）——模型会读取它们以选择正确的工具。糟糕的工具描述是导致选错工具故障的首要原因。

### 参数验证（Argument validation）

不要信任任何工具调用。必须进行验证：

1. **类型强制转换（Type coercion）**。模型可能在模式要求整型（int）时返回字符串 `"5"`。若含义明确则进行转换；否则拒绝。
2. **枚举验证（Enum validation）**。如果模式规定 `status in {"open", "closed"}` 而模型输出 `"in_progress"`，则拒绝并返回描述性错误。
3. **必填字段（Required fields）**。缺少必填字段 -> 立即向模型返回错误观察结果（error observation），而非直接崩溃。
4. **格式验证（Format validation）**。日期、电子邮件、URL —— 使用具体的解析器进行验证，而非依赖正则表达式（regex）。

每次验证失败都应返回结构化的观察结果，以便模型能够以正确的格式重试。

### 并行工具调用（Parallel tool calls）

现代服务提供商支持在单次助手回复轮次（assistant turn）中进行并行工具调用。执行循环如下：

1. 模型发出 3 个带有不同 `tool_use_id` 的工具调用。
2. 运行时（Runtime）执行它们（若相互独立则并行执行）。
3. 每个结果作为 `tool_result` 块返回，并通过 `tool_use_id` 进行关联。

工程准则：将关联 ID（correlation IDs）视为关键负载。一旦混淆，就会导致工具与结果错误路由。

### 沙盒隔离（Sandboxing）

工具执行构成了沙盒边界。详见第 09 课。简而言之：每个工具都应明确指定读写范围、网络访问权限、超时时间和内存上限。通用的 `run_shell(cmd)` 是危险信号；而具体的 `git_status()` 则更为安全。

## 构建
`code/main.py` 实现了一个具备生产环境形态的工具注册表（tool registry）：
- JSON Schema 子集验证器（仅使用标准库 stdlib）。
- 工具注册功能，包含描述、输入模式（input schema）、超时设置和执行器（executor）。
- 参数强制转换（argument coercion）与枚举验证（enum validation）。
- 基于关联 ID（correlation ID）的并行工具分发（parallel tool dispatch）。
- 以结构化字符串形式记录错误观测信息（error observation）。

运行方式：
python3 code/main.py
追踪日志（trace）展示了一个微型智能体（mini agent）在单轮交互中调用三个工具的过程，其中包含一个故意构造的格式错误调用。该调用会被系统拒绝，并返回描述性错误信息，以便模型据此采取相应操作。

## 使用
每家服务提供商（provider）都有其专属的工具模式（tool schema）——例如 Anthropic、OpenAI、Gemini 和 Bedrock。如果需要支持多提供商，请使用转换层（translation layer）（如 OpenAI Agents SDK、Vercel AI SDK 或 LangChain 工具适配器）。BFCL 是参考基准测试（reference benchmark）——如果工具调用是产品的核心功能，请在发布前针对你的智能体运行该测试。

## 发布
`outputs/skill-tool-registry.md` 会为指定的任务领域生成工具目录（tool catalog）、模式（schema）和注册表。其中包含描述质量检查（description-quality check）（例如：每个工具的描述是否明确告知模型何时该使用它？）。

## 练习
1. 添加一个“空操作”（no-op）工具，允许模型明确拒绝使用其他任何工具。在类似 BFCL 的幻觉测试（hallucination test）中进行评估。
2. 实现针对“字符串转整数”和“字符串转浮点数”的参数强制转换。参数转换在什么情况下会开始掩盖真实的缺陷（bug）？
3. 为每个工具添加超时设置和熔断器（circuit breaker）（连续失败 3 次后，60 秒内拒绝调用该工具）。这会如何改变模型的恢复机制？
4. 阅读 BFCL V4 的说明文档。选择一个类别（例如“多轮对话” multi-turn），将 10 个示例提示词（prompt）输入你的智能体进行测试。报告通过率。
5. 将标准库验证器移植到 Pydantic 或 Zod。Pydantic/Zod 捕获到了哪些简易实现（toy）遗漏的问题？

## 关键术语
| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 函数调用（Function calling） | “工具使用” | 基于已验证模式的结构化输出工具调用 |
| Toolformer | “自监督工具标注” | Schick 2023 提出——保留那些能降低下一个词元损失（next-token loss）的工具调用 |
| BFCL | “伯克利函数调用排行榜” | 2026 年基准测试：40% 智能体任务（agentic），30% 多轮对话，10% 实时环境（live），10% 非实时环境（non-live），10% 幻觉检测 |
| 工具模式（Tool schema） | “面向模型的函数签名” | 包含名称、描述及参数的 JSON Schema |
| tool_use_id | “关联 ID” | 将工具调用与其结果绑定；对并行分发至关重要 |
| 幻觉检测（Hallucination detection） | “知道何时不该调用” | V4 类别：当没有合适工具时拒绝调用 |
| 参数强制转换（Argument coercion） | “字符串转整数修复” | 针对可预见的模式不匹配进行精准修复；若存在歧义则拒绝 |
| 沙盒隔离（Sandboxing） | “工具执行边界” | 针对每个工具的读写范围、网络访问、超时限制和内存上限 |

## 延伸阅读

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — 自监督工具标注（Self-supervised Tool Annotation）
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — 2026年评估基准（2026 Evaluation Benchmark）
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Agent SDK 中的生产级工具定义（Production Tool Schema）
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 函数工具类型（Function Tool Type）与安全护栏（Guardrails）