# 综合实战项目 15 — 宪法安全护栏 + 红队靶场

> Anthropic 的宪法分类器（Constitutional Classifiers）、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 内容安全模型（Nemotron 3 Content Safety）以及支持多语言覆盖的 X-Guard，共同定义了 2026 年的安全分类器技术栈（safety-classifier stack）。garak、PyRIT、NVIDIA Aegis 和 promptfoo 已成为标准的对抗性评估工具（adversarial evaluation tools）。NeMo Guardrails v0.12 将它们整合至生产流水线（production pipeline）中。本综合实战项目将所有组件串联起来：围绕目标应用构建分层安全护栏（layered safety harness），部署运行 6 种以上攻击族（attack families）的自主红队智能体（autonomous red-team agent），并执行基于宪法的自我审查流程（constitutional self-critique run），以产出可量化的无害性差异指标（harmlessness delta）。

**类型：** 综合实战项目
**语言：** Python（安全流水线、红队），YAML（策略配置）
**前置要求：** 第 10 阶段（从零构建大语言模型）、第 11 阶段（大语言模型工程）、第 13 阶段（工具）、第 14 阶段（智能体）、第 18 阶段（伦理、安全与对齐）
**涉及阶段：** P10 · P11 · P13 · P14 · P18
**耗时：** 25 小时

## 问题描述

2026 年大语言模型（Large Language Model, LLM）安全的前沿挑战已不再是分类器是否有效（它们大致有效），而是如何围绕生产环境应用正确组合这些分类器，以避免过度拒绝（over-refusing）或留下明显漏洞。Llama Guard 4 负责处理英文策略违规。X-Guard（支持 132 种语言）应对多语言越狱攻击（jailbreak）。ShieldGemma-2 用于拦截基于图像的提示词注入（prompt injection）。NVIDIA Nemotron 3 Content Safety 覆盖企业级分类场景。Anthropic 的宪法分类器则是一种独立的方法，主要用于训练阶段而非推理服务阶段。

攻击手段的演进同样关键。PAIR 和 TAP 实现了越狱攻击的自动化发现。GCG 执行基于梯度的后缀攻击（suffix attacks）。多轮对话与语码切换攻击（code-switch attacks）则利用智能体记忆进行突破。任何已部署的 LLM 都需要配备红队靶场——garak 和 PyRIT 是标准的驱动工具——并附带文档化的缓解策略（mitigations）及基于 CVSS 评分的漏洞发现报告。

你将对一个目标应用（可以是 8B 指令微调模型（instruction-tuned model），或其他综合实战项目中的检索增强生成（Retrieval-Augmented Generation, RAG）聊天机器人）进行安全加固，对其运行 6 种以上的攻击族测试，并输出加固前后的无害性（harmlessness）对比评估指标。

## 核心概念

安全管道（Safety Pipeline）共分为五层。**输入清洗（Input Sanitize）**：剥离零宽字符（Zero-width Chars），解码 Base64/ROT13，规范化 Unicode。**策略层（Policy Layer）**：NeMo Guardrails v0.12 护栏（Off-domain、毒性检测、个人身份信息（PII）提取）。**分类器网关（Classifier Gate）**：输入端使用 Llama Guard 4，非英语内容使用 X-Guard，图像输入使用 ShieldGemma-2。**模型（Model）**：目标大语言模型（LLM）。**输出过滤器（Output Filter）**：输出端使用 Llama Guard 4，Presidio PII 脱敏，并在适用场景下强制引用检查。**人机协同层（HITL Tier）**：标记为高风险的输出将进入 Slack 队列。

红队演练环境（Red-team Range）由调度器驱动运行。PAIR 和 TAP 可自主发现越狱（Jailbreak）漏洞。GCG 执行基于梯度的后缀攻击（Suffix Attacks）。此外还包括 ASCII/Base64/ROT13 编码攻击、多轮对话攻击（角色代入、记忆利用）以及代码切换攻击（Code-switch Attacks，混合英语与斯瓦希里语或泰语等）。每次运行都会生成一份结构化的发现报告文件，其中包含通用漏洞评分系统（CVSS）评分与披露时间线。

宪法式自我批判（Constitutional Self-Critique）流程是一种训练期干预手段。选取 1000 条有害尝试提示词（Prompts），让模型生成初步回复，随后依据书面宪法（即“不造成伤害”规则）对其进行批判，并基于该批判循环进行再训练。最后在独立评估集（Held-out Eval）上测量干预前后的无害性（Harmlessness）差异。

## 架构（Architecture）

request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta

## 技术栈（Stack）

- 安全分类器（Safety Classifiers）：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- 护栏框架（Guardrail Framework）：NeMo Guardrails v0.12 + OPA
- 红队驱动工具（Red-team Drivers）：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- 越狱代理（Jailbreak Agents）：PAIR（Chao et al., 2023）、Tree-of-Attacks（TAP）、GCG 后缀攻击
- 宪法式训练（Constitutional Training）：Anthropic 风格的自我批判循环 + 基于批判的监督微调（SFT）
- PII 脱敏（PII Scrub）：Presidio
- 目标模型（Target）：一个 8B 指令微调（Instruction-tuned）模型，或其他毕业项目中的检索增强生成（RAG）聊天机器人之一

## 动手构建（Build It）

1. **目标环境搭建。** 在 vLLM 上部署一个 8B 指令微调模型（instruction-tuned model）（或复用其他综合实践项目（capstone）中的 RAG 聊天机器人）。该模型即为被测应用。

2. **安全流水线封装。** 围绕目标模型接入五层安全流水线（safety pipeline）。验证每一层均可独立观测（在 Langfuse 中为每层配置独立的追踪跨度（span））。

3. **分类器覆盖范围。** 加载 Llama Guard 4、X-Guard（多语言版）和 ShieldGemma-2（图像版）。在小型标注数据集上分别运行各模型，以建立性能基线（baseline）。

4. **红队调度器。** 调度运行 garak、PyRIT、PAIR 智能体（agent）、TAP 智能体、GCG 运行器、多轮攻击器（multi-turn attacker）以及语码转换攻击器（code-switch attacker）。每个攻击器均在独立的队列中运行。

5. **攻击套件。** 包含六大攻击类别（attack families）：(1) PAIR 自动化越狱（jailbreak），(2) TAP 攻击树（tree-of-attacks），(3) GCG 梯度后缀（gradient suffix），(4) ASCII / Base64 / ROT13 编码，(5) 多轮人设诱导（multi-turn persona），(6) 多语言语码转换（code-switch）。需报告每个类别的攻击成功率。

6. **宪法式自我批判。** 整理 1000 条包含有害意图的提示词（prompts）。针对每条提示词，目标模型生成回复草稿。由一个批评者大语言模型（critic LLM）根据书面“宪法”（如“不造成伤害”、“引用证据”、“拒绝非法请求”）进行评分。对批评者提出异议的样本进行重写；目标模型随后在经批判改进的配对数据上进行微调（fine-tune）。最后在独立评估集（held-out eval）上测量微调前后的无害性（harmlessness）指标。

7. **过度拒绝评估。** 在良性提示词集（如 XSTest）上追踪假阳性率（false-positive rate）。目标模型必须在回答良性问题时保持有益性（helpfulness）。

8. **CVSS 评分。** 针对每次成功越狱（jailbreak），使用 CVSS 4.0 标准（涵盖攻击向量、复杂度、影响）进行评分。并制定漏洞披露时间线与缓解计划（mitigation plan）。

9. **靶场自动化。** 上述所有流程均通过 cron 定时任务自动运行；测试结果写入消息队列；过度拒绝回归（regression）警报将自动推送至 Slack。

## 使用方法

$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)

## 交付成果

`outputs/skill-safety-harness.md` 为最终交付物。包含一套生产级分层安全流水线，以及一个可复现的红队演练靶场，并附带微调前后的无害性差异指标。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 攻击面覆盖度 | 覆盖 6 种以上攻击类别，支持 2 种以上语言 |
| 20 | 真阳性/假阳性权衡 | 攻击拦截率与 XSTest 良性通过率对比 |
| 20 | 自我批判改进幅度 | 独立评估集上微调前后的无害性差异 |
| 20 | 文档与披露 | 附带时间线的 CVSS 评分发现报告 |
| 15 | 自动化与可复现性 | 全流程 cron 定时运行并配备告警 |
| **100** | | |

## 练习

1. 在检索增强生成（RAG）聊天机器人上运行 garak 的提示注入（prompt-injection）插件，对比启用与禁用输出过滤层（output-filter layer）时的攻击成功率。

2. 新增第七类攻击族（attack family）：基于检索文档的间接提示注入（indirect prompt injection）。评估所需的额外防御措施。

3. 实现“引导式拒绝”（refuse-with-help）模式：当安全护栏（guardrail）触发拦截时，目标模型提供一条更安全的相关回答，而非生硬拒绝。测量 XSTest 基准测试的指标变化（delta）。

4. 多语言覆盖缺口（multilingual coverage gap）：找出 X-Guard 表现欠佳的语种，并构建针对该语种的微调（fine-tune）数据集。

5. 在 30B 参数模型上运行宪法式自我批判（constitutional self-critique），并评估该指标变化（delta）是否呈现规模效应（scales）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 分层安全（Layered safety） | “纵深防御（Defense in depth）” | 在输入、网关、输出及人工介入（HITL）环节部署多重安全护栏 |
| Llama Guard 4 | “Meta 的安全分类器” | 2026 年参考级输入/输出内容分类器 |
| PAIR | “越狱智能体” | 关于大语言模型驱动越狱发现的论文（Chao 等人） |
| TAP | “攻击树” | PAIR 的树搜索变体 |
| GCG | “贪婪坐标梯度” | 基于梯度的对抗后缀攻击 |
| 宪法式自我批判（Constitutional self-critique） | “Anthropic 风格训练” | 目标生成草稿 -> 批评者打分 -> 重写 -> 重新训练 |
| XSTest | “良性探测集” | 用于评估过度拒绝回归现象的基准测试 |
| CVSS 4.0 | “严重性评分” | 安全漏洞发现的标准评分体系 |

## 延伸阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 训练阶段参考
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像与多模态安全
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业级参考
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 支持 132 种语言的多语言安全模型
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 红队测试工具包
- [PyRIT](https://github.com/Azure/PyRIT) — 微软红队测试框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 护栏框架
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 越狱智能体相关论文