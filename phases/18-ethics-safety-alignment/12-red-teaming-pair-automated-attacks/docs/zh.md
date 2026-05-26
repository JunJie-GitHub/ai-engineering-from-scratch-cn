# 红队测试（Red-Teaming）：PAIR 与自动化攻击

> Chao, Robey, Dobriban, Hassani, Pappas, Wong (NeurIPS 2023, arXiv:2310.08419)。PAIR（Prompt Automatic Iterative Refinement，提示自动迭代优化）是一种典型的自动化黑盒（black-box）越狱（jailbreak）方法。配备红队系统提示词（red-team system prompt）的攻击者大语言模型（Attacker LLM）会为目标大语言模型（Target LLM）迭代生成越狱提示词，并将尝试过程与响应累积在其自身的聊天历史中，作为上下文内反馈（in-context feedback）。PAIR 通常能在 20 次查询内成功，其效率比 GCG（Zou 等人提出的词元级梯度搜索）高出数个数量级，且无需白盒访问权限（white-box access）。目前，PAIR 已成为 JailbreakBench（arXiv:2404.01318）和 HarmBench 中的标准基线（standard baseline），与 GCG、AutoDAN、TAP 以及说服性对抗提示（Persuasive Adversarial Prompt, PAP）并列。

**类型：** 构建
**语言：** Python（标准库，针对示例目标的 PAIR 循环模拟）
**前置要求：** 第 18 阶段 · 01（指令遵循），第 14 阶段（智能体工程）
**耗时：** 约 75 分钟

## 学习目标

- 描述 PAIR 算法：攻击者系统提示词、迭代优化、上下文内反馈。
- 解释当目标为黑盒时，为何 PAIR 的效率严格优于 GCG。
- 列举另外四种自动化攻击基线（GCG、AutoDAN、TAP、PAP），并说明各自的一个显著特征。
- 描述 JailbreakBench 和 HarmBench 的评估协议，并阐明在这两种协议下“攻击成功率”的具体含义。

## 问题背景

红队测试过去主要依赖人工操作。少数专家测试人员会手动构造对抗性提示词（adversarial prompts），并记录哪些提示词能够生效。这种方式难以规模化扩展：攻击成功率需要具有统计意义的样本量，且随着每次模型版本的发布，目标模型本身也在不断变化。PAIR 将红队测试转化为一个针对黑盒目标的优化问题（optimization problem），从而实现了流程的自动化与工程化。

## 核心概念

### PAIR 算法 (PAIR Algorithm)

输入：
- 目标大语言模型 (Target LLM) T（即被攻击的模型）。
- 评判大语言模型 (Judge LLM) J（用于评估回复是否构成越狱 (jailbreak)）。
- 攻击者大语言模型 (Attacker LLM) A（即红队优化器）。
- 目标字符串 (Goal string) G：“回复 [有害指令]。”
- 预算 (Budget) K（通常为 20 次查询）。

循环，k 从 1 到 K：
1. 向 A 输入目标 G 以及迄今为止的（提示词，回复）对历史记录。
2. A 生成一个新的提示词 p_k。
3. 将 p_k 提交给 T；接收回复 r_k。
4. J 根据目标对 (p_k, r_k) 进行评分。
5. 若评分 >= 阈值，则终止——越狱成功。
6. 否则，将 (p_k, r_k) 追加至 A 的历史记录中；继续循环。

实证结果（NeurIPS 2023）：针对 GPT-3.5-turbo 和 Llama-2-7B-chat 的攻击成功率 (Attack Success Rate, ASR) 超过 50%；成功所需的平均查询次数在 10 到 20 次之间。

### PAIR 算法高效的原因

GCG（Zou 等人，2023）通过梯度 (gradient) 搜索对抗性词元后缀 (adversarial token suffixes)；它需要白盒模型 (white-box model) 访问权限，且生成的后缀难以阅读。PAIR 属于黑盒 (black-box) 攻击，能生成可跨模型迁移 (transfer) 的自然语言攻击。PAIR 的上下文反馈 (in-context feedback) 机制使攻击者能够从每次拒绝中学习；而 GCG 缺乏等效机制（每次新的词元更新都必须重新发现之前的进展）。

### 相关的自动化攻击

- **GCG（Zou 等人，2023，arXiv:2307.15043）。** 针对对抗性后缀的词元级 (token-level) 梯度搜索。白盒攻击，具备可迁移性，生成难以阅读的字符串。
- **AutoDAN（Liu 等人，2023）。** 在提示词上进行进化搜索 (evolutionary search)，由分层目标引导。
- **TAP（Mehrotra 等人，2024）。** 带剪枝 (pruning) 的攻击树 (Tree-of-attacks)——分支出多个类似 PAIR 的推演 (rollouts)。
- **PAP（Zeng 等人，2024）。** 说服性对抗提示词 (Persuasive Adversarial Prompts)——将人类说服技巧编码为提示词模板 (prompt templates)。

### JailbreakBench 与 HarmBench

两者（均发布于 2024 年）旨在标准化评估 (standardize evaluation)：

- JailbreakBench（arXiv:2404.01318）。涵盖 10 个 OpenAI 政策类别的 100 种有害行为 (harmful behaviors)。以攻击成功率 (ASR) 作为主要指标。需要依赖评判器 (judge)（如 GPT-4-turbo、Llama Guard 或 StrongREJECT）。
- HarmBench（Mazeika 等人，2024）。涵盖 7 个类别的 510 种行为，包含语义与功能性危害测试 (semantic and functional harm tests)。对比了 18 种攻击方法在 33 个模型上的表现。

ASR 通常在固定的查询预算 (query budget) 下报告。比较不同攻击方法时必须匹配预算；在 200 次查询下达到 90% 的 ASR 与在 20 次查询下达到 85% 的 ASR 不具备可比性。

### 为何这对 2026 年的部署至关重要

如今，每家前沿实验室 (frontier lab) 在发布生产级模型 (production models) 前，都会使用 PAIR 和 TAP 对其进行测试。ASR 变化轨迹会出现在模型卡片 (model cards)（第 26 课）和安全论证附录 (safety-case appendices)（第 18 课）中。此类攻击已不再新奇——它已成为标准基础设施的一部分。

### 本内容在第 18 阶段中的定位

第 12 课是自动化攻击基础 (automated-attack foundation)。第 13 课（多样本越狱 / Many-Shot Jailbreaking）是一种互补的长度利用 (length-exploit) 攻击。第 14 课（ASCII 艺术 / 视觉攻击）属于编码攻击 (encoding attack)。第 15 课（间接提示词注入 / Indirect Prompt Injection）是 2026 年生产环境攻击面 (production attack surface)。第 16 课涵盖了对应的防御工具链 (defensive-tooling counterparts)（如 Llama Guard、Garak、PyRIT）。

## 实践应用

`code/main.py` 构建了一个简易的提示词自动迭代优化（PAIR）循环。目标模型是一个模拟分类器，用于拒绝“明显”的有害提示词（关键词过滤，keyword-filter）。攻击者是一个基于规则的优化器，会尝试同义改写、角色扮演框架设定和编码转换。裁判大语言模型（Judge LLM）负责对响应进行评分。你将观察到，攻击者在面对关键词过滤器时，约在 5-15 次迭代内即可成功，但在面对语义过滤器（semantic filter）时则会失败。

## 交付成果

本实验将生成 `outputs/skill-attack-audit.md`。给定一份红队评估报告（red-team evaluation report）后，该脚本将审计以下内容：运行了哪些攻击方法（PAIR、GCG、TAP、AutoDAN、PAP），各自的查询预算（query budget）是多少，使用了哪个裁判模型，以及基于哪个有害行为数据集（harmful-behaviour set）（JailbreakBench、HarmBench 或内部数据集）。

## 练习

1. 运行 `code/main.py`。测量三种内置攻击策略的平均成功查询次数（mean-queries-to-success）。解释每种策略分别利用了目标防御机制的哪项假设。
2. 实现第四种攻击策略（例如翻译为其他语言、Base64 编码）。报告该策略针对关键词过滤目标和语义过滤目标的新平均成功查询次数。
3. 阅读 Chao 等人 2023 年论文的图 5（PAIR 与 GCG 对比）。描述尽管 PAIR 具有效率优势，但在实际中仍优先选择 GCG 的两种场景。
4. JailbreakBench 报告的是针对固定目标集的攻击成功率（Attack Success Rate, ASR）。请设计一个额外的指标来衡量攻击多样性（成功提示词的方差）。解释为何多样性对防御评估至关重要。
5. TAP（Mehrotra 2024）在 PAIR 的基础上引入了分支与剪枝（branching + pruning）机制。请为 `code/main.py` 草拟一个 TAP 风格的扩展方案，并描述计算成本与成功率之间的权衡关系。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| PAIR | “自动化越狱” | 提示词自动迭代优化（Prompt Automatic Iterative Refinement）；攻击者大语言模型（Attacker-LLM）与裁判大语言模型（Judge-LLM）的交互循环 |
| GCG | “梯度越狱” | 针对对抗性后缀的白盒词元级（token-level）梯度搜索 |
| 攻击成功率（ASR） | “k 次查询内的越狱百分比” | 核心评估指标；报告时必须附带查询预算和裁判模型身份 |
| 裁判大语言模型（Judge LLM） | “评分器” | 用于评估响应是否满足有害目标的大语言模型 |
| JailbreakBench | “标准评估集” | 带有分类标签的标准化有害行为数据集 |
| HarmBench | “更广泛的基准测试” | 包含 510 种行为，涵盖功能性与语义性危害测试 |
| TAP | “攻击树” | 引入分支与剪枝机制的 PAIR；在更高算力下可获得更优的 ASR |

## 延伸阅读

- [Chao 等人 — 二十次查询内越狱黑盒大语言模型 (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — PAIR 论文，NeurIPS 2023
- [Zou 等人 — 针对对齐大语言模型的通用且可迁移对抗攻击 (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — GCG 论文
- [Chao 等人 — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — 标准化评估基准
- [Mazeika 等人 — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — 更广泛的评估基准