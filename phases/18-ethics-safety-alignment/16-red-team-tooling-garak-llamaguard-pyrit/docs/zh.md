# 红队工具（Red-Team Tooling）—— Garak、Llama Guard、PyRIT

> 三款生产级工具构成了 2026 年的红队技术栈（Red-Team Stack）。Llama Guard（Meta）—— 基于 14 个 MLCommons 危害类别（Hazard Categories）微调的 Llama-3.1-8B 分类器；2025 年发布的 Llama Guard 4 是从 Llama 4 Scout 剪枝而来的 12B 原生多模态分类器。Garak（NVIDIA）—— 开源的大语言模型（LLM）漏洞扫描器，内置针对幻觉（Hallucination）、数据泄露（Data Leakage）、提示词注入（Prompt Injection）、毒性内容（Toxicity）和越狱（Jailbreak）的静态、动态与自适应探针（Probes）。PyRIT（Microsoft）—— 支持多轮红队演练（Multi-turn Red-Team Campaigns）的框架，集成 Crescendo、TAP 及自定义转换器链（Converter Chains）以实现深度漏洞利用。Llama Guard 3 的相关文档见 Meta 的《Llama 3 Herd of Models》（arXiv:2407.21783）；Llama Guard 3-1B-INT4 见 arXiv:2411.17713；Garak 的探针架构见 github.com/NVIDIA/garak。这些工具构成了 2026 年连接红队研究（第 12-15 课）与生产部署（第 17 课及以后）的生产级接口。

**类型：** 构建（Build）
**语言：** Python（标准库、工具架构模拟器及 Llama Guard 风格分类器模拟程序）
**前置条件：** 第 18 阶段 · 第 12-15 课（越狱与 IPI）
**时长：** 约 75 分钟

## 学习目标

- 阐述 Llama Guard 3/4 在安全栈（Safety Stack）中的定位：输入分类器、输出分类器，或两者兼有。
- 列举 14 个 MLCommons 危害类别，并指出其中一个非显而易见的类别（代码解释器滥用，Code Interpreter Abuse）。
- 描述 Garak 的探针架构（Probe Architecture）：探针（Probes）、检测器（Detectors）与执行框架（Harnesses）。
- 描述 PyRIT 的多轮演练结构（Multi-turn Campaign Structure）及其如何与 Garak 探针组合使用。

## 问题背景

第 12-15 课介绍了攻击面（Attack Surface）。生产环境部署需要可重复、可扩展的评估机制。2026 年占据主导地位的三款工具分别是：Llama Guard（防御分类器）、Garak（扫描器）和 PyRIT（演练编排器）。它们分别针对红队生命周期（Red-Team Lifecycle）中的不同层级。

## 核心概念

### Llama Guard（Meta）

Llama Guard 3 是基于 Llama-3.1-8B 模型微调而来的，专门用于针对 MLCommons AILuminate 的 14 个类别进行输入/输出分类：
- 暴力犯罪、非暴力犯罪、性相关、儿童性虐待材料（CSAM）、诽谤
- 专业建议、隐私、知识产权（IP）、无差别武器、仇恨言论
- 自杀/自残、色情内容、选举、代码解释器滥用

支持 8 种语言。使用方式：可部署在大语言模型（LLM）之前（输入审核）、之后（输出审核），或同时部署。这两种用途会产生不同的训练数据分布——Llama Guard 3 以单一模型的形式发布，同时兼顾这两种任务。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，在移动端 CPU 上约 30 tokens/s）是其面向边缘设备的量化版本。

Llama Guard 4（2025 年 4 月发布）参数量为 12B，原生支持多模态，由 Llama 4 Scout 剪枝而来。它通过一个同时接收文本和图像的分类器，取代了此前 8B 的文本模型和 11B 的视觉模型。

### Garak（NVIDIA）

开源漏洞扫描工具。架构如下：
- **探针（Probes）**。用于生成针对幻觉、数据泄露、提示词注入、毒性内容及越狱攻击的测试用例。分为静态（固定提示词）、动态（生成提示词）和自适应（根据目标输出进行响应）三种类型。
- **检测器（Detectors）**。根据预期的失败模式（如毒性、泄露、越狱）对模型输出进行评分。
- **测试框架（Harnesses）**。管理探针与检测器的配对，执行测试活动，并生成报告。

TrustyAI 将 Garak 与 Llama-Stack 防护盾（Prompt-Guard-86M 输入分类器、Llama-Guard-3-8B 输出分类器）集成，实现端到端的防护目标评估。基于等级的评分（Tier-based scoring, TBSA）取代了传统的二元通过/失败判定——同一探针测试下，模型可能在严重等级 3 通过，但在严重等级 5 失败。

### PyRIT（Microsoft）

Python 风险识别工具包（Python Risk Identification Toolkit）。专注于多轮红队测试活动。核心架构围绕以下组件构建：
- **转换器（Converters）**。对种子提示词进行变换——包括改写、编码、翻译和角色扮演。
- **编排器（Orchestrators）**。执行测试活动：包括 Crescendo（逐步升级）、TAP（分支策略）和 RedTeaming（自定义循环）。
- **评分（Scoring）**。采用大语言模型作为裁判（LLM-as-judge）或分类器作为裁判（classifier-as-judge）。

PyRIT 可视为 Garak 的“重型”版本。Garak 运行成千上万的单轮探针测试；而 PyRIT 则执行深度的多轮测试活动，旨在突破特定的失败模式。

### 技术栈组合

在模型两侧部署 Llama Guard。每晚运行 Garak 进行回归测试。在发布前使用 PyRIT 执行测试活动。这是 2026 年大多数生产环境部署的默认配置。

### 评估陷阱

- **裁判身份（Judge identity）**。这三款工具均可使用大语言模型作为裁判；裁判的校准程度会直接影响报告的攻击成功率（ASR，见第 12 课）。在使用工具时需明确指定所用的裁判模型。
- **探针过时（Probe staleness）**。随着模型针对特定探针进行修复，Garak 的探针会逐渐失效。自适应探针（如 PAIR 架构）的失效速度慢于静态探针。
- **Llama Guard 对良性内容的误报率（FPR）**。早期版本的 Llama Guard 会过度标记政治和 LGBTQ+ 相关内容；Llama Guard 3/4 的校准已有所改进，但尚未针对具体部署场景进行单独校准。

### 在第 18 阶段中的定位

第 12-15 课涵盖攻击家族分类。第 16 课介绍生产级工具链。第 17 课（WMDP）聚焦双重用途能力的评估。第 18 课则探讨前沿安全框架，将这些工具整合到策略架构中。

## 使用它

`code/main.py` 构建了一个玩具级的 Llama Guard 风格分类器（基于 14 个类别的关键词与语义特征）、一个玩具级的 Garak 测试框架（harness，即探测-检测循环），以及一个 PyRIT 风格的多轮转换器链（converter chain）。你可以针对模拟目标运行这三个工具，并观察它们不同的覆盖特征（coverage signatures）。

## 交付它

本课时将生成 `outputs/skill-red-team-stack.md`。给定部署描述后，它会指明这三个工具中哪些适用、各工具需配置哪些参数，以及应以何种节奏（regression cadence）执行回归测试。

## 练习

1. 运行 `code/main.py`。比较 Llama Guard 风格分类器在单轮攻击与多轮攻击上的检测率。

2. 实现一个新的 Garak 探针（probe）：一个经过 base64 编码的有害请求。测量 Llama Guard 风格分类器对其的检测情况。

3. 在 PyRIT 风格转换器链中扩展一个“先翻译为法语，再进行改写”的转换器。重新测量攻击成功率。

4. 阅读 Llama Guard 3 的危害类别（hazard-category）列表。找出两个类别，在这些类别中，训练数据在现实情况下会对合法的开发者内容产生较高的误报率（false-positive rates）。

5. 比较 Garak 和 PyRIT 的设计原则。论证在何种部署场景下，各自才是正确的工具选择。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| Llama Guard | “分类器” | 经过微调的 Llama-3.1-8B/4-12B 安全分类器，涵盖 14 个危害类别（hazard categories） |
| Garak | “扫描器” | NVIDIA 开源漏洞扫描器；包含探针（probes）、检测器（detectors）和测试框架（harnesses） |
| PyRIT | “演练工具” | 微软多轮红队编排器（orchestrator）；包含转换器（converters）、编排器和评分模块 |
| Prompt-Guard | “小型分类器” | Meta 的 86M 参数提示注入（prompt-injection）分类器，与 Llama Guard 搭配使用 |
| TBSA | “基于层级的评分” | Garak 采用的基于层级的通过/失败机制（tier-based scoring），用于替代二元结果 |
| Converter chain | “改写 + 编码 + ...” | PyRIT 的组合原语（composition primitive），用于构建多步骤攻击 |
| MLCommons hazard categories | “14 种分类法” | Llama Guard 所针对的行业标准分类体系 |

## 延伸阅读

- [Meta — Llama Guard 3（出自 Llama 3 Herd 论文，arXiv:2407.21783）](https://arxiv.org/abs/2407.21783) — 8B 参数分类器
- [Meta — Llama Guard 3-1B-INT4（arXiv:2411.17713）](https://arxiv.org/abs/2411.17713) — 量化移动端分类器
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — 扫描器代码库与文档
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — 演练工具包